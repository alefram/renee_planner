#!/usr/bin/env python3
"""Boat-like base navigation around the machine (no ROS).

In boat mode the base drives forward along a closed loop of straight lanes
with circular arcs at the corners (``LoopPath``): heading along the path,
turning only while driving, never in place. This module holds everything
about that motion that does not need ROS:

- ``LoopPath``: the sampled loop (x, y, heading), stretches of it for
  Nav2's FollowPath, and progress along it.
- ``BasePlanner``: where the base stops so the arm reaches every target
  (a base pose on the loop has one free variable, the distance along it;
  a straight sideways shift towards the machine is only a fallback). For
  each target ``{frame, position, look_at}`` it samples the loop, solves
  the arm IK locally (damped least squares on the Pinocchio model, base
  fixed) and keeps the spots where the frame reaches the pose within the
  joint limits with the arm clear of the collision boxes, scored by
  manipulability and joint margin; then it groups the targets into the
  fewest stops along the driving direction (greedy interval stabbing), so
  the base drives the loop once, forward.
- ``LocalizationMonitor``: localize first, then move (stable correction,
  inside the workspace, no jumps while driving).
- ``gate_twist``: the straight, slow base motion to a spot, aligned with
  the lane, used for arrival and the sideways shift.

The defect detection mission (no HQP, no ``BasePlanner``) uses three rules,
one function each:

- boat mode: ``LoopPath.boat_path`` (forward stretch of the loop, never
  turning in place), the loop itself generated from the machine's boxes
  (``loop_from_boxes``);
- clearance: ``footprint_clearance`` / ``clearance_ok``, the base footprint
  never closer than ``min_clearance_m`` to the machine;
- lateral omni approach, only when needed: ``base_candidates`` (loop spots,
  offset 0 first, sideways offsets as a fallback) and
  ``lateral_approach_twist`` (stops when the next step breaks the clearance).
"""
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..hqp.tasks import aim_error, camera_orientation


def wrap_angle(angle: float) -> float:
    """Output: angle wrapped to (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


# -----------------------------------------------------------------------------
# The loop
# -----------------------------------------------------------------------------

class LoopPath:
    """Rounded-corner closed polyline sampled every `spacing` meters."""

    def __init__(self, corners: Sequence[Sequence[float]], radius: float, spacing: float = 0.05) -> None:
        """Sample the loop.

        Input:
            corners: lane intersections (x, y) in driving order, >= 3 (the
                loop closes from the last back to the first).
            radius: arc radius (m) at every corner; must fit in the shorter
                adjacent lane halves.
            spacing: distance (m) between consecutive samples.

        Output:
            None.

        Raises:
            ValueError: if there are fewer than 3 corners or a radius does
                not fit a lane.
        """
        corners = np.asarray(corners, dtype=float)

        if len(corners) < 3:
            raise ValueError("loop_path needs at least 3 corners")

        n = len(corners)
        arcs = []  # (arc start, arc end, center, start angle, signed sweep)
        for i in range(n):
            prev, here, nxt = corners[i - 1], corners[i], corners[(i + 1) % n]
            d_in = (here - prev) / np.linalg.norm(here - prev)
            d_out = (nxt - here) / np.linalg.norm(nxt - here)
            if radius > min(np.linalg.norm(here - prev), np.linalg.norm(nxt - here)) / 2:
                raise ValueError(f"loop_path radius {radius} does not fit at corner {i}")
            turn = math.atan2(d_in[0] * d_out[1] - d_in[1] * d_out[0], float(d_in @ d_out))
            # Tangent points at distance r*tan(|turn|/2) from the corner.
            offset = radius * math.tan(abs(turn) / 2)
            start, end = here - offset * d_in, here + offset * d_out
            normal = np.array([-d_in[1], d_in[0]]) * math.copysign(1.0, turn)  # towards the center
            center = start + radius * normal
            arcs.append((start, end, center,
                         math.atan2(start[1] - center[1], start[0] - center[0]), turn))

        points, curvature = [], []
        for i in range(n):
            start, end, center, angle0, turn = arcs[i]
            # Arc at corner i.
            steps = max(2, int(math.ceil(abs(turn) * radius / spacing)))
            for t in np.linspace(0.0, 1.0, steps, endpoint=False):
                a = angle0 + t * turn
                heading = math.atan2(math.sin(a), math.cos(a)) + math.copysign(math.pi / 2, turn)
                points.append((center[0] + radius * math.cos(a), center[1] + radius * math.sin(a), heading))
                curvature.append(math.copysign(1.0 / radius, turn))
            # Straight lane to the next corner's arc.
            nxt_start = arcs[(i + 1) % n][0]
            length = float(np.linalg.norm(nxt_start - end))
            heading = math.atan2(nxt_start[1] - end[1], nxt_start[0] - end[0])
            steps = max(1, int(math.ceil(length / spacing)))
            for t in np.linspace(0.0, 1.0, steps, endpoint=False):
                p = end + t * (nxt_start - end)
                points.append((p[0], p[1], heading))
                curvature.append(0.0)

        self.points = np.array(points)
        self.points[:, 2] = np.arctan2(np.sin(self.points[:, 2]), np.cos(self.points[:, 2]))
        # Signed curvature (1/m, + = left turn) and arc length (m) of each
        # sample; total_length closes the loop back to sample 0.
        self.curvature = np.array(curvature)
        steps = np.hypot(np.diff(self.points[:, 0], append=self.points[0, 0]),
                         np.diff(self.points[:, 1], append=self.points[0, 1]))
        self.arc_length = np.concatenate([[0.0], np.cumsum(steps[:-1])])
        self.total_length = float(np.sum(steps))
        # Straight lane (m) driven since the last arc sample, per sample (0 on an arc):
        # Nav2 needs some straight lane after a corner to settle on it before a stop.
        self.straight_before = np.zeros(len(self.points))

        curved = np.flatnonzero(self.curvature != 0.0)
        if len(curved):
            for index in range(len(self.points)):
                if self.curvature[index] != 0.0:
                    continue
                previous = curved[curved < index].max() if np.any(curved < index) else curved.max()
                self.straight_before[index] = (self.arc_length[index] - self.arc_length[previous]) % self.total_length
        else:
            self.straight_before[:] = np.inf

    def nearest(self, x: float, y: float) -> int:
        """Output: index of the loop sample closest to (x, y)."""
        return int(np.argmin(np.hypot(self.points[:, 0] - x, self.points[:, 1] - y)))

    def forward_distance(self, i: int, j: int) -> float:
        """Output: distance (m) along the loop from sample i forward to sample j."""
        return float((self.arc_length[j] - self.arc_length[i]) % self.total_length)

    def segment(self, start_xy: Sequence[float], goal_xy: Sequence[float]) -> np.ndarray:
        """Stretch of the loop from the sample nearest start to the one nearest goal.

        Input:
            start_xy, goal_xy: (x, y) of the current base and the goal.

        Output:
            np.ndarray (k, 3): (x, y, heading) samples in driving order,
                wrapping around the loop if needed; a single sample when both
                project onto the same one.
        """
        i, j = self.nearest(*start_xy), self.nearest(*goal_xy)
        if j >= i:
            return self.points[i:j + 1]
        return np.vstack([self.points[i:], self.points[:j + 1]])

    def boat_path(self, start_xy: Sequence[float], goal_xy: Sequence[float],
                  max_behind_m: float = 0.6) -> np.ndarray:
        """Forward stretch of the loop (counterclockwise) for Nav2's FollowPath.

        Boat mode: always forward, heading along the path, no turning in
        place. A goal less than max_behind_m behind the start (the base
        overshot it) is not a lap away: the path is that one sample, and the
        caller moves straight along the lane instead (Nav2 only drives forward).

        Input:
            start_xy, goal_xy: (x, y) of the current base and the goal.
            max_behind_m: how far behind the start a goal may be and still count as reached.

        Output:
            np.ndarray (k, 3): (x, y, heading) samples in driving order; a
                single sample when there is nothing to drive forward.
        """
        i, j = self.nearest(*start_xy), self.nearest(*goal_xy)
        if self.total_length - self.forward_distance(i, j) < max_behind_m:
            return self.points[j:j + 1]
        return self.segment(start_xy, goal_xy)


# -----------------------------------------------------------------------------
# Footprint clearance to the machine
# -----------------------------------------------------------------------------

def aabb_boxes(collision_boxes: Sequence[dict]) -> list:
    """Axis-aligned boxes (top view matters) of MoveIt collision objects.

    Input:
        collision_boxes: [{"box": [sx, sy, sz], "position": [x, y, z], optional "yaw"}].

    Output:
        list[(low (3,), high (3,))]; a turned box counts as its bounding box.
    """
    boxes = []
    for b in collision_boxes:
        center, (sx, sy, sz) = np.asarray(b["position"], dtype=float), np.asarray(b["box"], dtype=float)
        c, s = abs(math.cos(float(b.get("yaw", 0.0)))), abs(math.sin(float(b.get("yaw", 0.0))))
        half = np.array([c * sx + s * sy, s * sx + c * sy, sz]) / 2
        boxes.append((center - half, center + half))
    return boxes


def footprint_clearance(x: float, y: float, yaw: float, boxes: Sequence, footprint: Sequence[float] = (1.2, 0.7)) -> float:
    """Least distance (m, top view) from the base footprint at (x, y, yaw) to the boxes.

    Input:
        boxes: [(low, high)] from aabb_boxes().
        footprint: base [length, width] (m), centered on the base frame.

    Output:
        float: 0 when they overlap, inf without boxes.
    """
    length, width = footprint
    c, s = math.cos(yaw), math.sin(yaw)
    u, v = np.meshgrid(np.linspace(-length / 2, length / 2, 25), np.linspace(-width / 2, width / 2, 15))
    u, v = u.ravel(), v.ravel()
    points = np.column_stack([x + c * u - s * v, y + s * u + c * v])
    clearance = float("inf")
    for low, high in boxes:
        gap = np.maximum(np.maximum(low[:2] - points, points - high[:2]), 0.0)
        clearance = min(clearance, float(np.min(np.linalg.norm(gap, axis=1))))
    return clearance


def clearance_ok(pose: Sequence[float], boxes: Sequence, min_clearance_m: float = 0.30,
                 footprint: Sequence[float] = (1.2, 0.7)) -> bool:
    """Output: True if the footprint at pose (x, y, yaw) keeps min_clearance_m from every box."""
    return footprint_clearance(pose[0], pose[1], pose[2], boxes, footprint) >= min_clearance_m


def shifted_pose(pose: Sequence[float], offset: float, side_sign: float) -> tuple:
    """Output: (x, y, yaw) of pose moved `offset` m sideways (side_sign +1: left, -1: right), heading kept."""
    x, y, yaw = pose
    lateral = side_sign * offset
    return (float(x - math.sin(yaw) * lateral), float(y + math.cos(yaw) * lateral), float(yaw))


def loop_corners_from_boxes(boxes: Sequence, clearance_m: float, footprint: Sequence[float] = (1.2, 0.7),
                            margin_m: float = 0.0):
    """Loop corners around the machine: its top-view bounding rectangle grown by d.

    d = clearance_m + footprint width / 2 + margin_m: driving a straight lane
    the footprint's machine-facing side is clearance_m + margin_m from the
    rectangle. A rectangle (convex) because in boat mode, counterclockwise,
    the base only turns left.

    Input:
        boxes: [(low, high)] from aabb_boxes().
        clearance_m: lane distance to the machine (m).
        footprint: base [length, width] (m).
        margin_m: tracking margin of the controller (m).

    Output:
        tuple[list[(x, y)], float]: corners counterclockwise starting at
            (x max, y max), and d.

    Raises:
        ValueError: without boxes.
    """
    if not boxes:
        raise ValueError("the loop is generated from the collision boxes: there are none")
    low = np.min([b[0][:2] for b in boxes], axis=0)
    high = np.max([b[1][:2] for b in boxes], axis=0)
    d = float(clearance_m + footprint[1] / 2 + margin_m)
    x0, y0, x1, y1 = low[0] - d, low[1] - d, high[0] + d, high[1] + d
    return [(float(x1), float(y1)), (float(x0), float(y1)), (float(x0), float(y0)), (float(x1), float(y0))], d


def loop_from_boxes(boxes: Sequence, *, lane_clearance_m: float, min_clearance_m: float,
                    tracking_margin_m: float, max_radius_m: float, footprint: Sequence[float] = (1.2, 0.7),
                    spacing: float = 0.05, workspace: Sequence[float] = None,
                    wall_margin_m: float = 0.0) -> "LoopPath":
    """The boat loop around the machine, checked against the clearance limit and the walls.

    Input:
        boxes: [(low, high)] from aabb_boxes().
        lane_clearance_m, tracking_margin_m: see loop_corners_from_boxes().
        min_clearance_m: hard limit every loop sample must keep.
        max_radius_m: corner arcs have radius min(d, max_radius_m).
        workspace: [x_min, x_max, y_min, y_max] (map) inside the walls, or None.
        wall_margin_m: extra distance from the walls beyond the robot's half width.

    Output:
        LoopPath.

    Raises:
        ValueError: if a loop sample is closer than min_clearance_m to the
            machine, or the loop does not fit the workspace (says by how many cm).
    """
    corners, d = loop_corners_from_boxes(boxes, lane_clearance_m, footprint, tracking_margin_m)
    loop = LoopPath(corners, min(d, float(max_radius_m)), spacing)
    worst = min(footprint_clearance(x, y, yaw, boxes, footprint) for x, y, yaw in loop.points)
    if worst < min_clearance_m:
        raise ValueError(f"the generated loop comes {worst:.3f} m from the machine, under "
                         f"min_clearance_m {min_clearance_m}")
    if workspace is not None:
        inset = footprint[1] / 2 + wall_margin_m
        x_min, x_max, y_min, y_max = workspace
        short = max(x_min + inset - loop.points[:, 0].min(), loop.points[:, 0].max() - (x_max - inset),
                    y_min + inset - loop.points[:, 1].min(), loop.points[:, 1].max() - (y_max - inset))
        if short > 0.0:
            raise ValueError(f"the loop does not fit navigation.workspace: {short * 100:.0f} cm short "
                             f"(robot half width {footprint[1] / 2:.2f} m + wall margin {wall_margin_m:.2f} m); "
                             "no loop keeps the clearance limit here")
    return loop


@dataclass
class BasePlacementConfig:
    """Settings of base_candidates() and lateral_approach_twist()."""
    window_m: float = 0.8                 # +- along the loop from the target's projection
    offsets_m: Sequence[float] = ()       # sideways fallback offsets (0 is always tried first)
    spacing_m: float = 0.05               # loop sampling of the search
    min_straight_before_stop_m: float = 0.65
    min_clearance_m: float = 0.30
    lane_clearance_m: float = 0.40        # the approach may use lane_clearance_m - min_clearance_m
    footprint: Sequence[float] = (1.2, 0.7)
    side_sign: float = 1.0                # +1: machine on the left
    max_speed: float = 0.1                # lateral approach
    max_yaw_rate: float = 0.1
    align_tolerance_rad: float = 0.02
    probe_m: float = 0.05                 # how far ahead the approach checks the clearance

    @property
    def max_offset_m(self) -> float:
        """Output: the sideways travel the clearance budget allows (m)."""
        return max(0.0, self.lane_clearance_m - self.min_clearance_m)


@dataclass
class BaseCandidate:
    """A base spot for a target: the lane sample, the sideways offset and the resulting pose."""
    index: int
    lane_pose: tuple
    offset: float
    pose: tuple


def base_candidates(target: dict, loop: LoopPath, boxes: Sequence, cfg: BasePlacementConfig) -> list:
    """Base spots to try for a target, best first.

    The loop sample nearest the target's XY projection and those within
    +-window_m along the loop (nearest first, forward first), outside the
    corners (min_straight_before_stop_m). All with offset 0 first; only then
    the sideways offsets (offsets_m, at most the clearance budget) as a
    fallback. Every spot keeps min_clearance_m.

    Input:
        target: {"position": (3,)} in the loop's frame.

    Output:
        list[BaseCandidate].
    """
    n = len(loop.points)
    home = loop.nearest(float(target["position"][0]), float(target["position"][1]))
    spacing = loop.total_length / n
    stride = max(1, int(round(cfg.spacing_m / spacing)))
    reach = int(cfg.window_m / spacing)
    ks = sorted(range(-(reach // stride) * stride, reach + 1, stride), key=lambda k: (abs(k), -k))
    indices = [(home + k) % n for k in ks if loop.straight_before[(home + k) % n] >= cfg.min_straight_before_stop_m]
    offsets = [0.0] + [o for o in sorted(float(v) for v in cfg.offsets_m) if 0.0 < o <= cfg.max_offset_m + 1e-9]
    found = []
    for offset in offsets:
        for index in indices:
            lane_pose = tuple(float(v) for v in loop.points[index])
            pose = shifted_pose(lane_pose, offset, cfg.side_sign)
            if clearance_ok(pose, boxes, cfg.min_clearance_m, cfg.footprint):
                found.append(BaseCandidate(index, lane_pose, offset, pose))
    return found


def lateral_approach_twist(base_xy_yaw: Sequence[float], lane_pose: Sequence[float], offset_m: float,
                           boxes: Sequence, cfg: BasePlacementConfig):
    """Omnidirectional twist to the lane spot shifted offset_m towards the machine, lane heading held.

    Input:
        base_xy_yaw: (x, y, yaw) of the base (map).
        lane_pose: (x, y, yaw) of the lane sample, offset 0.
        offset_m: sideways offset to reach (0: back onto the lane).
        boxes: [(low, high)] from aabb_boxes().

    Output:
        (vx, vy, wz) body twist (gate_twist), or None to stop: the next step
            (cfg.probe_m ahead) would break the clearance limit and get closer
            than now (moving away from a violation is always allowed).
    """
    x, y, yaw = base_xy_yaw
    goal = shifted_pose(lane_pose, offset_m, cfg.side_sign)
    distance = math.hypot(goal[0] - x, goal[1] - y)
    if distance > 1e-6:
        step = min(cfg.probe_m, distance)
        ahead = (x + (goal[0] - x) / distance * step, y + (goal[1] - y) / distance * step, yaw)
        if not clearance_ok(ahead, boxes, cfg.min_clearance_m, cfg.footprint) and \
                footprint_clearance(*ahead, boxes, cfg.footprint) < footprint_clearance(x, y, yaw, boxes, cfg.footprint):
            return None
    return gate_twist((x, y, yaw), goal[:2], wrap_angle(lane_pose[2] - yaw), cfg.max_speed,
                      cfg.max_yaw_rate, cfg.align_tolerance_rad)


# -----------------------------------------------------------------------------
# Base placement
# -----------------------------------------------------------------------------

# Arm frames whose origins (and the segments between consecutive ones) are
# checked against the collision boxes, with ARM_RADIUS_M of clearance.
ARM_FRAMES = ("robot_arm_shoulder_link", "robot_arm_upper_arm_link", "robot_arm_forearm_link",
              "robot_arm_wrist_1_link", "robot_arm_wrist_2_link", "robot_arm_wrist_3_link",
              "robot_arm_tool0")
ARM_RADIUS_M = 0.07
# Wrist camera housing, checked as one more point (MoveIt rejects IK solutions
# where it touches the machine even when the links are clear).
CAMERA_BODY_FRAME = "robot_arm_rgbd_camera_base_link"


@dataclass
class Target:
    """A frame pose to reach: frame origin at `position`, +Z aimed at `look_at`.

    roll: image rotation about the axis (rad, see camera_orientation); None
    leaves it free (5 DoF, e.g. a nozzle).
    """
    frame: str
    position: np.ndarray
    look_at: np.ndarray
    roll: float = 0.0
    name: str = ""
    down: np.ndarray = None   # image y reference (camera_orientation), None = world down


@dataclass
class Candidate:
    """A base spot from which a target is reachable."""
    index: int          # loop sample
    offset: float       # sideways shift towards the machine (m)
    score: float
    q: np.ndarray       # arm IK solution


@dataclass
class Stop:
    """A base stop and the targets reached from it, in driving order."""
    index: int
    offset: float
    pose: tuple          # (x, y, yaw) of the base
    distance: float      # along the loop from the start (m)
    targets: list = field(default_factory=list)   # target indices
    q: dict = field(default_factory=dict)          # target index -> arm IK solution


class BasePlanner:
    """Sample the loop, find reachable spots per target and group them into stops."""

    def __init__(self, robot, loop: LoopPath, *, collision_boxes: Sequence[dict] = (),
                 seeds: Sequence[Sequence[float]] = (), machine_side: str = "right",
                 offsets_m: Sequence[float] = (0.0, 0.1, 0.2, 0.3), max_reach_m: float = 0.95,
                 position_tolerance_m: float = 0.005, angle_tolerance_rad: float = 0.01,
                 joint_margin_rad: float = 0.05, ik_iterations: int = 200,
                 footprint: Sequence[float] = (1.2, 0.7), base_clearance_m: float = 0.1,
                 shoulder_pan_range_rad: Optional[Sequence[float]] = None,
                 min_straight_before_stop_m: float = 0.0) -> None:
        """Configure the search.

        Input:
            robot: RobotModel (with every target frame, e.g. the camera or
                nozzle); its state is overwritten while planning.
            loop: LoopPath the base drives, in the targets' frame.
            collision_boxes: [{"box": [sx, sy, sz], "position": [x, y, z], optional "yaw"}]
                (MoveIt collision_objects), same frame; a turned box counts
                as its axis-aligned bounding box.
            seeds: arm configurations to start the IK from (e.g. travel_arm_q).
            machine_side: "right" or "left", the robot side facing the machine
                (where the fallback shift goes).
            offsets_m: sideways shifts tried in order; the first that reaches
                a target at some spot is used for it (0 = on the loop).
            max_reach_m: targets farther than this plus the target frame's
                distance from tool0 (e.g. a nozzle tip, a camera) from the
                shoulder are skipped without IK; depends on the arm (the
                UR5e reaches 0.85 m to the flange, the UR15 1.3 m).
            position_tolerance_m, angle_tolerance_rad: IK convergence.
            joint_margin_rad: required distance from the joint limits.
            ik_iterations: max IK iterations per seed.
            footprint: base [length, width] (m), centered on the base frame.
            base_clearance_m: least distance (m) between the base footprint
                and the collision boxes (seen from above) at a stop shifted
                off the loop.
            shoulder_pan_range_rad: [min, max] shoulder pan (rad, in
                [-pi, pi]) of the accepted solutions, or None for any: the pan
                range the arm reaches from its travel posture without sweeping
                through the robot (e.g. a UR15's upper arm and the tool changer).
            min_straight_before_stop_m: least straight lane (m) driven after a
                corner before a stop (spots on an arc or closer to its end are
                rejected): Nav2 leaves a corner off the lane and overshoots a
                goal right after it.

        Output:
            None.
        """
        if machine_side not in ("right", "left"):
            raise ValueError("machine_side must be 'right' or 'left'")
        self.robot = robot
        self.loop = loop
        self.boxes = aabb_boxes(collision_boxes)
        seeds = [np.asarray(s, dtype=float) for s in seeds] or [robot.q[robot.arm_q_slice].copy()]

        # Each seed also mirrored about the base (shoulder pan + pi): the local
        # IK does not swing the arm to the other side from a far seed.
        mirrored = []
        for seed in seeds:
            flipped = seed.copy()
            flipped[0] = np.arctan2(np.sin(seed[0] + np.pi), np.cos(seed[0] + np.pi))
            mirrored.append(flipped)

        self.seeds = seeds + mirrored
        self.side_sign = -1.0 if machine_side == "right" else 1.0
        self.offsets_m = [float(v) for v in offsets_m]
        self.max_reach_m = max_reach_m
        self.shoulder_pan_range_rad = (None if shoulder_pan_range_rad is None
                                       else [float(v) for v in shoulder_pan_range_rad])
        self.position_tolerance_m = position_tolerance_m
        self.angle_tolerance_rad = angle_tolerance_rad
        self.joint_margin_rad = joint_margin_rad
        self.ik_iterations = ik_iterations
        self.limits = robot.joint_limits(robot.arm_joint_names)
        self.arm_frames = [f for f in ARM_FRAMES if robot.model.existFrame(f)]
        self.footprint = [float(v) for v in footprint]
        self.rejected = {}   # target name -> {reason: count} of rejected spots
        self.base_clearance_m = float(base_clearance_m)
        self.min_straight_before_stop_m = float(min_straight_before_stop_m)

    # -- geometry --------------------------------------------------------

    def base_pose(self, index: int, offset: float = 0.0) -> tuple:
        """Output: (x, y, yaw) of the base at loop sample `index`, shifted `offset` m sideways."""
        return shifted_pose(self.loop.points[index], offset, self.side_sign)

    def transit_poses(self, start: Stop, goal: Stop, spacing_m: float = 0.05) -> list:
        """Base poses from one stop to the next: shift back onto the loop, drive, shift out.

        Output:
            list[(x, y, yaw)], every ~spacing_m, in driving order.
        """
        n = len(self.loop.points)
        step = max(1, int(round(spacing_m / (self.loop.total_length / n))))
        poses = [self.base_pose(start.index, o) for o in np.arange(start.offset, 0.0, -spacing_m)]
        span = (goal.index - start.index) % n
        poses += [self.base_pose((start.index + k) % n) for k in range(0, span + 1, step)]
        poses += [self.base_pose(goal.index, o) for o in np.arange(spacing_m, goal.offset + 1e-9, spacing_m)]
        return poses

    def arm_collision_along(self, q_arm, poses, target_frame: str = None):
        """First base pose at which the arm, held at q_arm, comes near a collision box.

        Output:
            (x, y, yaw) | None: None if the arm is clear at every pose.
        """
        for x, y, yaw in poses:
            self.robot.set_base_state(x, y, yaw, 0.0, 0.0, 0.0)
            self.robot.set_arm_state(np.asarray(q_arm, dtype=float))
            if self._collides(target_frame):
                return (x, y, yaw)
        return None

    def footprint_clear(self, x: float, y: float, yaw: float) -> bool:
        """Output: True if the base footprint at (x, y, yaw) keeps base_clearance_m from every box (top view)."""
        return footprint_clearance(x, y, yaw, self.boxes, self.footprint) >= self.base_clearance_m

    def _tool_offset(self, frame: str) -> float:
        """Output: distance (m) from the flange (robot_arm_tool0) to `frame`, cached."""
        cache = self.__dict__.setdefault("_tool_offsets", {})
        if frame not in cache:
            tool0 = self.robot.frame_pose("robot_arm_tool0")
            cache[frame] = float(np.linalg.norm((tool0.inverse() * self.robot.frame_pose(frame)).translation))
        return cache[frame]

    def _collides(self, target_frame: str = None) -> bool:
        """Output: True if the arm comes within ARM_RADIUS_M of a box (current state).

        Checked: the arm links' chain, the wrist camera housing and the
        segment from the flange to target_frame (e.g. a nozzle tip).
        """
        if not self.boxes:
            return False

        points = [np.asarray(self.robot.frame_pose(f).translation) for f in self.arm_frames]
        segments = list(zip(points[:-1], points[1:]))

        if target_frame and self.robot.model.existFrame(target_frame):
            segments.append((points[-1], np.asarray(self.robot.frame_pose(target_frame).translation)))

        samples = [a + t * (b - a) for a, b in segments for t in np.linspace(0.0, 1.0, 5)]
        if self.robot.model.existFrame(CAMERA_BODY_FRAME):
            samples.append(np.asarray(self.robot.frame_pose(CAMERA_BODY_FRAME).translation))

        samples = np.asarray(samples)
        for low, high in self.boxes:
            gap = np.maximum(np.maximum(low - samples, samples - high), 0.0)
            if np.any(np.linalg.norm(gap, axis=1) < ARM_RADIUS_M):
                return True
        return False

    # -- IK ---------------------------------------------------------------

    def _ik(self, target: Target, seed: np.ndarray):
        """Damped least-squares arm IK with the base fixed (current base state).

        Output:
            np.ndarray (6,) or None: converged arm configuration within the
                joint limits (minus joint_margin_rad).
        """
        robot = self.robot
        arm = robot.arm_v_slice
        lower = self.limits.q_lower + self.joint_margin_rad
        upper = self.limits.q_upper - self.joint_margin_rad
        robot.set_arm_state(np.clip(seed, lower, upper))
        goal_rotation = (camera_orientation(target.position, target.look_at, target.roll, target.down)
                         if target.roll is not None else None)

        for _ in range(self.ik_iterations):
            pose = robot.frame_pose(target.frame)
            rotation = np.asarray(pose.rotation)
            position = np.asarray(pose.translation)
            jacobian = robot.frame_jacobian(target.frame)[:, arm]
            position_error = target.position - position
            if goal_rotation is None:
                aim_err, angle = aim_error(rotation, position, target.look_at)
                axes = rotation[:, :2].T
                A = np.vstack([jacobian[:3], axes @ jacobian[3:]])
                e = np.concatenate([position_error, axes @ aim_err])
            else:
                from scipy.spatial.transform import Rotation
                rotation_error = Rotation.from_matrix(goal_rotation @ rotation.T).as_rotvec()
                angle = float(np.linalg.norm(rotation_error))
                A = jacobian
                e = np.concatenate([position_error, rotation_error])
            if np.linalg.norm(position_error) < self.position_tolerance_m and angle < self.angle_tolerance_rad:
                return robot.q[robot.arm_q_slice].copy()
            dq = A.T @ np.linalg.solve(A @ A.T + 1e-3 * np.eye(A.shape[0]), e)
            dq = np.clip(dq, -0.5, 0.5)
            q = np.clip(robot.q[robot.arm_q_slice] + dq, lower, upper)
            robot.set_arm_state(q)
        return None

    def _score(self, target: Target, q: np.ndarray) -> float:
        """Output: manipulability (arm-only, target frame) times the joint-margin factor."""
        robot = self.robot
        robot.set_arm_state(q)
        jacobian = robot.frame_jacobian(target.frame)[:, robot.arm_v_slice]
        manipulability = math.sqrt(max(np.linalg.det(jacobian @ jacobian.T), 0.0))
        margin = float(np.min(np.minimum(q - self.limits.q_lower, self.limits.q_upper - q)))
        return manipulability * min(1.0, margin / 0.3)

    def candidates(self, target: Target, offset: float, stride: int = 1) -> list:
        """Every loop spot (shifted `offset`) from which `target` is reachable.

        Output:
            list[Candidate].
        """
        robot = self.robot
        found = []
        # Why spots were rejected (reported when a target is unreachable).
        rejected = self.rejected.setdefault(target.name, {})
        for index in range(0, len(self.loop.points), stride):
            if self.loop.straight_before[index] < self.min_straight_before_stop_m:
                # On a corner or too soon after it: Nav2 would not settle on the lane.
                rejected["near_corner"] = rejected.get("near_corner", 0) + 1
                continue
            x, y, yaw = self.base_pose(index, offset)
            if offset > 0.0 and not self.footprint_clear(x, y, yaw):
                rejected["footprint"] = rejected.get("footprint", 0) + 1
                continue  # the loop itself is collision-free; a shift off it may not be
            robot.set_base_state(x, y, yaw, 0.0, 0.0, 0.0)
            robot.set_arm_state(self.seeds[0])
            shoulder = np.asarray(robot.frame_pose("robot_arm_shoulder_link").translation)
            if np.linalg.norm(target.position - shoulder) > self.max_reach_m + self._tool_offset(target.frame):
                rejected["out_of_reach"] = rejected.get("out_of_reach", 0) + 1
                continue
            best = None
            for seed in self.seeds:
                q = self._ik(target, seed)
                if q is None:
                    rejected["no_ik"] = rejected.get("no_ik", 0) + 1
                    continue
                if self._collides(target.frame):
                    rejected["arm_collision"] = rejected.get("arm_collision", 0) + 1
                    continue
                if self.shoulder_pan_range_rad is not None:
                    pan = np.arctan2(np.sin(q[0]), np.cos(q[0]))
                    if not self.shoulder_pan_range_rad[0] <= pan <= self.shoulder_pan_range_rad[1]:
                        rejected["shoulder_pan_range"] = rejected.get("shoulder_pan_range", 0) + 1
                        continue
                score = self._score(target, q)
                if best is None or score > best.score:
                    best = Candidate(index, offset, score, q)
            if best is not None:
                found.append(best)
        return found

    # -- grouping ---------------------------------------------------------

    def plan(self, targets: Sequence[Target], start_xy: Sequence[float], stride: int = 1):
        """Group the targets into the fewest stops along the loop, driving forward once.

        Input:
            targets: targets to reach.
            start_xy: where the base starts (the loop is unrolled from its
                nearest sample, in driving order).
            stride: loop sample stride (1 = every `spacing` m).

        Output:
            tuple[list[Stop], list[int]]: stops in driving order, and the
                indices of targets no spot/offset reaches.
        """
        n = len(self.loop.points)
        start = self.loop.nearest(*start_xy)
        spacing = float(np.mean(np.hypot(*np.diff(self.loop.points[:, :2], axis=0).T)))

        def unrolled(index: int) -> int:
            return (index - start) % n

        per_target, unreachable = {}, []
        for t, target in enumerate(targets):
            options = []
            for offset in self.offsets_m:
                options = self.candidates(target, offset, stride)
                if options:
                    break
            if options:
                per_target[t] = {c.index: c for c in options}
            else:
                unreachable.append(t)

        # Each target's first reachable stretch in driving order (consecutive
        # samples, `stride` apart): a target reachable around the start is
        # reached there, not a lap later.
        first_run = {}
        for t, options in per_target.items():
            order = sorted(options, key=unrolled)
            run = [order[0]]
            for index in order[1:]:
                if unrolled(index) - unrolled(run[-1]) > stride:
                    break
                run.append(index)
            first_run[t] = run

        # Greedy stabbing in driving order: the uncovered target whose first
        # stretch ends first sets the stop; among its spots there, take the
        # one reaching most other uncovered targets (same offset), then the
        # best summed score.
        uncovered = set(per_target)
        stops = []
        while uncovered:
            first = min(uncovered, key=lambda t: unrolled(first_run[t][-1]))
            best = None
            for index in first_run[first]:
                candidate = per_target[first][index]
                covered = [t for t in uncovered if index in per_target[t]
                           and per_target[t][index].offset == candidate.offset]
                key = (len(covered), sum(per_target[t][index].score for t in covered))
                if best is None or key > best[0]:
                    best = (key, index, candidate.offset, covered)
            _, index, offset, covered = best
            stop = Stop(index=index, offset=offset, pose=self.base_pose(index, offset),
                        distance=unrolled(index) * spacing,
                        targets=sorted(covered), q={t: per_target[t][index].q for t in covered})
            stops.append(stop)
            uncovered -= set(covered)
        stops.sort(key=lambda s: s.distance)
        return stops, unreachable


# -----------------------------------------------------------------------------
# Localization checks and the base gate
# -----------------------------------------------------------------------------

def correction_change(a: np.ndarray, b: np.ndarray):
    """Output: (distance m, |angle| rad) between two (x, y, yaw) localization corrections."""
    return float(np.hypot(*(a[:2] - b[:2]))), abs(wrap_angle(a[2] - b[2]))


class LocalizationMonitor:
    """Localize first, then move.

    The localization correction is the pose (x, y, yaw) of the odom frame in
    the map frame. Goals are only sent while the localized base is inside
    ``workspace`` and once the correction has stayed within
    ``settle_tolerance`` for ``settle_s``; a change beyond
    ``jump_tolerance`` while driving is a jump (the goal is cancelled and
    retried).
    """

    def __init__(self, workspace: Sequence[float] = None, settle_s: float = 0.0,
                 settle_tolerance: Sequence[float] = (0.05, 0.03),
                 jump_tolerance: Sequence[float] = (0.3, 0.2),
                 timeout_s: float = 60.0) -> None:
        """Input: workspace [x_min, x_max, y_min, y_max] (map) or None; settle_s (s);
        settle_tolerance, jump_tolerance [m, rad]; timeout_s (s) outside the workspace.

        Raises:
            ValueError: on a malformed workspace.
        """
        if workspace is not None and len(workspace) != 4:
            raise ValueError("navigation.workspace must be [x_min, x_max, y_min, y_max]")
        self.workspace = workspace
        self.settle_s = float(settle_s)
        self.settle_tolerance = [float(v) for v in settle_tolerance]
        self.jump_tolerance = [float(v) for v in jump_tolerance]
        self.timeout_s = float(timeout_s)

    def inside(self, base_xy) -> bool:
        """Output: True when no workspace is set or base_xy (map) is inside it."""
        if self.workspace is None:
            return True
        if base_xy is None:
            return False
        x_min, x_max, y_min, y_max = self.workspace
        return x_min <= base_xy[0] <= x_max and y_min <= base_xy[1] <= y_max

    def settled(self, state: dict, correction, now: float, settle_s: float = None) -> bool:
        """Track the correction in `state`; True once stable for settle_s (default self.settle_s)."""
        settle_s = self.settle_s if settle_s is None else settle_s
        if settle_s <= 0.0:
            return True
        if correction is None:
            return False
        reference = state.get("settle_reference")
        if reference is None or any(change > tolerance for change, tolerance in zip(
                correction_change(correction, reference), self.settle_tolerance)):
            state["settle_reference"], state["settle_since"] = correction, now
            return False
        return now - state["settle_since"] >= settle_s

    def jump(self, previous, correction):
        """Output: (m, rad) of a correction change beyond jump_tolerance, else None."""
        if previous is None or correction is None:
            return None
        change = correction_change(correction, previous)
        if any(value > tolerance for value, tolerance in zip(change, self.jump_tolerance)):
            return change
        return None


def gate_twist(base_xy_yaw, goal_xy, heading_error: float, max_speed: float,
               max_yaw_rate: float, align_tolerance_rad: float):
    """Body-frame twist bringing the base straight to goal_xy, aligned with its lane.

    Input:
        base_xy_yaw: (x, y, yaw) of the base (odom).
        goal_xy: (x, y) goal (odom).
        heading_error: lane heading minus base heading (rad).
        max_speed: translation speed bound (m/s).
        max_yaw_rate: heading-correction rate bound (rad/s).
        align_tolerance_rad: no rotation below this heading error.

    Output:
        tuple[float, float, float]: (vx, vy, wz), proportional translation
            (gain 1/s, clipped to max_speed) plus a slow heading correction.
    """
    x, y, yaw = base_xy_yaw
    velocity = np.array([goal_xy[0] - x, goal_xy[1] - y])
    norm = float(np.linalg.norm(velocity))
    if norm > max_speed:
        velocity *= max_speed / norm
    wz = 0.0
    if abs(heading_error) > align_tolerance_rad:
        wz = float(np.clip(heading_error, -max_yaw_rate, max_yaw_rate))
    c, s = math.cos(yaw), math.sin(yaw)
    return (c * velocity[0] + s * velocity[1], -s * velocity[0] + c * velocity[1], wz)
