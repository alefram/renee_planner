#!/usr/bin/env python3
"""HQP task definitions.

Each ``Task`` turns the current robot state into a linear block for one
priority level of the whole-body HQP cascade solved in ``solver.py``. A task
contributes a least-squares row block (``A``, ``b``, ``weight``) and/or an
inequality row block (``C``, ``lower``, ``upper``); ``solver.HQPController``
stacks the blocks of every active task in a level into one ``solver.Level``.

Tasks (the `type` in an experiment's `levels`):

  joint_limits    hard: arm joint positions/velocities within their limits
  base_lane       hard: base locked, held at a point or driving its lane (boat mode)
  camera_view     camera at a position, optical axis aimed at a point
  line_path       a frame (nozzle, camera) along a line on a surface, from a standoff
  posture         arm towards a preferred configuration
  base_velocity   base twist towards a preferred value

Tasks never touch ROS or OSQP directly -- they only read ``RobotModel``.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .robot import RobotModel


@dataclass
class TaskBlock:
    A: np.ndarray = None
    b: np.ndarray = None
    weight: object = None
    C: np.ndarray = None
    lower: np.ndarray = None
    upper: np.ndarray = None
    # Per-level QP regularization this block requires; None means "no
    # opinion" (solver.Level's own default applies). A Jacobian-based task
    # raises it (damped least squares) so velocities stay bounded near
    # kinematic singularities.
    regularization: float = None


class Task(ABC):
    def __init__(self, name: str) -> None:
        """Input: name, identifier for this task instance (logs/reports)."""
        self.name = name
        # Inactive tasks contribute nothing (HQPController skips them); the
        # mission switches its target task on only while it runs.
        self.active = True

    @abstractmethod
    def build(self, robot: "RobotModel", dt: float) -> TaskBlock:
        """Turn the current robot state into this task's linear contribution.

        Input:
            robot: RobotModel holding the current whole-body state.
            dt: control period (s), used by tasks that linearize over a
                horizon (e.g. joint limits) or advance in time (a path).

        Output:
            TaskBlock: the least-squares (A, b, weight) and/or inequality
                (C, lower, upper) rows this task contributes this cycle.
        """
        ...


def camera_orientation(position, look_at, roll: float = 0.0, down=None) -> np.ndarray:
    """Optical-frame rotation (x right, y down, z forward) aimed at look_at.

    The optical axis points straight at look_at (the camera looks at it
    head-on), and roll 0 keeps the image upright: the image y axis points
    as close to `down` (default world down, -Z) as possible, so the image x
    axis is horizontal. Looking straight down that is undefined: give `down`
    another reference then (e.g. the direction of a line under the camera).

    Input:
        position: camera position, (3,).
        look_at: point to aim at, (3,), in the same frame.
        roll: rotation (rad) of the image about the optical axis.
        down: (3,) direction the image y axis should follow; None = -Z.

    Output:
        np.ndarray: (3, 3) rotation of the camera in that frame.
    """
    from scipy.spatial.transform import Rotation

    z = np.asarray(look_at, dtype=float) - np.asarray(position, dtype=float)
    z = z / np.linalg.norm(z)
    x = np.cross(np.array([0.0, 0.0, -1.0]) if down is None else np.asarray(down, dtype=float), z)
    if np.linalg.norm(x) < 1e-6:
        # Looking straight up/down: any axis orthogonal to z will do.
        x = np.cross(np.array([1.0, 0.0, 0.0]), z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z]) @ Rotation.from_rotvec([0.0, 0.0, roll]).as_matrix()


def aim_error(rotation: np.ndarray, position: np.ndarray, look_at: np.ndarray):
    """Rotation error that turns a frame's +Z axis towards look_at.

    Input:
        rotation: frame orientation in the world frame, shape (3, 3).
        position: frame origin in the world frame, shape (3,).
        look_at: point to aim at, shape (3,).

    Output:
        tuple[np.ndarray, float]: (axis * angle, angle), with angle in
            rad between +Z and the direction to look_at.
    """
    z_c = rotation[:, 2]
    direction = look_at - position
    direction = direction / max(np.linalg.norm(direction), 1e-9)
    cross = np.cross(z_c, direction)
    sin_angle = float(np.linalg.norm(cross))
    angle = float(np.arctan2(sin_angle, float(z_c @ direction)))
    if sin_angle > 1e-9:
        axis = cross / sin_angle
    else:
        # Aligned (angle 0, no error) or facing exactly away: any axis
        # orthogonal to z_c works, the frame's own x axis is one.
        axis = rotation[:, 0]
    return axis * angle, angle


def parse_scan_poses(scan_poses: Sequence[dict], label: str = "scan_poses",
                     view_distance: Sequence[float] = None) -> list:
    """Validate YAML scan poses and turn them into numpy pairs.

    Input:
        scan_poses: list of {"position": [x, y, z], "look_at": [x, y, z]}.
        label: name used in error messages (e.g. "stations[0].scan_poses").
        view_distance: optional [min, max] (m) allowed distance from
            position to look_at (the camera's usable depth range); None
            skips the check.

    Output:
        list[tuple[np.ndarray, np.ndarray]]: (position, look_at) per pose.

    Raises:
        ValueError: if an entry is malformed, its position equals look_at,
            or its position-to-look_at distance is outside view_distance.
    """
    poses = []
    for index, pose in enumerate(scan_poses or []):
        try:
            position = np.asarray(pose["position"], dtype=float)
            look_at = np.asarray(pose["look_at"], dtype=float)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{label}[{index}] needs 'position' and 'look_at'") from exc
        if position.shape != (3,) or look_at.shape != (3,):
            raise ValueError(f"{label}[{index}] must use 3-element vectors")
        if np.linalg.norm(look_at - position) < 1e-6:
            raise ValueError(f"{label}[{index}] position equals look_at")
        if view_distance is not None:
            distance = float(np.linalg.norm(look_at - position))
            if not view_distance[0] <= distance <= view_distance[1]:
                raise ValueError(
                    f"{label}[{index}] is {distance:.2f}m from its look_at, outside "
                    f"view_distance_m [{view_distance[0]}, {view_distance[1]}]")
        poses.append((position, look_at))
    return poses


class CameraViewTask(Task):
    """Place a camera optical frame at a position and aim it at a point.

    The target (set_target(), in the model's world frame, odom) is
    ``{position, look_at}``. With ``roll`` set (e.g. 0), the task fixes all 6
    DoF: 3 position rows and 3 orientation rows towards
    ``camera_orientation(position, look_at, roll)``, i.e. the optical axis
    (+Z) pointing straight at ``look_at`` and the image rotated by ``roll``
    (0 = upright). With ``roll: null`` only 5 DoF are fixed: the optical
    axis aim error (aim_error()) along the camera's own x/y axes, leaving
    roll to lower levels. ``view_distance_m`` keeps the camera-to-look_at
    distance in range with a velocity damper (hard inequality).
    """

    def __init__(self, name: str, frame: str, roll: float = 0.0,
                 reach_tolerance_m: float = 0.03, reach_tolerance_rad: float = 0.05,
                 max_dwell_s: float = None, position_gain: float = 1.0,
                 aim_gain: float = 1.0, weight: float = 1.0, damping: float = 1e-2,
                 view_distance_m: Sequence[float] = None,
                 view_distance_gain: float = 1.0) -> None:
        """Configure the camera frame, tolerances and gains.

        Input:
            name: identifier for this task instance.
            frame: camera optical frame in the RobotModel (z forward).
            roll: fixed image rotation (rad) about the optical axis, 0 =
                upright; None leaves roll free (5-DoF task).
            reach_tolerance_m, reach_tolerance_rad: position / aim errors
                below which the target counts as reached.
            max_dwell_s: longest refinement (s) before the mission gives up
                (timeout); None waits forever.
            position_gain, aim_gain: proportional gains.
            weight: least-squares weight for this task's rows.
            damping: damped-least-squares regularization of the level.
            view_distance_m: optional [min, max] (m) camera-to-look_at
                distance (the camera's depth range): targets are checked
                against it when loaded, and it is a hard inequality while
                tracking, k (d_min - d) <= d_dot <= k (d_max - d).
            view_distance_gain: k (1/s) of that damper.

        Output:
            None.

        Raises:
            ValueError: if view_distance_m is not [min, max] with 0 <= min < max.
        """
        super().__init__(name)
        self.frame = frame
        self.roll = None if roll is None else float(roll)
        self.reach_tolerance_m = reach_tolerance_m
        self.reach_tolerance_rad = reach_tolerance_rad
        self.max_dwell_s = max_dwell_s
        self.position_gain = position_gain
        self.aim_gain = aim_gain
        self.weight = weight
        self.damping = damping
        self.view_distance_gain = float(view_distance_gain)
        self.view_distance_m = None
        if view_distance_m is not None:
            self.view_distance_m = [float(v) for v in view_distance_m]
            if len(self.view_distance_m) != 2 or not 0.0 <= self.view_distance_m[0] < self.view_distance_m[1]:
                raise ValueError(f"{name}.view_distance_m must be [min, max] with 0 <= min < max")
        self.target = None
        self.last_camera_position = np.full(3, np.nan)
        self.last_position_error = float("nan")
        self.last_aim_angle = float("nan")
        self.last_view_distance = float("nan")

    def set_target(self, position, look_at) -> None:
        """Set the camera target (model world frame, odom) and forget the last errors.

        Input:
            position, look_at: (3,) target camera position and aim point.
        """
        self.target = (np.asarray(position, dtype=float), np.asarray(look_at, dtype=float))
        self.last_camera_position = np.full(3, np.nan)
        self.last_position_error = float("nan")
        self.last_aim_angle = float("nan")

    def build(self, robot: "RobotModel", dt: float) -> TaskBlock:
        """Compute this cycle's camera position and aim rows.

        Output:
            TaskBlock: 6 least-squares rows (3 position, 3 orientation) with
                a fixed roll, or 5 (3 position, 2 aim) with a free roll, plus
                the view-distance inequality row if configured. Updates
                last_camera_position / last_position_error / last_aim_angle.

        Raises:
            ValueError: without a target.
        """
        if self.target is None:
            raise ValueError(f"{self.name}: no target set")
        position, look_at = self.target
        pose = robot.frame_pose(self.frame)
        jacobian = robot.frame_jacobian(self.frame)
        rotation = np.asarray(pose.rotation)
        camera_position = np.asarray(pose.translation)
        position_error = position - camera_position
        if self.roll is None:
            rotation_error, angle = aim_error(rotation, camera_position, look_at)
            axes = rotation[:, :2].T  # rotation about the optical axis stays free
        else:
            from scipy.spatial.transform import Rotation
            goal = camera_orientation(position, look_at, self.roll)
            rotation_error = Rotation.from_matrix(goal @ rotation.T).as_rotvec()
            angle = float(np.linalg.norm(rotation_error))
            axes = np.eye(3)
        self.last_camera_position = camera_position.copy()
        self.last_position_error = float(np.linalg.norm(position_error))
        self.last_aim_angle = angle

        A = np.vstack([jacobian[:3], axes @ jacobian[3:]])
        b = np.concatenate([self.position_gain * position_error,
                            self.aim_gain * (axes @ rotation_error)])
        block = TaskBlock(A=A, b=b, weight=self.weight, regularization=self.damping)
        offset = camera_position - look_at
        distance = float(np.linalg.norm(offset))
        self.last_view_distance = distance
        if self.view_distance_m is not None and distance > 1e-6:
            # d_dot = u^T J_pos v; velocity damper keeps d in [d_min, d_max].
            d_min, d_max = self.view_distance_m
            block.C = (offset / distance) @ jacobian[:3]
            block.lower = np.array([self.view_distance_gain * (d_min - distance)])
            block.upper = np.array([self.view_distance_gain * (d_max - distance)])
        return block


class JointLimitsTask(Task):
    """Hard inequality: keep joint velocities within position/velocity limits.

    Linearized over the control horizon ``dt``:
    ``(q_min - q)/dt <= dq <= (q_max - q)/dt``, intersected with
    ``[-v_max, v_max]``. While ``locked`` (set by the mission, e.g. a
    pass with the arm in a fixed configuration) the joints do not move
    (dq = 0).
    """

    def __init__(self, name: str, joints: Sequence[str], margin_rad: float = 0.0) -> None:
        """Configure which joints to keep within their position/velocity limits.

        Input:
            name: identifier for this task instance.
            joints: joint names to constrain.
            margin_rad: safety margin (rad) subtracted from each side of the
                URDF position limits before linearizing.

        Output:
            None.
        """
        super().__init__(name)
        self.joints = list(joints)
        self.margin_rad = margin_rad
        self.locked = False

    def build(self, robot: "RobotModel", dt: float) -> TaskBlock:
        """Compute this cycle's linearized joint-velocity bounds.

        Input:
            robot: RobotModel holding the current whole-body state.
            dt: control period (s), used to linearize position limits into
                velocity bounds over this horizon.

        Output:
            TaskBlock: one inequality row per joint in self.joints, mapping
                to that joint's velocity-vector index.
        """
        limits = robot.joint_limits(self.joints)
        current_q = robot.q[robot.configuration_slice(self.joints)]
        n = len(self.joints)
        indices = np.arange(limits.v_slice.start, limits.v_slice.stop)
        C = np.zeros((n, robot.nv))
        C[np.arange(n), indices] = 1.0
        lower = np.maximum((limits.q_lower + self.margin_rad - current_q) / dt, -limits.v_limit)
        upper = np.minimum((limits.q_upper - self.margin_rad - current_q) / dt, limits.v_limit)
        if self.locked:
            lower, upper = np.zeros(n), np.zeros(n)
        return TaskBlock(C=C, lower=lower, upper=upper)


class BaseLaneTask(Task):
    """Hard inequality: the base in boat mode (never rotated in place).

    Three states, set by the mission:

    - locked (no anchor): the base twist is zero;
    - held (anchor + hold point, e.g. while the arm works at a stop): the
      base is pinned at the hold point (velocity damper, gain ``gain``,
      speed <= ``max_speed``) with its heading on the anchor's lane heading
      (slow correction <= ``max_yaw_rate`` beyond ``align_tolerance_rad``,
      else wz = 0);
    - pass (start_pass(), e.g. a cleaning pass): the base only drives
      along its lane, forward (vx in [min_forward_speed,
      max_forward_speed]) or, with reverse, backward (the same range
      negated); vy = 0 and wz the same heading correction (no turning).

    The mission moves the base between stops with Nav2 and shifts it
    sideways with shifted(): a straight move of up to ``max_offset_m``
    towards the machine (``side``: the robot side facing it). ``max_speed``,
    ``align_tolerance_rad`` and ``static_speed``/``static_time_s`` also set
    that shift and the "aligned + static" check before the arm moves.
    """

    def __init__(self, name: str, side: str = "right", max_offset_m: float = 0.3,
                 max_speed: float = 0.1, gain: float = 1.0, max_yaw_rate: float = 0.1,
                 align_tolerance_rad: float = 0.02, static_speed: float = 0.01,
                 static_time_s: float = 0.5, max_forward_speed: float = 0.1,
                 min_forward_speed: float = 0.0) -> None:
        """Configure the base limits.

        Input:
            name: identifier for this task instance.
            side: "right" or "left", the robot side facing the machine.
            max_offset_m: largest sideways shift (m) towards the machine.
            max_speed: speed bound (m/s) of the hold and the shift.
            gain: 1/s, hold damper and heading correction.
            max_yaw_rate: largest heading-correction rate (rad/s).
            align_tolerance_rad: heading error below which the base counts
                as aligned with its lane (no correction).
            static_speed, static_time_s: the base counts as static when its
                measured speeds stay below static_speed for static_time_s.
            max_forward_speed, min_forward_speed: vx range (m/s) during a
                pass (min 0: forward only, the base may stop but never back up).

        Output:
            None.

        Raises:
            ValueError: on a bad side or min_forward_speed > max_forward_speed.
        """
        super().__init__(name)
        if side not in ("right", "left"):
            raise ValueError(f"{name}: side must be 'right' or 'left'")
        if min_forward_speed > max_forward_speed:
            raise ValueError(f"{name}: min_forward_speed must be <= max_forward_speed")
        self.side_sign = -1.0 if side == "right" else 1.0  # lateral axis = base +y (left)
        self.max_offset_m = float(max_offset_m)
        self.max_speed = float(max_speed)
        self.gain = float(gain)
        self.max_yaw_rate = float(max_yaw_rate)
        self.align_tolerance_rad = float(align_tolerance_rad)
        self.static_speed = float(static_speed)
        self.static_time_s = float(static_time_s)
        self.max_forward_speed = float(max_forward_speed)
        self.min_forward_speed = float(min_forward_speed)
        self.anchor = None  # (x, y, yaw) in odom; yaw = lane heading
        self.hold = None    # (x, y) in odom
        self.pass_active = False
        self.pass_reverse = False

    def set_anchor(self, x: float, y: float, yaw: float) -> None:
        """Set the lane reference: a point (odom) and the lane heading."""
        self.anchor = (float(x), float(y), float(yaw))

    def set_hold(self, x: float, y: float) -> None:
        """Pin the base at (x, y) (odom), e.g. while the arm works."""
        self.hold = (float(x), float(y))

    def clear_hold(self) -> None:
        """Release the pin (the base is locked unless a pass runs)."""
        self.hold = None

    def start_pass(self, reverse: bool = False) -> None:
        """Let the base drive along the anchor's lane: forward, or backward
        (reverse: vx in [-max_forward_speed, -min_forward_speed], still
        without rotating), e.g. to come back along a lane without turning."""
        if self.anchor is None:
            raise ValueError(f"{self.name}: start_pass needs an anchor (lane heading)")
        self.hold = None
        self.pass_active = True
        self.pass_reverse = bool(reverse)

    def stop_pass(self) -> None:
        """End the pass (the base is locked, or held once set_hold() is called)."""
        self.pass_active = False

    def heading_error(self, robot: "RobotModel") -> float:
        """Output: lane heading minus base heading (rad, wrapped); 0 without anchor."""
        if self.anchor is None:
            return 0.0
        _, _, c, s = robot.q[robot.base_q_slice]
        error = self.anchor[2] - float(np.arctan2(s, c))
        return float(np.arctan2(np.sin(error), np.cos(error)))

    def shifted(self, offset_m: float):
        """Output: (x, y, yaw) in odom of the anchor shifted offset_m towards the machine."""
        x, y, yaw = self.anchor
        lateral = self.side_sign * offset_m
        return (x - np.sin(yaw) * lateral, y + np.cos(yaw) * lateral, yaw)

    def _heading_rate(self, robot: "RobotModel") -> float:
        """Output: wz (rad/s) correcting the heading towards the lane, 0 when aligned."""
        error = self.heading_error(robot)
        if abs(error) <= self.align_tolerance_rad:
            return 0.0
        return float(np.clip(self.gain * error, -self.max_yaw_rate, self.max_yaw_rate))

    def build(self, robot: "RobotModel", dt: float) -> TaskBlock:
        """Compute this cycle's bounds on the base twist (vx, vy, wz, body frame).

        Output:
            TaskBlock: 3 inequality rows on the base's velocity slice.
        """
        vs = robot.base_v_slice
        C = np.zeros((3, robot.nv))
        C[np.arange(3), np.arange(vs.start, vs.stop)] = 1.0
        if self.anchor is None or (self.hold is None and not self.pass_active):
            return TaskBlock(C=C, lower=np.zeros(3), upper=np.zeros(3))
        wz = self._heading_rate(robot)
        if self.pass_active:
            low, high = self.min_forward_speed, self.max_forward_speed
            if self.pass_reverse:
                low, high = -high, -low
            return TaskBlock(C=C, lower=np.array([low, 0.0, wz]), upper=np.array([high, 0.0, wz]))
        # Held: body twist towards the hold point, clipped to max_speed.
        x, y, c, s = robot.q[robot.base_q_slice]
        error = np.array([self.hold[0] - x, self.hold[1] - y])
        velocity = self.gain * np.array([c * error[0] + s * error[1], -s * error[0] + c * error[1]])
        norm = float(np.linalg.norm(velocity))
        if norm > self.max_speed:
            velocity *= self.max_speed / norm
        twist = np.array([velocity[0], velocity[1], wz])
        return TaskBlock(C=C, lower=twist, upper=twist.copy())


class LinePathTask(Task):
    """Move a frame (a nozzle, a camera) along a straight line over a surface.

    The aim point runs along the line ``a(s) = start + s u`` (``u`` unit,
    ``s`` in [0, L]). The frame's origin is kept at ``n(s) = a(s) + standoff
    w``, where ``w`` is the unit vector from the line towards the robot's
    side, tilted ``tilt_deg`` up from the horizontal (90: straight above the
    line), and its +Z axis aims at ``a(s)``. With ``roll: null`` the rotation
    about that axis is left free (5 DoF, e.g. a nozzle); with a roll (e.g. 0
    for a camera) the orientation is ``camera_orientation(n, a, roll,
    image_down)`` (6 DoF); for a steep view (``tilt_deg`` >= 60) the image y
    axis follows the line direction instead of the undefined "world down".

    ``s`` advances by itself at ``speed`` (with a linear ramp of ``ramp_s``
    at the start and a slow-down of the same length before the end), scaled
    down linearly as the position error grows and stopped at
    ``slow_error_m``: the frame may fall behind in time but not leave the
    line. The rows are ``J_pos v = s_dot u + kp (n - p)`` plus the aim rows.
    An inequality keeps the frame at least ``min_clearance_m`` above the line
    (velocity damper).
    """

    def __init__(self, name: str, frame: str, standoff_m: float = 0.2, tilt_deg: float = 45.0,
                 speed: float = 0.05, ramp_s: float = 3.0, slow_error_m: float = 0.05,
                 min_clearance_m: float = 0.08, clearance_gain: float = 1.0,
                 position_gain: float = 2.0, aim_gain: float = 2.0, roll: float = None,
                 weight: float = 1.0, damping: float = 1e-2) -> None:
        """Configure the frame geometry and the path speed.

        Input:
            name: identifier for this task instance.
            frame: tracked frame in the RobotModel, +Z = the axis aimed at the line.
            standoff_m: frame origin to aim point distance (m).
            tilt_deg: angle (deg) of the frame-to-aim direction above the
                horizontal (90 = straight down).
            speed: path speed (m/s) along the line.
            ramp_s: acceleration/deceleration time (s) of the path speed.
            slow_error_m: position error (m) at which the path stops
                advancing (linear slow-down from 0).
            min_clearance_m: minimum height (m) of the frame above the line.
            clearance_gain: velocity-damper gain (1/s) of that bound.
            position_gain, aim_gain: proportional gains.
            roll: rotation (rad) about +Z, as in camera_orientation(); None
                leaves it free.
            weight: least-squares weight for this task's rows.
            damping: damped-least-squares regularization of the level.

        Output:
            None.

        Raises:
            ValueError: on non-positive standoff/speed/slow_error_m or a tilt
                outside (0, 90].
        """
        super().__init__(name)
        if standoff_m <= 0.0 or speed <= 0.0 or slow_error_m <= 0.0:
            raise ValueError(f"{name}: standoff_m, speed and slow_error_m must be > 0")
        if not 0.0 <= tilt_deg <= 90.0:
            raise ValueError(f"{name}: tilt_deg must be in [0, 90]")
        self.frame = frame
        # Defaults; a path may set its own (set_path()).
        self.default_standoff_m = float(standoff_m)
        self.default_tilt = np.radians(float(tilt_deg))
        self.standoff_m = self.default_standoff_m
        self.tilt = self.default_tilt
        self.speed = float(speed)
        self.ramp_s = float(ramp_s)
        self.slow_error_m = float(slow_error_m)
        self.min_clearance_m = float(min_clearance_m)
        self.clearance_gain = float(clearance_gain)
        self.position_gain = float(position_gain)
        self.aim_gain = float(aim_gain)
        self.roll = None if roll is None else float(roll)
        self.weight = float(weight)
        self.damping = float(damping)
        self._start = None
        # While paused the path does not advance (the frame holds s).
        self.paused = False
        # With the arm fixed only the base (along the lane) moves the frame:
        # the error across the line cannot be corrected, so only the error
        # along it slows the path down.
        self.along_only = False
        self.clear_path()

    def clear_path(self) -> None:
        """Forget the path (build() then raises)."""
        self._start = None
        self.length = 0.0
        self.s = 0.0
        self.s_dot = 0.0
        self.elapsed = 0.0
        self.last_frame_position = np.full(3, np.nan)
        self.last_target = np.full(3, np.nan)
        self.last_aim_point = np.full(3, np.nan)
        self.last_position_error = float("nan")
        self.last_cross_track = float("nan")
        self.last_standoff = float("nan")
        self.last_aim_angle = float("nan")

    def set_path(self, start, end, robot_side_xy, standoff_m: float = None,
                 tilt_deg: float = None) -> None:
        """Set the aim line (model world frame, odom) and restart at s = 0.

        Input:
            start, end: (3,) aim line end points (m).
            robot_side_xy: (2,) a point on the robot's side of the line (e.g.
                the base position); the frame is offset towards it.
            standoff_m, tilt_deg: this path's geometry (None: the task's
                defaults); tilt 0 = horizontal (e.g. at a vertical face).

        Output:
            None.

        Raises:
            ValueError: if the line is shorter than 1 cm or vertical.
        """
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length < 0.01:
            raise ValueError(f"{self.name}: path shorter than 1 cm")
        u = direction / length
        side = np.cross([0.0, 0.0, 1.0], u)[:2]
        if np.linalg.norm(side) < 1e-6:
            raise ValueError(f"{self.name}: vertical paths are not supported")
        side /= np.linalg.norm(side)
        if side @ (np.asarray(robot_side_xy, dtype=float) - start[:2]) < 0.0:
            side = -side
        self.clear_path()
        self.standoff_m = self.default_standoff_m if standoff_m is None else float(standoff_m)
        self.tilt = self.default_tilt if tilt_deg is None else np.radians(float(tilt_deg))
        self._start, self.u, self.length = start, u, length
        self.w = np.array([side[0] * np.cos(self.tilt), side[1] * np.cos(self.tilt), np.sin(self.tilt)])

    def image_down(self):
        """Output: the image y reference of camera_orientation(): the line
        direction for a steep view (tilt_deg >= 60, "world down" is then
        undefined), else None (world down)."""
        return self.u if self.tilt >= np.radians(60.0) else None

    def aim_point(self, s: float) -> np.ndarray:
        """Output: (3,) aim point at path parameter s (m)."""
        return self._start + float(np.clip(s, 0.0, self.length)) * self.u

    def frame_target(self, s: float) -> np.ndarray:
        """Output: (3,) frame origin target at path parameter s (m)."""
        return self.aim_point(s) + self.standoff_m * self.w

    @property
    def finished(self) -> bool:
        """Output: True once s reached the end of the line."""
        return self._start is not None and self.s >= self.length - 1e-6

    def _nominal_speed(self) -> float:
        """Output: path speed (m/s) with the start ramp and the end slow-down."""
        ramp = 1.0
        if self.ramp_s > 0.0:
            ramp = min(1.0, self.elapsed / self.ramp_s)
            # Decelerate over the distance the ramp would take at full speed.
            brake = 0.5 * self.speed * self.ramp_s
            ramp = min(ramp, max(0.2, (self.length - self.s) / brake) if brake > 0 else 1.0)
        return self.speed * ramp

    def build(self, robot: "RobotModel", dt: float) -> TaskBlock:
        """Advance the path and compute the frame rows.

        Input:
            robot: RobotModel holding the current whole-body state.
            dt: control period (s), used to advance s.

        Output:
            TaskBlock: 5 least-squares rows (3 position with feed-forward, 2
                aim) with a free roll, 6 (3 + 3 orientation) with a fixed
                roll, and 1 clearance inequality row.

        Raises:
            ValueError: if no path is set.
        """
        if self._start is None:
            raise ValueError(f"{self.name}: no path set")
        pose = robot.frame_pose(self.frame)
        jacobian = robot.frame_jacobian(self.frame)
        rotation = np.asarray(pose.rotation)
        position = np.asarray(pose.translation)

        lag = self.frame_target(self.s) - position
        error = abs(float(lag @ self.u)) if self.along_only else float(np.linalg.norm(lag))
        scale = float(np.clip(1.0 - error / self.slow_error_m, 0.0, 1.0))
        running = not (self.finished or self.paused)
        self.s_dot = self._nominal_speed() * scale if running else 0.0
        self.s = min(self.length, self.s + self.s_dot * dt)
        if running:
            self.elapsed += dt

        target = self.frame_target(self.s)
        aim = self.aim_point(self.s)
        position_error = target - position
        if self.roll is None:
            rotation_error, angle = aim_error(rotation, position, aim)
            axes = rotation[:, :2].T  # rotation about +Z stays free
        else:
            from scipy.spatial.transform import Rotation
            goal = camera_orientation(target, aim, self.roll, self.image_down())
            rotation_error = Rotation.from_matrix(goal @ rotation.T).as_rotvec()
            angle = float(np.linalg.norm(rotation_error))
            axes = np.eye(3)
        A = np.vstack([jacobian[:3], axes @ jacobian[3:]])
        b = np.concatenate([self.s_dot * self.u + self.position_gain * position_error,
                            self.aim_gain * (axes @ rotation_error)])
        block = TaskBlock(A=A, b=b, weight=self.weight, regularization=self.damping)
        # Clearance above the line: z_dot >= k (z_min - z); never above this
        # path's own nominal height (minus 5 cm), e.g. a horizontal pass.
        z_min = aim[2] + min(self.min_clearance_m, self.standoff_m * np.sin(self.tilt) - 0.05)
        block.C = jacobian[2:3]
        block.lower = np.array([self.clearance_gain * (z_min - position[2])])
        block.upper = np.array([np.inf])

        offset = position - aim
        along = float(offset @ self.u)
        self.last_frame_position = position.copy()
        self.last_target = target
        self.last_aim_point = aim
        self.last_position_error = float(np.linalg.norm(position_error))
        # Distance from the frame to the target line (the path n(s), any s).
        across = (position - self.frame_target(0.0))
        across = across - float(across @ self.u) * self.u
        self.last_cross_track = float(np.linalg.norm(across))
        self.last_standoff = float(np.linalg.norm(offset - along * self.u))
        self.last_aim_angle = angle
        return block


class PostureTask(Task):
    """Least-squares: pull arm joints towards a preferred configuration.

    A simple proxy for avoiding singularities/joint-limit crowding.
    """

    def __init__(self, name: str, joints: Sequence[str], preferred_q_rad: Sequence[float],
                 gain: float = 0.5, weight: float = 0.1) -> None:
        """Configure the preferred posture and this task's gain/weight.

        Input:
            name: identifier for this task instance.
            joints: joint names to regularize.
            preferred_q_rad: preferred position (rad) for each joint in
                joints, same order.
            gain: proportional gain on posture error.
            weight: least-squares weight for this task's rows.

        Output:
            None.
        """
        super().__init__(name)
        self.joints = list(joints)
        self.preferred_q_rad = np.asarray(preferred_q_rad, dtype=float)
        self.gain = gain
        self.weight = weight

    def build(self, robot: "RobotModel", dt: float) -> TaskBlock:
        """Compute this cycle's posture-regularization rows.

        Input:
            robot: RobotModel holding the current whole-body state.
            dt: control period (s); unused (this task has no dynamics of
                its own), kept for the common Task.build() signature.

        Output:
            TaskBlock: one least-squares row per joint in self.joints,
                pulling it towards preferred_q_rad.
        """
        v_slice = robot.velocity_slice(self.joints)
        current_q = robot.q[robot.configuration_slice(self.joints)]
        n = len(self.joints)
        indices = np.arange(v_slice.start, v_slice.stop)
        A = np.zeros((n, robot.nv))
        A[np.arange(n), indices] = 1.0
        b = self.gain * (self.preferred_q_rad - current_q)
        return TaskBlock(A=A, b=b, weight=self.weight)

    def set_preferred(self, preferred_q_rad: Sequence[float]) -> None:
        """Replace the posture target (e.g. with a MoveIt IK solution).

        Input:
            preferred_q_rad: new preferred position (rad) for each joint in
                self.joints, same order.

        Output:
            None.

        Raises:
            ValueError: if the length doesn't match self.joints.
        """
        preferred = np.asarray(preferred_q_rad, dtype=float)
        if preferred.shape != (len(self.joints),):
            raise ValueError(f"{self.name}: expected {len(self.joints)} values")
        self.preferred_q_rad = preferred


class BaseVelocityTask(Task):
    """Least-squares: pull the base twist towards a preferred value.

    Used as a low-priority regularizer so the base only moves when a
    higher-priority task actually needs it to.
    """

    def __init__(self, name: str, preferred_twist: Sequence[float] = (0.0, 0.0, 0.0),
                 weight: float = 1.0) -> None:
        """Configure the preferred base twist and this task's weight.

        Input:
            name: identifier for this task instance.
            preferred_twist: [vx, vy, wz] the base should settle towards
                when no higher-priority task needs it to move.
            weight: least-squares weight for this task's rows.

        Output:
            None.
        """
        super().__init__(name)
        self.preferred_twist = np.asarray(preferred_twist, dtype=float)
        self.weight = weight

    def build(self, robot: "RobotModel", dt: float) -> TaskBlock:
        """Compute this cycle's base-twist regularization rows.

        Input:
            robot: RobotModel holding the current whole-body state.
            dt: control period (s); unused (this task has no dynamics of
                its own), kept for the common Task.build() signature.

        Output:
            TaskBlock: 3 least-squares rows mapping to the base's velocity
                slice, pulling it towards preferred_twist.
        """
        A = np.zeros((3, robot.nv))
        indices = np.arange(robot.base_v_slice.start, robot.base_v_slice.stop)
        A[np.arange(3), indices] = 1.0
        return TaskBlock(A=A, b=self.preferred_twist, weight=self.weight)


TASK_REGISTRY = {
    "joint_limits": JointLimitsTask,
    "base_lane": BaseLaneTask,
    "camera_view": CameraViewTask,
    "line_path": LinePathTask,
    "posture": PostureTask,
    "base_velocity": BaseVelocityTask,
}


def build_tasks(config: dict, robot: "RobotModel") -> dict:
    """Translate a loaded experiment YAML dict into task instances per level.

    Input:
        config: parsed YAML dict with a "levels" list, each a
            {"name": str, "tasks": [{"type": str, ...params}, ...]} entry,
            ordered highest priority first.
        robot: RobotModel passed through for tasks that need it at
            construction time (currently unused by the built-in task
            types, but part of the interface for custom ones).

    Output:
        dict[str, list[Task]]: {level_name: [Task, ...]}, in config order,
            ready to pass to HQPController.

    Raises:
        ValueError: if a level is missing "name", a task's "type" is not in
            TASK_REGISTRY, a task's parameters don't match its class, or no
            levels are defined at all.
    """
    levels = {}
    for level in config.get("levels", []):
        level_name = level.get("name")
        if not level_name:
            raise ValueError("Every level requires a 'name'")
        tasks = []
        for index, spec in enumerate(level.get("tasks", [])):
            spec = dict(spec)
            task_type = spec.pop("type", None)
            task_cls = TASK_REGISTRY.get(task_type)
            if task_cls is None:
                raise ValueError(
                    f"Unknown task type '{task_type}' in level '{level_name}' "
                    f"(known types: {sorted(TASK_REGISTRY)})")
            task_name = spec.pop("name", f"{level_name}_{task_type}_{index}")
            try:
                tasks.append(task_cls(name=task_name, **spec))
            except TypeError as exc:
                raise ValueError(
                    f"Invalid parameters for task '{task_type}' in level "
                    f"'{level_name}': {exc}") from exc
        levels[level_name] = tasks
    if not levels:
        raise ValueError("config must define at least one level")
    return levels
