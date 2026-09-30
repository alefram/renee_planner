"""Defect detection mission: camera pose targets around the machine, Nav2 for the base, MoveIt for the arm (no ROS, no HQP).

Targets (`targets` in the experiment YAML, targets_frame):

  - {name, frame: camera, position: [x, y, z], look_at: [x, y, z]}

The camera's optical frame goes to `position` with its optical axis (+Z)
straight at `look_at`. The mission is a flat list of steps per target, in
driving order along the loop (navigation.py holds the rules, this class only
sequences them):

  choose_base    for each navigation.base_candidates(): MoveIt IK from that base; the
                 first with a valid IK gives (lane pose, offset, arm q); none: unreachable
  stow           arm to travel_arm_q, only if the base has to move
  back_to_lane   if the base is off its lane, lateral_approach_twist back to offset 0
  drive          Nav2 FollowPath along loop.boat_path (or, a few cm away or past the stop,
                 a straight move along the lane)
  approach       only if offset > 0: lateral_approach_twist towards the machine
  reach          IK again from the real base pose (seeded with the chosen q), MoveIt
                 motion (Pilz PTP, OMPL fallback), arm held
  record         reached / out_of_tolerance / unreachable / nav_failed / arm_failed /
                 clearance_violation, with the camera pose error measured through TF

While the base drives or approaches the footprint clearance to the machine is
checked on the real pose every tick: a violation cancels the drive, stops the
base and marks the target clearance_violation.

Results: defect_detection_run.npz (samples, plan and results; scripts/plot_run.py
draws them), summary.txt and trajectory.yaml. Image capture is not part of this
version.

The mission does not build the Pinocchio model or the HQP: it runs without the venv.
"""
import math
import os
from types import SimpleNamespace

import numpy as np
import yaml

from ..mission import save_run_config, write_summary
from ..navigation import (BasePlacementConfig, LocalizationMonitor, aabb_boxes, base_candidates, clearance_ok,
                          lateral_approach_twist, loop_from_boxes, shifted_pose, wrap_angle)
from ..recorders import CameraPoseRecorder, format_camera_summary
from ..tasks import aim_error, parse_scan_poses

ARM_JOINT_NAMES = ["robot_arm_shoulder_pan_joint", "robot_arm_shoulder_lift_joint", "robot_arm_elbow_joint",
                   "robot_arm_wrist_1_joint", "robot_arm_wrist_2_joint", "robot_arm_wrist_3_joint"]
# UR position limits (rad): every joint +-2 pi, the elbow +-pi.
ARM_JOINT_LIMIT_RAD = np.array([2 * np.pi, 2 * np.pi, np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi])

# The arm counts as already at a joint goal when every joint is this close (rad).
ARM_AT_GOAL_RAD = 0.05
# The base counts as at a stop when it is less than this ahead of it along the loop (m).
STOP_GAP_M = 0.1
# Arrived at a lane spot: within this distance (m) and the alignment tolerance, then still.
ARRIVE_M = 0.015
STATIC_SPEED = 0.01
STATIC_TIME_S = 0.5
MOVE_TIMEOUT_S = 30.0
# Off the lane by more than this (m) counts as an approach offset to undo.
OFFSET_EPS_M = 0.03
# The base is still this long (s) before the IK, and the arm held this long before the camera is measured.
STILL_S = 1.0
POSE_SETTLE_S = 1.0
TF_TIMEOUT_S = 5.0
RECORD_PERIOD_S = 0.1


class DefectDetectionMission:
    """Camera pose targets: the base to a spot on the loop, then the arm to the pose."""

    name = "defect_detection"

    def __init__(self, config: dict, *, log, dt: float = 0.01, run_dir: str = None, config_file: str = None,
                 node_parameters: dict = None, experiment_name: str = "", trajectory_dir: str = None,
                 **_unused) -> None:
        """Read the YAML, build the loop and order the targets.

        Input:
            config: parsed experiment YAML; moveit.collision_objects already
                holds the machine's boxes (the node fills them from its model).
            log: logger (info/warning/error with throttle_duration_sec).
            run_dir, config_file, node_parameters, experiment_name, trajectory_dir: outputs, as Mission.
            _unused: Mission-only arguments (xacro_path, HQP settings, ...), ignored.

        Raises:
            ValueError: on an invalid experiment (missing sections, a bad target, a
                loop that breaks the clearance limit or does not fit the workspace).
        """
        self.log = log
        self.dt = dt
        self.run_dir = run_dir
        self.experiment_name = experiment_name
        self.trajectory_dir = trajectory_dir
        self.moveit_config = config.get("moveit") or {}
        if not self.moveit_config:
            raise ValueError("the moveit: section is required")
        self.travel_q = np.asarray(config["travel_arm_q"], dtype=float)
        self.targets_frame = config.get("targets_frame", "robot_map")
        self.arm_joint_names = list(ARM_JOINT_NAMES)

        nav = config.get("navigation") or {}
        placement = config.get("base_placement") or {}
        missing = [k for k in ("min_clearance_m", "lane_clearance_m", "tracking_margin_m", "max_radius_m")
                   if k not in nav]
        if missing:
            raise ValueError("navigation needs " + ", ".join(missing))
        footprint = [float(v) for v in placement.get("footprint", [1.2, 0.7])]
        approach = placement.get("approach") or {}
        self.cfg = BasePlacementConfig(
            window_m=float(placement.get("window_m", 0.8)), offsets_m=placement.get("offsets_m", []),
            spacing_m=float(placement.get("spacing_m", 0.05)),
            min_straight_before_stop_m=float(placement.get("min_straight_before_stop_m", 0.0)),
            min_clearance_m=float(nav["min_clearance_m"]), lane_clearance_m=float(nav["lane_clearance_m"]),
            footprint=footprint, side_sign=-1.0 if placement.get("machine_side", "left") == "right" else 1.0,
            max_speed=float(approach.get("max_speed_mps", 0.1)),
            max_yaw_rate=float(approach.get("max_yaw_rate", 0.1)),
            align_tolerance_rad=float(approach.get("align_tolerance_rad", 0.02)))
        if self.cfg.lane_clearance_m < self.cfg.min_clearance_m:
            raise ValueError("navigation.lane_clearance_m must be >= min_clearance_m")
        self.boxes = aabb_boxes(self.moveit_config.get("collision_objects", []))
        loop = nav.get("loop_path") or {}
        self.loop = loop_from_boxes(
            self.boxes, lane_clearance_m=self.cfg.lane_clearance_m, min_clearance_m=self.cfg.min_clearance_m,
            tracking_margin_m=float(nav["tracking_margin_m"]), max_radius_m=float(nav["max_radius_m"]),
            footprint=footprint, spacing=float(loop.get("spacing", 0.05)), workspace=nav.get("workspace"),
            wall_margin_m=float(nav.get("wall_margin_m", 0.0)))
        self.loop_overshoot_m = float(loop.get("max_overshoot_m", 0.6))
        self.nav_retries = int(nav.get("retries", 0))
        self.nav_retry_delay_s = float(nav.get("retry_delay_s", 5.0))
        self.localization = LocalizationMonitor(
            nav.get("workspace"), float(nav.get("settle_s", 0.0)),
            nav.get("settle_tolerance", [0.05, 0.03]), nav.get("jump_tolerance", [0.3, 0.2]),
            float(nav.get("localization_timeout_s", 60.0)))
        self.start_pose = nav.get("start_pose")
        self.start_tolerance = nav.get("start_tolerance", [0.3, 0.2])
        self.start_settle_s = float(nav.get("start_settle_s", 5.0))
        self.seeds = [self.travel_q] + [np.asarray(s, dtype=float) for s in placement.get("seeds", [])]

        self.targets = self._load_targets(config)
        if not self.targets:
            raise ValueError("the experiment needs a non-empty `targets` list")
        self.order = self._driving_order()
        self.recorder = CameraPoseRecorder([], reach_tolerance_m=self.reach_tolerance_m,
                                           reach_tolerance_rad=self.reach_tolerance_rad)
        self.recorder.set_plan(
            [{"pose_id": t, "station": f"stop_{k + 1}", "position": self.targets[t]["position"],
              "look_at": self.targets[t]["look_at"]} for k, t in enumerate(self.order)],
            [{"name": f"stop_{k + 1}", "base": None} for k in range(len(self.order))])
        save_run_config(log, run_dir, config_file, node_parameters)
        self._save_plan()
        log.info(f"defect_detection: {len(self.targets)} target(s), loop {self.loop.total_length:.1f} m "
                 f"({self.cfg.lane_clearance_m:.2f} m lane clearance, {self.cfg.min_clearance_m:.2f} m limit); "
                 "order: " + ", ".join(self.targets[t]["name"] for t in self.order))

        self.ready = False
        self.done = False
        self.q_arm = None
        self.base_speed = float("inf")
        self.trajectory = []
        self._tool0_to = {}
        self._collision_call = None
        self._cursor = 0
        self._phase = None
        self._d = {}
        self._chosen = None       # (BaseCandidate, arm q) of the current target
        self._q_goal = None       # arm q of the current target's MoveIt motion
        self._offset_m = 0.0      # sideways offset of the base off its lane
        self._results = {}
        self._last_record = -np.inf
        self._run_start = None
        self._active_pose = -1
        self._active_error = (float("nan"), float("nan"))

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_targets(self, config: dict) -> list:
        """Read the camera pose targets and the camera settings.

        Output:
            list[dict]: {"id", "name", "frame", "position", "look_at", "section"}.

        Raises:
            ValueError: on a malformed target or a pose outside the view distance.
        """
        camera = config.get("camera_view") or {}
        self.roll = camera.get("roll", 0.0)
        self.reach_tolerance_m = float(camera.get("reach_tolerance_m", 0.03))
        self.reach_tolerance_rad = float(camera.get("reach_tolerance_rad", 0.05))
        view_distance = camera.get("view_distance_m", [0.3, 1.5])
        self.rolls = ([float(self.roll)] if self.roll is not None else
                      np.linspace(0.0, 2.0 * np.pi, int(self.moveit_config.get("ik_roll_samples", 12)),
                                  endpoint=False))
        frames = config.get("frames") or {}
        targets = []
        for index, spec in enumerate(config.get("targets") or []):
            name = spec.get("name", f"target_{index + 1}")
            frame = frames.get(spec.get("frame", ""), spec.get("frame", ""))
            (position, look_at), = parse_scan_poses([spec], f"targets[{index}] '{name}'", view_distance)
            targets.append({"id": index, "name": name, "frame": frame, "position": position,
                            "look_at": look_at, "section": spec.get("section")})
        return targets

    def _driving_order(self) -> list:
        """Target indices in driving order along the loop from start_pose (forward only).

        A target whose projection is just behind the start (less than
        base_placement window_m) goes first: the base reaches it straight
        along the lane instead of a lap later.
        """
        n = len(self.loop.points)
        start = self.loop.nearest(*(self.start_pose[:2] if self.start_pose is not None
                                    else self.loop.points[0][:2]))
        behind = int(self.cfg.window_m / (self.loop.total_length / n))

        def key(t: int) -> int:
            index = self.loop.nearest(float(self.targets[t]["position"][0]), float(self.targets[t]["position"][1]))
            ahead = (index - start + n // 2) % n - n // 2
            return ahead if ahead >= -behind else ahead + n

        return sorted(range(len(self.targets)), key=key)

    def joint_limits(self, names=None):
        """Output: object with q_lower / q_upper (rad) of the arm joints (UR limits)."""
        return SimpleNamespace(q_lower=-ARM_JOINT_LIMIT_RAD, q_upper=ARM_JOINT_LIMIT_RAD)

    # ------------------------------------------------------------------
    # Measurements
    # ------------------------------------------------------------------

    def on_arm_state(self, q_arm, dq_arm, now: float) -> None:
        """Store the measured arm joints."""
        self.q_arm = np.asarray(q_arm, dtype=float)

    def on_base_state(self, x: float, y: float, yaw: float, vx: float, vy: float, wz: float) -> None:
        """Store the base speed from odometry (the pose comes from TF, in targets_frame)."""
        self.base_speed = max(math.hypot(vx, vy), abs(wz))

    # ------------------------------------------------------------------
    # Executor
    # ------------------------------------------------------------------

    def tick(self, io) -> None:
        """Advance the mission by one control period."""
        if self.done:
            return
        if not self.ready:
            self._setup(io)
            return
        if self._run_start is None:
            self._run_start = io.now()
        getattr(self, f"_phase_{self._phase}")(io)
        self._record_tick(io)

    def _goto(self, phase: str, **data) -> None:
        """Move to a phase with fresh phase data."""
        self._phase, self._d = phase, data

    def _target(self) -> dict:
        return self.targets[self.order[self._cursor]]

    def _setup(self, io) -> None:
        """Wait for MoveIt, read the tool0 -> camera transforms, add the collision objects to the scene."""
        objects = self.moveit_config.get("collision_objects", [])
        if self._collision_call is None:
            if not io.moveit_ready():
                io.log.warning("Waiting for MoveIt services", throttle_duration_sec=5.0)
                return
            ik_link = self.moveit_config.get("ik_link", "robot_arm_tool0")
            for frame in {t["frame"] for t in self.targets}:
                pose = io.frame_pose(ik_link, frame)
                if pose is None:
                    return
                self._tool0_to[frame] = pose
            self._collision_call = io.apply_collision_objects(objects)
            return
        result = self._collision_call.poll()
        if result is None:
            return
        if result[0]:
            io.log.info(f"Added {len(objects)} collision object(s) to the MoveIt planning scene")
        else:
            io.log.error(f"Could not add the collision objects to the MoveIt planning scene: {result[1]}")
        self.ready = True
        self._goto("localize")

    # -- localize -------------------------------------------------------

    def _phase_localize(self, io) -> None:
        """Wait until localization is stable, inside the workspace and at start_pose."""
        d = self._d
        io.publish_twist(0.0, 0.0, 0.0)
        base = io.base_in_map()
        if not self._inside_workspace(io, base):
            d.pop("settle_reference", None)
            if io.now() - d["outside_since"] > self.localization.timeout_s:
                self._abort(io, "nav_failed", "localization outside the navigation workspace")
            return
        correction = io.localization_correction()
        if correction is None or base is None:
            return
        x, y, yaw = base
        if self.start_pose is not None:
            sx, sy, syaw = (float(v) for v in self.start_pose)
            distance, dyaw = math.hypot(x - sx, y - sy), abs(wrap_angle(yaw - syaw))
            if distance > self.start_tolerance[0] or dyaw > self.start_tolerance[1]:
                d.pop("settle_reference", None)
                io.log.warning(f"Localized at ({x:.2f}, {y:.2f}, {yaw:+.2f}), {distance:.2f} m / "
                               f"{dyaw:.2f} rad from the start pose; waiting for localization",
                               throttle_duration_sec=5.0)
                return
        if not self.localization.settled(d, correction, io.now(), max(self.start_settle_s, 1e-6)):
            return
        io.log.info(f"Localized at ({x:.2f}, {y:.2f}, {yaw:+.2f})")
        self._goto("choose_base")

    def _inside_workspace(self, io, base) -> bool:
        """Output: whether the localized base is inside the workspace (tracks how long it isn't)."""
        d = self._d
        if self.localization.inside(None if base is None else base[:2]):
            d["outside_since"] = None
            return True
        if d.get("outside_since") is None:
            d["outside_since"] = io.now()
        where = "unknown" if base is None else f"({base[0]:.2f}, {base[1]:.2f})"
        io.log.warning(f"Localized base {where} is outside the navigation workspace; waiting",
                       throttle_duration_sec=5.0)
        return False

    # -- choose_base ----------------------------------------------------

    def _phase_choose_base(self, io) -> None:
        """MoveIt IK from each base candidate; the first valid one sets the stop."""
        d, target = self._d, self._target()
        if "candidates" not in d:
            base = io.base_in_map()
            if base is None:
                return
            found = base_candidates(target, self.loop, self.boxes, self.cfg)
            candidates = [c for c in found if not self._lap_away(base, c.lane_pose)]
            if not candidates:
                return self._end_target(io, target, "unreachable",
                                        f"no base spot within {self.cfg.window_m} m keeps the "
                                        f"{self.cfg.min_clearance_m} m clearance without driving back a lap "
                                        f"({len(found)} spots before the lap filter)")
            d.update(candidates=candidates, i=0, seed=0, call=None)
        if d["i"] >= len(d["candidates"]):
            return self._end_target(io, target, "unreachable",
                                    f"no MoveIt IK from any of {len(d['candidates'])} base spots")
        candidate = d["candidates"][d["i"]]
        if d["call"] is None:
            d["call"] = io.request_ik(target["position"], target["look_at"], self._tool0_to[target["frame"]],
                                      self.seeds[d["seed"]], self.rolls, None, candidate.pose)
            return
        result = d["call"].poll()
        if result is None:
            return
        d["call"] = None
        if not result[0]:
            d["seed"] += 1
            if d["seed"] >= len(self.seeds):
                d["seed"], d["i"] = 0, d["i"] + 1
            return
        self._chosen = (candidate, np.asarray(result[1], dtype=float))
        io.log.info(f"Target '{target['name']}': base spot {d['i'] + 1}/{len(d['candidates'])} "
                    f"({candidate.pose[0]:+.2f}, {candidate.pose[1]:+.2f}), offset {candidate.offset:.2f} m")
        self.recorder.stations[self._cursor]["base"] = candidate.pose
        base = io.base_in_map()
        if base is None or math.hypot(base[0] - candidate.pose[0], base[1] - candidate.pose[1]) <= 0.03:
            self._goto("still")
        else:
            self._goto("stow", next="back_to_lane" if self._offset_m > OFFSET_EPS_M else "drive")

    def _lap_away(self, base, lane_pose) -> bool:
        """Output: True if the lane spot is behind the base by more than the overshoot (Nav2 would drive a lap)."""
        i, j = self.loop.nearest(*base[:2]), self.loop.nearest(*lane_pose[:2])
        ahead = self.loop.forward_distance(i, j)
        behind = self.loop.total_length - ahead
        return behind < ahead and behind > self.loop_overshoot_m

    # -- stow -----------------------------------------------------------

    def _phase_stow(self, io) -> None:
        """Arm to travel_arm_q before the base moves."""
        d, target = self._d, self._target()
        if "motion" not in d:
            if self.q_arm is None or np.max(np.abs(self.q_arm - self.travel_q)) < ARM_AT_GOAL_RAD:
                return self._goto(d["next"])
            d["motion"] = {}
            io.publish_twist(0.0, 0.0, 0.0)
        outcome = self._poll_arm(io, d["motion"], self.travel_q)
        if outcome is None:
            return
        if not outcome[0]:
            return self._end_target(io, target, "arm_failed", f"stow: {outcome[1]}")
        self._goto(d["next"])

    # -- back_to_lane, drive, approach ----------------------------------

    def _phase_back_to_lane(self, io) -> None:
        """Undo the last approach: sideways back onto the lane."""
        d = self._d
        base = io.base_in_map()
        if base is None:
            return
        if "lane" not in d:
            d["lane"] = tuple(float(v) for v in self.loop.points[self.loop.nearest(*base[:2])])
        outcome = self._omni_move(io, d, base, d["lane"], 0.0, guard=False)
        if outcome is None:
            return
        self._offset_m = 0.0
        self._goto("drive")

    def _phase_drive(self, io) -> None:
        """Nav2 FollowPath along the loop to the lane spot (or a straight move along the lane)."""
        d, target = self._d, self._target()
        candidate = self._chosen[0]
        base = io.base_in_map()
        if base is None:
            return
        if "mode" not in d:
            i, j = self.loop.nearest(*base[:2]), self.loop.nearest(*candidate.lane_pose[:2])
            ahead = self.loop.forward_distance(i, j)
            straight = ahead < STOP_GAP_M or self.loop.total_length - ahead < self.loop_overshoot_m
            d.update(mode="straight" if straight else "follow", attempts=0, retry_at=-np.inf, call=None)
        if d["mode"] == "straight":
            outcome = self._omni_move(io, d, base, candidate.lane_pose, 0.0, guard=True)
            if outcome is None:
                return
            if outcome in ("violation", "blocked"):
                return self._end_target(io, target, "clearance_violation", f"driving along the lane ({outcome})")
            return self._after_drive(io)
        if d["call"] is None:
            if io.now() < d["retry_at"]:
                return
            if not io.follow_path_ready():
                io.log.warning("Waiting for Nav2 (FollowPath)", throttle_duration_sec=5.0)
                return
            if not self._inside_workspace(io, base):
                if io.now() - d["outside_since"] > self.localization.timeout_s:
                    self._end_target(io, target, "nav_failed", "localization outside the navigation workspace")
                return
            if not self.localization.settled(d, io.localization_correction(), io.now()):
                io.log.info("Waiting for localization to settle", throttle_duration_sec=5.0)
                return
            path = self.loop.boat_path(base[:2], candidate.lane_pose[:2], self.loop_overshoot_m)
            samples = [tuple(p) for p in path[:-1]] + [tuple(candidate.lane_pose)]
            ahead = self.loop.forward_distance(self.loop.nearest(*base[:2]), self.loop.nearest(*candidate.lane_pose[:2]))
            io.log.info(f"Target '{target['name']}': driving {ahead:.2f} m along the loop")
            io.publish_twist(0.0, 0.0, 0.0)
            d["call"] = io.follow_path(samples)
            return
        if not clearance_ok(base, self.boxes, self.cfg.min_clearance_m, self.cfg.footprint):
            d["call"].cancel()
            io.publish_twist(0.0, 0.0, 0.0)
            return self._end_target(io, target, "clearance_violation",
                                    f"base at ({base[0]:.2f}, {base[1]:.2f}) under {self.cfg.min_clearance_m} m "
                                    "from the machine while driving")
        result = d["call"].poll()
        if result is None:
            io.log.info(f"Driving to the stop: base ({base[0]:.2f}, {base[1]:.2f})", throttle_duration_sec=15.0)
            return
        if not result[0]:
            d["call"] = None
            if d["attempts"] < self.nav_retries:
                d["attempts"] += 1
                d["retry_at"] = io.now() + self.nav_retry_delay_s
                io.log.warning(f"{result[1]}; retrying in {self.nav_retry_delay_s:.0f} s "
                               f"({d['attempts']}/{self.nav_retries})")
                return
            return self._end_target(io, target, "nav_failed", result[1])
        io.log.info("Arrived at the base stop")
        self._after_drive(io)

    def _after_drive(self, io) -> None:
        self._goto("approach" if self._chosen[0].offset > OFFSET_EPS_M else "still")

    def _phase_approach(self, io) -> None:
        """Sideways towards the machine (lane heading kept), up to the chosen offset."""
        d, target = self._d, self._target()
        candidate = self._chosen[0]
        base = io.base_in_map()
        if base is None:
            return
        outcome = self._omni_move(io, d, base, candidate.lane_pose, candidate.offset, guard=True)
        if outcome is None:
            return
        self._offset_m = self._lane_offset(base)
        if outcome == "violation":
            return self._end_target(io, target, "clearance_violation",
                                    f"base at ({base[0]:.2f}, {base[1]:.2f}) during the approach")
        # arrived, blocked by the clearance or timed out: the next IK tells if this is enough.
        io.log.info(f"Approach {outcome}: {self._offset_m:.2f} m off the lane")
        self._goto("still")

    def _lane_offset(self, base) -> float:
        """Output: distance (m) from the base to the loop, 0 below OFFSET_EPS_M."""
        x, y, _ = self.loop.points[self.loop.nearest(*base[:2])]
        offset = math.hypot(base[0] - x, base[1] - y)
        return offset if offset > OFFSET_EPS_M else 0.0

    def _omni_move(self, io, d: dict, base, lane_pose, offset: float, guard: bool):
        """One tick of a straight omnidirectional move to the lane spot shifted `offset` (lane heading held).

        Output:
            None while moving; "arrived" (within ARRIVE_M, aligned, still for
            STATIC_TIME_S), "blocked" (the next step would break the clearance),
            "violation" (guard: the real pose is under the limit) or "timeout".
        """
        now = io.now()
        d.setdefault("move_start", now)
        if guard and not clearance_ok(base, self.boxes, self.cfg.min_clearance_m, self.cfg.footprint):
            io.publish_twist(0.0, 0.0, 0.0)
            return "violation"
        goal = shifted_pose(lane_pose, offset, self.cfg.side_sign)
        distance = math.hypot(goal[0] - base[0], goal[1] - base[1])
        if distance < ARRIVE_M and abs(wrap_angle(lane_pose[2] - base[2])) < self.cfg.align_tolerance_rad:
            io.publish_twist(0.0, 0.0, 0.0)
            if self.base_speed > STATIC_SPEED:
                d["still_since"] = None
                return None
            if d.get("still_since") is None:
                d["still_since"] = now
            return "arrived" if now - d["still_since"] >= STATIC_TIME_S else None
        d["still_since"] = None
        if now - d["move_start"] > MOVE_TIMEOUT_S:
            io.publish_twist(0.0, 0.0, 0.0)
            io.log.warning(f"Base not in place after {MOVE_TIMEOUT_S:.0f} s ({distance:.3f} m); continuing")
            return "timeout"
        twist = lateral_approach_twist(base, lane_pose, offset, self.boxes, self.cfg)
        if twist is None:
            io.publish_twist(0.0, 0.0, 0.0)
            return "blocked"
        io.publish_twist(*twist)
        return None

    # -- reach ----------------------------------------------------------

    def _phase_still(self, io) -> None:
        """Wait for a still base before asking MoveIt for the IK from its real pose."""
        d = self._d
        io.publish_twist(0.0, 0.0, 0.0)
        now = io.now()
        if self.base_speed > STATIC_SPEED:
            d["since"] = None
            return
        if d.get("since") is None:
            d["since"] = now
            return
        if now - d["since"] >= STILL_S:
            self._goto("ik", seed=0, call=None)

    def _phase_ik(self, io) -> None:
        """IK from the real base pose, seeded with the chosen q, then the current arm."""
        d, target = self._d, self._target()
        seeds = [self._chosen[1], self.q_arm if self.q_arm is not None else self.travel_q]
        if d["call"] is None:
            d["call"] = io.request_ik(target["position"], target["look_at"], self._tool0_to[target["frame"]],
                                      seeds[d["seed"]], self.rolls, None)
            return
        result = d["call"].poll()
        if result is None:
            return
        d["call"] = None
        if not result[0]:
            d["seed"] += 1
            if d["seed"] >= len(seeds):
                self._end_target(io, target, "unreachable", f"IK from the real base pose: {result[1]}")
            return
        io.log.info(f"Target '{target['name']}': MoveIt goal q={np.round(result[1], 3).tolist()}")
        self._goto("move", q=np.asarray(result[1], dtype=float), motion={})

    def _phase_move(self, io) -> None:
        """MoveIt joint motion to the IK solution (Pilz PTP, OMPL fallback), then hold."""
        d, target = self._d, self._target()
        io.publish_twist(0.0, 0.0, 0.0)
        outcome = self._poll_arm(io, d["motion"], d["q"])
        if outcome is None:
            return
        if not outcome[0]:
            return self._end_target(io, target, "arm_failed", outcome[1])
        self._q_goal = d["q"]
        self._goto("hold", q=d["q"], since=io.now())

    def _phase_hold(self, io) -> None:
        """Hold the arm (and stop the base) while the camera settles."""
        d = self._d
        io.hold_arm(d["q"])
        if io.now() - d["since"] >= POSE_SETTLE_S:
            self._goto("record", start=io.now())

    def _phase_record(self, io) -> None:
        """Measure the camera pose through TF and end the target."""
        d, target = self._d, self._target()
        io.hold_arm(self._q_goal)
        pose = io.frame_pose(self.targets_frame, target["frame"])
        if pose is None:
            if io.now() - d["start"] < TF_TIMEOUT_S:
                return
            return self._end_target(io, target, "arm_failed", "no TF of the camera to measure the pose")
        position = pose[:3, 3]
        error = float(np.linalg.norm(position - target["position"]))
        aim = aim_error(pose[:3, :3], position, target["look_at"])[1]
        within = error < self.reach_tolerance_m and aim < self.reach_tolerance_rad
        self._end_target(io, target, "reached" if within else "out_of_tolerance", camera=position,
                         error=error, aim=aim)

    def _poll_arm(self, io, motion: dict, q_goal):
        """Advance a MoveIt joint motion, retried once with the fallback planner.

        Output:
            None while in progress; otherwise (success: bool, detail: str).
        """
        if motion.get("call") is None:
            motion["call"] = io.move_arm(q_goal, motion.get("fallback", False))
            if motion["call"] is None:
                io.log.warning("Waiting for MoveIt move_action", throttle_duration_sec=5.0)
            return None
        result = motion["call"].poll()
        if result is None:
            return None
        if not result[0] and self.moveit_config.get("fallback_pipeline") and not motion.get("fallback"):
            io.log.warning(f"{result[1]}; retrying the motion with {self.moveit_config['fallback_pipeline']}")
            motion.clear()
            motion["fallback"] = True
            return None
        return result

    # -- end of a target, end of the run --------------------------------

    def _end_target(self, io, target: dict, status: str, detail: str = "", camera=None,
                    error: float = float("nan"), aim: float = float("nan")) -> None:
        """Record the target's result and go on with the next one (or the final stow)."""
        io.publish_twist(0.0, 0.0, 0.0)
        pose_id = target["id"]
        self._active_error = (error, aim)
        self.recorder.add_result(pose_id, status, camera, error, aim, detail)
        self._results[pose_id] = status
        base = io.base_in_map()
        if base is not None:
            self._offset_m = self._lane_offset(base)
        self.trajectory.append({
            "target": target["name"], "status": status, "detail": detail,
            "base_map": None if base is None else [round(float(v), 4) for v in base],
            "lane_offset_m": round(float(self._offset_m), 3),
            "arm_q": None if self.q_arm is None else [round(float(v), 4) for v in self.q_arm]})
        message = (f"Target '{target['name']}': {status} (pos_err={error:.4f} m, aim={aim:.4f} rad)"
                   f"{' ' + detail if detail else ''}")
        # One call site per severity: rclpy raises if a call site changes severity.
        if status == "reached":
            io.log.info(message)
        else:
            io.log.warning(message)
        self._chosen = None
        self._cursor += 1
        self._goto("choose_base" if self._cursor < len(self.order) else "final_stow")

    def _phase_final_stow(self, io) -> None:
        """Arm back to travel_arm_q, then the run ends (a failed stow does not change the results)."""
        d = self._d
        if "motion" not in d:
            if self.q_arm is None or np.max(np.abs(self.q_arm - self.travel_q)) < ARM_AT_GOAL_RAD:
                return self._finish(io)
            d["motion"] = {}
        outcome = self._poll_arm(io, d["motion"], self.travel_q)
        if outcome is None:
            return
        if not outcome[0]:
            io.log.warning(f"Final stow failed: {outcome[1]}")
        self._finish(io)

    def _abort(self, io, status: str, detail: str) -> None:
        """End the run: every target not done gets `status`."""
        for target in self.targets:
            if target["id"] not in self._results:
                self.recorder.add_result(target["id"], status, detail=detail)
                self._results[target["id"]] = status
        io.log.error(f"Run ended: {detail}")
        self._finish(io)

    def _finish(self, io) -> None:
        """Stop the robot and write the results."""
        self.done = True
        io.publish_twist(0.0, 0.0, 0.0)
        io.log.info("All targets done; writing the run results.")
        self.save_results()
        self.save_trajectory()

    # ------------------------------------------------------------------
    # Recording and files
    # ------------------------------------------------------------------

    def _record_tick(self, io) -> None:
        """Store a sample (camera, base in targets_frame, arm) every RECORD_PERIOD_S."""
        now = io.now()
        if now - self._last_record < RECORD_PERIOD_S or self.q_arm is None:
            return
        self._last_record = now
        base = io.base_in_map()
        target = self._target() if self._cursor < len(self.order) else None
        camera = io.frame_pose(self.targets_frame, target["frame"]) if target is not None else None
        if base is None or camera is None:
            return
        recording = self._phase in ("hold", "record")
        self.recorder.record(now - self._run_start, camera[:3, 3], [], base, self.q_arm,
                             pose_id=target["id"] if recording else -1)

    def _save_plan(self) -> None:
        """Write the loop and the target order to run_dir/plan.yaml."""
        if self.run_dir is None:
            return
        try:
            with open(os.path.join(self.run_dir, "plan.yaml"), "w") as handle:
                yaml.safe_dump({"mission": self.name,
                                "loop_length_m": round(self.loop.total_length, 2),
                                "loop_bounds": {"min": np.round(self.loop.points[:, :2].min(axis=0), 3).tolist(),
                                                "max": np.round(self.loop.points[:, :2].max(axis=0), 3).tolist()},
                                "order": [self.targets[t]["name"] for t in self.order],
                                "collision_objects": self.moveit_config.get("collision_objects", [])},
                               handle, sort_keys=False)
        except OSError as exc:
            self.log.error(f"Could not write plan.yaml: {exc}")

    def save_results(self) -> None:
        """Write <mission>_run.npz and the results table (summary.txt)."""
        if self.run_dir is None or not len(self.recorder):
            return
        try:
            path = self.recorder.save(self.run_dir, self.name, "Defect detection: camera poses")
            self.log.info(f"Run data written to {path} (plot: scripts/plot_run.py {self.run_dir})")
        except (OSError, ValueError) as exc:
            self.log.error(f"Could not write the run data: {exc}")
        summary = format_camera_summary(self.recorder.summary())
        write_summary(self.log, self.run_dir, [("Camera targets", summary)] if summary else [])

    def save_trajectory(self) -> None:
        """Write the per-target outcome to run_dir/trajectory.yaml and trajectory_dir/<experiment>.yaml."""
        if not self.trajectory:
            return
        document = {"experiment": self.experiment_name, "mission": self.name,
                    "run": os.path.basename(self.run_dir or ""), "frame": self.targets_frame,
                    "joints": list(self.arm_joint_names), "targets": self.trajectory}
        paths = [os.path.join(self.run_dir, "trajectory.yaml")] if self.run_dir else []
        if self.trajectory_dir:
            paths.append(os.path.join(self.trajectory_dir, f"{self.experiment_name or self.name}.yaml"))
        for path in paths:
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as handle:
                    yaml.safe_dump(document, handle, sort_keys=False, default_flow_style=None)
                self.log.info(f"Trajectory written to {path}")
            except OSError as exc:
                self.log.error(f"Could not write the trajectory to {path}: {exc}")
