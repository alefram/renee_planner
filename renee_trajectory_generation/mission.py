# /usr/bin/env python3
"""What every mission shares: robot, HQP, base stops and the step executor (no ROS).

A mission (missions/: cleaning.py, screw_detection.py, defect_detection.py) reaches a list of targets
around the machine. Each target has a frame pose `{frame, position, look_at}` in
`targets_frame` (the frame's origin at `position`, its +Z aimed at
`look_at`). There are no base poses in the YAML: navigation.BasePlanner
places the base stops on navigation.loop_path (boat mode), and the mission
turns them into a list of generic steps, run one after the other:

  localize                       wait for stable localization at start_pose
  move_arm {q}                   MoveIt joint motion (e.g. travel_arm_q)
  move_base {pose}               Nav2 FollowPath along the loop, then aligned + static
  shift_base {offset}            straight sideways shift, aligned + static
  move_frame {target}            MoveIt IK + joint motion to the target's frame pose

With `arm_between_stops: hold` no move_arm to travel_arm_q comes between the
stops: the arm keeps the last target's configuration while Nav2 drives; with
`section` only between stops whose targets are of the same `section` (e.g. a
machine part), the arm folding to travel_arm_q from one section to the next;
a target with `travel_before: true` is always driven to in travel_arm_q, and
with `travel_through_corners: true` so is every drive through one of the loop's arcs.

followed, per target, by the mission's own steps (e.g. defect_detection's
refine_pose, cleaning's and screw_detection's follow_path). A failed step marks its target(s) and
skips the rest of its target or stop; steps marked `always` (stowing,
shifting back onto the lane) still run. Nav2 and MoveIt execute the large
motions; the HQP only works close to the targets.

The ROS side (trajectory_controller_node.py) feeds measurements
(on_arm_state, on_base_state) and calls tick(io) every control period with
an `io` object that reads ROS data and sends commands:

  io.now() -> float                     node time (s)
  io.log                                logger (info/warning/error, throttle_duration_sec)
  io.to_odom() -> (R, t, yaw) | None    targets_frame -> odom transform
  io.base_in_map() -> (x, y, yaw)|None  base pose in targets_frame
  io.localization_correction() -> (x, y, yaw) | None   odom pose in targets_frame
  io.follow_path_ready() -> bool
  io.follow_path(samples) -> call
  io.navigate_to_pose_ready() -> bool
  io.navigate_to_pose(x, y, yaw) -> call   Nav2 NavigateToPose (navigation.drive: navigate_to_pose)
                                        Nav2 FollowPath [(x, y, yaw)] in targets_frame
  io.moveit_ready() -> bool
  io.apply_collision_objects(objects) -> call
  io.request_ik(position, look_at, tool0_to_frame, seed, rolls, down) -> call | None
  io.move_arm(q_goal, fallback) -> call | None
  io.moveit_base_pose() -> call | None     move_group's base pose (x, y, yaw) in its planning frame
  io.check_validity(q_arm) -> call | None
  io.toggle_localization_pause() -> call | None   (navigation.pause_localization_at_stops)
  io.capture_ready() -> bool            (only with the YAML's `capture:` section)
  io.capture_rgbd(waypoint_id, session_dir, frame_count) -> call
  io.publish_twist(vx, vy, wz), io.publish_arm(q_arm), io.hold_arm(q_arm)

A `call` is non-blocking: call.poll() returns None while pending, else a
tuple (ok, value): (ok, detail) for motions, (True, q) / (False, detail) for
IK, (valid, contacts) for validity checks; call.cancel() aborts a motion.

The experiment YAML picks the mission with `experiment.mission`
(create_mission()).
"""
import math
import os
import shutil
import time
from typing import TYPE_CHECKING

import numpy as np
import yaml

from .navigation import (BasePlanner, LocalizationMonitor, LoopPath, Target, correction_change, gate_twist,
                         wrap_angle)
from .tasks import BaseLaneTask, PostureTask, build_tasks

if TYPE_CHECKING:
    from .robot import RobotModel

# The arm counts as already at a joint goal when every joint is this close (rad).
ARM_AT_GOAL_RAD = 0.05
# Max seconds for a base gate (aligned + static at a spot) or a sideways shift.
BASE_MOVE_TIMEOUT_S = 30.0
# How far ahead (s) along the commanded arm velocity each MoveIt validity
# check looks, so a collision is caught before the arm gets there.
VALIDITY_LOOKAHEAD_S = 0.25
# The base counts as at a stop when it is less than this ahead of it along the loop.
STOP_GAP_M = 0.1


def build_robot(config: dict, xacro_path: str, mappings: dict, ee_frame: str) -> "RobotModel":
    """Output: RobotModel with the YAML's virtual tool frame (nozzle), if any."""
    # Imported here: only a Mission builds the Pinocchio model, defect_detection needs neither it nor the HQP.
    from .robot import RobotModel
    robot = RobotModel.from_xacro(xacro_path, mappings, ee_frame=ee_frame)
    nozzle = config.get("nozzle")
    if nozzle is not None:
        robot.add_frame(nozzle["frame"], nozzle.get("parent", "robot_arm_tool0"),
                        nozzle.get("offset", [0.0, 0.0, 0.0]), nozzle.get("rpy", [0.0, 0.0, 0.0]))
    return robot


def save_run_config(log, run_dir: str, config_file: str, node_parameters: dict) -> None:
    """Copy the experiment YAML and the node parameters into run_dir."""
    if run_dir is None:
        return
    try:
        os.makedirs(run_dir, exist_ok=True)
        if config_file:
            shutil.copyfile(config_file, os.path.join(run_dir, "experiment.yaml"))
        if node_parameters is not None:
            with open(os.path.join(run_dir, "node_parameters.yaml"), "w") as handle:
                yaml.safe_dump({"trajectory_controller_node": {"ros__parameters": node_parameters}}, handle)
    except OSError as exc:
        log.error(f"Could not save the run config to {run_dir}: {exc}")


def write_summary(log, run_dir: str, tables: list) -> None:
    """Log (title, table) pairs and write them to run_dir/summary.txt."""
    for title, table in tables:
        log.info(f"{title}:\n{table}")
    if run_dir is None:
        return
    try:
        with open(os.path.join(run_dir, "summary.txt"), "w") as handle:
            handle.write("".join(f"{title}:\n{table}\n\n" for title, table in tables))
    except OSError as exc:
        log.error(f"Could not write summary.txt: {exc}")


def create_mission(config: dict, **kwargs):
    """Build the mission named by the YAML's experiment.mission.

    Input:
        config: parsed experiment YAML.
        kwargs: mission constructor arguments (defect_detection ignores the
            HQP ones: it builds no Pinocchio model and needs no venv).

    Output:
        the mission: e.g. missions.cleaning.CleaningMission, or
            missions.defect_detection.DefectDetectionMission (not a Mission).

    Raises:
        ValueError: on an unknown or missing mission name.
    """
    from .missions.cleaning import CleaningMission
    from .missions.defect_detection import DefectDetectionMission
    from .missions.screw_detection import ScrewDetectionMission
    missions = {"cleaning": CleaningMission, "screw_detection": ScrewDetectionMission,
                "defect_detection": DefectDetectionMission}
    name = (config.get("experiment") or {}).get("mission")
    if name not in missions:
        raise ValueError(f"experiment.mission must be one of {sorted(missions)} (got {name!r})")
    return missions[name](config, **kwargs)


class Mission:
    """Base mission: plan at construction, then tick(io) every control period.

    Subclasses implement load_targets(), target_steps(), record_result()
    and save_results(), their own `_step_<type>` methods, and optionally
    on_setup() and record_tick().
    """

    name = "mission"

    def __init__(self, config: dict, *, xacro_path: str, mappings: dict, log,
                 ee_frame: str = "robot_arm_tool0", dt: float = 0.01,
                 equality_tolerance: float = 1e-4, osqp_settings: dict = None,
                 arm_reference_reset_rad: float = 0.15, run_dir: str = None,
                 config_file: str = None, node_parameters: dict = None,
                 experiment_name: str = "", trajectory_dir: str = None) -> None:
        """Build the robot/tasks/solver, load the targets and plan the steps.

        Input:
            config: parsed experiment YAML.
            xacro_path, mappings: robot description (robot.resolve_default_xacro).
            log: logger (info/warning/error with throttle_duration_sec).
            ee_frame: arm tool frame of the model.
            dt: control period (s).
            equality_tolerance, osqp_settings: HQP solver settings.
            arm_reference_reset_rad: while the HQP streams arm commands the
                model keeps its own integrated arm reference (a position-
                controlled arm tracks with a lag; resetting to the lagging
                measurement every cycle stalls the motion); it is resynced
                when they differ by more than this (rad).
            run_dir: where the outputs go (created), or None.
            config_file: experiment YAML path, copied to run_dir.
            node_parameters: ROS parameters of the run, saved to run_dir.
            experiment_name: name of the experiment (trajectory file name).
            trajectory_dir: where the executed trajectory is written as
                <experiment_name>.yaml (e.g. the package's config/trajectories),
                besides run_dir/trajectory.yaml; None: run_dir only.

        Raises:
            ValueError: on an invalid experiment (missing sections, a bad
                target, a target no base spot reaches).
        """
        self.log = log
        self.dt = dt
        self.ee_frame = ee_frame
        self.arm_reference_reset_rad = arm_reference_reset_rad
        self.run_dir = run_dir
        self.experiment_name = experiment_name
        self.trajectory_dir = trajectory_dir
        # Executed trajectory (robot_map): one record per step, pass samples.
        self.trajectory = {"steps": [], "passes": {}}

        from .solver import HQPController
        self.robot = build_robot(config, xacro_path, mappings, ee_frame)
        self.level_tasks = build_tasks(config, self.robot)
        self.controller = HQPController(self.robot, self.level_tasks,
                                        equality_tolerance=equality_tolerance,
                                        osqp_settings=osqp_settings)
        self.all_tasks = [task for tasks in self.level_tasks.values() for task in tasks]
        self.posture_task = next((t for t in self.all_tasks if isinstance(t, PostureTask)), None)
        self.base_task = next((t for t in self.all_tasks if isinstance(t, BaseLaneTask)), None)
        if self.base_task is None:
            raise ValueError("levels need a base_lane task (base hold, shift and pass)")

        self.moveit_config = config.get("moveit") or {}
        if not self.moveit_config:
            raise ValueError("the moveit: section is required")
        self.travel_q = np.asarray(config["travel_arm_q"], dtype=float)
        # travel: the arm folds into travel_arm_q before every drive; hold: it
        # keeps the last target's configuration (the camera stays on the
        # machine), checked collision-free along each drive when planning.
        self.arm_between_stops = config.get("arm_between_stops", "travel")
        self.preposition = {}  # stop index -> arm configuration taken before driving on
        self.held_drives = set()  # stop indices k whose drive to stop k + 1 keeps the arm
        if self.arm_between_stops not in ("travel", "hold", "section"):
            raise ValueError("arm_between_stops must be travel, hold or section")
        self._read_navigation(config.get("navigation") or {})

        self.targets = self.load_targets(config)
        if not self.targets:
            raise ValueError("the experiment needs a non-empty `targets` list")
        self.stops = self._plan_stops(config, xacro_path, mappings)
        self.check_plan()
        self.steps = self._build_steps()
        self._save_run_config(config_file, node_parameters)
        self._save_plan()

        self.ready = False
        self.done = False
        self.q_goals = {}       # target -> arm configuration MoveIt moved to
        self._collision_call = None
        self._tool0_to = {}
        self._step_index = -1
        self._step = None
        self._results = set()   # targets whose result is recorded
        self._shift_m = 0.0     # current sideways shift of the base off its lane
        self._run_start = None
        self._validity_call = None
        self._last_validity_check = -np.inf
        self._step_correction = None   # SLAM correction at the end of the last step
        self._last_reports = []
        self._last_command = None
        self._last_arm_command = -np.inf
        self._have_arm = False
        self.arm_reference_lag = 0.0

    # ------------------------------------------------------------------
    # Mission hooks
    # ------------------------------------------------------------------

    def load_targets(self, config: dict) -> list:
        """Output: list[dict] with at least {"id", "name", "frame", "position", "look_at", "roll"}."""
        raise NotImplementedError

    def target_steps(self, t: int, stop_number: int) -> list:
        """Output: the mission's steps for target t after move_frame (list of step dicts)."""
        raise NotImplementedError

    def record_result(self, t: int, status: str, detail: str = "") -> None:
        """Store how target t ended (called once per target)."""
        raise NotImplementedError

    def save_results(self) -> None:
        """Write the mission's plots and summary to run_dir."""

    def check_plan(self) -> None:
        """Mission-specific checks of the planned stops; raise ValueError to refuse the run."""

    def on_setup(self, io) -> bool:
        """Mission setup once TF is available; False to be called again next tick."""
        return True

    def record_tick(self, io, reports) -> None:
        """Store one control tick (reports: this tick's HQP level reports, or [])."""

    # ------------------------------------------------------------------
    # Loading and planning
    # ------------------------------------------------------------------

    def _read_navigation(self, nav: dict) -> None:
        """Store the navigation section (loop, localization checks, retries).

        Raises:
            ValueError: without navigation.loop_path.
        """
        loop = nav.get("loop_path")
        if loop is None:
            raise ValueError("navigation.loop_path is required (base stops lie on it)")
        self.loop = LoopPath(loop["corners"], float(loop["radius"]), float(loop.get("spacing", 0.05)))
        self.loop_overshoot_m = float(loop.get("max_overshoot_m", 0.6))
        self.nav_retries = int(nav.get("retries", 0))
        # follow_loop: FollowPath along the loop (boat mode, its arcs); navigate_to_pose:
        # Nav2 plans the path to each stop itself (NavigateToPose, its behavior tree).
        self.drive = nav.get("drive", "follow_loop")
        if self.drive not in ("follow_loop", "navigate_to_pose"):
            raise ValueError("navigation.drive must be follow_loop or navigate_to_pose")
        self.nav_retry_delay_s = float(nav.get("retry_delay_s", 5.0))
        self.localization = LocalizationMonitor(
            nav.get("workspace"), float(nav.get("settle_s", 0.0)),
            nav.get("settle_tolerance", [0.05, 0.03]), nav.get("jump_tolerance", [0.3, 0.2]),
            float(nav.get("localization_timeout_s", 60.0)))
        # Pause SLAM's new measurements while working at a stop (sideways
        # shifts, arm motions, passes, straight moves along the lane): along a
        # corridor-like lane slow motion lets the scan matcher slide by metres.
        # Odometry alone is accurate over that work; SLAM resumes for every
        # Nav2 drive (and the initial localization).
        self.pause_localization = bool(nav.get("pause_localization_at_stops",
                                               nav.get("pause_localization_during_passes", False)))
        self.localization_paused = False
        self.start_pose = nav.get("start_pose")
        self.start_tolerance = nav.get("start_tolerance", [0.3, 0.2])
        self.start_settle_s = float(nav.get("start_settle_s", 5.0))

    def frame_alias(self, config: dict, frame: str) -> str:
        """Output: model frame for a target's `frame` (an alias in `frames:` or a frame name)."""
        return (config.get("frames") or {}).get(frame, frame)

    def _plan_stops(self, config: dict, xacro_path: str, mappings: dict) -> list:
        """Place the base stops for every target (navigation.BasePlanner, boat mode).

        The planner gets its own robot model, so the live one is untouched.

        Output:
            list[Stop]: stops in driving order from navigation.start_pose.

        Raises:
            ValueError: if a target is unreachable from every loop spot.
        """
        placement = config.get("base_placement") or {}
        self.base_placement = placement
        offsets = [float(v) for v in placement.get("offsets_m", [0.0, 0.1, 0.2, 0.3])]
        if any(v > self.base_task.max_offset_m + 1e-9 for v in offsets):
            raise ValueError(f"base_placement.offsets_m must be <= the base task's max_offset_m "
                             f"({self.base_task.max_offset_m})")
        planner = BasePlanner(build_robot(config, xacro_path, mappings, self.ee_frame), self.loop,
                              collision_boxes=self.moveit_config.get("collision_objects", []),
                              seeds=[self.travel_q.tolist()] + list(placement.get("seeds", [])),
                              machine_side=placement.get("machine_side", "right"), offsets_m=offsets,
                              footprint=placement.get("footprint", [1.2, 0.7]),
                              max_reach_m=float(placement.get("max_reach_m", 0.95)),
                              shoulder_pan_range_rad=placement.get("shoulder_pan_range_rad"),
                              base_clearance_m=float(placement.get("base_clearance_m", 0.1)),
                              min_straight_before_stop_m=float(placement.get("min_straight_before_stop_m", 0.0)))
        self.planner = planner
        targets = [Target(frame=t["frame"], position=t["position"], look_at=t["look_at"],
                          roll=t["roll"], name=t["name"], down=t.get("down")) for t in self.targets]
        start = self.start_pose[:2] if self.start_pose is not None else self.loop.points[0][:2]
        spacing = float(np.mean(np.hypot(*np.diff(self.loop.points[:, :2], axis=0).T)))
        stride = max(1, int(round(float(placement.get("spacing_m", 0.05)) / spacing)))
        started = time.monotonic()
        stops, unreachable = planner.plan(targets, start, stride=stride)
        if unreachable:
            raise ValueError("unreachable from every base spot: " + "; ".join(
                f"{self.targets[t]['name']} (rejected spots: {planner.rejected.get(self.targets[t]['name'], {})})"
                for t in unreachable))
        for number, stop in enumerate(stops, start=1):
            x, y, yaw = stop.pose
            self.log.info(f"Base stop {number}: ({x:+.2f}, {y:+.2f}, {math.degrees(yaw):+.0f} deg), "
                          f"{stop.distance:.2f} m along the loop, shift {stop.offset:.2f} m: "
                          + ", ".join(self.targets[t]["name"] for t in stop.targets))
        if self.arm_between_stops == "hold":
            self.held_drives = set(range(len(stops) - 1))
        elif self.arm_between_stops == "section":
            section = lambda t: self.targets[t].get("section")  # noqa: E731
            self.held_drives = {k for k in range(len(stops) - 1)
                                if section(stops[k].targets[-1]) is not None
                                and section(stops[k].targets[-1]) == section(stops[k + 1].targets[0])}
        # A target with travel_before: true is always driven to with the arm in
        # travel_arm_q (e.g. around a corner where the held arm would hit).
        self.held_drives -= {k for k in range(len(stops) - 1)
                             if any(self.targets[t].get("travel_before") for t in stops[k + 1].targets)}
        # travel_through_corners: a drive through one of the loop's arcs is done
        # with the arm in travel_arm_q (the base turns with the arm stretched out).
        if config.get("travel_through_corners", False):
            n = len(self.loop.points)
            def through_corner(k):
                span = (stops[k + 1].index - stops[k].index) % n
                return any(self.loop.curvature[(stops[k].index + i) % n] != 0.0 for i in range(span + 1))
            cornering = {k for k in self.held_drives if through_corner(k)}
            if cornering:
                self.log.info("travel_through_corners: arm in travel posture on the drives through a corner: "
                              + ", ".join(f"stop {k + 1} -> {k + 2}" for k in sorted(cornering)))
            self.held_drives -= cornering
        if self.held_drives:
            self._check_held_arm(planner, stops)
        self.log.info(f"Base placement planned in {time.monotonic() - started:.1f} s")
        return stops

    def _check_held_arm(self, planner: BasePlanner, stops: list) -> None:
        """Pick the arm configuration of each drive that keeps the arm (self.held_drives).

        Checked with the planner's IK solutions (MoveIt's goals are seeded with
        them) against the collision boxes, from the shift back at one stop to
        the shift out at the next: the last target's configuration is kept if
        it stays clear, else the next stop's first target's one is taken
        before driving (the camera still faces the machine); self.preposition
        maps the stop index to that configuration.

        Raises:
            ValueError: listing each drive where neither configuration is clear.
        """
        self.preposition, blocked = {}, []
        for number, (stop, following) in enumerate(zip(stops[:-1], stops[1:]), start=1):
            if number - 1 not in self.held_drives:
                continue
            last, first = stop.targets[-1], following.targets[0]
            poses = planner.transit_poses(stop, following)
            hit = planner.arm_collision_along(stop.q[last], poses, self.targets[last]["frame"])
            if hit is None:
                continue
            if planner.arm_collision_along(following.q[first], poses, self.targets[first]["frame"]) is None:
                self.preposition[number - 1] = [float(v) for v in following.q[first]]
                self.log.info(f"Drive stop {number} -> {number + 1}: the arm at '{self.targets[last]['name']}' "
                              f"would hit a box at ({hit[0]:+.2f}, {hit[1]:+.2f}); it takes "
                              f"'{self.targets[first]['name']}''s configuration before driving")
                continue
            blocked.append(f"stop {number} -> {number + 1} (arm at '{self.targets[last]['name']}' or "
                           f"'{self.targets[first]['name']}', base at ({hit[0]:+.2f}, {hit[1]:+.2f}))")
        if blocked:
            raise ValueError(f"arm_between_stops: {self.arm_between_stops}: the arm hits a collision box "
                             "driving " + "; ".join(blocked))
        self.log.info(f"arm_between_stops: {self.arm_between_stops}: the arm is kept on "
                      f"{len(self.held_drives)} of {len(stops) - 1} drives, clear on all "
                      f"({len(self.preposition)} with the next target's configuration)")

    def _build_steps(self) -> list:
        """Turn the stops into the step list (see the module docstring).

        Output:
            list[dict]: {"type", ..., "stop", "target" (or None), "always"}.
        """
        steps = [{"type": "localize", "stop": None, "target": None, "always": True}]
        for number, stop in enumerate(self.stops):
            lane_pose = tuple(self.loop.points[stop.index])
            if number == 0 or number - 1 not in self.held_drives:
                steps.append({"type": "move_arm", "q": "travel", "stop": number, "target": None})
            steps.append({"type": "move_base", "pose": lane_pose, "stop": number, "target": None})
            if stop.offset > 0.0:
                steps.append({"type": "shift_base", "offset": stop.offset, "yaw": lane_pose[2],
                              "stop": number, "target": None})
            for t in stop.targets:
                steps.append({"type": "move_frame", "target": t, "seed": stop.q[t], "stop": number})
                steps.extend(self.target_steps(t, number))
            if number in self.preposition:
                steps.append({"type": "move_arm", "q": self.preposition[number], "stop": number,
                              "target": None, "always": True})
            # With approach_step_m a stop may end shifted even when none was
            # planned (a closer approach after an IK failure): always shift back.
            if stop.offset > 0.0 or float(self.base_placement.get("approach_step_m", 0.0)) > 0.0:
                if number not in self.held_drives:
                    steps.append({"type": "move_arm", "q": "travel", "stop": number, "target": None,
                                  "always": True})
                # Undoes the shift actually done (none if the stop was skipped).
                steps.append({"type": "shift_base", "back": True, "yaw": lane_pose[2],
                              "stop": number, "target": None, "always": True})
        steps.append({"type": "move_arm", "q": "travel", "stop": None, "target": None, "always": True})
        return steps

    # ------------------------------------------------------------------
    # Measurements
    # ------------------------------------------------------------------

    def on_arm_state(self, q_arm, dq_arm, now: float) -> None:
        """Update the model's arm from the measured joints (see arm_reference_reset_rad)."""
        q_arm = np.asarray(q_arm, dtype=float)
        if self._have_arm and now - self._last_arm_command < 0.1:
            lag = float(np.max(np.abs(self.robot.q[self.robot.arm_q_slice] - q_arm)))
            if lag < self.arm_reference_reset_rad:
                self.arm_reference_lag = max(self.arm_reference_lag, lag)
                return
            self.log.warning(f"Arm {lag:.3f} rad away from the HQP reference; resyncing to the "
                             "measurement", throttle_duration_sec=2.0)
        self.robot.set_arm_state(q_arm, dq_arm)
        self._have_arm = True

    def on_base_state(self, x: float, y: float, yaw: float, vx: float, vy: float, wz: float) -> None:
        """Update the model's base from odometry (pose in odom, body-frame twist)."""
        self.robot.set_base_state(x, y, yaw, vx, vy, wz)

    @property
    def arm_joint_names(self) -> list:
        """Output: the arm joint names, in HQP order."""
        return self.robot.arm_joint_names

    def joint_limits(self, names=None):
        """Output: the arm joints' limits (robot.JointLimits)."""
        return self.robot.joint_limits(names)

    def base_pose(self):
        """Output: (x, y, yaw) of the base in odom (model state)."""
        x, y, c, s = self.robot.q[self.robot.base_q_slice]
        return float(x), float(y), math.atan2(s, c)

    # ------------------------------------------------------------------
    # Executor
    # ------------------------------------------------------------------

    def tick(self, io) -> None:
        """Advance the mission by one control period."""
        if not self.ready:
            self._setup(io)
            return
        if self.done:
            return
        if self._step is None:
            self._next_step(io)
            if self.done:
                return
        step = self._step
        result = getattr(self, f"_step_{step['type']}")(io, step, step["data"])
        if self._run_start is None:
            self._run_start = io.now()
        self.record_tick(io, self._last_reports)
        self._last_reports = []
        if result is not None:
            self._end_step(io, *result)

    def _setup(self, io) -> None:
        """Mission setup, then add the collision objects to MoveIt's planning scene."""
        if not self.on_setup(io):
            return
        if self._collision_call is None:
            if not io.moveit_ready():
                io.log.warning("Waiting for MoveIt services", throttle_duration_sec=5.0)
                return
            tool0 = self.robot.frame_pose(self.moveit_config.get("ik_link", "robot_arm_tool0"))
            for frame in {t["frame"] for t in self.targets}:
                self._tool0_to[frame] = (tool0.inverse() * self.robot.frame_pose(frame)).homogeneous
            self._collision_call = io.apply_collision_objects(
                self.moveit_config.get("collision_objects", []))
            return
        if self._collision_call.poll() is None:
            return
        io.log.info(f"Added {len(self.moveit_config.get('collision_objects', []))} collision "
                    "object(s) to the MoveIt planning scene")
        self.ready = True

    def _next_step(self, io) -> None:
        """Start the next step that is not skipped, or finish the run."""
        while True:
            self._step_index += 1
            if self._step_index >= len(self.steps):
                self._finish(io)
                return
            step = self.steps[self._step_index]
            if not step.get("skipped"):
                break
        step["data"] = {"start": io.now()}
        self._step = step
        # SLAM runs for the initial localization and Nav2 drives only.
        self.set_localization_paused(io, step["type"] not in ("localize", "move_base"))
        target = step.get("target")
        label = f" '{self.targets[target]['name']}'" if target is not None else ""
        where = f" (stop {step['stop'] + 1}/{len(self.stops)})" if step.get("stop") is not None else ""
        io.log.info(f"Step {self._step_index + 1}/{len(self.steps)}: {step['type']}{label}{where}")

    def _end_step(self, io, ok: bool, detail: str = "", status: str = None) -> None:
        """Close the current step; on failure record it and skip what depends on it.

        A failed move_base/shift_base/move_arm skips the rest of its stop, a
        failed target step the rest of its target (steps with `always` still
        run); a failed localize or final stow ends the run.
        """
        step = self._step
        self._step = None
        io.publish_twist(0.0, 0.0, 0.0)
        self._record_step(io, step, ok, status)
        # Localization guard: SLAM's correction (odom in the map) must not
        # jump during a step (e.g. lost along a lane during a long pass); a
        # base driven on a wrong map pose ends in the machine or a wall.
        correction = io.localization_correction()
        jump = self.localization.jump(self._step_correction, correction)
        if correction is not None:
            self._step_correction = correction
        if jump is not None and step["type"] != "localize":
            io.log.error(f"Localization jumped {jump[0]:.2f} m / {jump[1]:.2f} rad during step "
                         f"{step['type']}: the map pose is not trusted; ending the run")
            for target in self.targets:
                self._result(target["id"], "localization_jump", f"{jump[0]:.2f} m during {step['type']}")
            io.hold_arm(self.robot.q[self.robot.arm_q_slice])
            self._finish(io)
            return
        if ok:
            return
        status = status or f"{step['type']}_failed"
        io.log.error(f"Step {step['type']} failed: {status} ({detail})")
        if step["type"] == "localize" or (step["type"] == "move_arm" and step.get("always")):
            # Never drive or shift the base with the arm out.
            for target in self.targets:
                self._result(target["id"], "not_run", detail)
            self._finish(io)
            return
        if step["type"] in ("move_base", "shift_base", "move_arm"):
            key, value = "stop", step["stop"]
            affected = self.stops[value].targets if value is not None else []
        else:
            key, value = "target", step["target"]
            affected = [value]
        for later in self.steps[self._step_index + 1:]:
            if later.get(key) == value and not later.get("always"):
                later["skipped"] = True
        for t in affected:
            self._result(t, status, detail)

    def _finish(self, io) -> None:
        """Stop the robot and write the results and the executed trajectory."""
        self.done = True
        io.publish_twist(0.0, 0.0, 0.0)
        for target in self.targets:
            self._result(target["id"], "not_run")
        io.log.info("All steps done; writing the run results.")
        self.save_results()
        self.save_trajectory()

    # ------------------------------------------------------------------
    # Executed trajectory
    # ------------------------------------------------------------------

    @staticmethod
    def _rounded(values, digits: int = 4):
        """Output: list of floats rounded for the YAML (None stays None)."""
        return None if values is None else [round(float(v), digits) for v in values]

    def _record_step(self, io, step: dict, ok: bool, status: str = None) -> None:
        """Store where a step left the robot: base (robot_map), arm joints, MoveIt goal."""
        target = step.get("target")
        record = {"step": self._step_index + 1, "type": step["type"],
                  "target": None if target is None else self.targets[target]["name"],
                  "stop": None if step.get("stop") is None else step["stop"] + 1,
                  "result": "ok" if ok else (status or "failed"),
                  "base_map": self._rounded(io.base_in_map()),
                  "arm_q": self._rounded(self.robot.q[self.robot.arm_q_slice])}
        if step["type"] == "move_frame" and target in self.q_goals:
            record["moveit_goal_q"] = self._rounded(self.q_goals[target])
        self.trajectory["steps"].append(record)

    def record_pass_sample(self, io, name: str, s: float, spacing_m: float = 0.02) -> None:
        """Store a pass sample (path parameter, base in robot_map, arm joints) every spacing_m."""
        samples = self.trajectory["passes"].setdefault(name, [])
        if samples and s - samples[-1]["s_m"] < spacing_m:
            return
        samples.append({"s_m": round(float(s), 4), "base_map": self._rounded(io.base_in_map()),
                        "arm_q": self._rounded(self.robot.q[self.robot.arm_q_slice])})

    def save_trajectory(self) -> None:
        """Write the executed trajectory to run_dir/trajectory.yaml and trajectory_dir/<experiment>.yaml."""
        if not self.trajectory["steps"]:
            return
        document = {
            "experiment": self.experiment_name, "mission": self.name,
            "run": os.path.basename(self.run_dir or ""), "frame": "robot_map",
            "joints": list(self.robot.arm_joint_names),
            "stops": [{"stop": number + 1, "base_pose": self._rounded(stop.pose, 3),
                       "shift_m": float(stop.offset),
                       "targets": [self.targets[t]["name"] for t in stop.targets]}
                      for number, stop in enumerate(self.stops)],
            "steps": self.trajectory["steps"],
            "passes": self.trajectory["passes"],
        }
        paths = [os.path.join(self.run_dir, "trajectory.yaml")] if self.run_dir else []
        if self.trajectory_dir:
            paths.append(os.path.join(self.trajectory_dir, f"{self.experiment_name or self.name}.yaml"))
        for path in paths:
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as handle:
                    handle.write(f"# Executed trajectory of {self.experiment_name or self.name} "
                                 f"(run {document['run']}), written by the mission at the end of the run.\n")
                    yaml.safe_dump(document, handle, sort_keys=False, default_flow_style=None)
                self.log.info(f"Trajectory written to {path}")
            except OSError as exc:
                self.log.error(f"Could not write the trajectory to {path}: {exc}")

    def _result(self, t: int, status: str, detail: str = "") -> None:
        """Record target t's result once (the first status wins)."""
        if t in self._results:
            return
        self._results.add(t)
        self.record_result(t, status, detail)

    # ------------------------------------------------------------------
    # Generic steps
    # ------------------------------------------------------------------

    def _step_localize(self, io, step: dict, data: dict):
        """Wait until localization is stable, inside the workspace and at start_pose."""
        io.publish_twist(0.0, 0.0, 0.0)
        base = io.base_in_map()
        if not self._inside_workspace(io, base, data):
            data.pop("settle_reference", None)
            return self._workspace_timeout(io, data)
        correction = io.localization_correction()
        if correction is None or base is None:
            return None
        x, y, yaw = base
        if self.start_pose is not None:
            sx, sy, syaw = (float(v) for v in self.start_pose)
            distance, dyaw = math.hypot(x - sx, y - sy), abs(wrap_angle(yaw - syaw))
            if distance > self.start_tolerance[0] or dyaw > self.start_tolerance[1]:
                data.pop("settle_reference", None)
                io.log.warning(f"Localized at ({x:.2f}, {y:.2f}, {yaw:+.2f}), {distance:.2f} m / "
                               f"{dyaw:.2f} rad from the start pose; waiting for localization",
                               throttle_duration_sec=5.0)
                return None
        if not self.localization.settled(data, correction, io.now(), max(self.start_settle_s, 1e-6)):
            return None
        io.log.info(f"Localized at ({x:.2f}, {y:.2f}, {yaw:+.2f})")
        return True, ""

    def _step_move_arm(self, io, step: dict, data: dict):
        """MoveIt joint motion to step['q'] ("travel" = travel_arm_q)."""
        q_goal = self.travel_q if step["q"] == "travel" else np.asarray(step["q"], dtype=float)
        if "motion" not in data:
            if np.max(np.abs(self.robot.q[self.robot.arm_q_slice] - q_goal)) < ARM_AT_GOAL_RAD:
                return True, ""
            data["motion"] = {}
            io.publish_twist(0.0, 0.0, 0.0)
        outcome = self._poll_arm_motion(io, data["motion"], q_goal)
        if outcome is None:
            return None
        return (True, "") if outcome[0] else (False, outcome[1], "stow_failed")

    def _step_move_base(self, io, step: dict, data: dict):
        """Drive the loop to step['pose'] with Nav2, then gate: aligned with the lane, static."""
        x, y, yaw = step["pose"]
        if data.get("phase") is None:
            data.update(phase="nav", attempts=0, retry_at=-np.inf, call=None)
        if data["phase"] == "nav":
            if data["call"] is None:
                if io.now() < data["retry_at"]:
                    return None
                ready = io.navigate_to_pose_ready() if self.drive == "navigate_to_pose" else io.follow_path_ready()
                if not ready:
                    io.log.warning(f"Waiting for Nav2 ({self.drive})", throttle_duration_sec=5.0)
                    return None
                base = io.base_in_map()
                if base is None:
                    return None
                if not self._inside_workspace(io, base, data):
                    return self._workspace_timeout(io, data)
                if not self.localization.settled(data, io.localization_correction(), io.now()):
                    io.log.info("Waiting for localization to settle", throttle_duration_sec=5.0)
                    return None
                if self.drive == "navigate_to_pose":
                    distance = math.hypot(x - base[0], y - base[1])
                    if distance < STOP_GAP_M:
                        io.log.info(f"Base {distance:.2f} m from the stop; moving straight to it")
                        data["phase"] = "gate"
                        self.set_localization_paused(io, True)
                    else:
                        io.log.info(f"Nav2 NavigateToPose to the stop ({x:+.2f}, {y:+.2f}, "
                                    f"{math.degrees(yaw):+.0f} deg), {distance:.2f} m away")
                        io.publish_twist(0.0, 0.0, 0.0)
                        data["call"] = io.navigate_to_pose(x, y, yaw)
                        data["last_correction"] = io.localization_correction()
                        return None
                if self.drive == "follow_loop":
                    # Distance to the stop ahead along the loop (driving order);
                    # a base slightly past it (overshoot) works from where it is.
                    ahead = self.loop.forward_distance(self.loop.nearest(*base[:2]), self.loop.nearest(x, y))
                    behind = self.loop.total_length - ahead
                    if ahead < STOP_GAP_M or behind < self.loop_overshoot_m:
                        # Close to (or past) the stop on its lane: go there in a
                        # straight line along the lane (backing up if past), heading
                        # held, no turning; Nav2 only drives forward.
                        io.log.info(f"Base {0.0 if ahead < STOP_GAP_M else behind:.2f} m past the stop; "
                                    "moving straight to it along the lane")
                        data["phase"] = "gate"
                        self.set_localization_paused(io, True)
                    else:
                        io.log.info(f"Driving {ahead:.2f} m along the loop to the stop")
                        samples = [tuple(p) for p in self.loop.segment(base[:2], (x, y))[:-1]] + [(x, y, yaw)]
                        io.publish_twist(0.0, 0.0, 0.0)
                        data["call"] = io.follow_path(samples)
                        data["last_correction"] = io.localization_correction()
                        return None
            else:
                # A localization jump while driving: stop and retry.
                correction = io.localization_correction()
                jump = self.localization.jump(data.get("last_correction"), correction)
                result = data["call"].poll()
                if jump is not None and result is None:
                    data["call"].cancel()
                    result = (False, f"localization jumped {jump[0]:.2f} m / {jump[1]:.2f} rad while driving")
                if correction is not None:
                    data["last_correction"] = correction
                if result is None:
                    base = io.base_in_map()
                    if base is not None:
                        left = (math.hypot(x - base[0], y - base[1]) if self.drive == "navigate_to_pose" else
                                self.loop.forward_distance(self.loop.nearest(*base[:2]), self.loop.nearest(x, y)))
                        io.log.info(f"Driving to the stop: base ({base[0]:.2f}, {base[1]:.2f}), "
                                    f"{left:.2f} m to go", throttle_duration_sec=15.0)
                    return None
                if not result[0]:
                    data["call"] = None
                    if data["attempts"] < self.nav_retries:
                        data["attempts"] += 1
                        data["retry_at"] = io.now() + self.nav_retry_delay_s
                        io.log.warning(f"{result[1]}; retrying in {self.nav_retry_delay_s:.0f} s "
                                       f"({data['attempts']}/{self.nav_retries})")
                        return None
                    return False, result[1], "nav_failed"
                io.log.info("Arrived at the base stop")
                data["phase"] = "gate"
                self.set_localization_paused(io, True)
        if "gate" not in data:
            # Nav2 stops within its goal tolerance (~0.1 m): the gate then
            # takes the base the rest of the way to the planned stop in a
            # straight line (the arm reach at a stop can depend on a few cm),
            # and anchors the lane (and the sideways shift) there.
            to_odom = io.to_odom()
            if to_odom is None:
                return None
            rotation, translation, _ = to_odom
            goal = rotation @ np.array([x, y, 0.0]) + translation
            data["gate"] = self._begin_gate(io, float(goal[0]), float(goal[1]), yaw)
            bx, by, _ = self.base_pose()
            io.log.info(f"Moving the base {math.hypot(goal[0] - bx, goal[1] - by):.3f} m straight "
                        "to the planned stop")
        return self._poll_gate(io, data["gate"])

    def _step_shift_base(self, io, step: dict, data: dict):
        """Straight sideways shift of step['offset'] m towards the machine.

        With step['back'] it undoes the shift done so far, back onto the
        lane; nothing to do if there was none.
        """
        if "gate" not in data:
            offset = -self._shift_m if step.get("back") else float(step["offset"])
            if abs(offset) < 1e-6:
                return True, ""
            bx, by, _ = self.base_pose()
            data["gate"] = self._begin_gate(io, bx, by, step["yaw"])
            goal = self.base_task.shifted(offset)
            data["gate"]["goal"] = (goal[0], goal[1])
            data["offset"] = offset
            io.log.info(f"Shifting the base {offset:+.2f} m towards the machine")
        result = self._poll_gate(io, data["gate"])
        if result is not None and result[0]:
            self._shift_m += data["offset"]
        return result

    def _step_move_frame(self, io, step: dict, data: dict):
        """MoveIt IK (seeded with the planner's solution, all rolls if free) and joint motion.

        If MoveIt finds no collision-free solution, or cannot plan the motion
        to it, the IK is asked again with other seeds: the arm's current
        configuration, then both seeds with the shoulder on the other side
        (pan + pi; IK solutions near one seed can all touch the robot itself).
        On success the arm configuration is stored in self.q_goals[target].
        """
        target = self.targets[step["target"]]
        if data.get("phase") == "approach":
            return self._poll_ik_approach(io, target, data)
        if data.get("phase") is None and self.moveit_config.get("base_sync") and not data.get("synced"):
            data.update(phase="sync", sync_start=io.now())
        if data.get("phase") == "sync":
            return self._poll_base_sync(io, target, data)
        if data.get("phase") is None:
            rolls = ([target["roll"]] if target["roll"] is not None else np.linspace(
                0.0, 2.0 * np.pi, int(self.moveit_config.get("ik_roll_samples", 12)), endpoint=False))
            seed = self._ik_seeds(step)[data.get("attempt", 0)]
            call = io.request_ik(target["position"], target["look_at"], self._tool0_to[target["frame"]],
                                 seed, rolls, target.get("down"))
            if call is None:
                return None
            data.update(phase="ik", call=call)
            self.hold_base()
            return None
        if data["phase"] == "ik":
            result = data["call"].poll()
            if result is None:
                return None
            if not result[0]:
                return self._next_ik_seed(io, step, data, result[1], "unreachable_ik")
            io.log.info(f"Target '{target['name']}': MoveIt goal q={np.round(result[1], 3).tolist()}")
            data.update(phase="motion", q_goal=np.asarray(result[1], dtype=float), motion={})
            return None
        outcome = self._poll_arm_motion(io, data["motion"], data["q_goal"])
        if outcome is None:
            return None
        if not outcome[0]:
            return self._next_ik_seed(io, step, data, f"no motion to that IK solution ({outcome[1]})",
                                      "approach_failed")
        self.q_goals[step["target"]] = data["q_goal"]
        return True, ""

    def _poll_base_sync(self, io, target: dict, data: dict):
        """moveit.base_sync: wait for a still base before the IK of a target.

        The base must be static (base_lane static_speed), the localization
        correction (robot_map <- robot_odom) stable within tolerance_m /
        tolerance_rad for settle_s, and move_group's own base pose (its
        collision checks) within tolerance_m / tolerance_rad of the node's.
        After timeout_s the target fails (base_not_synced).
        """
        sync = self.moveit_config["base_sync"]
        tolerance = (float(sync.get("tolerance_m", 0.03)), float(sync.get("tolerance_rad", 0.03)))
        now = io.now()
        if now - data["sync_start"] > float(sync.get("timeout_s", 30.0)):
            return False, f"base not still/synced after {sync.get('timeout_s', 30.0)} s ({data.get('sync_detail', '')})", \
                "base_not_synced"
        vx, vy, wz = self.robot.dq[self.robot.base_v_slice]
        correction, base = io.localization_correction(), io.base_in_map()
        if correction is None or base is None:
            data["sync_detail"] = "no localization"
            return None
        reference = data.get("sync_reference")
        if (max(math.hypot(vx, vy), abs(wz)) > self.base_task.static_speed or reference is None
                or any(c > t for c, t in zip(correction_change(correction, reference), tolerance))):
            data.update(sync_reference=correction, sync_since=now, sync_detail="base or localization moving")
            return None
        if now - data["sync_since"] < float(sync.get("settle_s", 1.0)):
            return None
        call = data.get("scene_call")
        if call is None:
            data["scene_call"] = io.moveit_base_pose()
            return None
        result = call.poll()
        if result is None:
            return None
        data["scene_call"] = None
        if not result[0]:
            data["sync_detail"] = result[1]
            return None
        mx, my, myaw = result[1]
        offset = (math.hypot(mx - base[0], my - base[1]), abs(wrap_angle(myaw - base[2])))
        if offset[0] > tolerance[0] or offset[1] > tolerance[1]:
            data["sync_detail"] = f"move_group's base {offset[0] * 100:.1f} cm / {offset[1]:.3f} rad off"
            return None
        io.log.info(f"Target '{target['name']}': base still, move_group's base within "
                    f"{offset[0] * 100:.1f} cm / {offset[1]:.3f} rad (after {now - data['sync_start']:.1f} s)")
        data.update(phase=None, synced=True)
        return None

    def _ik_seeds(self, step: dict) -> list:
        """Output: IK seeds tried in order: the planner's, the current arm, both with the pan + pi."""
        seeds = [np.asarray(step["seed"], dtype=float), self.robot.q[self.robot.arm_q_slice].copy()]
        for seed in list(seeds):
            flipped = seed.copy()
            flipped[0] = math.atan2(math.sin(seed[0] + math.pi), math.cos(seed[0] + math.pi))
            seeds.append(flipped)
        return seeds

    def _next_ik_seed(self, io, step: dict, data: dict, detail: str, status: str):
        """Retry move_frame with the next IK seed, or fail it with `status` after the last."""
        attempt = data.get("attempt", 0) + 1
        if attempt >= len(self._ik_seeds(step)):
            if status == "unreachable_ik" and self._begin_ik_approach(io, step, data, detail):
                return None
            return False, detail, status
        io.log.warning(f"Target '{self.targets[step['target']]['name']}': {detail}; retrying the IK "
                       f"with seed {attempt + 1}/{len(self._ik_seeds(step))}")
        data.clear()
        data["attempt"] = attempt
        return None

    def _begin_ik_approach(self, io, step: dict, data: dict, detail: str) -> bool:
        """No IK from this spot: plan a closer sideways (omnidirectional) approach, boat-like.

        base_placement.approach_step_m further towards the machine (lane heading
        kept), up to the base task's max_offset_m, if the footprint keeps
        base_clearance_m from the collision boxes and the arm, as it is or else
        in travel_arm_q, stays clear of them over the shift.

        Output:
            bool: True if an approach was started (move_frame then retries its
                IK from the first seed), False if none is possible.
        """
        step_m = float(self.base_placement.get("approach_step_m", 0.0))
        if step_m <= 0.0 or step.get("stop") is None:
            return False
        stop = self.stops[step["stop"]]
        offset = self._shift_m + step_m
        name = self.targets[step["target"]]["name"]
        if offset > self.base_task.max_offset_m + 1e-9:
            io.log.warning(f"Target '{name}': {detail}; no closer approach (shift {self._shift_m:.2f} m "
                           f"is the base's max_offset_m)")
            return False
        poses = [self.planner.base_pose(stop.index, o) for o in np.arange(self._shift_m, offset + 1e-9, 0.05)]
        if not self.planner.footprint_clear(*poses[-1]):
            io.log.warning(f"Target '{name}': {detail}; no closer approach (the footprint would come within "
                           f"{self.planner.base_clearance_m} m of the machine at shift {offset:.2f} m)")
            return False
        frame = self.targets[step["target"]]["frame"]
        q_arm = self.robot.q[self.robot.arm_q_slice].copy()
        stow = None
        if self.planner.arm_collision_along(q_arm, poses, frame) is not None:
            if self.planner.arm_collision_along(self.travel_q, poses, frame) is not None:
                io.log.warning(f"Target '{name}': {detail}; no closer approach (the arm would hit the machine)")
                return False
            stow = {}
        io.log.warning(f"Target '{name}': {detail}; approaching the machine {step_m:.2f} m sideways "
                       f"(shift {offset:.2f} m){', arm in travel posture first' if stow is not None else ''}, "
                       "then retrying the IK")
        data.clear()
        data.update(phase="approach", stow=stow, shift={}, shift_step={"offset": step_m,
                                                                       "yaw": float(self.loop.points[stop.index][2])})
        return True

    def _poll_ik_approach(self, io, target: dict, data: dict):
        """Stow the arm if needed, shift the base sideways, then restart the target's IK."""
        if data["stow"] is not None:
            outcome = self._poll_arm_motion(io, data["stow"], self.travel_q)
            if outcome is None:
                return None
            if not outcome[0]:
                return False, f"approach: no arm motion to travel_arm_q ({outcome[1]})", "approach_failed"
            data["stow"] = None
        result = self._step_shift_base(io, data["shift_step"], data["shift"])
        if result is None:
            return None
        if not result[0]:
            return False, f"approach: {result[1]}", "approach_failed"
        io.log.info(f"Target '{target['name']}': base now {self._shift_m:.2f} m off its lane; retrying the IK")
        data.clear()
        return None

    # ------------------------------------------------------------------
    # HQP stepping, base gate, arm motions
    # ------------------------------------------------------------------

    def _optional_levels(self, task) -> set:
        """Output: names of the levels below `task`'s level (regularization only)."""
        names = list(self.level_tasks)
        for index, name in enumerate(names):
            if task in self.level_tasks[name]:
                return set(names[index + 1:])
        return set()

    def reset_validity(self) -> None:
        """Forget pending MoveIt validity checks (a new HQP motion starts)."""
        self._validity_call = None
        self._last_validity_check = -np.inf

    def hqp_step(self, io, task):
        """One HQP step, published, with the MoveIt lookahead collision check.

        Input:
            task: the step's main task; failures of the levels below it only
                are tolerated (the higher levels' solution is still safe).

        Output:
            tuple[str, str]: ("ok", ""), ("failed", reason) (zero twist, the
                arm holds its last point) or ("collision", contacts) (nothing
                published).
        """
        self._last_reports = []
        if self._validity_call is not None:
            result = self._validity_call.poll()
            if result is not None:
                self._validity_call = None
                if not result[0]:
                    io.publish_twist(0.0, 0.0, 0.0)
                    return "collision", result[1]
        try:
            command, reports = self.controller.step(self.dt)
        except (ValueError, RuntimeError) as exc:
            io.log.error(f"HQP step failed: {exc}", throttle_duration_sec=1.0)
            io.publish_twist(0.0, 0.0, 0.0)
            return "failed", str(exc)
        self._last_reports = reports
        failed = [r for r in reports if r.status != "solved"]
        optional = self._optional_levels(task)
        if failed and all(report.name in optional for report in failed):
            io.log.warning(f"level {failed[0].name}: {failed[0].status}; sending the "
                           "higher-priority levels' solution", throttle_duration_sec=2.0)
        elif failed:
            for report in failed:
                io.log.warning(f"level {report.name}: {report.status} "
                               f"(residual={report.residual_norm:.4g})", throttle_duration_sec=1.0)
            io.publish_twist(0.0, 0.0, 0.0)
            return "failed", f"level {failed[0].name} {failed[0].status}"
        now = io.now()
        rate = float(self.moveit_config.get("validity_rate_hz", 20.0))
        if self._validity_call is None and now - self._last_validity_check >= 1.0 / rate:
            q_ahead = command.arm_q_target + command.arm_dq * VALIDITY_LOOKAHEAD_S
            self._validity_call = io.check_validity(q_ahead)
            if self._validity_call is not None:
                self._last_validity_check = now
        self._last_command = command
        io.publish_twist(*command.base_twist)
        io.publish_arm(command.arm_q_target)
        self._last_arm_command = now
        return "ok", ""

    def _begin_gate(self, io, x: float, y: float, lane_yaw: float) -> dict:
        """Anchor the base task at (x, y) (odom) with the lane heading; gate to that spot.

        Input:
            x, y: spot in odom.
            lane_yaw: lane heading in targets_frame.

        Output:
            dict: gate state for _poll_gate() (goal = the spot).
        """
        yaw = lane_yaw
        to_odom = io.to_odom()
        if to_odom is not None:
            yaw = wrap_angle(lane_yaw + to_odom[2])
        self.base_task.set_anchor(x, y, yaw)
        self.base_task.clear_hold()
        return {"goal": (float(x), float(y)), "start": io.now(), "static_since": None}

    def _poll_gate(self, io, gate: dict):
        """Bring the base to gate['goal'] in a straight line, aligned with the lane and static.

        Output:
            None while moving; (True, "") once within 1.5 cm and
                align_tolerance_rad and static for static_time_s (also after
                BASE_MOVE_TIMEOUT_S, with a warning). The base is then held.
        """
        task = self.base_task
        pose = self.base_pose()
        distance = math.hypot(gate["goal"][0] - pose[0], gate["goal"][1] - pose[1])
        heading_error = task.heading_error(self.robot)
        now = io.now()
        timed_out = now - gate["start"] > BASE_MOVE_TIMEOUT_S
        if (distance < 0.015 and abs(heading_error) < task.align_tolerance_rad) or timed_out:
            io.publish_twist(0.0, 0.0, 0.0)
            vx, vy, wz = self.robot.dq[self.robot.base_v_slice]
            if max(math.hypot(vx, vy), abs(wz)) > task.static_speed and not timed_out:
                gate["static_since"] = None
                return None
            if gate["static_since"] is None:
                gate["static_since"] = now
            if now - gate["static_since"] < task.static_time_s and not timed_out:
                return None
            if timed_out:
                io.log.warning(f"Base not in place/aligned/static after {BASE_MOVE_TIMEOUT_S:.0f} s "
                               f"({distance:.3f} m, {heading_error:+.3f} rad); continuing anyway")
            self.hold_base()
            return True, ""
        gate["static_since"] = None
        io.publish_twist(*gate_twist(pose, gate["goal"], heading_error, task.max_speed,
                                     task.max_yaw_rate, task.align_tolerance_rad))
        return None

    def set_localization_paused(self, io, paused: bool) -> None:
        """Pause / resume SLAM's new measurements (a toggle service) if configured."""
        if not self.pause_localization or paused == self.localization_paused:
            return
        if io.toggle_localization_pause() is None:
            io.log.warning("SLAM pause service not available; localization keeps running")
            return
        self.localization_paused = paused
        io.log.info(f"SLAM measurements {'paused (work at a stop)' if paused else 'resumed'}")

    def hold_base(self) -> None:
        """Pin the base where it is (base task hold), e.g. while the arm works."""
        if self.base_task.anchor is not None:
            x, y, _ = self.base_pose()
            self.base_task.set_hold(x, y)

    def _poll_arm_motion(self, io, motion: dict, q_goal):
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

    def _inside_workspace(self, io, base, data: dict) -> bool:
        """Output: whether the localized base is inside the workspace (tracks how long it isn't)."""
        if self.localization.inside(None if base is None else base[:2]):
            data["outside_since"] = None
            return True
        if data.get("outside_since") is None:
            data["outside_since"] = io.now()
        where = "unknown" if base is None else f"({base[0]:.2f}, {base[1]:.2f})"
        io.log.warning(f"Localized base {where} is outside the navigation workspace; waiting",
                       throttle_duration_sec=5.0)
        return False

    def _workspace_timeout(self, io, data: dict):
        """Output: a failure once the base has been outside the workspace too long, else None."""
        if io.now() - data["outside_since"] > self.localization.timeout_s:
            return False, "localization outside the navigation workspace", "nav_failed"
        return None

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------

    def _save_run_config(self, config_file: str, node_parameters: dict) -> None:
        """Copy the experiment YAML and the node parameters into run_dir."""
        save_run_config(self.log, self.run_dir, config_file, node_parameters)

    def _save_plan(self) -> None:
        """Write the planned base stops and steps to run_dir/plan.yaml."""
        if self.run_dir is None:
            return
        stops = [{"stop": number + 1, "base_pose": [round(float(v), 3) for v in stop.pose],
                  "along_loop_m": round(float(stop.distance), 2), "shift_m": float(stop.offset),
                  "targets": [self.targets[t]["name"] for t in stop.targets]}
                 for number, stop in enumerate(self.stops)]
        steps = [{"type": s["type"], "stop": None if s.get("stop") is None else s["stop"] + 1,
                  "target": None if s.get("target") is None else self.targets[s["target"]]["name"]}
                 for s in self.steps]
        try:
            with open(os.path.join(self.run_dir, "plan.yaml"), "w") as handle:
                # The collision objects too: they may come from the machine's
                # published model rather than from experiment.yaml.
                yaml.safe_dump({"mission": self.name, "stops": stops, "steps": steps,
                                "collision_objects": self.moveit_config.get("collision_objects", [])},
                               handle, sort_keys=False)
        except OSError as exc:
            self.log.error(f"Could not write plan.yaml: {exc}")

    def write_summary(self, tables: list) -> None:
        """Log (title, table) pairs and write them to run_dir/summary.txt."""
        write_summary(self.log, self.run_dir, tables)
