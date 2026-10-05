#!/usr/bin/env python3
"""Arm-only tool-frame targets: aim the wrist camera at, or touch with the pointer, target frames; base fixed.

For each target frame T (from TF, or a pose written in the experiment YAML)
the tool frame (`frames.tool`, e.g. the pointer tip, else `frames.camera`,
the camera optical frame) is placed `view.standoff_m` along T's +Z, its +Z
pointing back at T's origin:

    tool = T * Trans(0, 0, standoff) * RotX(pi) * RotZ(roll)

so the tool +Z is -T.z, its x follows T.x and its y -T.y (roll 0). A camera
looks at the target from that distance; the pointer tip with standoff 0
touches it. `standoff_m` may be a list, run in order (e.g. [0.05, 0.0]: 5 cm
above the target along its axis, then on it). With `view.roll_samples: N`
the roll about the tool's +Z is left free instead and the IK solution
closest to the current arm is taken. With `view.x_reference: [x, y, z]` (in
reference_frame) the target's X is re-aimed along that axis, its +Z kept: the
roll then comes from the robot, not from whatever set the target's X.

Only the 6 arm joints move: the base stays where it is (its current TF pose is
given to MoveIt through the SRDF's virtual_joint), there is no Nav2, no HQP
and no Pinocchio. MoveIt computes a collision-free IK and plans and executes
the joint motion (the experiment's planning_pipeline, retried with
fallback_pipeline if it fails), so it runs the same in sim and on the real
arm. The tool-frame pose reached is measured from TF after each motion and the
results go to <run_dir>/results.yaml.

    ros2 launch renee_trajectory_generation arm_static_planner.launch.py experiment:=arm_static_planner_sim
"""
import math
import os
import threading
import time

import numpy as np
import rclpy
import tf2_ros
import yaml
from moveit_msgs.msg import MoveItErrorCodes
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState

from renee_trajectory_generation.legacy.arm_motion import ArmMotion
from renee_trajectory_generation.legacy.arm_static_planner import (
    pose_error, pose_matrix, standoffs, target_goals, yaml_target_pose)

DEFAULT_ARM_JOINTS = ["robot_arm_shoulder_pan_joint", "robot_arm_shoulder_lift_joint", "robot_arm_elbow_joint",
                      "robot_arm_wrist_1_joint", "robot_arm_wrist_2_joint", "robot_arm_wrist_3_joint"]
# UR joint limits (rad): +-2 pi, the elbow +-pi.
DEFAULT_Q_LOWER = [-2 * math.pi, -2 * math.pi, -math.pi, -2 * math.pi, -2 * math.pi, -2 * math.pi]
DEFAULT_Q_UPPER = [2 * math.pi, 2 * math.pi, math.pi, 2 * math.pi, 2 * math.pi, 2 * math.pi]


class ArmStaticPlannerNode(Node):
    """Reads the experiment YAML and runs its targets one after the other in a worker thread."""

    def __init__(self) -> None:
        super().__init__("arm_static_planner_node")
        declare = lambda name, default: self.declare_parameter(name, default).value  # noqa: E731
        config_file = declare("tasks_config_file", "")
        self.run_dir = declare("run_dir", "") or os.path.join(
            os.path.expanduser("~"), ".ros", "hqp_runs", "arm_static_planner", time.strftime("%Y%m%d_%H%M%S"))
        joint_states_topic = declare("joint_states_topic", "/robot/joint_states")
        self.startup_timeout_s = float(declare("startup_timeout_s", 30.0))
        if not config_file:
            raise RuntimeError("no tasks_config_file (run it with arm_static_planner.launch.py experiment:=<name>)")
        with open(config_file) as handle:
            self.config = yaml.safe_load(handle)
        self.config_file = config_file

        self.reference_frame = self.config.get("reference_frame", "robot_arm_base_link")
        # The frame placed on each goal: frames.tool (e.g. the pointer tip), else frames.camera.
        frames = self.config.get("frames") or {}
        self.tool_frame = frames.get("tool") or frames.get(
            "camera", "robot_arm_rgbd_camera_left_camera_optical_frame")
        # Optional tool frame in ik_link (tool0), {position, rpy_deg}, instead of
        # its TF (the URDF): e.g. the measured pointer length.
        offset = frames.get("tool_offset")
        self.tool_offset = None if offset is None else yaml_target_pose(
            {"position": offset["position"], "rpy_deg": offset.get("rpy_deg", [0.0, 0.0, 0.0])})
        view = self.config.get("view") or {}
        self.view = view
        self.standoffs = standoffs(view.get("standoff_m", 0.4))
        self.roll_samples = int(view.get("roll_samples", 0) or 0)
        self.settle_s = float(view.get("settle_s", 1.0))
        self.reach_tolerance_m = float(view.get("reach_tolerance_m", 0.01))
        self.reach_tolerance_rad = float(view.get("reach_tolerance_rad", 0.03))
        arm = self.config.get("arm") or {}
        self.arm_joint_names = list(arm.get("joint_names", DEFAULT_ARM_JOINTS))
        self.q_lower = np.asarray(arm.get("q_lower", DEFAULT_Q_LOWER), dtype=float)
        self.q_upper = np.asarray(arm.get("q_upper", DEFAULT_Q_UPPER), dtype=float)
        self.home_q = arm.get("home_q")
        self.moveit_config = self.config.get("moveit") or {}
        # moveit.lock_wrist_3: wrist_3 never turns: every
        # MoveIt goal keeps its current angle. The tool frame must lie on the
        # wrist_3 axis (tool0 Z), so only its roll about +Z is given up.
        self.lock_wrist_3 = bool(self.moveit_config.get("lock_wrist_3", False))
        self.wrist_3 = next((i for i, name in enumerate(self.arm_joint_names) if "wrist_3" in name), None)
        if self.lock_wrist_3 and self.wrist_3 is None:
            raise ValueError(f"{config_file}: lock_wrist_3 but no wrist_3 joint in {self.arm_joint_names}")
        self.targets = list(self.config.get("targets") or [])
        if not self.targets:
            raise ValueError(f"{config_file}: no targets")
        for index, target in enumerate(self.targets):
            if ("tf_frame" in target) == ("position" in target):
                raise ValueError(f"{config_file}: target {index} needs either tf_frame or position (+ rpy_deg "
                                 f"/ quaternion) in reference_frame")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.moveit = ArmMotion(
            self, self.arm_joint_names,
            namespace=self.moveit_config.get("namespace", "/robot"),
            group=self.moveit_config.get("group", "arm"),
            ik_link=self.moveit_config.get("ik_link", "robot_arm_tool0"),
            virtual_joint=self.moveit_config.get("virtual_joint", "virtual_joint"),
            planning_frame=self.moveit_config.get("planning_frame", "robot_map"),
            ik_timeout_s=float(self.moveit_config.get("ik_timeout_s", 0.1)))
        self._q_arm = None
        self.create_subscription(JointState, joint_states_topic, self._on_joint_state, qos_profile_sensor_data)
        self.done = threading.Event()
        self.log = self.get_logger()

    def _on_joint_state(self, msg: JointState) -> None:
        positions = dict(zip(msg.name, msg.position))
        if all(name in positions for name in self.arm_joint_names):
            self._q_arm = np.array([positions[name] for name in self.arm_joint_names])

    # ------------------------------------------------------------------
    # Waiting on ROS (the executor spins in the main thread)
    # ------------------------------------------------------------------

    def _wait(self, condition, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while rclpy.ok() and time.monotonic() < deadline:
            if condition():
                return True
            time.sleep(0.02)
        return condition()

    def _lookup(self, target_frame: str, source_frame: str, timeout_s: float = 2.0):
        """Output: (4, 4) pose of source_frame in target_frame, or None."""
        found = {}

        def available():
            try:
                found["t"] = self.tf_buffer.lookup_transform(target_frame, source_frame, Time()).transform
                return True
            except tf2_ros.TransformException:
                return False
        if not self._wait(available, timeout_s):
            self.log.error(f"no TF {target_frame} <- {source_frame} in {timeout_s:.0f}s")
            return None
        t = found["t"]
        return pose_matrix([t.translation.x, t.translation.y, t.translation.z],
                           quaternion=[t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w])

    def _base_transform(self):
        """Output: Transform planning_frame -> base (the fixed base given to MoveIt), or None."""
        planning_frame = self.moveit_config.get("planning_frame", "robot_map")
        base_frame = self.moveit_config.get("base_frame", "robot_base_footprint")
        if self._lookup(planning_frame, base_frame, self.startup_timeout_s) is None:
            return None
        return self.tf_buffer.lookup_transform(planning_frame, base_frame, Time()).transform

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _target_pose(self, target: dict):
        """Output: (4, 4) target frame in reference_frame (TF looked up now), or None."""
        if "tf_frame" in target:
            return self._lookup(self.reference_frame, target["tf_frame"], float(target.get("tf_timeout_s", 5.0)))
        return yaml_target_pose(target)

    def _solve_ik(self, goal: np.ndarray, tool0_to_tool: np.ndarray, base_transform):
        """Collision-free arm configuration putting the tool frame at goal; (q, detail)."""
        # request_camera_ik aims position -> look_at with the image y as close to
        # `down` as possible: down = the goal's own y gives exactly goal at
        # roll 0, and the extra rolls turn about its +Z.
        position = goal[:3, 3]
        look_at = position + goal[:3, 2]
        down = goal[:3, 1]
        rolls = ([2.0 * math.pi * k / self.roll_samples for k in range(self.roll_samples)]
                 if self.roll_samples > 0 else [0.0])
        seed = self._q_arm.copy()
        futures = self.moveit.request_camera_ik(position, look_at, self.reference_frame, tool0_to_tool,
                                                base_transform, seed, rolls, down)
        self._wait(lambda: all(f.done() for f in futures), 5.0)
        done = [f for f in futures if f.done()]
        q = self.moveit.pick_ik_solution(done, seed, self.q_lower, self.q_upper)
        if q is None:
            codes = sorted({f.result().error_code.val for f in done if f.result() is not None})
            return None, f"no collision-free IK (MoveIt codes {codes}, {len(done)}/{len(futures)} replies)"
        return q, ""

    def _check_on_wrist_3_axis(self) -> None:
        """Raise unless the tool frame is on the wrist_3 axis with its +Z along it (lock_wrist_3)."""
        tool0_frame = self.moveit_config.get("tool0_frame", "robot_arm_tool0")
        ik_link = self.moveit_config.get("ik_link", "robot_arm_tool0")
        if self.tool_offset is not None:
            ik_link_pose = self._lookup(tool0_frame, ik_link, self.startup_timeout_s)
            tool = None if ik_link_pose is None else ik_link_pose @ self.tool_offset
        else:
            tool = self._lookup(tool0_frame, self.tool_frame, self.startup_timeout_s)
        if tool is None:
            raise RuntimeError(f"lock_wrist_3: no TF {tool0_frame} <- {self.tool_frame}")
        off_axis_m = float(np.linalg.norm(tool[:2, 3]))
        tilt_rad = float(np.arccos(np.clip(tool[2, 2], -1.0, 1.0)))
        if off_axis_m > 1e-3 or tilt_rad > 1e-3:
            raise RuntimeError(f"lock_wrist_3: {self.tool_frame} is {off_axis_m * 1000:.1f} mm off / "
                               f"{math.degrees(tilt_rad):.2f} deg tilted from the wrist_3 axis ({tool0_frame} Z): "
                               f"holding wrist_3 would move it")
        self.log.info(f"lock_wrist_3: wrist_3 held at its current angle, {self.tool_frame} roll left free")

    def _move(self, q_goal) -> tuple:
        """MoveIt joint motion to q_goal (planning_pipeline, then fallback_pipeline); (ok, detail)."""
        config = self.moveit_config
        locked = []
        if self.lock_wrist_3:
            # The tool is on the wrist_3 axis: the current wrist_3 angle keeps
            # its position and +Z, only its roll changes.
            q_goal = np.array(q_goal, dtype=float)
            q_goal[self.wrist_3] = self._q_arm[self.wrist_3]
            locked = [self.arm_joint_names[self.wrist_3]]
        attempts = [(config.get("planning_pipeline", "pilz_industrial_motion_planner"),
                     config.get("planner_id", "PTP"))]
        if config.get("fallback_pipeline"):
            attempts.append((config["fallback_pipeline"], config.get("fallback_planner_id", "")))
        detail = ""
        for pipeline, planner in attempts:
            goal_future = self.moveit.request_joint_motion(
                q_goal, velocity_scaling=float(config.get("velocity_scaling", 0.1)),
                planning_time_s=float(config.get("planning_time_s", 5.0)),
                pipeline_id=pipeline, planner_id=planner,
                goal_tolerance_rad=float(config.get("joint_goal_tolerance_rad", 0.001)),
                locked_joints=locked)
            if goal_future is None:
                return False, "move_action not available"
            if not self._wait(goal_future.done, 10.0) or not goal_future.result().accepted:
                detail = f"{pipeline}: goal not accepted"
                continue
            result_future = goal_future.result().get_result_async()
            if not self._wait(result_future.done, float(config.get("motion_timeout_s", 120.0))):
                goal_future.result().cancel_goal_async()
                detail = f"{pipeline}: motion timed out"
                continue
            code = result_future.result().result.error_code.val
            if code == MoveItErrorCodes.SUCCESS:
                return True, pipeline
            detail = f"{pipeline}: MoveIt error code {code}"
            self.log.warning(f"motion failed ({detail})" + (", retrying" if pipeline != attempts[-1][0] else ""))
        return False, detail

    def run(self) -> None:
        """Run every target in order, then write results.yaml and stop."""
        results = []
        try:
            results = self._run_targets()
        except Exception as exc:  # noqa: BLE001 -- report and stop, never leave the node hanging
            self.log.error(f"arm_static_planner stopped: {exc}")
        finally:
            self._save(results)
            self.done.set()

    def _run_targets(self) -> list:
        if not self._wait(lambda: self._q_arm is not None, self.startup_timeout_s):
            raise RuntimeError("no arm joint_states")
        if not self._wait(lambda: self.moveit.ready() and self.moveit.move_client.server_is_ready(),
                          self.startup_timeout_s):
            raise RuntimeError(f"move_group not available under {self.moveit_config.get('namespace', '/robot')}")
        ik_link = self.moveit_config.get("ik_link", "robot_arm_tool0")
        tool0_to_tool = (self.tool_offset if self.tool_offset is not None
                         else self._lookup(ik_link, self.tool_frame, self.startup_timeout_s))
        if self.tool_offset is not None:
            self.log.info(f"{self.tool_frame}: tool_offset {self.tool_offset[:3, 3].round(4).tolist()} m in "
                          f"{ik_link} (not its TF)")
        base_transform = self._base_transform()
        if tool0_to_tool is None or base_transform is None:
            raise RuntimeError(f"missing TF of {self.tool_frame} or of the base")
        if self.lock_wrist_3:
            self._check_on_wrist_3_axis()
        objects = self.moveit_config.get("collision_objects") or []
        if objects:
            future = self.moveit.apply_collision_objects(objects)
            if not self._wait(future.done, 10.0):
                raise RuntimeError("apply_planning_scene timed out")
            self.log.info(f"{len(objects)} collision object(s) added to the planning scene")
        self.log.info(f"{len(self.targets)} target(s) in {self.reference_frame}, {self.tool_frame} at "
                      f"{self.standoffs} m along each target's +Z; base fixed")
        if self.home_q is not None:
            ok, detail = self._move(self.home_q)
            self.log.info(f"home: {'ok' if ok else 'FAILED ' + detail}")

        results = []
        for index, target in enumerate(self.targets):
            name = target.get("name", f"target_{index}")
            result = {"name": name, "reached": False, "stages": []}
            results.append(result)
            target_pose = self._target_pose(target)
            if target_pose is None:
                result["detail"] = "target frame not available"
                continue
            target_pose, goals = target_goals(target, target_pose, self.view)
            result["target"] = _pose_dict(target_pose)
            # Approach stages one after the other (e.g. 5 cm above, then on the
            # target): a failed stage skips the rest of this target.
            for standoff, goal in goals:
                stage = self._reach(f"{name} @ {standoff:.3f} m", goal, tool0_to_tool, base_transform)
                stage["standoff_m"] = standoff
                result["stages"].append(stage)
                if not stage["reached"]:
                    result["detail"] = f"stage {standoff:.3f} m: {stage.get('detail', 'off target')}"
                    break
            else:
                result["reached"] = True
        reached = sum(r["reached"] for r in results)
        self.log.info(f"done: {reached}/{len(results)} target(s) reached; results in {self.run_dir}")
        return results

    def _reach(self, label: str, goal: np.ndarray, tool0_to_tool: np.ndarray, base_transform) -> dict:
        """IK, MoveIt motion and TF check of one tool-frame goal (reference_frame); the stage's result dict."""
        stage = {"goal": _pose_dict(goal), "reached": False}
        q, detail = self._solve_ik(goal, tool0_to_tool, base_transform)
        if q is None:
            stage["detail"] = detail
            self.log.error(f"[{label}] {detail}")
            return stage
        if self.lock_wrist_3:
            q[self.wrist_3] = self._q_arm[self.wrist_3]
        stage["q_goal"] = [round(float(v), 5) for v in q]
        self.log.info(f"[{label}] IK ok, moving (|dq| {np.linalg.norm(q - self._q_arm):.2f} rad)")
        ok, detail = self._move(q)
        if not ok:
            stage["detail"] = detail
            self.log.error(f"[{label}] {detail}")
            return stage
        time.sleep(self.settle_s)
        if self.tool_offset is None:
            reached = self._lookup(self.reference_frame, self.tool_frame)
        else:
            reached = self._lookup(self.reference_frame, self.moveit_config.get("ik_link", "robot_arm_tool0"))
            reached = None if reached is None else reached @ self.tool_offset
        if reached is None:
            stage["detail"] = f"{self.tool_frame} TF not available after the motion"
            return stage
        if self.roll_samples > 0 or self.lock_wrist_3:
            # Roll free: only the position and the +Z axis count.
            error_m = float(np.linalg.norm(reached[:3, 3] - goal[:3, 3]))
            error_rad = float(np.arccos(np.clip(reached[:3, 2] @ goal[:3, 2], -1.0, 1.0)))
        else:
            error_m, error_rad = pose_error(reached, goal)
        stage.update(reached_pose=_pose_dict(reached), error_m=round(error_m, 4),
                     error_deg=round(math.degrees(error_rad), 3), planner=detail,
                     reached=error_m <= self.reach_tolerance_m and error_rad <= self.reach_tolerance_rad)
        self.log.info(f"[{label}] {'reached' if stage['reached'] else 'OFF TARGET'}: {self.tool_frame} error "
                      f"{error_m * 1000:.1f} mm, {math.degrees(error_rad):.2f} deg")
        return stage

    def _save(self, results: list) -> None:
        os.makedirs(self.run_dir, exist_ok=True)
        with open(os.path.join(self.run_dir, "results.yaml"), "w") as handle:
            yaml.safe_dump({"experiment": os.path.basename(self.config_file), "reference_frame": self.reference_frame,
                            "tool_frame": self.tool_frame, "standoff_m": self.standoffs,
                            "targets": results}, handle, sort_keys=False)
        with open(os.path.join(self.run_dir, os.path.basename(self.config_file)), "w") as handle:
            yaml.safe_dump(self.config, handle, sort_keys=False)


def _pose_dict(pose: np.ndarray) -> dict:
    """Output: {"position": [x, y, z], "quaternion": [x, y, z, w]} of a (4, 4) pose (YAML-friendly)."""
    return {"position": [round(float(v), 4) for v in pose[:3, 3]],
            "quaternion": [round(float(v), 5) for v in Rotation.from_matrix(pose[:3, :3]).as_quat()]}


def main() -> None:
    rclpy.init()
    node = None
    try:
        node = ArmStaticPlannerNode()
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        threading.Thread(target=node.run, daemon=True).start()
        while rclpy.ok() and not node.done.is_set():
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
