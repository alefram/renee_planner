#!/usr/bin/env python3
"""ROS 2 interface of the whole-body trajectory missions (RB-VOGUI+ with a UR arm).

This node only reads the ROS data the mission needs and sends its
commands: /joint_states and odometry into the robot model, TF lookups,
Nav2 FollowPath goals, MoveIt requests (IK, joint motions, validity checks,
planning-scene objects), and the base twist / arm joint trajectory
publishers. Everything else -- targets, base placement, the step executor,
the HQP, the results -- is the mission (renee_trajectory_generation.mission
and its missions/ modules: cleaning, screw_detection, defect_detection; no ROS),
picked by the experiment YAML's experiment.mission. The Pinocchio model and the
HQP are built only by `Mission` (cleaning, screw_detection): defect_detection
runs without them (no venv).
"""
import math
import os
import time

import numpy as np
import rclpy
import tf2_ros
import yaml
from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped, Transform, Twist, TwistStamped
from moveit_msgs.msg import MoveItErrorCodes
from nav2_msgs.action import FollowPath, NavigateToPose
from nav_msgs.msg import Odometry, Path
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from renee_bringup_utils.wait_for_ros import wait_for_topic
from renee_trajectory_generation.arm_motion import ArmMotion
from renee_trajectory_generation.campetella import to_collision_objects, urdf_collision_boxes
from renee_trajectory_generation.mission import create_mission
from renee_trajectory_generation.robot import resolve_default_xacro

# Wall-clock limits for MoveIt replies: a lost reply must not stall a step.
IK_REPLY_TIMEOUT_S = 5.0
VALIDITY_REPLY_TIMEOUT_S = 1.0


def _check_arm_model(node: Node, topic: str, ur_type: str, timeout_s: float) -> None:
    """Check that the running robot's arm is the experiment's ur_type.

    The HQP model comes from the xacro with the YAML's ur_type; a different
    arm in Gazebo/on the robot would give wrong kinematics. The published
    robot_description names the arm's meshes (".../meshes/<ur_type>/...").

    Raises:
        RuntimeError: if the published description has another arm.
    """
    received = []
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    sub = node.create_subscription(String, topic, lambda msg: received.append(msg.data), qos)
    deadline = time.monotonic() + timeout_s
    while not received and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_subscription(sub)
    if not received:
        node.get_logger().warn(f"{topic} not received in {timeout_s}s: arm model ({ur_type}) not checked")
    elif f"/meshes/{ur_type}/" not in received[0]:
        raise RuntimeError(f"the experiment's ur_type is {ur_type} but {topic} has another arm "
                           f"(start the sim with UR_TYPE={ur_type})")


def _collision_objects_from_description(node: Node, spec: dict, frame: str, timeout_s: float) -> list:
    """MoveIt collision_objects from the Campetella's published robot_description (campetella.py).

    Input:
        spec: the YAML's moveit.collision_from_description: {topic, root_frame,
            margin_m, id_prefix}.
        frame: frame of the objects (targets_frame); root_frame is looked up in it.

    Raises:
        RuntimeError: if the description or the transform doesn't arrive in
            timeout_s, or the machine is tilted (the boxes are planar).
    """
    topic = spec.get("topic", "/campetella_robot_description")
    root_frame = spec.get("root_frame", "campetella_base_link")
    received = []
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    sub = node.create_subscription(String, topic, lambda msg: received.append(msg.data), qos)
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer, node)
    deadline = time.monotonic() + timeout_s
    transform = None
    while time.monotonic() < deadline and (not received or transform is None):
        rclpy.spin_once(node, timeout_sec=0.1)
        if transform is None and tf_buffer.can_transform(frame, root_frame, Time()):
            transform = tf_buffer.lookup_transform(frame, root_frame, Time()).transform
    node.destroy_subscription(sub)
    tf_listener.unregister()
    if not received:
        raise RuntimeError(f"{topic} not received in {timeout_s}s: no machine collision model")
    if transform is None:
        raise RuntimeError(f"no transform {frame} -> {root_frame} in {timeout_s}s")
    q = transform.rotation
    roll, pitch, yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")
    if max(abs(roll), abs(pitch)) > 0.01:
        raise RuntimeError(f"{root_frame} is tilted in {frame} (roll {roll:.3f}, pitch {pitch:.3f} rad)")
    t = transform.translation
    boxes = urdf_collision_boxes(received[0], root_frame)
    objects = to_collision_objects(boxes, (t.x, t.y, t.z, yaw), frame,
                                   margin_m=float(spec.get("margin_m", 0.0)),
                                   id_prefix=spec.get("id_prefix", ""))
    node.get_logger().info(f"{len(objects)} collision boxes from {topic} ({root_frame} at "
                           f"({t.x:.3f}, {t.y:.3f}, {t.z:.3f}), yaw {math.degrees(yaw):.1f} deg in {frame})")
    return objects


def _yaw_from_quaternion(q) -> float:
    """Output: yaw (rad) of a planar orientation quaternion."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class _ActionCall:
    """A running action goal: poll() -> None | (ok, detail); cancel()."""

    def __init__(self, goal_future, succeeded) -> None:
        """Input: goal_future (send_goal_async); succeeded(result) -> (ok, detail)."""
        self._goal_future = goal_future
        self._succeeded = succeeded
        self._handle = None
        self._result_future = None

    def poll(self):
        if self._result_future is None:
            if not self._goal_future.done():
                return None
            self._handle = self._goal_future.result()
            if self._handle is None or not self._handle.accepted:
                return False, "goal rejected"
            self._result_future = self._handle.get_result_async()
            return None
        if not self._result_future.done():
            return None
        return self._succeeded(self._result_future.result())

    def cancel(self) -> None:
        if self._handle is not None:
            self._handle.cancel_goal_async()


class _ServiceCall:
    """A pending service request: poll() -> None | interpret(response), or on_timeout."""

    def __init__(self, future, interpret, timeout_s: float, on_timeout) -> None:
        self._future = future
        self._interpret = interpret
        self._deadline = time.monotonic() + timeout_s
        self._on_timeout = on_timeout

    def poll(self):
        if self._future.done():
            return self._interpret(self._future.result())
        if time.monotonic() > self._deadline:
            return self._on_timeout
        return None


class _IkCall:
    """compute_ik for several rolls: poll() -> None | (True, q) | (False, detail)."""

    def __init__(self, futures, pick) -> None:
        self._futures = futures
        self._pick = pick
        self._deadline = time.monotonic() + IK_REPLY_TIMEOUT_S

    def poll(self):
        done = [future for future in self._futures if future.done()]
        if len(done) < len(self._futures) and time.monotonic() < self._deadline:
            return None
        q = self._pick(done)
        if q is None:
            codes = sorted({f.result().error_code.val for f in done if f.result() is not None})
            return False, (f"no collision-free arm configuration from this base stop (MoveIt IK codes "
                           f"{codes}, {len(done)}/{len(self._futures)} replies)")
        return True, q


class TrajectoryControllerNode(Node):
    def __init__(self) -> None:
        """Read the parameters and the experiment YAML, connect to ROS, start the timer.

        Raises:
            RuntimeError: if joint_states or odom don't appear in time.
            ValueError: on an invalid experiment YAML.
        """
        super().__init__("trajectory_controller_node")
        declare = lambda name, default: self.declare_parameter(name, default).value  # noqa: E731
        control_rate_hz = float(declare("control_rate_hz", 100.0))
        ee_frame = declare("ee_frame", "robot_arm_tool0")
        cmd_vel_topic = declare("cmd_vel_topic", "/robot/robotnik_base_control/cmd_vel")
        # Gazebo's base controller takes TwistStamped; the real base's
        # twist_mux (through vogui_ros1_ros2_bridge) takes plain Twist.
        self.cmd_vel_stamped = bool(declare("cmd_vel_stamped", True))
        self.base_frame_id = declare("base_frame_id", "robot_base_footprint")
        joint_trajectory_topic = declare("joint_trajectory_topic",
                                         "/robot/joint_trajectory_controller/joint_trajectory")
        joint_states_topic = declare("joint_states_topic", "/robot/joint_states")
        odom_topic = declare("odom_topic", "/robot/robotnik_base_control/odom")
        robot_description_topic = declare("robot_description_topic", "/robot/robot_description")
        startup_timeout_s = float(declare("startup_timeout_s", 30.0))
        self.trajectory_horizon_factor = float(declare("trajectory_horizon_factor", 2.0))
        arm_reference_reset_rad = float(declare("arm_reference_reset_rad", 0.15))
        equality_tolerance = float(declare("equality_tolerance", 1e-4))
        osqp_settings = {"eps_abs": float(declare("osqp_eps_abs", 1e-5)),
                         "eps_rel": float(declare("osqp_eps_rel", 1e-5)),
                         "max_iter": int(declare("osqp_max_iter", 4000)),
                         "polish": bool(declare("osqp_polish", True))}
        config_file = declare("tasks_config_file", "") or os.path.join(
            get_package_share_directory("renee_trajectory_generation"),
            "config", "experiments", "defect_detection_sim.yaml")
        # Outputs go to <output_dir>/<YYYYmmdd_HHMMSS>/, or run_dir (launch file).
        experiment_name = declare("experiment_name", "")
        output_dir = declare("output_dir", "") or os.path.join(
            os.path.expanduser("~"), ".ros", "hqp_runs", experiment_name)
        # The run's own folder; the launch file sets it (its bag goes there too).
        run_dir = declare("run_dir", "") or os.path.join(output_dir, time.strftime("%Y%m%d_%H%M%S"))
        xacro_path = declare("xacro_path", "")
        # The executed trajectory is also written there as <experiment>.yaml
        # (the launch file: the package's config/trajectories).
        trajectory_dir = declare("trajectory_dir", "")

        for topic in (joint_states_topic, odom_topic):
            if not wait_for_topic(self, topic, startup_timeout_s, require_msg=True):
                raise RuntimeError(f"{topic} not available after {startup_timeout_s}s")
        with open(config_file, "r") as handle:
            config = yaml.safe_load(handle)
        ur_type = config.get("ur_type", "ur5e")
        _check_arm_model(self, robot_description_topic, ur_type, startup_timeout_s)
        # The machine's collision boxes come from its published model (with its
        # cart supports), ahead of the YAML's own collision_objects, so the
        # planning scene and the base planner always match the current model.
        moveit_section = config.get("moveit") or {}
        if moveit_section.get("collision_from_description"):
            moveit_section["collision_objects"] = _collision_objects_from_description(
                self, moveit_section["collision_from_description"], config.get("targets_frame", "robot_map"),
                startup_timeout_s) + list(moveit_section.get("collision_objects") or [])
            config["moveit"] = moveit_section
        default_xacro, mappings = resolve_default_xacro(config.get("wrist_camera", "none"), ur_type)
        self.log = self.get_logger()
        self._dt = 1.0 / control_rate_hz
        names = self.list_parameters([], 0).names
        self.mission = create_mission(
            config, xacro_path=xacro_path or default_xacro, mappings=mappings, log=self.log,
            ee_frame=ee_frame, dt=self._dt, equality_tolerance=equality_tolerance,
            osqp_settings=osqp_settings, arm_reference_reset_rad=arm_reference_reset_rad,
            run_dir=run_dir,
            config_file=config_file, experiment_name=experiment_name,
            trajectory_dir=trajectory_dir or None,
            node_parameters={p.name: p.value for p in self.get_parameters(names)})
        self.arm_joint_names = self.mission.arm_joint_names

        # Frames: targets_frame (map), odom (from the odometry messages), base.
        self._targets_frame = config.get("targets_frame", "robot_map")
        nav = config.get("navigation") or {}
        self._base_frame = nav.get("base_frame", "robot_base_footprint")
        self._odom_frame = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Nav2 and MoveIt.
        loop = nav.get("loop_path") or {}
        self._loop_controller_id = loop.get("controller_id", "FollowPath")
        self._loop_goal_checker_id = loop.get("goal_checker_id", "general_goal_checker")
        # Named explicitly: with more than one progress checker on the
        # server, an empty id makes Nav2 abort FollowPath.
        self._loop_progress_checker_id = loop.get("progress_checker_id", "progress_checker")
        self.follow_client = ActionClient(self, FollowPath, loop.get("follow_path_action", "/robot/follow_path"))
        # navigation.drive: navigate_to_pose -- Nav2 plans and drives to each stop itself.
        self.navigate_client = ActionClient(self, NavigateToPose,
                                            nav.get("navigate_to_pose_action", "/robot/navigate_to_pose"))
        # SLAM measurements paused while working at a stop (navigation.pause_localization_at_stops).
        self.localization_pause_client = None
        if nav.get("pause_localization_at_stops") or nav.get("pause_localization_during_passes"):
            from slam_toolbox.srv import Pause
            self._pause_type = Pause
            self.localization_pause_client = self.create_client(
                Pause, nav.get("localization_pause_service", "/slam_toolbox/pause_new_measurements"))
        # Image capture (renee_action_servers /capture_rgbd), for missions that record images.
        capture = config.get("capture")
        self.capture_client = None
        if capture is not None:
            from renee_action_servers.action import CaptureRGBD
            self._capture_type = CaptureRGBD
            self.capture_client = ActionClient(self, CaptureRGBD, capture.get("action", "/capture_rgbd"))
        self._moveit_config = config.get("moveit") or {}
        self.moveit = ArmMotion(
            self, self.arm_joint_names,
            namespace=self._moveit_config.get("namespace", "/robot"),
            group=self._moveit_config.get("group", "arm"),
            ik_link=self._moveit_config.get("ik_link", "robot_arm_tool0"),
            virtual_joint=self._moveit_config.get("virtual_joint", "virtual_joint"),
            planning_frame=self._moveit_config.get("planning_frame", "robot_map"))

        # Robot state in, commands out.
        self._have_joint_state = False
        self._have_odom = False
        self.cmd_vel_pub = self.create_publisher(TwistStamped if self.cmd_vel_stamped else Twist,
                                                 cmd_vel_topic, 10)
        self.joint_traj_pub = self.create_publisher(JointTrajectory, joint_trajectory_topic, 10)
        self.create_subscription(JointState, joint_states_topic, self.on_joint_state, qos_profile_sensor_data)
        self.create_subscription(Odometry, odom_topic, self.on_odom, qos_profile_sensor_data)
        self.create_timer(self._dt, self.on_control_timer)
        self.log.info(f"trajectory_controller_node ready: {self.mission.name} mission, "
                      f"{len(self.mission.targets)} target(s); outputs in {self.mission.run_dir}")

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def on_joint_state(self, msg: JointState) -> None:
        """Pass the arm joint positions/velocities to the mission."""
        try:
            positions = dict(zip(msg.name, msg.position))
            velocities = dict(zip(msg.name, msg.velocity)) if msg.velocity else {}
            q_arm = np.array([positions[name] for name in self.arm_joint_names])
            dq_arm = np.array([velocities[name] for name in self.arm_joint_names]) if velocities else None
        except (KeyError, ValueError) as exc:
            self.log.warning(f"Invalid joint_states: {exc}", throttle_duration_sec=5.0)
            return
        self.mission.on_arm_state(q_arm, dq_arm, self.now())
        self._have_joint_state = True

    def on_odom(self, msg: Odometry) -> None:
        """Pass the base pose (odom) and body-frame twist to the mission."""
        position, twist = msg.pose.pose.position, msg.twist.twist
        self.mission.on_base_state(position.x, position.y,
                                      _yaw_from_quaternion(msg.pose.pose.orientation),
                                      twist.linear.x, twist.linear.y, twist.angular.z)
        self._odom_frame = msg.header.frame_id
        self._have_odom = True

    def on_control_timer(self) -> None:
        """Advance the mission by one control period."""
        if self._have_joint_state and self._have_odom:
            self.mission.tick(self)

    # ------------------------------------------------------------------
    # Reads for the mission (io interface, see mission.py)
    # ------------------------------------------------------------------

    def now(self) -> float:
        """Output: current node time (s; sim time when use_sim_time)."""
        return self.get_clock().now().nanoseconds * 1e-9

    def _lookup(self, target_frame: str, source_frame: str):
        """Output: (R (3, 3), t (3,), Transform) of source_frame in target_frame, or None."""
        try:
            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, Time()).transform
        except tf2_ros.TransformException as exc:
            self.log.warning(f"Waiting for TF {target_frame} <- {source_frame}: {exc}",
                             throttle_duration_sec=2.0)
            return None
        rotation = Rotation.from_quat([transform.rotation.x, transform.rotation.y,
                                       transform.rotation.z, transform.rotation.w]).as_matrix()
        translation = np.array([transform.translation.x, transform.translation.y, transform.translation.z])
        return rotation, translation, transform

    def to_odom(self):
        """Output: (R, t, yaw offset) from targets_frame to odom, or None."""
        if self._odom_frame is None:
            return None
        found = self._lookup(self._odom_frame, self._targets_frame)
        if found is None:
            return None
        rotation, translation, _ = found
        return rotation, translation, math.atan2(rotation[1, 0], rotation[0, 0])

    def base_in_map(self):
        """Output: (x, y, yaw) of the base in targets_frame, or None."""
        found = self._lookup(self._targets_frame, self._base_frame)
        if found is None:
            return None
        rotation, translation, _ = found
        return float(translation[0]), float(translation[1]), math.atan2(rotation[1, 0], rotation[0, 0])

    def localization_correction(self):
        """Output: (x, y, yaw) of the odom frame in targets_frame, or None."""
        if self._odom_frame is None:
            return None
        found = self._lookup(self._targets_frame, self._odom_frame)
        if found is None:
            return None
        rotation, translation, _ = found
        return np.array([translation[0], translation[1], math.atan2(rotation[1, 0], rotation[0, 0])])

    def frame_pose(self, target_frame: str, source_frame: str):
        """Output: (4, 4) pose of source_frame in target_frame, or None without TF."""
        found = self._lookup(target_frame, source_frame)
        if found is None:
            return None
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = found[0], found[1]
        return pose

    def _base_transform(self):
        """Output: geometry_msgs Transform planning_frame -> base footprint (MoveIt), or None."""
        found = self._lookup(self._moveit_config.get("planning_frame", "robot_map"),
                             self._moveit_config.get("base_frame", "robot_base_footprint"))
        return None if found is None else found[2]

    # ------------------------------------------------------------------
    # Nav2 and MoveIt requests
    # ------------------------------------------------------------------

    def follow_path_ready(self) -> bool:
        return self.follow_client.server_is_ready()

    def follow_path(self, samples) -> _ActionCall:
        """Send (x, y, yaw) samples in targets_frame to Nav2 FollowPath, with the
        loop's controller, goal checker and progress checker (navigation.loop_path).
        """
        path = Path()
        path.header.frame_id = self._targets_frame
        path.header.stamp = self.get_clock().now().to_msg()
        for x, y, yaw in samples:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x, pose.pose.position.y = float(x), float(y)
            pose.pose.orientation.z, pose.pose.orientation.w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
            path.poses.append(pose)
        goal = FollowPath.Goal()
        goal.path = path
        goal.controller_id = self._loop_controller_id
        goal.goal_checker_id = self._loop_goal_checker_id
        goal.progress_checker_id = self._loop_progress_checker_id

        def succeeded(result):
            ok = result.status == GoalStatus.STATUS_SUCCEEDED
            return ok, "" if ok else f"Nav2 status {result.status}"
        return _ActionCall(self.follow_client.send_goal_async(goal), succeeded)

    def navigate_to_pose_ready(self) -> bool:
        return self.navigate_client.server_is_ready()

    def navigate_to_pose(self, x: float, y: float, yaw: float) -> _ActionCall:
        """Send a pose in targets_frame to Nav2 NavigateToPose (its planner and behavior tree)."""
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = self._targets_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = float(x), float(y)
        goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)

        def succeeded(result):
            ok = result.status == GoalStatus.STATUS_SUCCEEDED
            return ok, "" if ok else f"Nav2 status {result.status}"
        return _ActionCall(self.navigate_client.send_goal_async(goal), succeeded)

    def moveit_ready(self) -> bool:
        return self.moveit.ready()

    def apply_collision_objects(self, objects) -> _ServiceCall:
        """Add the collision boxes to MoveIt's planning scene."""
        return _ServiceCall(self.moveit.apply_collision_objects(objects),
                            lambda response: (True, ""), 10.0, (False, "apply_planning_scene timed out"))

    def request_ik(self, position, look_at, tool0_to_frame, seed, rolls, down=None, base_pose=None):
        """compute_ik for a frame pose (targets_frame); None without TF.

        The base is the current one (TF), or base_pose (x, y, yaw in targets_frame)
        to ask from a base spot the robot is not at.
        """
        if base_pose is None:
            base_transform = self._base_transform()
            if base_transform is None:
                return None
        else:
            x, y, yaw = base_pose
            base_transform = Transform()
            base_transform.translation.x, base_transform.translation.y = float(x), float(y)
            base_transform.rotation.z, base_transform.rotation.w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
        futures = self.moveit.request_camera_ik(position, look_at, self._targets_frame, tool0_to_frame,
                                                base_transform, seed, rolls, down)
        limits = self.mission.joint_limits(self.arm_joint_names)
        return _IkCall(futures, lambda done: self.moveit.pick_ik_solution(
            done, seed, limits.q_lower, limits.q_upper))

    def moveit_base_pose(self) -> _ServiceCall:
        """move_group's base pose: poll() -> (True, (x, y, yaw)) | (False, detail); None if down."""
        future = self.moveit.request_base_pose()
        if future is None:
            return None

        def interpret(response):
            pose = self.moveit.base_pose_from_scene(response) if response is not None else None
            return (True, pose) if pose is not None else (False, "no virtual_joint in move_group's state")
        return _ServiceCall(future, interpret, 2.0, (False, "get_planning_scene timed out"))

    def move_arm(self, q_goal, fallback: bool = False):
        """MoveIt joint motion (planning_pipeline, or fallback_pipeline); None if move_action is down."""
        config = self._moveit_config
        goal_future = self.moveit.request_joint_motion(
            q_goal, velocity_scaling=float(config.get("velocity_scaling", 0.3)),
            pipeline_id=(config["fallback_pipeline"] if fallback
                         else config.get("planning_pipeline", "pilz_industrial_motion_planner")),
            planner_id=(config.get("fallback_planner_id", "") if fallback
                        else config.get("planner_id", "PTP")),
            goal_tolerance_rad=float(config.get("joint_goal_tolerance_rad", 0.001)))
        if goal_future is None:
            return None

        def succeeded(result):
            code = result.result.error_code.val
            return code == MoveItErrorCodes.SUCCESS, f"MoveIt error code {code}"
        return _ActionCall(goal_future, succeeded)

    def check_validity(self, q_arm):
        """MoveIt check_state_validity of an arm configuration at the current base; None without TF."""
        base_transform = self._base_transform()
        if base_transform is None:
            return None

        def interpret(response):
            if response is None or response.valid:
                return True, ""
            return False, ", ".join(sorted({f"{c.contact_body_1}/{c.contact_body_2}"
                                            for c in response.contacts})[:3])
        # A lost reply counts as no finding (the next check follows).
        return _ServiceCall(self.moveit.request_validity(q_arm, base_transform), interpret,
                            VALIDITY_REPLY_TIMEOUT_S, (True, ""))

    def toggle_localization_pause(self):
        """Toggle SLAM's new measurements (slam_toolbox pause_new_measurements); None if not configured."""
        if self.localization_pause_client is None or not self.localization_pause_client.service_is_ready():
            return None
        return _ServiceCall(self.localization_pause_client.call_async(self._pause_type.Request()),
                            lambda response: (response is not None, ""), 5.0, (False, "no reply"))

    def capture_ready(self) -> bool:
        return self.capture_client is not None and self.capture_client.server_is_ready()

    def capture_rgbd(self, waypoint_id: str, session_dir: str, frame_count: int) -> _ActionCall:
        """Capture RGB-D keyframes (the base must be still); poll() -> (ok, detail)."""
        goal = self._capture_type.Goal()
        goal.waypoint_id, goal.session_dir, goal.frame_count = waypoint_id, session_dir, int(frame_count)

        def succeeded(result):
            result = result.result
            return bool(result.success), (result.rgb_path if result.success else result.message)
        return _ActionCall(self.capture_client.send_goal_async(goal), succeeded)

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def publish_twist(self, vx: float, vy: float, wz: float) -> None:
        """Publish a body-frame base twist (TwistStamped or Twist, see cmd_vel_stamped)."""
        twist = Twist()
        twist.linear.x, twist.linear.y, twist.angular.z = float(vx), float(vy), float(wz)
        if not self.cmd_vel_stamped:
            self.cmd_vel_pub.publish(twist)
            return
        stamped = TwistStamped()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.header.frame_id = self.base_frame_id
        stamped.twist = twist
        self.cmd_vel_pub.publish(stamped)

    def _publish_arm_point(self, q_arm, horizon_s: float) -> None:
        traj = JointTrajectory()
        traj.joint_names = self.arm_joint_names
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in q_arm]
        # joint_trajectory_controller rejects a last point with nonzero
        # velocity, and every point here is the last one: positions only.
        point.time_from_start = Duration(sec=int(horizon_s), nanosec=int((horizon_s % 1.0) * 1e9))
        traj.points = [point]
        self.joint_traj_pub.publish(traj)

    def publish_arm(self, q_arm) -> None:
        """Stream one arm position target (reached in trajectory_horizon_factor periods)."""
        self._publish_arm_point(q_arm, self.trajectory_horizon_factor * self._dt)

    def hold_arm(self, q_arm) -> None:
        """Stop the base and keep the arm at q_arm."""
        self.publish_twist(0.0, 0.0, 0.0)
        self._publish_arm_point(q_arm, 0.1)


def main() -> None:
    """Run the node until shutdown; the results are written on the way out too."""
    rclpy.init()
    node = None
    try:
        node = TrajectoryControllerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            if not node.mission.done:
                node.mission.save_results()
                node.mission.save_trajectory()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
