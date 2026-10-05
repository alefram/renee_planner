#!/usr/bin/env python3
"""Generates the camera poses around the Campetella (no robot motion).

Inputs:
  - the experiment YAML (parameter experiment_file, or the action goal's experiment)
  - the machine URDF: /campetella_robot_description (machine.description_topic) or machine.urdf_file
  - TF robot_map -> campetella_base_link (machine.pose_source: tf), or machine.pose in the YAML
  - action goal /generate_scan_poses (renee_trajectory_generation/action/GenerateScanPoses)
Outputs:
  - <trajectory_dir>/<experiment>.yaml + _coverage.npz + _machine.glb (pipeline.generate)
  - /scan_trajectory (CameraTrajectory, transient local) and the action result

    ros2 launch renee_trajectory_generation scan_pose_generator.launch.py experiment:=campetella_scan_cad_sim
    ros2 action send_goal /generate_scan_poses renee_trajectory_generation/action/GenerateScanPoses "{use_cache: true}"
"""
import os
import subprocess
import threading
import time
import traceback

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Pose
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from renee_trajectory_generation.action import GenerateScanPoses
from renee_trajectory_generation.machine import quaternion_matrix
from renee_trajectory_generation.msg import CameraPose, CameraTrajectory
from renee_trajectory_generation.pipeline import generate, load_experiment, machine_pose_from_config

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)


class ScanPoseGeneratorNode(Node):
    def __init__(self):
        super().__init__("scan_pose_generator_node")
        self.experiment_file = self.declare_parameter("experiment_file", "").value
        self.experiments_dir = self.declare_parameter("experiments_dir", "").value
        self.trajectory_dir = self.declare_parameter("trajectory_dir", "").value
        self.cache_dir = self.declare_parameter("cache_dir", "").value or os.path.join(
            os.path.expanduser("~"), ".ros", "scan_poses", "cache")
        self.tf_timeout_s = float(self.declare_parameter("tf_timeout_s", 10.0).value)
        self.description_timeout_s = float(self.declare_parameter("description_timeout_s", 15.0).value)
        run_on_start = bool(self.declare_parameter("run_on_start", False).value)

        self.group = ReentrantCallbackGroup()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
        self.publisher = self.create_publisher(CameraTrajectory, "scan_trajectory", LATCHED)
        self.descriptions = {}          # topic -> latest URDF
        self.description_subscriptions = {}
        self.busy = threading.Lock()
        self.server = ActionServer(self, GenerateScanPoses, "generate_scan_poses", self.execute,
                                   callback_group=self.group)
        self.get_logger().info(f"ready: experiment '{self.experiment_file or '(goal)'}', "
                               f"trajectories -> {self.trajectory_dir or '(experiment output.trajectory_dir)'}")
        try:
            import open3d  # noqa: F401  (used by the pipeline's surface and visibility stages)
        except ImportError:
            self.get_logger().error(
                "open3d is not installed for this Python: goals will fail. Install it "
                "(pip install --break-system-packages \"open3d>=0.19\" \"numpy<2\", README 'Dependencies').")
        if run_on_start:
            threading.Thread(target=self._run_on_start, daemon=True).start()

    # -- inputs ---------------------------------------------------------------

    def _experiment_path(self, name: str) -> str:
        """Output: the YAML path of an experiment name or path ('' = the node's experiment)."""
        name = name or self.experiment_file
        if not name:
            raise ValueError("no experiment: set the goal's experiment or the experiment_file parameter")
        if os.path.isfile(name):
            return os.path.abspath(name)
        path = os.path.join(self.experiments_dir, f"{name}.yaml")
        if not os.path.isfile(path):
            raise ValueError(f"no experiment '{name}' ({path})")
        return path

    def _urdf(self, cfg) -> str:
        """Output: the machine URDF XML, from machine.urdf_file (xacro is run) or the description topic."""
        if cfg.machine.urdf_file:
            if cfg.machine.urdf_file.endswith(".xacro"):
                return subprocess.run(["xacro", cfg.machine.urdf_file], check=True, capture_output=True,
                                      text=True).stdout
            with open(cfg.machine.urdf_file) as handle:
                return handle.read()
        topic = cfg.machine.description_topic
        if topic not in self.description_subscriptions:
            self.description_subscriptions[topic] = self.create_subscription(
                String, topic, lambda msg, t=topic: self.descriptions.__setitem__(t, msg.data), LATCHED,
                callback_group=self.group)
        deadline = time.monotonic() + self.description_timeout_s
        while topic not in self.descriptions:
            if time.monotonic() > deadline:
                raise TimeoutError(f"no URDF on {topic} after {self.description_timeout_s:.0f} s "
                                   "(launch campetella_sim spawn_campetella.launch.py, or set machine.urdf_file)")
            time.sleep(0.1)
        return self.descriptions[topic]

    def _machine_pose(self, cfg) -> np.ndarray:
        """Output: frame -> machine root 4x4, from TF or the YAML (machine.pose_source)."""
        if cfg.machine.pose_source == "yaml":
            return machine_pose_from_config(cfg.machine)
        deadline = time.monotonic() + self.tf_timeout_s
        while True:
            try:
                transform = self.tf_buffer.lookup_transform(cfg.frame, cfg.machine.root_link, Time(),
                                                            timeout=Duration(seconds=0.5))
                break
            except Exception as error:  # tf2 raises several exception types
                if time.monotonic() > deadline:
                    raise TimeoutError(f"no TF {cfg.frame} -> {cfg.machine.root_link} after "
                                       f"{self.tf_timeout_s:.0f} s ({error}); or use machine.pose_source: yaml")
        t, q = transform.transform.translation, transform.transform.rotation
        return quaternion_matrix([t.x, t.y, t.z], [q.x, q.y, q.z, q.w])

    # -- generation -----------------------------------------------------------

    def _generate(self, experiment: str, source: str, captures_dir: str, use_cache: bool, progress):
        path = self._experiment_path(experiment)
        cfg = load_experiment(path, source, captures_dir)
        output_dir = self.trajectory_dir or cfg.output_dir
        if not output_dir:
            raise ValueError("no output folder: set the trajectory_dir parameter or output.trajectory_dir")
        self.get_logger().info(f"generating '{cfg.name}' ({cfg.mode} mode) from {path}")
        urdf = self._urdf(cfg)
        T_map_machine = self._machine_pose(cfg)
        self.get_logger().info(f"machine at {np.round(T_map_machine[:3, 3], 3).tolist()} in {cfg.frame}")
        trajectory, yaml_path = generate(cfg, urdf, T_map_machine, output_dir, os.path.join(self.cache_dir, cfg.name),
                                         use_cache, progress, log=self.get_logger().info)
        message = self._to_msg(trajectory, yaml_path)
        self.publisher.publish(message)
        return message, yaml_path

    def _to_msg(self, trajectory, yaml_path: str) -> CameraTrajectory:
        message = CameraTrajectory()
        message.header.frame_id = trajectory.frame
        message.header.stamp = self.get_clock().now().to_msg()
        message.experiment = trajectory.experiment
        message.mode = trajectory.mode
        message.camera_frame = trajectory.camera["frame"]
        message.coverage_ratio = float(trajectory.coverage["ratio"])
        message.surface_points = int(trajectory.coverage["surface_points"])
        message.uncovered_points = int(trajectory.coverage["uncovered_points"])
        message.trajectory_file = yaml_path
        for pose in trajectory.poses:
            item = CameraPose(id=pose.id, name=pose.name, section=pose.section, covered_points=pose.covered,
                              arc_length_m=float(pose.arc_m))
            item.pose = Pose()
            item.pose.position.x, item.pose.position.y, item.pose.position.z = (float(v) for v in pose.position)
            (item.pose.orientation.x, item.pose.orientation.y, item.pose.orientation.z,
             item.pose.orientation.w) = (float(v) for v in pose.orientation)
            item.look_at = Point(x=float(pose.look_at[0]), y=float(pose.look_at[1]), z=float(pose.look_at[2]))
            message.poses.append(item)
        return message

    def execute(self, goal_handle):
        request = goal_handle.request
        result = GenerateScanPoses.Result()
        if not self.busy.acquire(blocking=False):
            goal_handle.abort()
            result.message = "a generation is already running"
            return result

        def progress(stage: str, fraction: float):
            goal_handle.publish_feedback(GenerateScanPoses.Feedback(stage=stage, progress=float(fraction)))

        try:
            message, yaml_path = self._generate(request.experiment, request.source, request.captures_dir,
                                                request.use_cache, progress)
            result.success, result.trajectory, result.trajectory_file = True, message, yaml_path
            result.message = (f"{len(message.poses)} poses, coverage {message.coverage_ratio:.1%}")
            goal_handle.succeed()
        except Exception as error:
            self.get_logger().error(f"generation failed: {error}\n{traceback.format_exc()}")
            result.message = str(error)
            goal_handle.abort()
        finally:
            self.busy.release()
        return result

    def _run_on_start(self):
        with self.busy:
            try:
                message, _ = self._generate("", "", "", True, None)
                self.get_logger().info(f"done: {len(message.poses)} poses, coverage {message.coverage_ratio:.1%}")
            except Exception as error:
                self.get_logger().error(f"generation failed: {error}\n{traceback.format_exc()}")


def main():
    rclpy.init()
    node = ScanPoseGeneratorNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
