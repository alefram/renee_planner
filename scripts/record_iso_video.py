#!/usr/bin/env python3
"""Record an isometric-style video of a running Gazebo experiment.

Spawns a static camera sensor into the running world, bridges its image
topic to ROS and writes every frame to an MP4 until Ctrl+C / SIGTERM.
Frames are written at --fps in sim time, so the video plays back at sim
speed regardless of the real-time factor.

With --follow (default), the camera follows the robot at a three-quarter
view from outside the machine side the robot is on: it sits --distance m
away from the robot, away from --machine-center, turned by --azimuth, at
--cam-height, and looks between the robot and the machine (--view
outside), or, with --view arm, sits --arm-ahead m ahead of the robot along
its heading and --arm-side m towards the machine side (the robot's right,
clockwise loop) at --arm-height, looking at the arm. The camera is kept
inside --room (world x_min x_max y_min y_max, the scanning walls). Without it,
the camera stays at --eye looking at --target.

Usage (inside the simulation container, with install/setup.bash sourced):
    python3 record_iso_video.py --output /fnh_pkgs/hqp_videos/loop.mp4
"""
import argparse
import math
import signal
import subprocess
import sys

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformListener

CAMERA_SDF = """<?xml version="1.0"?>
<sdf version="1.9">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <sensor name="camera" type="camera">
        <topic>{topic}</topic>
        <update_rate>{fps}</update_rate>
        <always_on>true</always_on>
        <camera>
          <horizontal_fov>{hfov}</horizontal_fov>
          <image><width>{width}</width><height>{height}</height><format>R8G8B8</format></image>
          <clip><near>0.1</near><far>50</far></clip>
        </camera>
      </sensor>
    </link>
  </model>
</sdf>"""


def look_at_pose(eye, target):
    """Return (x, y, z, roll, pitch, yaw) of a camera at eye looking at target.

    Input:
        eye: (3,) camera position (world frame, m).
        target: (3,) point to look at (world frame, m).

    Output:
        tuple: pose for ros_gz_sim create, camera +X axis towards target.
    """
    dx, dy, dz = (t - e for t, e in zip(target, eye))
    yaw = math.atan2(dy, dx)
    pitch = math.atan2(-dz, math.hypot(dx, dy))  # positive pitch looks down
    return (*eye, 0.0, pitch, yaw)


def rpy_to_quaternion(roll, pitch, yaw):
    """Output: (w, x, y, z) of the ZYX (yaw, pitch, roll) rotation."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


class Follower:
    """Move the camera entity to a three-quarter view of the robot."""

    def __init__(self, node: Node, args) -> None:
        self.node = node
        self.args = args
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, node)
        self.eye = None
        self.target = None
        self.pending = None
        node.create_timer(1.0 / args.fps, self.update)

    def update(self) -> None:
        a = self.args
        try:
            tf = self.tf_buffer.lookup_transform(a.map_frame, a.follow_frame, Time())
        except Exception:  # TF not ready yet
            return
        rx = tf.transform.translation.x + a.map_offset[0]
        ry = tf.transform.translation.y + a.map_offset[1]
        mx, my = a.machine_center
        if a.view == "arm":
            q = tf.transform.rotation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            fx, fy = math.cos(yaw), math.sin(yaw)   # heading
            sx, sy = math.sin(yaw), -math.cos(yaw)  # robot's right (machine side)
            eye = [rx + a.arm_ahead * fx + a.arm_side * sx,
                   ry + a.arm_ahead * fy + a.arm_side * sy, a.arm_height]
            target = [rx + a.arm_look_side * sx, ry + a.arm_look_side * sy, a.arm_look_height]
        else:
            # Outward direction from the machine, turned for a three-quarter view.
            out = math.atan2(ry - my, rx - mx) + a.azimuth
            eye = [rx + a.distance * math.cos(out), ry + a.distance * math.sin(out), a.cam_height]
            # Look between the robot and the machine.
            target = [rx + a.look_ratio * (mx - rx), ry + a.look_ratio * (my - ry), a.look_height]
        x_min, x_max, y_min, y_max = a.room
        eye[0] = min(max(eye[0], x_min + 0.15), x_max - 0.15)
        eye[1] = min(max(eye[1], y_min + 0.15), y_max - 0.15)
        if self.eye is None:
            self.eye, self.target = eye, target
        else:  # low-pass filter for a smooth camera path
            k = a.smoothing
            self.eye = [e + k * (n - e) for e, n in zip(self.eye, eye)]
            self.target = [t + k * (n - t) for t, n in zip(self.target, target)]
        if self.pending is not None and self.pending.poll() is None:
            return  # previous set_pose still running
        x, y, z, roll, pitch, yaw = look_at_pose(self.eye, self.target)
        w, qx, qy, qz = rpy_to_quaternion(roll, pitch, yaw)
        req = (f'name: "{a.name}" position {{x: {x} y: {y} z: {z}}} '
               f'orientation {{w: {w} x: {qx} y: {qy} z: {qz}}}')
        self.pending = subprocess.Popen(
            ["gz", "service", "-s", f"/world/{a.world}/set_pose", "--reqtype", "gz.msgs.Pose",
             "--reptype", "gz.msgs.Boolean", "--timeout", "1000", "--req", req],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class Recorder(Node):
    """Write every received image to an MP4 file."""

    def __init__(self, topic: str, output: str, fps: float) -> None:
        # Sim time: frames and TF lookups follow the Gazebo clock.
        super().__init__("iso_video_recorder",
                         parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.bridge = CvBridge()
        self.output = output
        self.fps = fps
        self.writer = None
        self.frames = 0
        self.create_subscription(Image, topic, self.on_image, 10)

    def on_image(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if self.writer is None:
            height, width = frame.shape[:2]
            self.writer = cv2.VideoWriter(
                self.output, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
            self.get_logger().info(f"Recording {width}x{height} to {self.output}")
        self.writer.write(frame)
        self.frames += 1
        if self.frames % (int(self.fps) * 10) == 0:
            self.get_logger().info(f"{self.frames} frames ({self.frames / self.fps:.0f}s sim)")

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
        self.get_logger().info(f"Saved {self.frames} frames to {self.output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True, help="MP4 file to write")
    parser.add_argument("--world", default="table_cube_world")
    parser.add_argument("--name", default="iso_video_camera")
    # Defaults frame the Campetella loop (navigation.loop_path of the experiments) in the scanning
    # world (world = robot_map + (3, 3)), seen from the +X/+Y corner at 30 deg, ~3.8m away.
    parser.add_argument("--eye", type=float, nargs=3, default=[2.43, 1.93, 2.7])
    parser.add_argument("--target", type=float, nargs=3, default=[0.1, -0.4, 0.8])
    parser.add_argument("--hfov", type=float, default=1.2, help="rad")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--no-follow", dest="follow", action="store_false",
                        help="Keep the camera fixed at --eye instead of following the robot")
    parser.add_argument("--follow-frame", default="robot_base_footprint")
    parser.add_argument("--map-frame", default="robot_map")
    parser.add_argument("--map-offset", type=float, nargs=2, default=[3.0, 3.0],
                        help="world position of map-frame's origin (scanning_entrypoint.sh)")
    parser.add_argument("--machine-center", type=float, nargs=2, default=[0.35, -0.3],
                        help="Campetella center in world (x, y)")
    parser.add_argument("--distance", type=float, default=2.8, help="camera-robot distance (m)")
    parser.add_argument("--azimuth", type=float, default=0.6,
                        help="rotation (rad) of the camera away from straight outward")
    parser.add_argument("--cam-height", type=float, default=2.2)
    parser.add_argument("--look-ratio", type=float, default=0.35,
                        help="look-at point: fraction of the way from the robot to the machine")
    parser.add_argument("--look-height", type=float, default=0.9)
    parser.add_argument("--room", type=float, nargs=4, default=[-3.7, 3.0, -2.85, 2.15],
                        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
                        help="inner faces of the walls in the world frame (scanning.sdf)")
    parser.add_argument("--view", choices=["arm", "outside"], default="arm")
    parser.add_argument("--arm-ahead", type=float, default=1.6)
    parser.add_argument("--arm-side", type=float, default=0.3)
    parser.add_argument("--arm-height", type=float, default=1.9)
    parser.add_argument("--arm-look-side", type=float, default=0.35)
    parser.add_argument("--arm-look-height", type=float, default=1.0)
    parser.add_argument("--smoothing", type=float, default=0.15, help="0-1, camera low-pass gain")
    args = parser.parse_args()

    topic = f"/{args.name}/image"
    x, y, z, roll, pitch, yaw = look_at_pose(args.eye, args.target)
    sdf = CAMERA_SDF.format(name=args.name, topic=topic, fps=args.fps, hfov=args.hfov,
                            width=args.width, height=args.height)
    subprocess.run(["ros2", "run", "ros_gz_sim", "create", "-world", args.world,
                    "-name", args.name, "-string", sdf,
                    "-x", str(x), "-y", str(y), "-z", str(z),
                    "-R", str(roll), "-P", str(pitch), "-Y", str(yaw)], check=True)
    bridge = subprocess.Popen(["ros2", "run", "ros_gz_bridge", "parameter_bridge",
                               f"{topic}@sensor_msgs/msg/Image[gz.msgs.Image"])

    rclpy.init()
    node = Recorder(topic, args.output, args.fps)
    if args.follow:
        node.follower = Follower(node, args)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        bridge.terminate()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
