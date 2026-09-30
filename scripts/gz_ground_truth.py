#!/usr/bin/env python3
"""Publish a Gazebo entity's true pose on ROS, to compare with the localization.

Reads a gz-transport Pose_V topic (e.g. /world/<world>/dynamic_pose/info),
picks the entity named --entity (the robot model) and publishes its pose as
geometry_msgs/PoseStamped on --topic in the `world` frame, stamped with
Gazebo's sim time. ros_gz_bridge's Pose_V -> TFMessage conversion drops the
entity names, so it cannot tell the robot apart.

    python3 gz_ground_truth.py --gz-topic /world/table_cube_world/dynamic_pose/info
"""
import argparse
import threading

import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node as GzNode


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gz-topic", required=True, help="Gazebo Pose_V topic")
    parser.add_argument("--entity", default="robot", help="entity name in the Pose_V")
    parser.add_argument("--topic", default="/ground_truth/robot_pose", help="ROS topic")
    parser.add_argument("--frame", default="world")
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = rclpy.create_node("gz_ground_truth")
    pub = node.create_publisher(PoseStamped, args.topic, 10)
    lock = threading.Lock()

    def on_poses(msg: Pose_V) -> None:
        for pose in msg.pose:
            if pose.name != args.entity:
                continue
            out = PoseStamped()
            out.header.frame_id = args.frame
            out.header.stamp = Time(sec=msg.header.stamp.sec, nanosec=msg.header.stamp.nsec)
            out.pose.position.x, out.pose.position.y, out.pose.position.z = (
                pose.position.x, pose.position.y, pose.position.z)
            out.pose.orientation.x, out.pose.orientation.y = pose.orientation.x, pose.orientation.y
            out.pose.orientation.z, out.pose.orientation.w = pose.orientation.z, pose.orientation.w
            with lock:
                pub.publish(out)
            return

    gz_node = GzNode()
    if not gz_node.subscribe(Pose_V, args.gz_topic, on_poses):
        raise SystemExit(f"could not subscribe to {args.gz_topic}")
    node.get_logger().info(f"'{args.entity}' of {args.gz_topic} -> {args.topic} ({args.frame})")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
