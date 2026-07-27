#!/usr/bin/env python3

"""
Launch file for the scan planner node.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    planner_share = os.path.join(get_package_share_directory("renee_planner"))
    config_file = LaunchConfiguration("config_file")
    output_file = LaunchConfiguration("output_file")

    return LaunchDescription([
        DeclareLaunchArgument(
            "config_file",
            default_value=os.path.join(planner_share, "config", "scan_config.yaml"),
            description="Path to the scan planner YAML configuration.",
        ),
        DeclareLaunchArgument(
            "output_file",
            default_value="/tmp/renee_scan_plan.yaml",
            description="Path where the generated scan plan will be written.",
        ),
        Node(
            package="renee_planner",
            executable="planner_node",
            name="planner_node",
            output="screen",
            parameters=[{
                "config_file": config_file,
                "output_file": output_file,
            }],
        ),
    ])
