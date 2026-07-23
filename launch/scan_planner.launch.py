from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config_file = LaunchConfiguration("config_file")
    output_file = LaunchConfiguration("output_file")

    return LaunchDescription([
        DeclareLaunchArgument(
            "config_file",
            default_value="",
            description="Path to the scan planner YAML configuration.",
        ),
        DeclareLaunchArgument(
            "output_file",
            default_value="",
            description="Path where the generated scan plan will be written.",
        ),
        Node(
            package="renee_planner",
            executable="scan_planner_node",
            name="scan_planner_node",
            output="screen",
            parameters=[{
                "config_file": config_file,
                "output_file": output_file,
            }],
        ),
    ])
