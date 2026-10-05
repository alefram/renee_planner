# /usr/bin/env python3
"""Start the scan-pose web viewer: its HTTP server and rosbridge.

    ros2 launch renee_trajectory_generation scan_viewer.launch.py     # open http://<host>:8091

    It shows the trajectories in config/trajectories (symlink install) or
    ~/.ros/scan_poses/trajectories, the same folder the generator writes to; with
    rosbridge it also follows /scan_trajectory and /tf live. rosbridge:=false
    starts only the HTTP server (offline review). Without the rosbridge_server
    package (apt: ros-jazzy-rosbridge-suite) it starts only the HTTP server too.
"""
import os

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
IN_SOURCE = os.path.isfile(os.path.join(PACKAGE_DIR, "package.xml")) and os.path.isdir(
    os.path.join(PACKAGE_DIR, "renee_trajectory_generation"))
TRAJECTORY_DIR = (os.path.join(PACKAGE_DIR, "config", "trajectories") if IN_SOURCE
                  else os.path.join(os.path.expanduser("~"), ".ros", "scan_poses", "trajectories"))


def _rosbridge(context):
    if LaunchConfiguration("rosbridge").perform(context) != "true":
        return []
    try:
        get_package_share_directory("rosbridge_server")
    except PackageNotFoundError:
        return [LogInfo(msg="rosbridge_server is not installed (apt install ros-jazzy-rosbridge-suite): "
                            "the viewer runs without live data (/scan_trajectory, /tf)")]
    return [Node(package="rosbridge_server", executable="rosbridge_websocket", name="scan_viewer_rosbridge",
                 output="log", parameters=[{"port": int(LaunchConfiguration("rosbridge_port").perform(context))}])]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("port", default_value="8091", description="HTTP port of the viewer"),
        DeclareLaunchArgument("rosbridge_port", default_value="9090"),
        DeclareLaunchArgument("rosbridge", default_value="true", choices=["true", "false"]),
        DeclareLaunchArgument("trajectory_dir", default_value=TRAJECTORY_DIR),
        Node(package="renee_trajectory_generation", executable="scan_viewer_server_node.py",
             name="scan_viewer_server_node", output="screen", emulate_tty=True,
             parameters=[{"web_dir": os.path.join(PACKAGE_DIR, "web", "dist"),
                          "trajectory_dir": LaunchConfiguration("trajectory_dir"),
                          "port": LaunchConfiguration("port"),
                          "rosbridge_port": LaunchConfiguration("rosbridge_port")}]),
        OpaqueFunction(function=_rosbridge),
    ])
