# /usr/bin/env python3
"""Run one arm-only camera-view experiment with arm_static_planner_node (base fixed, MoveIt moves the arm).

    ros2 launch renee_trajectory_generation arm_static_planner.launch.py experiment:=arm_static_planner_sim

experiment: a name (config/experiments/<name>.yaml) or a path to a YAML.
The node's ROS parameters come from the YAML's `experiment.node_parameters`
(sim or real robot). The results (results.yaml and a copy of the YAML) go to
the package's data/<experiment name>/<YYYYmmdd_HHMMSS>/ with a symlink
install (git-ignored), else to ~/.ros/hqp_runs/<experiment name>/. Each entry
of the YAML's `publish_frames` ({name, parent, position, rpy_deg}) is
published as a static TF frame, e.g. to try TF targets in sim.

Needs move_group running (e.g. `docker compose up --no-deps moveit`); no Nav2.
"""
import math
import os
import time

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# The package's source folder when this file is a symlink into it (colcon
# --symlink-install), else its installed share folder.
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
IN_SOURCE = os.path.isfile(os.path.join(PACKAGE_DIR, "package.xml")) and os.path.isdir(
    os.path.join(PACKAGE_DIR, "renee_trajectory_generation"))


def _launch_node(context):
    experiment = LaunchConfiguration("experiment").perform(context)
    path = experiment if os.path.isfile(experiment) else os.path.join(
        PACKAGE_DIR, "config", "experiments", f"{experiment}.yaml")
    if not os.path.isfile(path):
        raise RuntimeError(f"No experiment '{experiment}' ({path})")
    path = os.path.realpath(path)
    with open(path) as handle:
        config = yaml.safe_load(handle)
    section = config.get("experiment") or {}
    name = section.get("name") or os.path.splitext(os.path.basename(path))[0]
    data_dir = (os.path.join(PACKAGE_DIR, "data") if IN_SOURCE
                else os.path.join(os.path.expanduser("~"), ".ros", "hqp_runs"))
    parameters = dict(section.get("node_parameters") or {})
    parameters.update(tasks_config_file=path,
                      run_dir=os.path.join(data_dir, name, time.strftime("%Y%m%d_%H%M%S")))
    use_sim_time = bool(parameters.get("use_sim_time", False))
    actions = []
    for frame in config.get("publish_frames") or []:
        x, y, z = (str(float(v)) for v in frame["position"])
        roll, pitch, yaw = (str(math.radians(float(v))) for v in frame.get("rpy_deg", [0.0, 0.0, 0.0]))
        actions.append(Node(package="tf2_ros", executable="static_transform_publisher",
                            name=f"frame_{frame['name']}", output="log",
                            parameters=[{"use_sim_time": use_sim_time}],
                            arguments=["--x", x, "--y", y, "--z", z, "--roll", roll, "--pitch", pitch,
                                       "--yaw", yaw, "--frame-id", frame["parent"],
                                       "--child-frame-id", frame["name"]]))
    actions.append(Node(package="renee_trajectory_generation", executable="arm_static_planner_node.py",
                        name="arm_static_planner_node", output="screen", emulate_tty=True,
                        parameters=[parameters],
                        # The run ends with the node: stop the frame publishers too.
                        on_exit=Shutdown()))
    return actions


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("experiment", default_value="arm_static_planner_sim",
                              description="config/experiments/<name>.yaml, or a path to an experiment YAML"),
        OpaqueFunction(function=_launch_node),
    ])
