# /usr/bin/env python3
"""Start the scan-pose generator (camera poses around the Campetella; never moves the robot).

    ros2 launch renee_trajectory_generation scan_pose_generator.launch.py experiment:=campetella_scan_cad_sim
    ros2 action send_goal /generate_scan_poses renee_trajectory_generation/action/GenerateScanPoses "{use_cache: true}"

experiment: a name (config/experiments/<name>.yaml) or a path to a YAML; the
goal can name another one. run_on_start:=true generates once at startup.
`experiment.node_parameters` in the YAML sets the node's ROS parameters (sim
or real robot, e.g. use_sim_time). With a symlink install the YAMLs are read
from the source tree and the results go to its config/trajectories (cache in
data/scan_cache, git-ignored); otherwise to ~/.ros/scan_poses/.

Needs the machine description and its TF (campetella_sim spawn_campetella.launch.py,
in sim or with gazebo:=false parent_frame:=robot_map on the real robot), or
machine.urdf_file + machine.pose_source: yaml in the experiment.
"""
import os

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# The package's source folder when this file is a symlink into it (colcon
# --symlink-install), else its installed share folder.
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
IN_SOURCE = os.path.isfile(os.path.join(PACKAGE_DIR, "package.xml")) and os.path.isdir(
    os.path.join(PACKAGE_DIR, "renee_trajectory_generation"))
EXPERIMENTS_DIR = os.path.join(PACKAGE_DIR, "config", "experiments")
OUTPUT_DIR = PACKAGE_DIR if IN_SOURCE else os.path.join(os.path.expanduser("~"), ".ros", "scan_poses")


def _launch_node(context):
    experiment = LaunchConfiguration("experiment").perform(context)
    path = experiment if os.path.isfile(experiment) else os.path.join(EXPERIMENTS_DIR, f"{experiment}.yaml")
    if not os.path.isfile(path):
        available = sorted(f[:-5] for f in os.listdir(EXPERIMENTS_DIR) if f.endswith(".yaml"))
        raise RuntimeError(f"No experiment '{experiment}' ({path}); available: {', '.join(available)}")
    path = os.path.realpath(path)
    with open(path) as handle:
        config = yaml.safe_load(handle) or {}
    parameters = dict((config.get("experiment") or {}).get("node_parameters") or {})
    parameters.update(
        experiment_file=path, experiments_dir=EXPERIMENTS_DIR,
        trajectory_dir=os.path.join(OUTPUT_DIR, "config", "trajectories") if IN_SOURCE
        else os.path.join(OUTPUT_DIR, "trajectories"),
        cache_dir=os.path.join(OUTPUT_DIR, "data", "scan_cache") if IN_SOURCE else os.path.join(OUTPUT_DIR, "cache"),
        run_on_start=LaunchConfiguration("run_on_start").perform(context).lower() == "true")
    return [Node(package="renee_trajectory_generation", executable="scan_pose_generator_node.py",
                 name="scan_pose_generator_node", output="screen", emulate_tty=True, parameters=[parameters])]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("experiment", default_value="campetella_scan_cad_sim",
                              description="config/experiments/<name>.yaml, or a path to an experiment YAML"),
        DeclareLaunchArgument("run_on_start", default_value="false", choices=["true", "false"],
                              description="Generate once at startup (else wait for an action goal)"),
        OpaqueFunction(function=_launch_node),
    ])
