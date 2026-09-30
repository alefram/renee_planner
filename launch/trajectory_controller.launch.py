# /usr/bin/env python3
"""Run one experiment (a scanning / cleaning mission) with trajectory_controller_node.

    ros2 launch renee_trajectory_generation trajectory_controller.launch.py experiment:=defect_detection_sim

experiment: a name (config/experiments/<name>.yaml) or a path to a YAML.
The YAML holds everything for the run, including under
`experiment.node_parameters` the node's ROS parameters (sim or real robot
interfaces). With a symlink install the YAML is read from the source tree
(edits need no rebuild) and the outputs (plots, summary.txt, plan.yaml, a
copy of the YAML and the node parameters) go to the package's
data/<experiment name>/<YYYYmmdd_HHMMSS>/ (git-ignored); otherwise to
~/.ros/hqp_runs/<experiment name>/. With `capture.start_server: true` (e.g.
screw_detection) the launch also starts renee_action_servers'
capture_rgbd_action_server with `capture.server_params` (its config folder)
and `capture.server_overrides`. With a `record:` section it records a ros2 bag
of the topics matching `record.regex` into <run folder>/bag; with `record.ground_truth`
(a Gazebo Pose_V topic) scripts/gz_ground_truth.py publishes the robot's true pose
on /ground_truth/robot_pose (world frame) for the bag.

Needs Nav2 and move_group running (e.g. `docker compose up scanning` and
`docker compose up --no-deps moveit`). Pinocchio/OSQP come from the HQP
venv (README "Dependencies"; HQP_VENV, default /tmp/renee-hqp-venv) when it
exists.
"""
import os
import time

import yaml
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# The package's source folder when this file is a symlink into it (colcon
# --symlink-install), else its installed share folder.
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
IN_SOURCE = os.path.isfile(os.path.join(PACKAGE_DIR, "package.xml")) and os.path.isdir(
    os.path.join(PACKAGE_DIR, "renee_trajectory_generation"))


def _venv_environment() -> dict:
    """Output: PYTHONPATH / LD_LIBRARY_PATH putting the HQP venv's Pinocchio first, if the venv exists."""
    venv = os.environ.get("HQP_VENV", "/tmp/renee-hqp-venv")
    site = os.path.join(venv, "lib", "python3.12", "site-packages")
    if not os.path.isdir(site):
        return {}
    cmeel = os.path.join(site, "cmeel.prefix")
    return {
        "PYTHONPATH": os.pathsep.join(filter(None, [
            os.path.join(cmeel, "lib", "python3.12", "site-packages"), site,
            os.environ.get("PYTHONPATH", "")])),
        "LD_LIBRARY_PATH": os.pathsep.join(filter(None, [
            os.path.join(cmeel, "lib"), os.environ.get("LD_LIBRARY_PATH", "")])),
    }


def _launch_node(context):
    experiment = LaunchConfiguration("experiment").perform(context)
    path = experiment if os.path.isfile(experiment) else os.path.join(
        PACKAGE_DIR, "config", "experiments", f"{experiment}.yaml")
    if not os.path.isfile(path):
        available = sorted(f[:-5] for f in os.listdir(os.path.join(PACKAGE_DIR, "config", "experiments"))
                           if f.endswith(".yaml"))
        raise RuntimeError(f"No experiment '{experiment}' ({path}); available: {', '.join(available)}")
    path = os.path.realpath(path)
    with open(path) as handle:
        config = yaml.safe_load(handle)
    section = config.get("experiment") or {}
    name = section.get("name") or os.path.splitext(os.path.basename(path))[0]
    data_dir = (os.path.join(PACKAGE_DIR, "data") if IN_SOURCE
                else os.path.join(os.path.expanduser("~"), ".ros", "hqp_runs"))
    run_dir = os.path.join(data_dir, name, time.strftime("%Y%m%d_%H%M%S"))
    parameters = dict(section.get("node_parameters") or {})
    parameters.update(tasks_config_file=path, experiment_name=name,
                      output_dir=os.path.join(data_dir, name), run_dir=run_dir,
                      # The executed trajectory, kept with the configs (source tree).
                      trajectory_dir=os.path.join(PACKAGE_DIR, "config", "trajectories") if IN_SOURCE else "")
    actions = [Node(package="renee_trajectory_generation", executable="trajectory_controller_node.py",
                    name="trajectory_controller_node", output="screen", emulate_tty=True,
                    parameters=[parameters], additional_env=_venv_environment(),
                    # The run ends with the node (done, or a config error):
                    # stop the helpers (the capture server) too.
                    on_exit=Shutdown(),
                    # On Ctrl+C the node still writes its plots and trajectory:
                    # give it time before the SIGTERM/SIGKILL escalation.
                    sigterm_timeout="60", sigkill_timeout="30")]
    capture = config.get("capture") or {}
    if capture.get("start_server") and capture.get("enabled", True):
        server_params = os.path.join(get_package_share_directory("renee_action_servers"), "config",
                                     capture.get("server_params", "rgbd_capture_sim.yaml"))
        actions.append(Node(package="renee_action_servers", executable="capture_rgbd_action_server",
                            name="capture_rgbd_action_server", output="screen",
                            parameters=[server_params, dict(capture.get("server_overrides") or {},
                                        use_sim_time=bool(parameters.get("use_sim_time", False)))]))
    record = config.get("record") or {}
    if record.get("regex"):
        # Stopped (SIGINT, the bag is closed) with the rest when the node exits.
        os.makedirs(run_dir, exist_ok=True)
        command = ["ros2", "bag", "record", "-o", os.path.join(run_dir, "bag"), "--include-hidden-topics",
                   "-e", record["regex"]]
        if parameters.get("use_sim_time", False):
            command.append("--use-sim-time")
        actions.append(ExecuteProcess(cmd=command, name="run_bag", output="log"))
    if record.get("ground_truth"):
        # Gazebo's true robot pose (scripts/gz_ground_truth.py), e.g. to compare with the localization.
        actions.append(ExecuteProcess(
            cmd=["python3", os.path.join(PACKAGE_DIR, "scripts", "gz_ground_truth.py"),
                 "--gz-topic", record["ground_truth"], "--entity", record.get("ground_truth_entity", "robot")],
            name="ground_truth", output="log"))
    return actions


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("experiment", default_value="defect_detection_sim",
                              description="config/experiments/<name>.yaml, or a path to an experiment YAML"),
        OpaqueFunction(function=_launch_node),
    ])
