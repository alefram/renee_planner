# /usr/bin/env python3
"""Draw an arm_static_planner experiment's targets and tool goals as TF frames, for RViz.

    ros2 launch renee_trajectory_generation arm_static_planner_frames.launch.py experiment:=screwPoseTest01

experiment: a name (config/experiments/<name>.yaml) or a path to a YAML.
Static TF frames, with the same goals arm_static_planner_node computes (renee_trajectory_generation.arm_static_planner):

- `<target>`: each YAML target pose (after `x_reference`), in reference_frame.
- `<target>_goal_<standoff mm>mm`: where the tool frame (`frames.tool` /
  `frames.camera`) goes at each approach stage. For a TF target (tf_frame)
  it hangs from that frame, without x_reference (applying it needs the
  frame's pose, only known at run time).
- the YAML's `publish_frames`.

Nothing moves; add them to any open RViz's TF display (e.g. the navigation one).
"""
import math
import os

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from scipy.spatial.transform import Rotation

from renee_trajectory_generation.arm_static_planner import target_goals, yaml_target_pose

# The package's source folder when this file is a symlink into it (colcon
# --symlink-install), else its installed share folder.
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def _static_tf(name: str, parent: str, pose, use_sim_time: bool) -> Node:
    """static_transform_publisher of a (4, 4) pose as `name` in `parent`."""
    qx, qy, qz, qw = Rotation.from_matrix(pose[:3, :3]).as_quat()
    x, y, z = pose[:3, 3]
    return Node(package="tf2_ros", executable="static_transform_publisher", name=f"frame_{name}",
                output="log", parameters=[{"use_sim_time": use_sim_time}],
                arguments=["--x", repr(float(x)), "--y", repr(float(y)), "--z", repr(float(z)),
                           "--qx", repr(float(qx)), "--qy", repr(float(qy)), "--qz", repr(float(qz)),
                           "--qw", repr(float(qw)), "--frame-id", parent, "--child-frame-id", name])


def _launch(context):
    experiment = LaunchConfiguration("experiment").perform(context)
    path = experiment if os.path.isfile(experiment) else os.path.join(
        PACKAGE_DIR, "config", "experiments", f"{experiment}.yaml")
    if not os.path.isfile(path):
        raise RuntimeError(f"No experiment '{experiment}' ({path})")
    with open(path) as handle:
        config = yaml.safe_load(handle)
    use_sim_time = bool(((config.get("experiment") or {}).get("node_parameters") or {}).get("use_sim_time", False))
    reference_frame = config.get("reference_frame", "robot_arm_base_link")
    frames_section = config.get("frames") or {}
    tool_frame = frames_section.get("tool") or frames_section.get(
        "camera", "robot_arm_rgbd_camera_left_camera_optical_frame")
    view = config.get("view") or {}

    actions, lines = [], []
    for frame in config.get("publish_frames") or []:
        actions.append(Node(package="tf2_ros", executable="static_transform_publisher",
                            name=f"frame_{frame['name']}", output="log",
                            parameters=[{"use_sim_time": use_sim_time}],
                            arguments=["--x", str(float(frame["position"][0])), "--y", str(float(frame["position"][1])),
                                       "--z", str(float(frame["position"][2])),
                                       "--roll", str(math.radians(float(frame.get("rpy_deg", [0, 0, 0])[0]))),
                                       "--pitch", str(math.radians(float(frame.get("rpy_deg", [0, 0, 0])[1]))),
                                       "--yaw", str(math.radians(float(frame.get("rpy_deg", [0, 0, 0])[2]))),
                                       "--frame-id", frame["parent"], "--child-frame-id", frame["name"]]))
    for index, target in enumerate(config.get("targets") or []):
        name = target.get("name", f"target_{index}")
        if "tf_frame" in target:
            # Goals relative to the TF frame itself (identity target).
            parent = target["tf_frame"]
            local = {key: value for key, value in target.items() if key != "x_reference"}
            local_view = {key: value for key, value in view.items() if key != "x_reference"}
            _, goals = target_goals(local, yaml_target_pose({"position": [0.0, 0.0, 0.0]}), local_view)
        else:
            parent = reference_frame
            target_pose, goals = target_goals(target, yaml_target_pose(target), view)
            actions.append(_static_tf(name, parent, target_pose, use_sim_time))
        for standoff, goal in goals:
            goal_name = f"{name}_goal_{round(standoff * 1000):d}mm"
            actions.append(_static_tf(goal_name, parent, goal, use_sim_time))
            lines.append(f"  {goal_name}: {tool_frame} goal in {parent}, position "
                         f"{[round(float(v), 4) for v in goal[:3, 3]]}")
    actions.insert(0, LogInfo(msg=f"{os.path.basename(path)}: {len(lines)} goal frame(s) of {tool_frame}\n"
                                  + "\n".join(lines)))
    return actions


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("experiment", default_value="arm_static_planner_sim",
                              description="config/experiments/<name>.yaml, or a path to an experiment YAML"),
        OpaqueFunction(function=_launch),
    ])
