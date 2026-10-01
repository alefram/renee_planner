"""Goal poses of the arm-only targets (arm_static_planner_node), without ROS.

A target frame T (from TF, or a pose in the experiment YAML) gives one goal
of the tool frame per approach stage:

    tool = T' * Trans(0, 0, standoff) * RotX(pi) * RotZ(roll)

with T' = T, or T with its X re-aimed along `x_reference` (+Z kept). Used by
arm_static_planner_node.py (to move there) and arm_static_planner_frames.launch.py (to draw them).
"""
import math

import numpy as np
from scipy.spatial.transform import Rotation


def view_pose(target: np.ndarray, standoff_m: float, roll: float = 0.0) -> np.ndarray:
    """Camera optical-frame pose that looks at a target frame from its +Z.

    Input:
        target: (4, 4) pose of the target frame.
        standoff_m: distance (m) from the target's origin along its +Z.
        roll: rotation (rad) of the image about the optical axis.

    Output:
        np.ndarray: (4, 4) camera pose, in the target's parent frame.
    """
    offset = np.eye(4)
    offset[:3, :3] = (Rotation.from_euler("x", math.pi) * Rotation.from_euler("z", roll)).as_matrix()
    offset[2, 3] = standoff_m
    return target @ offset


def with_x_reference(target: np.ndarray, x_reference) -> np.ndarray:
    """Target with its +Z kept and its X re-aimed along x_reference (projected onto its XY plane).

    A detected target's roll often comes from the camera that saw it (e.g.
    screw_pose takes X from the camera's x axis), so the camera mount leaks
    into it; this ties the roll to a robot axis instead.

    Input:
        target: (4, 4) target pose.
        x_reference: (3,) direction, same frame as target; must not be along its +Z.

    Output:
        np.ndarray: (4, 4) target, same origin and +Z.
    """
    z = target[:3, 2]
    x = np.asarray(x_reference, dtype=float)
    x = x - (x @ z) * z
    if np.linalg.norm(x) < 1e-6:
        raise ValueError(f"x_reference {list(x_reference)} is along the target's +Z")
    x = x / np.linalg.norm(x)
    out = target.copy()
    out[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    return out


def pose_matrix(position, rpy=None, quaternion=None) -> np.ndarray:
    """Output: (4, 4) pose from a position and rpy (rad, extrinsic xyz) or a quaternion (x, y, z, w)."""
    pose = np.eye(4)
    if quaternion is not None:
        pose[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    elif rpy is not None:
        pose[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    pose[:3, 3] = position
    return pose


def pose_error(reached: np.ndarray, goal: np.ndarray) -> tuple:
    """Output: (position error m, orientation error rad) between two (4, 4) poses."""
    position = float(np.linalg.norm(reached[:3, 3] - goal[:3, 3]))
    angle = float(Rotation.from_matrix(goal[:3, :3].T @ reached[:3, :3]).magnitude())
    return position, angle


def standoffs(value) -> list:
    """Output: list of standoffs (m) from a number or a list of numbers (approach stages, in order)."""
    return [float(v) for v in value] if isinstance(value, (list, tuple)) else [float(value)]


def yaml_target_pose(target: dict) -> np.ndarray:
    """Output: (4, 4) pose of a YAML target ({position, rpy_deg | quaternion}) in reference_frame."""
    rpy = [math.radians(v) for v in target["rpy_deg"]] if "rpy_deg" in target else None
    return pose_matrix(target["position"], rpy=rpy, quaternion=target.get("quaternion"))


def target_goals(target: dict, target_pose: np.ndarray, view: dict) -> tuple:
    """The tool-frame goals of one target, one per approach stage.

    Input:
        target: the YAML target entry (its standoff_m / roll_deg / x_reference
            override the view's).
        target_pose: (4, 4) target frame (YAML pose, or looked up on TF).
        view: the experiment's `view` section.

    Output:
        tuple: ((4, 4) target pose after x_reference, [(standoff_m, (4, 4) goal), ...]).
    """
    x_reference = target.get("x_reference", view.get("x_reference"))
    if x_reference is not None:
        target_pose = with_x_reference(target_pose, x_reference)
    roll = math.radians(float(target.get("roll_deg", view.get("roll_deg", 0.0))))
    stages = standoffs(target.get("standoff_m", view.get("standoff_m", 0.4)))
    return target_pose, [(d, view_pose(target_pose, d, roll)) for d in stages]
