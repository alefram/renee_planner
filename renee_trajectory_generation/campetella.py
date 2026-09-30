#!/usr/bin/env python3
"""The Campetella's collision boxes, for the MoveIt planning scene and the base planner (no ROS).

The Campetella (with its cart supports) is its own robot_description
(campetella_sim: /campetella_robot_description), published in simulation and on
the real robot. Its box collisions, walked down its fixed-joint tree from the
root link, become the experiment's `moveit.collision_objects` (see
trajectory_controller_node.py, `moveit.collision_from_description`), so the
planning scene always matches the current model.

Only box collisions are used (the parts' `simple` collisions); a box rotated
other than about z is replaced by its axis-aligned bounding box. Boxes entirely
inside another one are dropped.
"""
import math
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def _origin(element) -> np.ndarray:
    """Output: 4x4 transform of an element's <origin> (identity without one)."""
    transform = np.eye(4)
    origin = element.find("origin") if element is not None else None
    if origin is not None:
        transform[:3, :3] = Rotation.from_euler(
            "xyz", [float(v) for v in origin.get("rpy", "0 0 0").split()]).as_matrix()
        transform[:3, 3] = [float(v) for v in origin.get("xyz", "0 0 0").split()]
    return transform


def urdf_collision_boxes(urdf_xml: str, root_link: str = None) -> list:
    """Box collisions of a URDF in its root link's frame.

    Input:
        urdf_xml: the robot_description.
        root_link: frame of the output (default: the URDF's root link).

    Output:
        list of {"link", "center" (3,), "size" (3,), "yaw"} (m, rad), boxes
        entirely inside another one dropped.

    Raises:
        ValueError: if root_link is not in the URDF, or a link is reached through
            a non-fixed joint (the pose of a moving part is not known here).
    """
    robot = ET.fromstring(urdf_xml)
    links = {link.get("name"): link for link in robot.findall("link")}
    children = {}
    for joint in robot.findall("joint"):
        children.setdefault(joint.find("parent").get("link"), []).append(joint)
    if root_link is None:
        child_links = {j.find("child").get("link") for j in robot.findall("joint")}
        roots = [name for name in links if name not in child_links]
        if len(roots) != 1:
            raise ValueError(f"URDF has {len(roots)} root links; pass root_link")
        root_link = roots[0]
    if root_link not in links:
        raise ValueError(f"{root_link} is not a link of the URDF")

    boxes, stack = [], [(root_link, np.eye(4))]
    while stack:
        name, link_pose = stack.pop()
        for collision in links[name].findall("collision"):
            box = collision.find("geometry/box")
            if box is None:
                continue
            pose = link_pose @ _origin(collision)
            size = np.array([float(v) for v in box.get("size").split()])
            rotation = pose[:3, :3]
            if abs(rotation[2, 2] - 1.0) < 1e-9:  # rotated about z only
                yaw = math.atan2(rotation[1, 0], rotation[0, 0])
            else:  # axis-aligned bounding box of the rotated box
                size, yaw = np.abs(rotation) @ size, 0.0
            boxes.append({"link": name, "center": pose[:3, 3].copy(), "size": size, "yaw": yaw})
        for joint in children.get(name, []):
            if joint.get("type") != "fixed":
                raise ValueError(f"joint {joint.get('name')} is {joint.get('type')}, not fixed")
            stack.append((joint.find("child").get("link"), link_pose @ _origin(joint)))
    return _drop_contained(boxes)


def _drop_contained(boxes: list) -> list:
    """Output: boxes without those entirely inside another axis-aligned one of the same yaw."""
    def bounds(box):
        return box["center"] - box["size"] / 2, box["center"] + box["size"] / 2

    kept = []
    for index, box in enumerate(boxes):
        low, high = bounds(box)
        inside = False
        for other_index, other in enumerate(boxes):
            if other_index == index or abs(other["yaw"] - box["yaw"]) > 1e-9:
                continue
            other_low, other_high = bounds(other)
            if np.all(other_low <= low + 1e-9) and np.all(other_high >= high - 1e-9):
                # Identical boxes: keep the first one only.
                if not (np.allclose(other_low, low) and np.allclose(other_high, high)) or other_index < index:
                    inside = True
                    break
        if not inside:
            kept.append(box)
    return kept


def to_collision_objects(boxes: list, root_pose, frame: str, margin_m: float = 0.0,
                         id_prefix: str = "") -> list:
    """Place root-frame boxes in `frame` as MoveIt collision_objects.

    Input:
        boxes: urdf_collision_boxes() output.
        root_pose: (x, y, z, yaw) of the root link in `frame` (planar: roll and
            pitch are not supported, as for the base planner's boxes).
        frame: frame of the objects (the experiment's targets_frame).
        margin_m: added on every side of each box.
        id_prefix: prefix of the object ids (the link name follows).

    Output:
        list of {"id", "frame", "box", "position", "yaw"} (arm_motion.apply_collision_objects).
    """
    x, y, z, yaw = (float(v) for v in root_pose)
    c, s = math.cos(yaw), math.sin(yaw)
    objects = []
    for box in boxes:
        bx, by, bz = box["center"]
        objects.append({
            "id": f"{id_prefix}{box['link']}",
            "frame": frame,
            # Plain floats (not numpy): the objects are also written to plan.yaml.
            "box": [round(float(v) + 2 * margin_m, 4) for v in box["size"]],
            "position": [round(float(v), 4) for v in (x + c * bx - s * by, y + s * bx + c * by, z + bz)],
            "yaw": round(float(math.atan2(math.sin(yaw + box["yaw"]), math.cos(yaw + box["yaw"]))), 6),
        })
    return objects
