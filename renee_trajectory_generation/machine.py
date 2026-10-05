"""The machine to scan: its mesh parts, collision boxes and pose in the map.

Input: the Campetella URDF (the XML published on /campetella_robot_description,
or xacro's output), optionally campetella_config.yaml (only parts with
`spawn: true`), and T_map_machine (robot_map -> campetella_base_link, from TF
or the experiment YAML).
Output: `Machine` with the mesh parts (STL path, scale, pose in the machine
frame), the collision boxes (machine frame) and T_map_machine; helpers give
everything in the map frame.

All CAD meshes share the Campetella CAD frame (mm, scaled by 0.001 in the
URDF), so a part's pose is its fixed-joint chain down to the root link.
"""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
import yaml


@dataclass
class MeshPart:
    """One visual mesh of the machine.

    name: link name; section: machine section (extraction, transversal, ...);
    path: mesh file on disk; scale_xyz: URDF mesh scale; T_machine_mesh: 4x4
    pose of the mesh in the machine root frame.
    """
    name: str
    section: str
    path: str
    scale_xyz: np.ndarray
    T_machine_mesh: np.ndarray


@dataclass
class Box:
    """One collision box. center/rotation in the frame of the owning Machine call
    (machine frame from parse_urdf, map frame from Machine.boxes_in_map)."""
    name: str
    section: str
    center: np.ndarray
    size: np.ndarray
    rotation: np.ndarray


@dataclass
class Machine:
    root_link: str
    parts: list[MeshPart]
    boxes: list[Box]
    T_map_machine: np.ndarray = field(default_factory=lambda: np.eye(4))

    @property
    def sections(self) -> list[str]:
        """Output: the section names, sorted (index = section id in SurfaceModel)."""
        return sorted({p.section for p in self.parts} | {b.section for b in self.boxes})

    def boxes_in_map(self) -> list[Box]:
        """Output: the collision boxes in the map frame."""
        R, t = self.T_map_machine[:3, :3], self.T_map_machine[:3, 3]
        return [Box(b.name, b.section, R @ b.center + t, b.size.copy(), R @ b.rotation) for b in self.boxes]


# ---------------------------------------------------------------------------
# Small rigid-transform helpers (shared by the other modules).
# ---------------------------------------------------------------------------

def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Output: 3x3 rotation for URDF fixed-axis roll/pitch/yaw (Rz @ Ry @ Rx)."""
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def pose_matrix(xyz, rpy=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Input: position [x, y, z] (m), [roll, pitch, yaw] (rad). Output: 4x4 transform."""
    T = np.eye(4)
    T[:3, :3] = rpy_matrix(*[float(v) for v in rpy])
    T[:3, 3] = [float(v) for v in xyz]
    return T


def quaternion_matrix(xyz, quat_xyzw) -> np.ndarray:
    """Input: position, quaternion [qx, qy, qz, qw]. Output: 4x4 transform."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = [float(v) for v in xyz]
    return T


def section_of(link_name: str) -> str:
    """Output: the section of a link: its leading letters (vertical27_link -> vertical,
    cart_support_extraction_south_link -> cart)."""
    match = re.match(r"[a-zA-Z]+", link_name)
    return match.group(0).lower() if match else "machine"


# ---------------------------------------------------------------------------
# URDF parsing
# ---------------------------------------------------------------------------

def _origin(element) -> np.ndarray:
    origin = element.find("origin") if element is not None else None
    if origin is None:
        return np.eye(4)
    xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
    return pose_matrix(xyz, rpy)


def resolve_mesh_path(uri: str, path_prefix_map: list[tuple[str, str]] | None = None) -> str:
    """Input: a URDF mesh filename (file://..., package://pkg/..., or a path) and
    [from_prefix, to_prefix] pairs (e.g. a container path -> the host path).
    Output: the path on disk.
    Raises: FileNotFoundError if it does not exist after the mapping."""
    path = uri
    if uri.startswith("file://"):
        path = uri[len("file://"):]
    elif uri.startswith("package://"):
        package, _, relative = uri[len("package://"):].partition("/")
        from ament_index_python.packages import get_package_share_directory  # only for package:// URIs
        path = os.path.join(get_package_share_directory(package), relative)
    for source, target in path_prefix_map or []:
        if path.startswith(source):
            path = target + path[len(source):]
            break
    if not os.path.isfile(path):
        raise FileNotFoundError(f"mesh '{uri}' not found at '{path}' (machine.mesh_path_map maps path prefixes)")
    return path


def parse_urdf(urdf_xml: str, root_link: str = "campetella_base_link",
               enabled_links: set[str] | None = None,
               path_prefix_map: list[tuple[str, str]] | None = None) -> Machine:
    """Input: the URDF XML, its root link, the links to keep (None: all), mesh path mapping.
    Output: Machine (parts and boxes in the root-link frame, T_map_machine = identity).
    Raises: ValueError if the root link is missing."""
    robot = ET.fromstring(urdf_xml)
    links = {link.get("name"): link for link in robot.findall("link")}
    if root_link not in links:
        raise ValueError(f"root link '{root_link}' not in the URDF (machine.root_link)")
    # Parent of each child link and the joint origin; joints other than fixed
    # are taken at zero (the Campetella description is static for the scan).
    parent = {}
    for joint in robot.findall("joint"):
        child = joint.find("child").get("link")
        parent[child] = (joint.find("parent").get("link"), _origin(joint))

    def link_pose(name: str) -> np.ndarray | None:
        T = np.eye(4)
        seen = set()
        while name != root_link:
            if name not in parent or name in seen:
                return None  # not under the root link
            seen.add(name)
            name, T_parent_child = parent[name]
            T = T_parent_child @ T
        return T

    parts, boxes = [], []
    for name, link in links.items():
        if enabled_links is not None and name not in enabled_links:
            continue
        T_machine_link = link_pose(name)
        if T_machine_link is None:
            continue
        section = section_of(name)
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            scale = np.array([float(v) for v in mesh.get("scale", "1 1 1").split()])
            parts.append(MeshPart(name, section, resolve_mesh_path(mesh.get("filename"), path_prefix_map),
                                  scale, T_machine_link @ _origin(visual)))
        for collision in link.findall("collision"):
            box = collision.find("geometry/box")
            if box is None:
                continue
            T = T_machine_link @ _origin(collision)
            boxes.append(Box(name, section, T[:3, 3].copy(), np.array([float(v) for v in box.get("size").split()]),
                             T[:3, :3].copy()))
    return Machine(root_link, parts, boxes)


def enabled_links_from_config(config_path: str) -> set[str]:
    """Input: campetella_config.yaml. Output: the link names of the parts with `spawn: true`."""
    with open(config_path) as handle:
        config = yaml.safe_load(handle) or {}
    return {part.get("link", f"{name}_link") for name, part in (config.get("parts") or {}).items()
            if part.get("spawn", False)}


def load_machine(urdf_xml: str, T_map_machine: np.ndarray, root_link: str = "campetella_base_link",
                 config_path: str = "", path_prefix_map: list[tuple[str, str]] | None = None) -> Machine:
    """Input: URDF XML, robot_map -> root link 4x4, optional campetella_config.yaml, mesh path mapping.
    Output: Machine placed in the map.
    Raises: ValueError if no mesh part is left."""
    enabled = enabled_links_from_config(config_path) if config_path else None
    machine = parse_urdf(urdf_xml, root_link, enabled, path_prefix_map)
    if not machine.parts:
        raise ValueError("the machine has no mesh parts (check machine.config_file / the URDF)")
    machine.T_map_machine = np.asarray(T_map_machine, dtype=float)
    return machine
