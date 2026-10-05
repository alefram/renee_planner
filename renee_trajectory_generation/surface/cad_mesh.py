"""CAD mode: the surface to inspect from the machine's STL meshes and box visuals (cart supports).

Input: `Machine` (machine.py: mesh parts + T_map_machine) and CadSurfaceConfig
(sampling spacing, decimation size, filters).
Output: `SurfaceModel` (sample points with normals and section, occluder mesh)
and the decimated, section-coloured mesh for the web viewer (`export_glb`).

Steps: each STL is scaled and placed in the map, decimated (quadric) to its
share of `max_triangles`, merged; points are Poisson-disk sampled on the
merged mesh (~`spacing_m` apart) and take the normal and section of their
closest triangle. Points facing down (bottom faces) or enclosed by other parts
(contact faces) are dropped: no camera pose can see them.
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass

import numpy as np

from ..machine import Machine
from .base import SurfaceModel

# Section colours for the viewer mesh (RGB 0..1), cycled by section id.
SECTION_COLORS = np.array([[0.36, 0.55, 0.80], [0.85, 0.55, 0.25], [0.45, 0.70, 0.45], [0.75, 0.40, 0.60],
                           [0.60, 0.60, 0.60], [0.80, 0.75, 0.35], [0.40, 0.70, 0.75]])


@dataclass
class CadSurfaceConfig:
    spacing_m: float = 0.025        # distance between sample points
    max_triangles: int = 150_000    # merged mesh after decimation (sampling + occlusion)
    web_triangles: int = 60_000     # mesh exported for the viewer
    min_normal_z: float = -0.7      # drop points whose normal points further down (bottom faces)
    min_height_m: float | None = None  # drop points below this map z (floor contact), None: keep all
    enclosed_distance_m: float = 0.03  # drop points whose every probe ray hits within this (contact faces)
    seed: int = 0

    def __post_init__(self):
        if self.spacing_m <= 0.0:
            raise ValueError(f"surface.spacing_m must be > 0 (got {self.spacing_m})")
        if self.max_triangles < 1000 or self.web_triangles < 1000:
            raise ValueError("surface.max_triangles and surface.web_triangles must be >= 1000")
        if not -1.0 <= self.min_normal_z <= 1.0:
            raise ValueError(f"surface.min_normal_z must be in [-1, 1] (got {self.min_normal_z})")


def _stl_triangle_count(path: str) -> int:
    """Output: the triangle count of a binary STL (header), or an estimate for ASCII STL."""
    size = os.path.getsize(path)
    with open(path, "rb") as handle:
        header = handle.read(84)
    if len(header) == 84:
        count = struct.unpack("<I", header[80:84])[0]
        if 84 + 50 * count == size:
            return count
    return max(1, size // 250)


def _load_part(part, T_map_machine: np.ndarray):
    """Output: the part's open3d mesh in the map frame, welded (shared vertices) for decimation."""
    import open3d as o3d
    if part.box_size is not None:
        mesh = o3d.geometry.TriangleMesh.create_box(*part.box_size)
        mesh.translate(-0.5 * part.box_size)  # create_box starts at the origin; URDF boxes are centred
    else:
        mesh = o3d.io.read_triangle_mesh(part.path)
    if len(mesh.triangles) == 0:
        raise ValueError(f"mesh '{part.path}' ({part.name}) has no triangles")
    vertices = np.asarray(mesh.vertices) * part.scale_xyz
    T = T_map_machine @ part.T_machine_mesh
    mesh.vertices = o3d.utility.Vector3dVector(vertices @ T[:3, :3].T + T[:3, 3])
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    return mesh


def merged_mesh(machine: Machine, max_triangles: int, log=print):
    """Input: Machine, triangle budget for the merged mesh.
    Output: (open3d TriangleMesh in the map frame with section vertex colours,
    triangle_section_ids (T,), section_names, triangle_part_ids (T,), part_names = link names)."""
    import open3d as o3d
    sections = machine.sections
    parts = sorted({p.name for p in machine.parts})
    counts = np.array([12 if p.box_size is not None else _stl_triangle_count(p.path) for p in machine.parts],
                      dtype=float)
    # Each part keeps its share of the budget, so small parts are not wiped out.
    ratio = min(1.0, max_triangles / counts.sum())
    vertices, triangles, colors, triangle_sections, triangle_parts = [], [], [], [], []
    offset = 0
    for part, count in zip(machine.parts, counts):
        mesh = _load_part(part, machine.T_map_machine)
        target = max(12, int(round(len(mesh.triangles) * ratio)))
        if target < len(mesh.triangles):
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target)
        section_id = sections.index(part.section)
        part_vertices = np.asarray(mesh.vertices)
        vertices.append(part_vertices)
        triangles.append(np.asarray(mesh.triangles) + offset)
        colors.append(np.tile(SECTION_COLORS[section_id % len(SECTION_COLORS)], (len(part_vertices), 1)))
        triangle_sections.append(np.full(len(mesh.triangles), section_id, dtype=np.int32))
        triangle_parts.append(np.full(len(mesh.triangles), parts.index(part.name), dtype=np.int32))
        offset += len(part_vertices)
    merged = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.vstack(vertices)),
                                       o3d.utility.Vector3iVector(np.vstack(triangles)))
    merged.vertex_colors = o3d.utility.Vector3dVector(np.vstack(colors))
    merged.compute_triangle_normals()
    merged.compute_vertex_normals()
    log(f"[surface] {len(machine.parts)} parts, {int(counts.sum())} -> {len(merged.triangles)} triangles")
    return merged, np.concatenate(triangle_sections), sections, np.concatenate(triangle_parts), parts


def _probe_directions(normals: np.ndarray) -> list[np.ndarray]:
    """Output: the normal and four directions 45 deg from it (one per tangent half-axis)."""
    helper = np.where(np.abs(normals[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
    u = np.cross(normals, helper)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(normals, u)
    c = np.cos(np.pi / 4)
    return [normals] + [c * normals + c * s * w for w in (u, v) for s in (1.0, -1.0)]


def build_cad_surface(machine: Machine, cfg: CadSurfaceConfig, log=print):
    """Input: Machine placed in the map, CadSurfaceConfig.
    Output: (SurfaceModel, open3d TriangleMesh for the viewer)."""
    import open3d as o3d
    mesh, triangle_sections, sections, triangle_parts, parts = merged_mesh(machine, cfg.max_triangles, log)
    o3d.utility.random.seed(cfg.seed)
    count = max(1000, int(mesh.get_surface_area() / cfg.spacing_m ** 2))
    cloud = mesh.sample_points_poisson_disk(number_of_points=count, init_factor=4)
    points = np.asarray(cloud.points)

    # Normal and section from the closest triangle (the sampled cloud's own
    # normals are interpolated vertex normals, blurred at the CAD's sharp edges).
    scene = o3d.t.geometry.RaycastingScene()
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    scene.add_triangles(o3d.core.Tensor(vertices.astype(np.float32)), o3d.core.Tensor(faces.astype(np.uint32)))
    closest = scene.compute_closest_points(o3d.core.Tensor(points.astype(np.float32)))
    primitive = closest["primitive_ids"].numpy().astype(np.int64)
    normals = np.asarray(mesh.triangle_normals)[primitive]
    section_ids = triangle_sections[primitive]
    part_ids = triangle_parts[primitive]

    keep = normals[:, 2] >= cfg.min_normal_z
    if cfg.min_height_m is not None:
        keep &= points[:, 2] >= cfg.min_height_m
    # Contact faces between parts: every probe ray hits another part right away.
    enclosed = np.ones(len(points), dtype=bool)
    origins = points + 1e-3 * normals
    for direction in _probe_directions(normals):
        rays = np.hstack([origins, direction]).astype(np.float32)
        t_hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        enclosed &= t_hit < cfg.enclosed_distance_m
    keep &= ~enclosed
    log(f"[surface] {len(points)} samples, kept {int(keep.sum())} "
        f"(dropped {int((normals[:, 2] < cfg.min_normal_z).sum())} facing down, {int(enclosed.sum())} enclosed)")
    surface = SurfaceModel(points[keep], normals[keep], section_ids[keep], sections, vertices, faces,
                           part_ids[keep], parts)
    return surface, mesh


def export_glb(mesh, path: str, max_triangles: int) -> str:
    """Input: open3d TriangleMesh (map frame), output .glb path, triangle budget.
    Output: path of the written .glb (decimated for the browser).
    Raises: RuntimeError if open3d cannot write it."""
    import open3d as o3d
    if len(mesh.triangles) > max_triangles:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=max_triangles)
        mesh.compute_vertex_normals()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not o3d.io.write_triangle_mesh(path, mesh, write_vertex_normals=True, write_vertex_colors=True):
        raise RuntimeError(f"could not write the viewer mesh {path}")
    return path
