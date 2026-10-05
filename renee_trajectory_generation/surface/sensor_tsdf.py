"""Sensor mode: the surface to inspect from a previous RGB-D scan.

Input: a capture session folder written by /capture_rgbd (frames.jsonl: per
keyframe the depth_mm PNG, the colour intrinsics and T_world_camera = the
optical frame in robot_map), SensorSurfaceConfig, and optionally the CAD
surface (cad_mesh.py) to align with ICP and to complete what the scan missed.
Output: `SensorSurface`: the SurfaceModel, the reconstructed mesh (viewer),
and the machine pose corrected by ICP (None without ICP).

Steps: the depth images are fused in a TSDF (open3d ScalableTSDFVolume) with
the recorded camera poses; marching cubes gives the mesh, which is
Poisson-disk sampled like the CAD. With the CAD, ICP point-to-plane aligns the
scan to the CAD; the inverse moves the CAD onto the real machine (corrected
T_map_machine), and CAD points the scan did not reach are added.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from .base import SurfaceModel


@dataclass
class SensorSurfaceConfig:
    captures_dir: str = ""
    voxel_m: float = 0.015           # TSDF voxel
    sdf_trunc_m: float = 0.06        # TSDF truncation (~4 voxels)
    depth_range_m: tuple = (0.2, 3.0)
    frame_stride: int = 1            # use every n-th keyframe
    spacing_m: float = 0.025         # sample point spacing, as in the CAD mode
    min_normal_z: float = -0.7
    crop_margin_m: float = 0.3       # with the CAD: drop the reconstruction outside its boxes + margin (floor, walls)
    icp: bool = True                 # align to the CAD (needs the CAD)
    icp_max_distance_m: float = 0.08
    icp_iterations: int = 60
    complete_with_cad: bool = True   # add the CAD points farther than complete_distance_m from the scan
    complete_distance_m: float = 0.04
    seed: int = 0

    def __post_init__(self):
        self.depth_range_m = tuple(float(v) for v in self.depth_range_m)
        if not self.captures_dir:
            raise ValueError("surface.captures_dir is required in sensor mode")
        if self.voxel_m <= 0.0 or self.sdf_trunc_m <= self.voxel_m:
            raise ValueError("surface.voxel_m must be > 0 and surface.sdf_trunc_m > surface.voxel_m")
        if self.frame_stride < 1:
            raise ValueError("surface.frame_stride must be >= 1")


@dataclass
class SensorSurface:
    surface: SurfaceModel
    mesh: object                 # open3d TriangleMesh (map frame), for the viewer
    T_map_machine: np.ndarray | None  # corrected by ICP, None without it
    icp_fitness: float | None
    icp_rmse_m: float | None
    frames_used: int


def read_session(captures_dir: str, stride: int = 1) -> list[dict]:
    """Input: capture session folder (frames.jsonl). Output: one dict per keyframe
    {depth, rgb (paths), K (3x3), T_map_camera (4x4)}.
    Raises: FileNotFoundError / ValueError for a missing or empty session."""
    path = os.path.join(captures_dir, "frames.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no frames.jsonl in '{captures_dir}' (surface.captures_dir)")
    frames = []
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            keyframe = record.get("session_keyframe") or {}
            depth = record.get("depth_mm") or keyframe.get("depth_path")
            if not depth or record.get("T_world_camera") is None or keyframe.get("intrinsics") is None:
                continue
            frames.append({"depth": os.path.join(captures_dir, depth),
                           "rgb": os.path.join(captures_dir, record["rgb"]) if record.get("rgb") else "",
                           "K": np.array(keyframe["intrinsics"], dtype=float),
                           "T_map_camera": np.array(record["T_world_camera"], dtype=float)})
    if not frames:
        raise ValueError(f"no usable keyframes in {path}")
    return frames[::stride]


def integrate_tsdf(frames: list[dict], cfg: SensorSurfaceConfig, log=print):
    """Input: keyframes (read_session), config. Output: open3d TriangleMesh (marching cubes), map frame."""
    import open3d as o3d
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=cfg.voxel_m, sdf_trunc=cfg.sdf_trunc_m,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for i, frame in enumerate(frames):
        depth = o3d.io.read_image(frame["depth"])
        height, width = np.asarray(depth).shape[:2]
        color = o3d.io.read_image(frame["rgb"]) if frame["rgb"] and os.path.isfile(frame["rgb"]) else None
        if color is None or np.asarray(color).shape[:2] != (height, width):
            # The depth is aligned to the colour camera; without a matching
            # image only the geometry matters.
            color = o3d.geometry.Image(np.full((height, width, 3), 160, dtype=np.uint8))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=1000.0, depth_trunc=cfg.depth_range_m[1], convert_rgb_to_intensity=False)
        K = frame["K"]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        volume.integrate(rgbd, intrinsic, np.linalg.inv(frame["T_map_camera"]))
        if (i + 1) % 20 == 0:
            log(f"[surface] fused {i + 1}/{len(frames)} keyframes")
    mesh = volume.extract_triangle_mesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    return mesh


def _crop_to_boxes(mesh, boxes, margin_m: float):
    """Output: the mesh without the triangles outside the boxes' map AABB + margin."""
    corners = []
    for box in boxes:
        half = box.size / 2.0
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        corners.append((signs * half) @ box.rotation.T + box.center)
    corners = np.vstack(corners)
    low, high = corners.min(axis=0) - margin_m, corners.max(axis=0) + margin_m
    vertices = np.asarray(mesh.vertices)
    outside = np.any((vertices < low) | (vertices > high), axis=1)
    mesh.remove_vertices_by_mask(outside)
    return mesh


def _sample(mesh, spacing_m: float, seed: int):
    """Output: (points, normals, triangle ids) Poisson-disk sampled on the mesh."""
    import open3d as o3d
    o3d.utility.random.seed(seed)
    count = max(1000, int(mesh.get_surface_area() / spacing_m ** 2))
    points = np.asarray(mesh.sample_points_poisson_disk(number_of_points=count, init_factor=4).points)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
                        o3d.core.Tensor(np.asarray(mesh.triangles, dtype=np.uint32)))
    primitive = scene.compute_closest_points(o3d.core.Tensor(points.astype(np.float32)))["primitive_ids"].numpy()
    return points, np.asarray(mesh.triangle_normals)[primitive.astype(np.int64)]


def build_sensor_surface(cfg: SensorSurfaceConfig, cad: SurfaceModel | None = None,
                         T_map_machine: np.ndarray | None = None, boxes=None, log=print) -> SensorSurface:
    """Input: config, optionally the CAD SurfaceModel placed at T_map_machine and the
    machine's collision boxes in the map (machine.Machine.boxes_in_map).
    Output: SensorSurface (surface, viewer mesh, corrected machine pose)."""
    import open3d as o3d
    from scipy.spatial import cKDTree

    frames = read_session(cfg.captures_dir, cfg.frame_stride)
    log(f"[surface] {len(frames)} keyframes from {cfg.captures_dir}")
    mesh = integrate_tsdf(frames, cfg, log)
    if boxes:
        mesh = _crop_to_boxes(mesh, boxes, cfg.crop_margin_m)
    if len(mesh.triangles) == 0:
        raise ValueError("the reconstruction is empty (check the captures, surface.depth_range_m, crop_margin_m)")
    mesh.compute_triangle_normals()
    points, normals = _sample(mesh, cfg.spacing_m, cfg.seed)
    log(f"[surface] reconstruction: {len(mesh.triangles)} triangles, {len(points)} samples")

    T_corrected, fitness, rmse = None, None, None
    section_names, part_names = ["scan"], ["scan"]
    section_ids = np.zeros(len(points), dtype=np.int32)
    part_ids = np.zeros(len(points), dtype=np.int32)
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if cad is not None:
        T_scan_to_cad = np.eye(4)
        if cfg.icp:
            source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cad.points_m))
            target.normals = o3d.utility.Vector3dVector(cad.normals)
            result = o3d.pipelines.registration.registration_icp(
                source, target, cfg.icp_max_distance_m, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=cfg.icp_iterations))
            T_scan_to_cad, fitness, rmse = result.transformation, float(result.fitness), float(result.inlier_rmse)
            if T_map_machine is not None:
                # CAD ~ T_scan_to_cad @ scan, so the real machine is the CAD moved back.
                T_corrected = np.linalg.inv(T_scan_to_cad) @ T_map_machine
            shift = np.linalg.norm(T_scan_to_cad[:3, 3])
            log(f"[surface] ICP fitness {fitness:.2f}, rmse {rmse * 1000:.1f} mm, CAD shift {shift * 1000:.0f} mm")
        # The CAD moved onto the scan (map frame).
        T_cad_to_scan = np.linalg.inv(T_scan_to_cad)
        cad_points = cad.points_m @ T_cad_to_scan[:3, :3].T + T_cad_to_scan[:3, 3]
        cad_normals = cad.normals @ T_cad_to_scan[:3, :3].T
        # Each scan point takes the section and part of its closest CAD point.
        _, nearest = cKDTree(cad_points).query(points)
        section_names, part_names = list(cad.section_names), list(cad.part_names)
        section_ids, part_ids = cad.section_ids[nearest], cad.part_ids[nearest]
        if cfg.complete_with_cad:
            distance, _ = cKDTree(points).query(cad_points)
            missing = distance > cfg.complete_distance_m
            points = np.vstack([points, cad_points[missing]])
            normals = np.vstack([normals, cad_normals[missing]])
            section_ids = np.concatenate([section_ids, cad.section_ids[missing]])
            part_ids = np.concatenate([part_ids, cad.part_ids[missing]])
            # Occluders: the scan plus the moved CAD (it hides what the scan missed).
            cad_vertices = cad.vertices_m @ T_cad_to_scan[:3, :3].T + T_cad_to_scan[:3, 3]
            triangles = np.vstack([triangles, cad.triangles + len(vertices)])
            vertices = np.vstack([vertices, cad_vertices])
            log(f"[surface] added {int(missing.sum())} CAD points the scan did not reach")

    keep = normals[:, 2] >= cfg.min_normal_z
    surface = SurfaceModel(points[keep], normals[keep], section_ids[keep], section_names, vertices, triangles,
                           part_ids[keep], part_names)
    return SensorSurface(surface, mesh, T_corrected, fitness, rmse, len(frames))
