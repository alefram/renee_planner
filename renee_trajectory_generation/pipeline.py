"""Runs the whole generator: experiment YAML -> ordered camera poses.

Input: the experiment YAML (mode cad | sensor and one section per stage), the
machine URDF XML and T_map_machine (robot_map -> campetella_base_link).
Output: the Trajectory (trajectory.py), written as <output_dir>/<experiment>.yaml
+ <experiment>_coverage.npz + <experiment>_machine.glb (viewer mesh).

Order: machine -> surface (cad_mesh | sensor_tsdf) -> target (the parts to
inspect) -> workspace -> candidates -> visibility -> set_cover -> ordering -> trajectory. The heavy results
(surface, candidates, visibility) are cached in cache_dir under a hash of
everything they depend on, so changing e.g. set_cover only redoes the cheap
stages. This is the only module the ROS node calls.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field

import numpy as np
import yaml
from scipy import sparse

from .camera import CameraConfig, rotation_to_quaternion
from .candidates import CandidateConfig, Candidates, generate_candidates
from .machine import load_machine, pose_matrix
from .ordering import OrderConfig, order_poses
from .ring import RingConfig, ring_poses
from .set_cover import CoverConfig, greedy_set_cover, select_all
from .target import TargetConfig, select_target
from .surface.base import SurfaceModel
from .surface.cad_mesh import CadSurfaceConfig, build_cad_surface, export_glb
from .surface.sensor_tsdf import SensorSurfaceConfig, build_sensor_surface
from .trajectory import CameraPose, Trajectory, save_trajectory
from .visibility import visibility_matrix
from .workspace import Workspace, WorkspaceConfig

MODES = ("cad", "sensor")
METHODS = ("coverage", "ring")   # poses.method: set cover over candidates | fixed ring
_SENSOR_KEYS = {f.name for f in dataclasses.fields(SensorSurfaceConfig)}
_CAD_KEYS = {f.name for f in dataclasses.fields(CadSurfaceConfig)}


@dataclass
class MachineConfig:
    root_link: str = "campetella_base_link"
    description_topic: str = "/campetella_robot_description"
    urdf_file: str = ""              # empty: the description topic; else a .urdf / .xacro
    config_file: str = ""            # campetella_config.yaml: keep only `spawn: true` parts; empty: all URDF links
    pose_source: str = "tf"          # tf: robot_map -> root_link from TF; yaml: `pose` below
    pose: dict = field(default_factory=lambda: {"xyz": [0.0, 0.0, 0.0], "rpy_deg": [0.0, 0.0, 0.0]})
    mesh_path_map: list = field(default_factory=list)  # [[from_prefix, to_prefix], ...] for mesh paths

    def __post_init__(self):
        if self.pose_source not in ("tf", "yaml"):
            raise ValueError(f"machine.pose_source must be tf or yaml (got {self.pose_source})")
        if len(self.pose.get("xyz", [])) != 3 or len(self.pose.get("rpy_deg", [0, 0, 0])) != 3:
            raise ValueError("machine.pose needs xyz: [x, y, z] and rpy_deg: [roll, pitch, yaw]")
        self.mesh_path_map = [tuple(pair) for pair in self.mesh_path_map]


@dataclass
class ExperimentConfig:
    name: str
    mode: str
    frame: str
    description: str
    node_parameters: dict
    machine: MachineConfig
    cad: CadSurfaceConfig
    sensor: SensorSurfaceConfig | None
    method: str
    ring: RingConfig
    target: TargetConfig
    camera: CameraConfig
    workspace: WorkspaceConfig
    candidates: CandidateConfig
    occlusion_tolerance_m: float
    cover: CoverConfig
    ordering: OrderConfig
    output_dir: str
    source_path: str


def resolve_path(path: str, base_dir: str = "") -> str:
    """Input: a path (package://pkg/..., ~/..., absolute, or relative to base_dir).
    Output: the absolute path ('' stays '')."""
    if not path:
        return ""
    if path.startswith("package://"):
        package, _, relative = path[len("package://"):].partition("/")
        from ament_index_python.packages import get_package_share_directory  # only for package:// paths
        return os.path.join(get_package_share_directory(package), relative)
    path = os.path.expanduser(path)
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base_dir, path))


def _section(cls, data: dict | None, name: str, allowed: set[str] | None = None):
    """Output: cls(**data), with an error naming the YAML section on unknown keys."""
    data = dict(data or {})
    known = allowed if allowed is not None else {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"unknown key(s) {unknown} in '{name}' (known: {sorted(known)})")
    return cls(**data)


def _method(section: dict | None) -> str:
    """Output: poses.method, validated (coverage by default)."""
    section = dict(section or {})
    unknown = sorted(set(section) - {"method"})
    if unknown:
        raise ValueError(f"unknown key(s) {unknown} in 'poses' (known: ['method'])")
    method = section.get("method", "coverage")
    if method not in METHODS:
        raise ValueError(f"poses.method must be one of {METHODS} (got {method})")
    return method


def load_experiment(path: str, mode: str = "", captures_dir: str = "") -> ExperimentConfig:
    """Input: experiment YAML path, optional mode / captures folder overrides (action goal).
    Output: ExperimentConfig, validated.
    Raises: ValueError naming the bad field."""
    with open(path) as handle:
        raw = yaml.safe_load(handle) or {}
    base_dir = os.path.dirname(os.path.abspath(path))
    head = raw.get("experiment") or {}
    name = head.get("name") or os.path.splitext(os.path.basename(path))[0]
    mode = mode or head.get("mode", "cad")
    if mode not in MODES:
        raise ValueError(f"experiment.mode must be one of {MODES} (got {mode})")
    surface = dict(raw.get("surface") or {})
    unknown = sorted(set(surface) - _CAD_KEYS - _SENSOR_KEYS)
    if unknown:
        raise ValueError(f"unknown key(s) {unknown} in 'surface'")
    # The CAD settings are used in both modes (sensor mode aligns to the CAD).
    cad = CadSurfaceConfig(**{k: v for k, v in surface.items() if k in _CAD_KEYS})
    sensor = None
    if mode == "sensor":
        sensor_fields = {k: v for k, v in surface.items() if k in _SENSOR_KEYS}
        if captures_dir:
            sensor_fields["captures_dir"] = captures_dir
        sensor_fields["captures_dir"] = resolve_path(sensor_fields.get("captures_dir", ""), base_dir)
        sensor = SensorSurfaceConfig(**sensor_fields)
    machine = _section(MachineConfig, raw.get("machine"), "machine")
    machine.urdf_file = resolve_path(machine.urdf_file, base_dir)
    machine.config_file = resolve_path(machine.config_file, base_dir)
    visibility = dict(raw.get("visibility") or {})
    unknown = sorted(set(visibility) - {"occlusion_tolerance_m"})
    if unknown:
        raise ValueError(f"unknown key(s) {unknown} in 'visibility'")
    output = raw.get("output") or {}
    return ExperimentConfig(
        name=name, mode=mode, frame=raw.get("frame", "robot_map"), description=head.get("description", ""),
        node_parameters=dict(head.get("node_parameters") or {}), machine=machine, cad=cad, sensor=sensor,
        method=_method(raw.get("poses")),
        ring=_section(RingConfig, raw.get("ring"), "ring"),
        target=_section(TargetConfig, raw.get("target"), "target"),
        camera=_section(CameraConfig, raw.get("camera"), "camera"),
        workspace=_section(WorkspaceConfig, raw.get("workspace"), "workspace"),
        candidates=_section(CandidateConfig, raw.get("candidates"), "candidates"),
        occlusion_tolerance_m=float(visibility.get("occlusion_tolerance_m", 0.02)),
        cover=_section(CoverConfig, raw.get("set_cover"), "set_cover"),
        ordering=_section(OrderConfig, raw.get("ordering"), "ordering"),
        output_dir=resolve_path(output.get("trajectory_dir", ""), base_dir), source_path=os.path.abspath(path))


def machine_pose_from_config(cfg: MachineConfig) -> np.ndarray:
    """Output: robot_map -> root link 4x4 from machine.pose (xyz, rpy_deg)."""
    return pose_matrix(cfg.pose["xyz"], np.radians(cfg.pose.get("rpy_deg", [0.0, 0.0, 0.0])))


def _hash(*parts) -> str:
    """Output: a short hash of JSON-able parts (dataclasses, arrays, strings)."""
    def plain(value):
        if dataclasses.is_dataclass(value):
            return plain(dataclasses.asdict(value))
        if isinstance(value, dict):
            return {str(k): plain(v) for k, v in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        if isinstance(value, np.ndarray):
            return np.round(value, 6).tolist()
        return value
    text = json.dumps([plain(p) for p in parts], sort_keys=True, default=str)
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def _session_signature(captures_dir: str) -> list:
    """Output: what identifies a capture session's content (frames.jsonl size and mtime)."""
    path = os.path.join(captures_dir, "frames.jsonl")
    stat = os.stat(path) if os.path.isfile(path) else None
    return [path, stat.st_size if stat else 0, stat.st_mtime if stat else 0]


def generate(cfg: ExperimentConfig, urdf_xml: str, T_map_machine: np.ndarray, output_dir: str, cache_dir: str,
             use_cache: bool = True, progress=None, log=print) -> tuple[Trajectory, str]:
    """Input: ExperimentConfig, machine URDF XML, robot_map -> root link 4x4, output and
    cache folders, whether to reuse cached stages, progress(stage, fraction) callback, logger.
    Output: (Trajectory, path of the written YAML).
    Raises: ValueError / FileNotFoundError with the stage's problem."""
    started = time.time()
    progress = progress or (lambda stage, fraction: None)
    os.makedirs(cache_dir, exist_ok=True)

    def cached(name: str) -> str:
        return os.path.join(cache_dir, name)

    def hit(path: str) -> bool:
        return use_cache and os.path.isfile(path)

    progress("machine", 0.0)
    machine = load_machine(urdf_xml, T_map_machine, cfg.machine.root_link, cfg.machine.config_file,
                           cfg.machine.mesh_path_map)
    log(f"[machine] {len(machine.parts)} mesh parts, {len(machine.boxes)} collision boxes, "
        f"sections {machine.sections}")
    urdf_key = hashlib.sha1(urdf_xml.encode()).hexdigest()[:12]

    progress("surface", 0.1)
    cad_key = _hash("cad-v3", urdf_key, cfg.machine.config_file, T_map_machine, cfg.cad)
    cad_npz, cad_glb = cached(f"surface_cad_{cad_key}.npz"), cached(f"machine_cad_{cad_key}.glb")
    if hit(cad_npz) and os.path.isfile(cad_glb):
        surface = SurfaceModel.load(cad_npz)
        log(f"[surface] CAD from cache ({len(surface)} points)")
    else:
        surface, mesh = build_cad_surface(machine, cfg.cad, log)
        surface.save(cad_npz)
        export_glb(mesh, cad_glb, cfg.cad.web_triangles)
    surface_key, viewer_glb = cad_key, cad_glb
    notes = {}
    if cfg.mode == "sensor":
        sensor_key = _hash("sensor", cad_key, cfg.sensor, _session_signature(cfg.sensor.captures_dir))
        sensor_npz, sensor_glb = cached(f"surface_sensor_{sensor_key}.npz"), cached(f"machine_sensor_{sensor_key}.glb")
        sensor_json = cached(f"surface_sensor_{sensor_key}.json")
        if hit(sensor_npz) and os.path.isfile(sensor_glb) and os.path.isfile(sensor_json):
            surface = SurfaceModel.load(sensor_npz)
            with open(sensor_json) as handle:
                notes = json.load(handle)
            log(f"[surface] sensor from cache ({len(surface)} points)")
        else:
            result = build_sensor_surface(cfg.sensor, surface, T_map_machine, machine.boxes_in_map(), log)
            surface = result.surface
            surface.save(sensor_npz)
            export_glb(result.mesh, sensor_glb, cfg.cad.web_triangles)
            notes = {"frames_used": result.frames_used, "icp_fitness": result.icp_fitness,
                     "icp_rmse_m": result.icp_rmse_m,
                     "T_map_machine_corrected": None if result.T_map_machine is None
                     else np.asarray(result.T_map_machine).tolist()}
            with open(sensor_json, "w") as handle:
                json.dump(notes, handle)
        if notes.get("T_map_machine_corrected") is not None:
            # The workspace follows the machine where the scan found it.
            machine.T_map_machine = np.array(notes["T_map_machine_corrected"])
        surface_key, viewer_glb = sensor_key, sensor_glb

    progress("candidates", 0.3)
    surface = select_target(surface, cfg.target, log)
    # The workspace uses every box: the parts not inspected still keep the robot away.
    workspace = Workspace(machine.boxes_in_map(), cfg.workspace)
    candidates_key = _hash("candidates", surface_key, cfg.target, machine.T_map_machine, cfg.workspace,
                           cfg.method, cfg.ring if cfg.method == "ring" else cfg.candidates)
    candidates_npz = cached(f"candidates_{candidates_key}.npz")
    if cfg.method == "ring":
        # Fixed pattern: cheap, never cached.
        candidates = ring_poses(surface, workspace, machine.boxes_in_map(), cfg.ring, log)
    elif hit(candidates_npz):
        candidates = Candidates.load(candidates_npz)
        log(f"[candidates] from cache ({len(candidates)})")
    else:
        candidates = generate_candidates(surface, workspace, cfg.candidates, log)
        candidates.save(candidates_npz)

    progress("visibility", 0.45)
    visibility_key = _hash("visibility", candidates_key, cfg.camera, cfg.occlusion_tolerance_m)
    visibility_npz = cached(f"visibility_{visibility_key}.npz")
    if hit(visibility_npz):
        visibility = sparse.load_npz(visibility_npz).tocsr()
        log(f"[visibility] from cache ({visibility.nnz} pairs)")
    else:
        visibility = visibility_matrix(candidates, surface, cfg.camera, cfg.occlusion_tolerance_m, log=log)
        sparse.save_npz(visibility_npz, visibility)

    progress("set_cover", 0.8)
    costs = 1.0 + cfg.cover.tilt_cost_per_rad * candidates.tilts_rad
    if cfg.method == "ring":
        selection = select_all(visibility, log)
    else:
        selection = greedy_set_cover(visibility, costs, cfg.cover, log)
    if len(selection.indices) == 0:
        raise ValueError("no pose was selected (check set_cover.min_new_points and the camera range)")

    progress("ordering", 0.9)
    chosen = candidates.take(selection.indices)
    order, arc = order_poses(chosen.positions_m, chosen.rotations, workspace, cfg.ordering)
    ordered = selection.indices[order]
    if len(ordered) < len(selection.indices):
        # ordering.end_xy dropped poses: the coverage counts only the kept ones.
        view_count = np.asarray(visibility[ordered].astype(np.int32).sum(axis=0)).reshape(-1).astype(np.int32)
        seen = view_count > 0
        selection = dataclasses.replace(
            selection, indices=ordered, view_count=view_count,
            coverage_ratio=float(seen.sum() / max(len(seen), 1)),
            coverable_ratio=float((seen & selection.coverable).sum() / max(selection.coverable.sum(), 1)))
        log(f"[ordering] end_xy: {len(chosen) - len(ordered)} poses after the end dropped")
    log(f"[ordering] {len(ordered)} poses over a {workspace.loop_length_m:.1f} m lap ({cfg.ordering.direction})")

    progress("write", 0.95)
    rows = visibility[ordered]
    poses = []
    for k, (index, arc_m) in enumerate(zip(ordered, arc)):
        section = surface.section_names[int(candidates.section_ids[index])]
        poses.append(CameraPose(
            id=k, name=f"{section}_{k:03d}", section=section,
            position=candidates.positions_m[index].tolist(), orientation=candidates.quaternions[index].tolist(),
            look_at=candidates.look_at_m[index].tolist(), covered=int(rows[k].nnz), arc_m=float(arc_m),
            standoff_m=float(candidates.standoffs_m[index]), tilt_deg=float(np.degrees(candidates.tilts_rad[index]))))
    covered = selection.view_count > 0
    T = machine.T_map_machine
    trajectory = Trajectory(
        experiment=cfg.name, mode=cfg.mode, frame=cfg.frame, camera=cfg.camera.as_dict(),
        coverage={"ratio": selection.coverage_ratio, "coverable_ratio": selection.coverable_ratio,
                  "surface_points": len(surface), "coverable_points": int(selection.coverable.sum()),
                  "uncovered_points": int((~covered).sum()), "views_per_point": cfg.cover.views_per_point,
                  "candidates": len(candidates), "stop_reason": selection.stop_reason},
        machine={"root_link": cfg.machine.root_link, "pose_source": cfg.machine.pose_source,
                 "corrected_by_icp": cfg.mode == "sensor" and notes.get("T_map_machine_corrected") is not None,
                 "position": T[:3, 3].tolist(), "orientation": rotation_to_quaternion(T[:3, :3])[0].tolist(),
                 "mesh": f"{cfg.name}_machine.glb"},
        poses=poses, generated=time.strftime("%Y-%m-%d %H:%M:%S"),
        notes={"description": cfg.description, "experiment_file": cfg.source_path,
               "method": cfg.method,
               "target": "whole machine" if cfg.target.whole_machine else dataclasses.asdict(cfg.target),
               "lap_length_m": workspace.loop_length_m, "duration_s": round(time.time() - started, 1),
               **{k: v for k, v in notes.items() if k != "T_map_machine_corrected"}})
    coverage_arrays = {
        "points_m": surface.points_m.astype(np.float32), "normals": surface.normals.astype(np.float32),
        "section_ids": surface.section_ids, "section_names": np.array(surface.section_names),
        "part_ids": surface.part_ids, "part_names": np.array(surface.part_names),
        "view_count": selection.view_count, "coverable": selection.coverable,
        "pose_indptr": rows.indptr.astype(np.int64), "pose_indices": rows.indices.astype(np.int32),
        "lane_xy": workspace.loop_xy.astype(np.float32)}
    path = save_trajectory(trajectory, output_dir, coverage_arrays)
    shutil.copyfile(viewer_glb, os.path.join(output_dir, trajectory.machine["mesh"]))
    log(f"[write] {path} ({len(poses)} poses, coverage {selection.coverage_ratio:.1%}, "
        f"{time.time() - started:.0f} s)")
    progress("done", 1.0)
    return trajectory, path
