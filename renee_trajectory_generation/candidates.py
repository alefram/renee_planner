"""Candidate camera poses: many possible views of every patch of the surface.

Input: SurfaceModel (surface/), Workspace (workspace.py), CandidateConfig
(patch size, standoff distances, tilts).
Output: `Candidates`: positions, look-at points, optical-frame rotations and
quaternions (camera.py), plus the patch, section, standoff and tilt of each.

The surface points are grouped into patches by position (a voxel grid of
`cluster_size_m`) and normal (26 directions), so the two sides of a plate or
the faces of a corner are separate patches. For each patch the camera looks at
its centroid from `standoff` along its mean normal, tilted by each of
`tilts_deg` around it. Poses the robot cannot reach (workspace) or whose view
of the centroid is blocked are dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .camera import look_at_rotations, rotation_to_quaternion
from .surface.base import SurfaceModel
from .workspace import Workspace

_NORMAL_BINS = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
                         if (i, j, k) != (0, 0, 0)], dtype=float)
_NORMAL_BINS /= np.linalg.norm(_NORMAL_BINS, axis=1, keepdims=True)


@dataclass
class CandidateConfig:
    cluster_size_m: float = 0.25
    min_cluster_points: int = 5
    standoffs_m: list = field(default_factory=lambda: [0.6, 0.8, 1.0])
    tilts_deg: list = field(default_factory=lambda: [0.0, 25.0])
    tilt_azimuths: int = 4            # directions per non-zero tilt, evenly around the normal
    max_candidates: int = 8000        # random subset above this (seeded)
    line_of_sight_tolerance_m: float = 0.03
    seed: int = 0

    def __post_init__(self):
        self.standoffs_m = [float(v) for v in self.standoffs_m]
        self.tilts_deg = [float(v) for v in self.tilts_deg]
        if self.cluster_size_m <= 0.0:
            raise ValueError("candidates.cluster_size_m must be > 0")
        if not self.standoffs_m or min(self.standoffs_m) <= 0.0:
            raise ValueError("candidates.standoffs_m must be a non-empty list of distances > 0")
        if not self.tilts_deg or not all(0.0 <= t < 90.0 for t in self.tilts_deg):
            raise ValueError("candidates.tilts_deg must be a non-empty list in [0, 90)")
        if self.tilt_azimuths < 1 or self.max_candidates < 1:
            raise ValueError("candidates.tilt_azimuths and candidates.max_candidates must be >= 1")


@dataclass
class Candidates:
    positions_m: np.ndarray     # (C, 3)
    look_at_m: np.ndarray       # (C, 3)
    rotations: np.ndarray       # (C, 3, 3) optical frame in the map
    quaternions: np.ndarray     # (C, 4) [qx, qy, qz, qw]
    cluster_ids: np.ndarray     # (C,) patch looked at
    section_ids: np.ndarray     # (C,)
    standoffs_m: np.ndarray     # (C,)
    tilts_rad: np.ndarray       # (C,)

    def __len__(self) -> int:
        return len(self.positions_m)

    def take(self, index: np.ndarray) -> "Candidates":
        """Input: indices or a boolean mask. Output: those candidates."""
        return Candidates(*(getattr(self, name)[index] for name in self.__dataclass_fields__))

    def save(self, path: str) -> None:
        np.savez_compressed(path, **{name: getattr(self, name) for name in self.__dataclass_fields__})

    @classmethod
    def load(cls, path: str) -> "Candidates":
        data = np.load(path)
        return cls(*(data[name] for name in cls.__dataclass_fields__))


def cluster_surface(surface: SurfaceModel, cfg: CandidateConfig):
    """Input: SurfaceModel, config.
    Output: (centroids (K, 3), mean normals (K, 3), majority section (K,), point labels (N,);
    label -1 for points of patches smaller than min_cluster_points)."""
    voxels = np.floor(surface.points_m / cfg.cluster_size_m).astype(np.int64)
    bins = np.argmax(surface.normals @ _NORMAL_BINS.T, axis=1)
    keys = np.column_stack([voxels, bins])
    _, labels, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    labels = labels.reshape(-1)
    count = len(counts)
    centroids = np.stack([np.bincount(labels, surface.points_m[:, k], count) for k in range(3)], axis=1)
    centroids /= counts[:, None]
    normals = np.stack([np.bincount(labels, surface.normals[:, k], count) for k in range(3)], axis=1)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
    sections = len(surface.section_names)
    votes = np.bincount(labels * sections + surface.section_ids, minlength=count * sections).reshape(count, sections)
    section_ids = np.argmax(votes, axis=1)
    # The centroid of a curved patch can float off the surface: look at the
    # patch point closest to it instead.
    distance = np.linalg.norm(surface.points_m - centroids[labels], axis=1)
    order = np.lexsort((distance, labels))
    first = order[np.searchsorted(labels[order], np.arange(count))]
    centroids = surface.points_m[first]
    big = counts >= cfg.min_cluster_points
    remap = np.full(count, -1)
    remap[big] = np.arange(int(big.sum()))
    return centroids[big], normals[big], section_ids[big], remap[labels]


def _view_directions(normals: np.ndarray, cfg: CandidateConfig):
    """Output: (directions (K, V, 3) from the patch towards the camera, tilts (V,) rad)."""
    helper = np.where(np.abs(normals[:, 2:3]) < 0.95, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
    u = np.cross(helper, normals)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(normals, u)
    directions, tilts = [], []
    for tilt in np.radians(cfg.tilts_deg):
        azimuths = [0.0] if tilt == 0.0 else np.linspace(0.0, 2 * np.pi, cfg.tilt_azimuths, endpoint=False)
        for azimuth in azimuths:
            directions.append(np.cos(tilt) * normals
                              + np.sin(tilt) * (np.cos(azimuth) * u + np.sin(azimuth) * v))
            tilts.append(tilt)
    return np.stack(directions, axis=1), np.array(tilts)


def generate_candidates(surface: SurfaceModel, workspace: Workspace, cfg: CandidateConfig, log=print) -> Candidates:
    """Input: SurfaceModel, Workspace, CandidateConfig.
    Output: Candidates (reachable, with a clear view of their patch).
    Raises: ValueError if none is left."""
    centroids, normals, sections, _ = cluster_surface(surface, cfg)
    directions, tilts = _view_directions(normals, cfg)
    count, views = directions.shape[:2]
    standoffs = np.array(cfg.standoffs_m)
    # Every (patch, direction, standoff) combination, flattened.
    look_at = np.repeat(np.repeat(centroids[:, None, :], views, axis=1)[:, :, None, :], len(standoffs), axis=2)
    positions = look_at + directions[:, :, None, :] * standoffs[None, None, :, None]
    shape = (count, views, len(standoffs))
    cluster_ids = np.broadcast_to(np.arange(count)[:, None, None], shape).reshape(-1)
    tilt_rad = np.broadcast_to(tilts[None, :, None], shape).reshape(-1)
    standoff_m = np.broadcast_to(standoffs[None, None, :], shape).reshape(-1)
    positions, look_at = positions.reshape(-1, 3), look_at.reshape(-1, 3)
    total = len(positions)

    keep = workspace.inside(positions)
    reachable = int(keep.sum())
    # Line of sight from the camera to the patch.
    index = np.flatnonzero(keep)
    rays = look_at[index] - positions[index]
    distance = np.linalg.norm(rays, axis=1)
    t_hit = surface.raycast(positions[index], rays / distance[:, None])
    keep[index[t_hit < distance - cfg.line_of_sight_tolerance_m]] = False
    index = np.flatnonzero(keep)
    log(f"[candidates] {len(centroids)} patches, {total} poses: {reachable} reachable, {len(index)} with a clear view")
    if len(index) == 0:
        raise ValueError("no candidate pose is reachable with a clear view (check workspace.* and candidates.*)")
    if len(index) > cfg.max_candidates:
        index = np.sort(np.random.default_rng(cfg.seed).choice(index, cfg.max_candidates, replace=False))
        log(f"[candidates] kept a random {cfg.max_candidates} (candidates.max_candidates)")
    rotations = look_at_rotations(positions[index], look_at[index])
    return Candidates(positions[index], look_at[index], rotations, rotation_to_quaternion(rotations),
                      cluster_ids[index], sections[cluster_ids[index]], standoff_m[index], tilt_rad[index])
