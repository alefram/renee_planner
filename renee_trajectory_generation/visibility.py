"""Which surface points each candidate pose sees.

Input: Candidates (candidates.py), SurfaceModel (surface/), CameraConfig (camera.py).
Output: sparse boolean matrix (candidates x points): True where the point is in
the pose's frustum and range, faces it within max_incidence_deg, and the ray
from the camera reaches it before any other surface (occlusion).
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from .camera import CameraConfig, sees
from .candidates import Candidates
from .surface.base import SurfaceModel


def visibility_matrix(candidates: Candidates, surface: SurfaceModel, camera: CameraConfig,
                      occlusion_tolerance_m: float = 0.02, ray_batch: int = 2_000_000, log=print) -> sparse.csr_matrix:
    """Input: candidates, surface, camera model, how far before the point a hit still
    counts as the point itself, rays per raycasting call.
    Output: (C, N) csr_matrix of bool."""
    tree = cKDTree(surface.points_m)
    rows, cols = [], []
    pending_rows, pending_cols = [], []
    pending = 0

    def flush():
        # Occlusion test of the pending (candidate, point) pairs in one raycasting call.
        nonlocal pending
        if not pending:
            return
        r, c = np.concatenate(pending_rows), np.concatenate(pending_cols)
        origins = candidates.positions_m[r]
        offset = surface.points_m[c] - origins
        distance = np.linalg.norm(offset, axis=1)
        t_hit = surface.raycast(origins, offset / distance[:, None])
        visible = t_hit >= distance - occlusion_tolerance_m
        rows.append(r[visible])
        cols.append(c[visible])
        pending_rows.clear()
        pending_cols.clear()
        pending = 0

    for i, (position, rotation) in enumerate(zip(candidates.positions_m, candidates.rotations)):
        near = np.asarray(tree.query_ball_point(position, camera.range_m[1]), dtype=np.int64)
        if len(near) == 0:
            continue
        seen = near[sees(position, rotation, surface.points_m[near], surface.normals[near], camera)]
        if len(seen):
            pending_rows.append(np.full(len(seen), i, dtype=np.int64))
            pending_cols.append(seen)
            pending += len(seen)
        if pending >= ray_batch:
            flush()
        if (i + 1) % 1000 == 0:
            log(f"[visibility] {i + 1}/{len(candidates)} candidates")
    flush()
    r = np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)
    c = np.concatenate(cols) if cols else np.empty(0, dtype=np.int64)
    matrix = sparse.csr_matrix((np.ones(len(r), dtype=bool), (r, c)), shape=(len(candidates), len(surface)))
    coverable = int((matrix.getnnz(axis=0) > 0).sum())
    log(f"[visibility] {matrix.nnz} visible pairs; {coverable}/{len(surface)} points seen by some candidate")
    return matrix
