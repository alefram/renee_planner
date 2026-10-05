"""Pick the fewest poses that cover the surface (view planning as set cover).

Input: the visibility matrix (visibility.py), per-candidate costs, CoverConfig
(target coverage, views per point, stopping rules).
Output: `Selection`: the chosen candidate indices (in pick order), the views
per point and the coverage reached.

Greedy weighted set cover (ln n approximation of the optimum): each step
takes the candidate with the most still-needed points per unit cost, until the
target share of the coverable points (seen by at least one candidate) has
`views_per_point` views, or no candidate adds `min_new_points`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class CoverConfig:
    target_coverage: float = 0.95   # of the coverable points
    views_per_point: int = 1        # > 1: overlap between neighbouring poses
    min_new_points: int = 20        # stop when the best pose adds fewer
    max_poses: int = 300
    tilt_cost_per_rad: float = 0.3  # prefer straight views at equal gain

    def __post_init__(self):
        if not 0.0 < self.target_coverage <= 1.0:
            raise ValueError(f"set_cover.target_coverage must be in (0, 1] (got {self.target_coverage})")
        if self.views_per_point < 1 or self.max_poses < 1 or self.min_new_points < 1:
            raise ValueError("set_cover.views_per_point, max_poses and min_new_points must be >= 1")


@dataclass
class Selection:
    indices: np.ndarray         # chosen candidates, in pick order
    view_count: np.ndarray      # (N,) views of each point by the chosen poses
    coverable: np.ndarray       # (N,) bool: seen by at least one candidate
    coverage_ratio: float       # points with a view / all points
    coverable_ratio: float      # points with a view / coverable points
    stop_reason: str


def select_all(visibility: sparse.csr_matrix, log=print) -> Selection:
    """Input: (C, N) bool visibility. Output: Selection with every candidate (fixed patterns, ring.py)."""
    matrix = visibility.tocsr()
    view_count = np.asarray(matrix.astype(np.int32).sum(axis=0)).reshape(-1).astype(np.int32)
    # No candidate pool: every point counts, so the viewer marks what the ring misses as not covered.
    coverable = np.ones(matrix.shape[1], dtype=bool)
    ratio = float((view_count > 0).sum() / max(matrix.shape[1], 1))
    log(f"[set_cover] fixed pattern: all {matrix.shape[0]} poses, coverage {ratio:.1%} of all points")
    return Selection(np.arange(matrix.shape[0], dtype=np.int64), view_count, coverable, ratio, ratio,
                     "fixed_pattern")


def greedy_set_cover(visibility: sparse.csr_matrix, costs: np.ndarray, cfg: CoverConfig, log=print) -> Selection:
    """Input: (C, N) bool visibility, (C,) costs (> 0), config. Output: Selection."""
    matrix = visibility.astype(np.float32).tocsr()
    points = matrix.shape[1]
    coverable = np.asarray(matrix.getnnz(axis=0) > 0).reshape(-1)
    target_points = cfg.target_coverage * coverable.sum()
    view_count = np.zeros(points, dtype=np.int32)
    available = np.ones(matrix.shape[0], dtype=bool)
    chosen = []
    stop = "max_poses"
    while len(chosen) < cfg.max_poses:
        done = int(np.sum(view_count >= cfg.views_per_point))
        if done >= target_points:
            stop = "target_coverage"
            break
        need = ((view_count < cfg.views_per_point) & coverable).astype(np.float32)
        gain = matrix @ need
        gain[~available] = 0.0
        best = int(np.argmax(gain / costs))
        if gain[best] < cfg.min_new_points:
            stop = "min_new_points"
            break
        chosen.append(best)
        available[best] = False
        view_count[matrix.indices[matrix.indptr[best]:matrix.indptr[best + 1]]] += 1
    covered = view_count > 0
    selection = Selection(np.array(chosen, dtype=np.int64), view_count, coverable,
                          float(covered.sum() / max(points, 1)), float(covered.sum() / max(coverable.sum(), 1)),
                          stop)
    log(f"[set_cover] {len(chosen)} poses, "
        f"coverage {selection.coverage_ratio:.1%} of all points, "
        f"{selection.coverable_ratio:.1%} of the coverable ones (stop: {stop})")
    return selection
