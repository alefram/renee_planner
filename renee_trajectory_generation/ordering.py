"""Visiting order of the chosen poses: one forward lap around the machine.

Input: the chosen poses (positions, optical-frame rotations), the Workspace
lane (workspace.py), OrderConfig (driving direction, start, reorder window).
Output: the order (indices) and each pose's arc length along the lap.

The poses are projected on the lane and sorted by arc length in the driving
direction, so the base only drives forward (Nav2 boat mode). Then 2-opt
reverses runs of poses that lie within `window_m` of lane: the base barely
moves there, and the arm travel (camera distance + rotation) gets shorter.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .workspace import Workspace


@dataclass
class OrderConfig:
    direction: str = "ccw"          # driving direction around the machine (ccw | cw, seen from above)
    start_xy: list | None = None    # the lap starts at this point's projection; None: at the widest gap
    window_m: float = 0.6
    angle_weight_m_per_rad: float = 0.3

    def __post_init__(self):
        if self.direction not in ("ccw", "cw"):
            raise ValueError(f"ordering.direction must be ccw or cw (got {self.direction})")
        if self.start_xy is not None and len(self.start_xy) != 2:
            raise ValueError("ordering.start_xy must be [x, y] or null")
        if self.window_m < 0.0:
            raise ValueError("ordering.window_m must be >= 0")


def _rotation_angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1.0) / 2.0, -1.0, 1.0)))


def order_poses(positions_m: np.ndarray, rotations: np.ndarray, workspace: Workspace, cfg: OrderConfig):
    """Input: (P, 3) positions, (P, 3, 3) rotations, Workspace, config.
    Output: (order (P,) indices into the inputs, arc_m (P,) lap arc length of each ordered pose)."""
    count = len(positions_m)
    if count == 0:
        return np.empty(0, dtype=np.int64), np.empty(0)
    length = workspace.loop_length_m
    arc = workspace.arc_length(positions_m)
    if cfg.direction == "cw":
        arc = length - arc
    if cfg.start_xy is not None:
        start = workspace.arc_length(np.array([cfg.start_xy], dtype=float))[0]
        if cfg.direction == "cw":
            start = length - start
    else:
        # Start just after the widest gap between consecutive poses.
        sorted_arc = np.sort(arc)
        gaps = np.diff(np.concatenate([sorted_arc, [sorted_arc[0] + length]]))
        start = sorted_arc[(int(np.argmax(gaps)) + 1) % count] - 1e-6
    lap = (arc - start) % length
    order = list(np.argsort(lap, kind="stable"))

    def cost(a: int, b: int) -> float:
        return float(np.linalg.norm(positions_m[a] - positions_m[b])
                     + cfg.angle_weight_m_per_rad * _rotation_angle(rotations[a], rotations[b]))

    # 2-opt on the open path, only inside the arc window.
    improved = True
    while improved:
        improved = False
        for i in range(1, count - 1):
            for j in range(i + 1, count):
                # The run's arc span only grows with j: the base stays within the window.
                if np.ptp(lap[order[i:j + 1]]) > cfg.window_m:
                    break
                before = cost(order[i - 1], order[i]) + (cost(order[j], order[j + 1]) if j + 1 < count else 0.0)
                after = cost(order[i - 1], order[j]) + (cost(order[i], order[j + 1]) if j + 1 < count else 0.0)
                if after < before - 1e-9:
                    order[i:j + 1] = order[i:j + 1][::-1]
                    improved = True
    order = np.array(order, dtype=np.int64)
    return order, lap[order]
