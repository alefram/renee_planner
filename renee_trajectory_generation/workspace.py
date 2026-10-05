"""Where the robot can put the camera (geometric, no IK).

Input: the machine's collision boxes in the map (machine.Machine.boxes_in_map)
and WorkspaceConfig (robot half width, lane clearance, arm reach, camera heights).
Output: `Workspace`: the base lane (closed loop around the machine, counter-
clockwise, plan view), `inside(positions)` (the camera can be there), and
`arc_length(positions)` (where along the lane a pose is, for ordering).

The lane is the convex hull of the machine's footprint grown by the robot's
half width + clearance: the base drives along it (Nav2, boat mode) and the arm
reaches `arm_reach_in_m` towards the machine and `arm_reach_out_m` away from it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull

from .machine import Box

_CHUNK = 2048


@dataclass
class WorkspaceConfig:
    robot_half_width_m: float = 0.45
    lane_clearance_m: float = 0.30      # base footprint to machine footprint
    arm_reach_in_m: float = 0.90        # camera from the lane towards the machine (plan view)
    arm_reach_out_m: float = 0.30       # camera from the lane away from the machine
    camera_height_m: tuple = (0.40, 1.90)
    min_clearance_m: float = 0.15       # camera to any machine box (3D)
    loop_resolution_m: float = 0.05

    def __post_init__(self):
        self.camera_height_m = tuple(float(v) for v in self.camera_height_m)
        if not self.camera_height_m[0] < self.camera_height_m[1]:
            raise ValueError(f"workspace.camera_height_m must be [min, max] (got {self.camera_height_m})")
        for name in ("robot_half_width_m", "lane_clearance_m", "arm_reach_in_m", "arm_reach_out_m",
                     "min_clearance_m", "loop_resolution_m"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"workspace.{name} must be >= 0")


class Workspace:
    def __init__(self, boxes: list[Box], cfg: WorkspaceConfig):
        if not boxes:
            raise ValueError("the machine has no collision boxes: the workspace needs them")
        self.cfg = cfg
        self.centers = np.array([b.center for b in boxes])
        self.half_sizes = np.array([b.size / 2.0 for b in boxes])
        self.rotations = np.array([b.rotation for b in boxes])
        self.lane_offset_m = cfg.robot_half_width_m + cfg.lane_clearance_m
        self.loop_xy = self._lane_loop()
        segments = np.diff(np.vstack([self.loop_xy, self.loop_xy[:1]]), axis=0)
        self._segment_lengths = np.linalg.norm(segments, axis=1)
        self._cumulative = np.concatenate([[0.0], np.cumsum(self._segment_lengths)[:-1]])
        self.loop_length_m = float(self._segment_lengths.sum())

    def _lane_loop(self) -> np.ndarray:
        """Output: (M, 2) counter-clockwise closed loop (last point != first), resampled."""
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        corners = (np.einsum("bij,bkj->bki", self.rotations, signs[None] * self.half_sizes[:, None, :])
                   + self.centers[:, None, :]).reshape(-1, 3)[:, :2]
        angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
        disc = self.lane_offset_m * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        grown = (corners[:, None, :] + disc[None]).reshape(-1, 2)
        hull = grown[ConvexHull(grown).vertices]   # counter-clockwise
        closed = np.vstack([hull, hull[:1]])
        lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(lengths)])
        samples = np.arange(0.0, s[-1], self.cfg.loop_resolution_m)
        return np.stack([np.interp(samples, s, closed[:, 0]), np.interp(samples, s, closed[:, 1])], axis=1)

    def box_distance(self, points_m: np.ndarray) -> np.ndarray:
        """Input: (N, 3) points. Output: (N,) distance to the closest machine box (0 inside one)."""
        points_m = np.atleast_2d(points_m)
        distance = np.empty(len(points_m))
        for i in range(0, len(points_m), _CHUNK):
            part = points_m[i:i + _CHUNK]
            local = np.einsum("bji,nbj->nbi", self.rotations, part[:, None, :] - self.centers[None])
            outside = np.maximum(np.abs(local) - self.half_sizes[None], 0.0)
            distance[i:i + _CHUNK] = np.linalg.norm(outside, axis=2).min(axis=1)
        return distance

    def lane_offset(self, points_m: np.ndarray) -> np.ndarray:
        """Input: (N, 2|3) points. Output: (N,) plan-view distance to the lane, negative
        inside it (towards the machine)."""
        xy = np.atleast_2d(points_m)[:, :2]
        _, distance = self._project(xy)
        closed = np.vstack([self.loop_xy, self.loop_xy[:1]])
        edges = np.diff(closed, axis=0)
        inside = np.empty(len(xy), dtype=bool)
        for i in range(0, len(xy), _CHUNK):
            part = xy[i:i + _CHUNK]
            cross = edges[None, :, 0] * (part[:, None, 1] - closed[None, :-1, 1]) - \
                edges[None, :, 1] * (part[:, None, 0] - closed[None, :-1, 0])
            inside[i:i + _CHUNK] = np.all(cross >= 0.0, axis=1)   # convex, counter-clockwise
        return np.where(inside, -distance, distance)

    def inside(self, positions_m: np.ndarray) -> np.ndarray:
        """Input: (N, 3) camera positions. Output: (N,) bool: height in range, clear of
        the machine boxes and within the arm's reach from the lane."""
        positions_m = np.atleast_2d(positions_m)
        z = positions_m[:, 2]
        offset = self.lane_offset(positions_m)
        return ((z >= self.cfg.camera_height_m[0]) & (z <= self.cfg.camera_height_m[1])
                & (offset >= -self.cfg.arm_reach_in_m) & (offset <= self.cfg.arm_reach_out_m)
                & (self.box_distance(positions_m) >= self.cfg.min_clearance_m))

    def _project(self, xy: np.ndarray):
        """Output: (arc length (N,), distance (N,)) of the closest lane point."""
        start = self.loop_xy
        edges = np.diff(np.vstack([self.loop_xy, self.loop_xy[:1]]), axis=0)
        lengths_sq = np.maximum(np.sum(edges ** 2, axis=1), 1e-12)
        arc, distance = np.empty(len(xy)), np.empty(len(xy))
        for i in range(0, len(xy), _CHUNK):    # (points x segments) arrays stay small
            part = xy[i:i + _CHUNK]
            t = np.clip(np.einsum("nsk,sk->ns", part[:, None, :] - start[None], edges) / lengths_sq, 0.0, 1.0)
            gap = np.linalg.norm(part[:, None, :] - (start[None] + t[..., None] * edges[None]), axis=2)
            segment = np.argmin(gap, axis=1)
            rows = np.arange(len(part))
            arc[i:i + _CHUNK] = self._cumulative[segment] + t[rows, segment] * self._segment_lengths[segment]
            distance[i:i + _CHUNK] = gap[rows, segment]
        return arc, distance

    def arc_length(self, points_m: np.ndarray) -> np.ndarray:
        """Input: (N, 2|3) points. Output: (N,) arc length (m) of their projection on the lane."""
        return self._project(np.atleast_2d(points_m)[:, :2])[0]

    def lane_point(self, arc_m: float) -> np.ndarray:
        """Input: arc length (m). Output: the (x, y) lane point there."""
        closed = np.vstack([self.loop_xy, self.loop_xy[:1]])
        s = np.concatenate([self._cumulative, [self.loop_length_m]])
        arc_m = arc_m % self.loop_length_m
        return np.array([np.interp(arc_m, s, closed[:, 0]), np.interp(arc_m, s, closed[:, 1])])
