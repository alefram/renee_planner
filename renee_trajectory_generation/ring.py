"""Fixed ring of camera poses around the machine (poses.method: ring).

Input: the Workspace lane (workspace.py), the machine's collision boxes (for
its height), the surface (only to name the section each pose looks at), and
RingConfig (spacing along the lap, camera height, pitch, inset from the lane).
Output: `Candidates` (candidates.py) with one pose per `spacing_m` of lane, in
lap order: the camera `inset_m` from the lane towards the machine, at
`height_m`, looking towards the machine (perpendicular to the lane) and
`pitch_deg` down from horizontal, image upright. No coverage optimisation:
every pose is kept (unless the robot cannot put the camera there).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .camera import look_at_rotations, rotation_to_quaternion
from .candidates import Candidates
from .machine import Box
from .surface.base import SurfaceModel
from .workspace import Workspace


@dataclass
class RingConfig:
    spacing_m: float = 0.4           # distance between poses along the lane
    pitch_deg: float = 45.0          # optical axis below horizontal
    height_m: float | None = None    # camera height (map z); null: machine top + above_top_m
    above_top_m: float = 0.3
    inset_m: float = 0.4             # camera from the lane towards the machine (plan view)
    look_distance_m: float = 1.0     # look_at point along the optical axis

    def __post_init__(self):
        if self.spacing_m <= 0.0 or self.look_distance_m <= 0.0:
            raise ValueError("ring.spacing_m and ring.look_distance_m must be > 0")
        if not 0.0 <= self.pitch_deg < 90.0:
            raise ValueError(f"ring.pitch_deg must be in [0, 90) (got {self.pitch_deg})")


def machine_top_m(boxes: list[Box]) -> float:
    """Output: the highest z of the boxes (map frame)."""
    tops = [box.center[2] + 0.5 * np.abs(box.rotation).dot(box.size)[2] for box in boxes]
    return float(max(tops))


def ring_poses(surface: SurfaceModel, workspace: Workspace, boxes: list[Box], cfg: RingConfig,
               log=print) -> Candidates:
    """Input: surface, Workspace, machine boxes in the map, RingConfig.
    Output: Candidates in counter-clockwise lap order (ordering.py applies the direction).
    Raises: ValueError if no pose is inside the workspace."""
    if cfg.height_m is None:
        top = machine_top_m(boxes)
        height = min(top + cfg.above_top_m, workspace.cfg.camera_height_m[1])
        log(f"[ring] machine top {top:.2f} m -> camera height {height:.2f} m")
    else:
        height = float(cfg.height_m)
    arcs = np.arange(0.0, workspace.loop_length_m, cfg.spacing_m)
    pitch = np.radians(cfg.pitch_deg)
    positions, look_at = [], []
    for arc in arcs:
        point = workspace.lane_point(arc)
        tangent = workspace.lane_point(arc + 0.15) - workspace.lane_point(arc - 0.15)
        tangent /= max(np.linalg.norm(tangent), 1e-9)
        inward = np.array([-tangent[1], tangent[0]])          # left of the counter-clockwise lane
        xy = point + cfg.inset_m * inward
        position = np.array([xy[0], xy[1], height])
        axis = np.array([np.cos(pitch) * inward[0], np.cos(pitch) * inward[1], -np.sin(pitch)])
        positions.append(position)
        look_at.append(position + cfg.look_distance_m * axis)
    positions, look_at = np.array(positions), np.array(look_at)
    keep = workspace.inside(positions)
    if not keep.any():
        raise ValueError("no ring pose is inside the workspace (check ring.height_m / inset_m and workspace.*)")
    if not keep.all():
        log(f"[ring] {int((~keep).sum())} of {len(keep)} poses outside the workspace dropped")
    positions, look_at = positions[keep], look_at[keep]
    rotations = look_at_rotations(positions, look_at)
    # Section: where the optical axis meets the surface (closest sample to the look-at point).
    _, nearest = cKDTree(surface.points_m).query(look_at)
    count = len(positions)
    log(f"[ring] {count} poses every {cfg.spacing_m:.2f} m over a {workspace.loop_length_m:.1f} m lap, "
        f"pitch {cfg.pitch_deg:.0f} deg")
    return Candidates(positions, look_at, rotations, rotation_to_quaternion(rotations),
                      np.full(count, -1), surface.section_ids[nearest],
                      np.full(count, cfg.look_distance_m), np.full(count, pitch))
