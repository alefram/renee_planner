"""Fixed ring of camera poses around the machine (poses.method: ring).

Input: the Workspace lane (workspace.py), the machine's collision boxes (for
its height), the surface (only to name the section each pose looks at), and
RingConfig (spacing along the lap, camera height, pitch, inset from the lane).
Output: `Candidates` (candidates.py) with one pose per `spacing_m` of lane, in
lap order: the camera `inset_m` from the lane towards the machine, at
`height_m`, looking towards the machine (perpendicular to the lane) and
`pitch_deg` down from horizontal, image upright. No coverage optimisation:
every pose is kept (unless the robot cannot put the camera there).

`extra_rows` add more poses at the same lane points, each with its own height
and pitch (negative: looking up), e.g. for a tall section the main row misses;
a row with `sections` only keeps the poses within `near_m` (plan view) of them.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np
from scipy.spatial import cKDTree

from .camera import look_at_rotations, rotation_to_quaternion
from .candidates import Candidates
from .machine import Box
from .surface.base import SurfaceModel
from .workspace import Workspace


@dataclass
class RingRow:
    """One extra row of ring.extra_rows (inset_m / look_distance_m: null = the main row's)."""
    height_m: float
    pitch_deg: float                 # optical axis below horizontal; negative: looking up
    sections: list = field(default_factory=list)  # only near these sections; empty: the whole lap
    near_m: float = 1.5              # camera to the sections' points (plan view)
    inset_m: float | None = None
    look_distance_m: float | None = None

    def __post_init__(self):
        if not -90.0 < self.pitch_deg < 90.0:
            raise ValueError(f"ring.extra_rows pitch_deg must be in (-90, 90) (got {self.pitch_deg})")
        if self.near_m <= 0.0 or (self.look_distance_m is not None and self.look_distance_m <= 0.0):
            raise ValueError("ring.extra_rows near_m and look_distance_m must be > 0")
        self.sections = [str(s) for s in self.sections]


@dataclass
class RingConfig:
    spacing_m: float = 0.4           # distance between poses along the lane
    pitch_deg: float = 45.0          # optical axis below horizontal; negative: looking up
    height_m: float | None = None    # camera height (map z); null: machine top + above_top_m
    above_top_m: float = 0.3
    inset_m: float = 0.4             # camera from the lane towards the machine (plan view)
    look_distance_m: float = 1.0     # look_at point along the optical axis
    extra_rows: list = field(default_factory=list)  # RingRow fields, one dict per row

    def __post_init__(self):
        if self.spacing_m <= 0.0 or self.look_distance_m <= 0.0:
            raise ValueError("ring.spacing_m and ring.look_distance_m must be > 0")
        if not -90.0 < self.pitch_deg < 90.0:
            raise ValueError(f"ring.pitch_deg must be in (-90, 90) (got {self.pitch_deg})")
        known = {f.name for f in fields(RingRow)}
        rows = []
        for row in self.extra_rows:
            if isinstance(row, dict):
                unknown = sorted(set(row) - known)
                if unknown:
                    raise ValueError(f"unknown key(s) {unknown} in 'ring.extra_rows' (known: {sorted(known)})")
                row = RingRow(**row)
            rows.append(row)
        self.extra_rows = rows


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
    # (height, pitch, inset, look distance, plan-view points the row stays near or None, near_m)
    rows = [(height, np.radians(cfg.pitch_deg), cfg.inset_m, cfg.look_distance_m, None, 0.0)]
    for row in cfg.extra_rows:
        near = None
        if row.sections:
            unknown = sorted(set(row.sections) - set(surface.section_names))
            if unknown:
                raise ValueError(f"ring.extra_rows sections {unknown} not in the surface "
                                 f"(sections: {surface.section_names}; check target)")
            ids = [surface.section_names.index(s) for s in row.sections]
            near = cKDTree(surface.points_m[np.isin(surface.section_ids, ids), :2])
        rows.append((float(row.height_m), np.radians(row.pitch_deg),
                     cfg.inset_m if row.inset_m is None else float(row.inset_m),
                     cfg.look_distance_m if row.look_distance_m is None else float(row.look_distance_m),
                     near, row.near_m))
    arcs = np.arange(0.0, workspace.loop_length_m, cfg.spacing_m)
    positions, look_at, pitches, look_distances, near_ok = [], [], [], [], []
    for arc in arcs:
        point = workspace.lane_point(arc)
        tangent = workspace.lane_point(arc + 0.15) - workspace.lane_point(arc - 0.15)
        tangent /= max(np.linalg.norm(tangent), 1e-9)
        inward = np.array([-tangent[1], tangent[0]])          # left of the counter-clockwise lane
        # The rows of one lane point stay together: same arc, the lap order keeps them in row order.
        for row_height, pitch, inset, look_distance, near, near_m in rows:
            xy = point + inset * inward
            position = np.array([xy[0], xy[1], row_height])
            axis = np.array([np.cos(pitch) * inward[0], np.cos(pitch) * inward[1], -np.sin(pitch)])
            positions.append(position)
            look_at.append(position + look_distance * axis)
            pitches.append(pitch)
            look_distances.append(look_distance)
            near_ok.append(near is None or near.query(xy)[0] <= near_m)
    positions, look_at = np.array(positions), np.array(look_at)
    pitches, look_distances = np.array(pitches), np.array(look_distances)
    near_ok = np.array(near_ok)
    positions, look_at, pitches, look_distances = (positions[near_ok], look_at[near_ok], pitches[near_ok],
                                                   look_distances[near_ok])
    keep = workspace.inside(positions)
    if not keep.any():
        raise ValueError("no ring pose is inside the workspace (check ring.height_m / inset_m and workspace.*)")
    if not keep.all():
        log(f"[ring] {int((~keep).sum())} of {len(keep)} poses outside the workspace dropped")
    positions, look_at, pitches, look_distances = positions[keep], look_at[keep], pitches[keep], look_distances[keep]
    rotations = look_at_rotations(positions, look_at)
    # Section: where the optical axis meets the surface (closest sample to the look-at point).
    _, nearest = cKDTree(surface.points_m).query(look_at)
    count = len(positions)
    log(f"[ring] {count} poses every {cfg.spacing_m:.2f} m over a {workspace.loop_length_m:.1f} m lap, "
        f"pitch {cfg.pitch_deg:.0f} deg" + (f", {len(cfg.extra_rows)} extra row(s)" if cfg.extra_rows else ""))
    return Candidates(positions, look_at, rotations, rotation_to_quaternion(rotations),
                      np.full(count, -1), surface.section_ids[nearest], look_distances, pitches)
