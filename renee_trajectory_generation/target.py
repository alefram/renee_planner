"""Which part of the machine to inspect.

Input: SurfaceModel (surface/) and TargetConfig: sections (extraction,
transversal, vertical, wrist, cart) and/or parts (URDF link names, e.g.
vertical27_link or vertical27, shell wildcards allowed: transversal2*).
Output: the SurfaceModel restricted to those points. Its occluder mesh stays
the whole machine, and the workspace still uses every collision box, so the
rest of the machine still blocks views and keeps the robot away.

A point is kept if its section is in `sections` or its part matches `parts`
(both empty: the whole machine), and its part does not match `exclude_parts`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch

import numpy as np

from .surface.base import SurfaceModel


@dataclass
class TargetConfig:
    sections: list = field(default_factory=list)
    parts: list = field(default_factory=list)
    exclude_parts: list = field(default_factory=list)

    def __post_init__(self):
        for name in ("sections", "parts", "exclude_parts"):
            value = getattr(self, name)
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise ValueError(f"target.{name} must be a list of names")

    @property
    def whole_machine(self) -> bool:
        return not (self.sections or self.parts or self.exclude_parts)


def _matches(name: str, patterns: list) -> bool:
    """Output: whether a link name matches a pattern, with or without its _link suffix."""
    short = name[:-5] if name.endswith("_link") else name
    return any(fnmatch(name, p) or fnmatch(short, p) for p in patterns)


def select_target(surface: SurfaceModel, cfg: TargetConfig, log=print) -> SurfaceModel:
    """Input: SurfaceModel, TargetConfig. Output: the targeted subset (same occluder).
    Raises: ValueError if nothing matches, listing the available sections and parts."""
    if cfg.whole_machine:
        return surface
    sections = np.array([s in cfg.sections for s in surface.section_names])
    parts = np.array([_matches(p, cfg.parts) for p in surface.part_names])
    excluded = np.array([_matches(p, cfg.exclude_parts) for p in surface.part_names])
    keep = sections[surface.section_ids] | parts[surface.part_ids] if (cfg.sections or cfg.parts) \
        else np.ones(len(surface), dtype=bool)
    keep &= ~excluded[surface.part_ids]
    if not keep.any():
        raise ValueError(f"target selects no surface point. Sections: {surface.section_names}; "
                         f"parts: {surface.part_names}")
    chosen = sorted({surface.part_names[i] for i in np.unique(surface.part_ids[keep])})
    log(f"[target] {int(keep.sum())}/{len(surface)} points on {len(chosen)} parts: "
        f"{', '.join(chosen[:12])}{' ...' if len(chosen) > 12 else ''}")
    return surface.subset(keep)
