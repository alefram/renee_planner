"""The surface to inspect, the same for the CAD and sensor modes.

Input: (none) -- defines the format.
Output: `SurfaceModel`: sample points with normals, section and part, plus the
triangle mesh used as occluder by `raycast()`; saved to / loaded from .npz.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SurfaceModel:
    """points_m (N, 3) and normals (N, 3, unit, pointing out of the surface) in
    the map frame; section_ids (N,) index into section_names, part_ids (N,) into
    part_names (the machine's links; default: one part per section). vertices_m
    (V, 3) and triangles (T, 3) are the occluder mesh (map frame), always the
    whole machine even when the points are a subset (target.py)."""
    points_m: np.ndarray
    normals: np.ndarray
    section_ids: np.ndarray
    section_names: list[str]
    vertices_m: np.ndarray
    triangles: np.ndarray
    part_ids: np.ndarray | None = None
    part_names: list[str] | None = None
    _scene: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self.points_m = np.asarray(self.points_m, dtype=np.float64).reshape(-1, 3)
        self.normals = np.asarray(self.normals, dtype=np.float64).reshape(-1, 3)
        self.section_ids = np.asarray(self.section_ids, dtype=np.int32).reshape(-1)
        self.vertices_m = np.asarray(self.vertices_m, dtype=np.float64).reshape(-1, 3)
        self.triangles = np.asarray(self.triangles, dtype=np.int32).reshape(-1, 3)
        if self.part_ids is None:
            self.part_ids, self.part_names = self.section_ids.copy(), list(self.section_names)
        self.part_ids = np.asarray(self.part_ids, dtype=np.int32).reshape(-1)
        self.part_names = [str(n) for n in self.part_names]
        if not (len(self.points_m) == len(self.normals) == len(self.section_ids) == len(self.part_ids)):
            raise ValueError("SurfaceModel: points, normals, section_ids and part_ids differ in length")

    def __len__(self) -> int:
        return len(self.points_m)

    def raycast(self, origins_m: np.ndarray, directions: np.ndarray) -> np.ndarray:
        """Input: ray origins (R, 3) and unit directions (R, 3), map frame.
        Output: distance to the first hit on the occluder mesh (R,), inf if none."""
        import open3d as o3d
        if self._scene is None:
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(o3d.core.Tensor(self.vertices_m.astype(np.float32)),
                                o3d.core.Tensor(self.triangles.astype(np.uint32)))
            self._scene = scene
        rays = np.hstack([origins_m, directions]).astype(np.float32)
        return self._scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().astype(np.float64)

    def save(self, path: str) -> None:
        """Output: the model as a compressed .npz at path."""
        np.savez_compressed(path, points_m=self.points_m, normals=self.normals, section_ids=self.section_ids,
                            section_names=np.array(self.section_names), vertices_m=self.vertices_m,
                            triangles=self.triangles, part_ids=self.part_ids, part_names=np.array(self.part_names))

    @classmethod
    def load(cls, path: str) -> "SurfaceModel":
        """Input: a .npz written by save(). Output: SurfaceModel."""
        data = np.load(path, allow_pickle=False)
        parts = "part_ids" in data.files
        return cls(data["points_m"], data["normals"], data["section_ids"], [str(s) for s in data["section_names"]],
                   data["vertices_m"], data["triangles"], data["part_ids"] if parts else None,
                   [str(s) for s in data["part_names"]] if parts else None)

    def subset(self, keep: np.ndarray) -> "SurfaceModel":
        """Input: boolean mask (N,). Output: a model with only those sample points (same occluder)."""
        model = SurfaceModel(self.points_m[keep], self.normals[keep], self.section_ids[keep], self.section_names,
                             self.vertices_m, self.triangles, self.part_ids[keep], self.part_names)
        model._scene = self._scene
        return model
