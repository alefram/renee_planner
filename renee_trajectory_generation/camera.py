"""Camera model: the optical frame orientation and what a pose can see.

Input: CameraConfig (frame, effective field of view, range, max incidence),
camera positions and the points they look at.
Output: optical-frame rotations / quaternions (+Z towards the look-at point,
+X right, +Y down, image upright) and `sees()`: which surface points are in
the frustum, in range and not seen too obliquely (occlusion is visibility.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CameraConfig:
    frame: str = "robot_arm_rgbd_camera_left_camera_optical_frame"
    # Effective field of view used for coverage: narrower than the ZED2i's
    # 110 x 70 deg, the image borders are distorted and low-resolution.
    fov_deg: tuple = (60.0, 40.0)
    range_m: tuple = (0.3, 1.5)
    max_incidence_deg: float = 60.0   # angle between the surface normal and the ray to the camera

    def __post_init__(self):
        self.fov_deg = tuple(float(v) for v in self.fov_deg)
        self.range_m = tuple(float(v) for v in self.range_m)
        if len(self.fov_deg) != 2 or not all(0.0 < v < 180.0 for v in self.fov_deg):
            raise ValueError(f"camera.fov_deg must be [horizontal, vertical] in (0, 180) (got {self.fov_deg})")
        if len(self.range_m) != 2 or not 0.0 < self.range_m[0] < self.range_m[1]:
            raise ValueError(f"camera.range_m must be [min, max] with 0 < min < max (got {self.range_m})")
        if not 0.0 < self.max_incidence_deg <= 90.0:
            raise ValueError(f"camera.max_incidence_deg must be in (0, 90] (got {self.max_incidence_deg})")

    def as_dict(self) -> dict:
        return {"frame": self.frame, "fov_deg": list(self.fov_deg), "view_distance_m": list(self.range_m),
                "max_incidence_deg": self.max_incidence_deg}


def look_at_rotations(positions_m: np.ndarray, look_at_m: np.ndarray, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Input: camera positions (C, 3), look-at points (C, 3), world up.
    Output: optical-frame rotations (C, 3, 3), columns = the frame's X, Y, Z axes in the map.
    The image stays upright (Y points down along -up); looking straight up or
    down, the image top points along +X of the map."""
    positions_m = np.atleast_2d(positions_m)
    z = np.atleast_2d(look_at_m) - positions_m
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    up = np.broadcast_to(np.asarray(up, dtype=float), z.shape).copy()
    vertical = np.abs(np.sum(z * up, axis=1)) > 0.98
    up[vertical] = [1.0, 0.0, 0.0]
    y = -(up - np.sum(up * z, axis=1, keepdims=True) * z)
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=2)


def rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Input: rotations (C, 3, 3). Output: quaternions (C, 4) as [qx, qy, qz, qw], qw >= 0."""
    R = np.asarray(R, dtype=float).reshape(-1, 3, 3)
    q = np.empty((len(R), 4))
    trace = np.trace(R, axis1=1, axis2=2)
    for i, (m, t) in enumerate(zip(R, trace)):
        if t > 0.0:
            s = 2.0 * np.sqrt(t + 1.0)
            q[i] = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
        else:
            k = int(np.argmax(np.diag(m)))
            a, b = (k + 1) % 3, (k + 2) % 3
            s = 2.0 * np.sqrt(1.0 + m[k, k] - m[a, a] - m[b, b])
            v = np.empty(4)
            v[k] = 0.25 * s
            v[a] = (m[a, k] + m[k, a]) / s
            v[b] = (m[b, k] + m[k, b]) / s
            v[3] = (m[b, a] - m[a, b]) / s
            q[i] = v
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 3] < 0.0] *= -1.0
    return q


def sees(position_m: np.ndarray, R_map_camera: np.ndarray, points_m: np.ndarray, normals: np.ndarray,
         cfg: CameraConfig) -> np.ndarray:
    """Input: one camera pose (position (3,), rotation (3, 3)), surface points and normals (N, 3).
    Output: (N,) bool: in the frustum, in range and facing the camera within
    max_incidence_deg. Occlusion is not checked here."""
    offset = points_m - position_m
    local = offset @ R_map_camera            # point in the optical frame
    depth = local[:, 2]
    tan_h = np.tan(np.radians(cfg.fov_deg[0]) / 2.0)
    tan_v = np.tan(np.radians(cfg.fov_deg[1]) / 2.0)
    distance = np.linalg.norm(offset, axis=1)
    in_view = ((depth > 1e-6) & (np.abs(local[:, 0]) <= tan_h * depth) & (np.abs(local[:, 1]) <= tan_v * depth)
               & (distance >= cfg.range_m[0]) & (distance <= cfg.range_m[1]))
    # cos(incidence) = normal . (camera - point) / distance
    cos_incidence = -np.sum(normals * offset, axis=1) / np.maximum(distance, 1e-9)
    return in_view & (cos_incidence >= np.cos(np.radians(cfg.max_incidence_deg)))
