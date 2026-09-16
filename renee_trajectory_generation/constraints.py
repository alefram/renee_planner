"""Hard scan constraints and the single extension point for new restrictions.

Add new nonlinear rows to ``build_constraints``. The optimizer differentiates
them and verifies their original values after every step. Keep row count and
ordering stable; never silently disable a constraint when data is unavailable.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def look_at(position, target):
    """Return optical rotation: +Z forward, +X right, +Y down (ROS optical)."""
    forward = np.asarray(target) - np.asarray(position)
    length = np.linalg.norm(forward)
    if length < 1e-10:
        raise ValueError('Camera position coincides with observation target')
    forward = forward / length
    up = np.array([0., 0., 1.])
    if abs(forward @ up) > 0.99:
        up = np.array([0., 1., 0.])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.column_stack((right, down, forward))


def build_constraints(joints, model, base_pose, targets, config):
    """Return nonlinear values and bounds for one stationary-base scan.

    Parameters
    ----------
    joints : ndarray, shape (N, 6)
        Arm positions in radians, in model joint-name order.
    model : object
        Implements camera_pose, clearances, lower and upper.
    base_pose : ndarray, shape (3,)
        Base-footprint x, y and yaw in the map frame (m, m, rad).
    targets : ndarray, shape (N, 3)
        Actual observed surface points in the map frame, metres.
    config : dict
        Scan depth/pointing and collision clearance configuration.

    Returns
    -------
    tuple of ndarray
        Values, lower bounds, upper bounds. Units follow each physical row.
    """
    joints, targets = np.asarray(joints, dtype=float), np.asarray(targets, dtype=float)
    if (joints.ndim != 2 or joints.shape[1] != 6 or not len(joints) or
            targets.shape != (len(joints), 3) or not np.all(np.isfinite(joints)) or
            not np.all(np.isfinite(targets))):
        raise ValueError('Expected finite joints (N, 6) and targets (N, 3), N > 0')
    values, lower, upper = [], [], []
    scan = config['scan']
    for q, target in zip(joints, targets):
        position, rotation = model.camera_pose(q, base_pose)
        offset = target - position
        depth = np.linalg.norm(offset)
        direction = offset / max(depth, 1e-12)
        # Chord error avoids arccos' singular derivative at perfect pointing.
        pointing = np.linalg.norm(rotation[:, 2] - direction)
        values.extend([depth, pointing])
        lower.extend([scan['depth_min_m'], 0.0])
        upper.extend([scan['depth_max_m'],
                      2 * np.sin(np.deg2rad(scan['pointing_tolerance_deg']) / 2)])
        values.extend(q)
        lower.extend(model.lower)
        upper.extend(model.upper)
        distances = model.clearances(q, base_pose)
        values.extend(distances)
        lower.extend(np.full(len(distances), config['collision']['clearance_m']))
        upper.extend(np.full(len(distances), np.inf))
    return tuple(np.asarray(a, dtype=float) for a in (values, lower, upper))


def make_tasks(initial, targets, model, base_pose, config):
    """Build ordered residual functions for observation, smoothness, reference."""
    shape = initial.shape

    def observation(flat):
        residual = []
        for q, target in zip(flat.reshape(shape), targets):
            position, rotation = model.camera_pose(q, base_pose)
            desired = look_at(position, target)
            residual.append(np.linalg.norm(position - target) -
                            config['scan']['depth_target_m'])
            residual.extend(Rotation.from_matrix(desired.T @ rotation).as_rotvec())
        return np.asarray(residual)

    def smoothness(flat):
        joints = flat.reshape(shape)
        points = np.asarray([model.camera_pose(q, base_pose)[0] for q in joints])
        # These are open local tracts; do not close a station's arm trajectory
        # around the entire machine. There is no timing/velocity claim here.
        return np.r_[np.diff(points, n=2, axis=0).ravel(),
                     np.diff(joints, axis=0).ravel()]

    def reference(flat):
        return flat - initial.ravel()

    functions = {'observation': observation, 'smoothness': smoothness,
                 'reference': reference}
    return [functions[name] for name in config['tasks']['priority_order']]
