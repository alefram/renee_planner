"""Map footprint checks, deterministic station search and local scan planning."""

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import yaml

from .constraints import build_constraints, look_at, make_tasks
from .hqp_optimizer import optimize_scan
from .scan_geometry import resolve_path, target_visible


def load_map(path):
    """Read a trinary ROS map YAML; return a bottom-up blocked-cell array."""
    with open(path, encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    if config.get('mode', 'trinary') != 'trinary':
        raise ValueError('Only trinary map files are supported')
    pixels = np.asarray(Image.open(resolve_path(config['image'], path.parent)).convert('L'))
    probability = pixels / 255.0 if config.get('negate', 0) else 1 - pixels / 255.0
    resolution = float(config['resolution'])
    origin = np.asarray(config['origin'], dtype=float)
    if (resolution <= 0 or not np.isfinite(resolution) or origin.shape != (3,) or
            not np.all(np.isfinite(origin)) or
            not 0 <= config['free_thresh'] < config['occupied_thresh'] <= 1):
        raise ValueError('Invalid map resolution, origin or occupancy thresholds')
    # Unknown is intentionally blocked too. Image rows run downward; map y up.
    return {'blocked': np.flipud(probability >= config['free_thresh']),
            'resolution': resolution, 'origin': origin}


def _rotation(yaw):
    return np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])


def footprint_free(grid, base, footprint, margin):
    """Check the complete convex footprint, including cells touched by edges."""
    polygon = np.asarray(footprint) @ _rotation(base[2]).T + base[:2]
    polygon = (polygon - grid['origin'][:2]) @ _rotation(grid['origin'][2])
    resolution, blocked = grid['resolution'], grid['blocked']
    low = np.floor((polygon.min(axis=0) - margin) / resolution).astype(int)
    high = np.floor((polygon.max(axis=0) + margin) / resolution).astype(int)
    if np.any(low < 0) or high[0] >= blocked.shape[1] or high[1] >= blocked.shape[0]:
        return False
    edges = np.roll(polygon, -1, axis=0) - polygon
    axes = np.vstack((np.eye(2), np.column_stack((-edges[:, 1], edges[:, 0]))))
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    projected = polygon @ axes.T
    for row in range(low[1], high[1] + 1):
        for col in range(low[0], high[0] + 1):
            if not blocked[row, col]:
                continue
            square = (np.array([[col, row], [col + 1, row],
                                [col + 1, row + 1], [col, row + 1]]) * resolution)
            cell = square @ axes.T
            separated = ((projected.max(axis=0) + margin < cell.min(axis=0)) |
                         (cell.max(axis=0) + margin < projected.min(axis=0)))
            if not np.any(separated):
                return False
    return True


def map_collision_objects(grid, height):
    """Extrude blocked row runs into conservative static Coal boxes (map z=0)."""
    import coal

    objects = []
    rotation = Rotation.from_euler('z', grid['origin'][2]).as_matrix()
    resolution = grid['resolution']
    for row, cells in enumerate(grid['blocked']):
        changes = np.diff(np.r_[False, cells, False].astype(int))
        for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
            position = rotation @ np.array([(start + end) * resolution / 2,
                                           (row + 0.5) * resolution, height / 2])
            position[:2] += grid['origin'][:2]
            geometry = coal.Box((end - start) * resolution, resolution, height)
            objects.append(coal.CollisionObject(geometry, coal.Transform3s(rotation, position)))
    # RobotModel checks arm-floor separation separately, allowing wheel contact.
    return objects


def station_candidates(structure, grid, config):
    """Sample perimeter rays, outward offsets and lateral omnidirectional shifts."""
    options = config['stations']
    low, high = structure['bounds']
    center = (low + high) / 2
    half = (high - low) / 2
    transform = structure['transform']
    step = np.deg2rad(options['candidate_spacing_deg'])
    for angle in np.arange(0.0, 2 * np.pi - 1e-9, step):
        direction = np.array([np.cos(angle), np.sin(angle), 0.0])
        intersection = np.divide(half[:2], np.abs(direction[:2]),
                                 out=np.full(2, np.inf), where=np.abs(direction[:2]) > 1e-9)
        axis = int(np.argmin(intersection))
        sign = 1 if direction[axis] >= 0 else -1
        face = {(0, 1): 0, (1, 1): 1, (0, -1): 2, (1, -1): 3}[axis, sign]
        normal = np.zeros(3)
        normal[axis] = sign
        tangent = np.array([-normal[1], normal[0], 0.0])
        for offset in options['surface_offsets_m']:
            for shift in options['lateral_offsets_m']:
                local = center + direction * intersection[axis] + normal * offset + tangent * shift
                world = transform[:3, :3] @ local + transform[:3, 3]
                world_normal = transform[:3, :3] @ normal
                yaw = np.arctan2(-world_normal[1], -world_normal[0])
                base = np.array([world[0], world[1], yaw])
                if footprint_free(grid, base, config['map']['footprint'],
                                  config['map']['footprint_margin_m']):
                    actual_angle = np.arctan2(local[1] - center[1], local[0] - center[0])
                    yield {'base': base, 'angle': actual_angle % (2 * np.pi),
                           'nominal_angle': angle, 'face': face}


def _angular_difference(first, second):
    return abs((first - second + np.pi) % (2 * np.pi) - np.pi)


def angular_gaps(stations):
    """Return circular station gaps in degrees, including last-to-first."""
    if not stations:
        return np.array([360.0])
    angles = np.sort([s['angle'] for s in stations])
    return np.rad2deg(np.diff(np.r_[angles, angles[0] + 2 * np.pi]))


def select_stations(candidates, required_ids, options):
    """Greedily cover views, then fill angular gaps from successful candidates."""
    remaining, selected, pool = set(required_ids), [], list(candidates)
    nominal = np.deg2rad(options['nominal_spacing_deg'])
    while remaining:
        useful = [c for c in pool if set(c['view_ids']) & remaining]
        if not useful:
            break
        # Prefer coverage, nominal 30-degree slots, then shorter base moves.
        previous = selected[-1]['base'][:2] if selected else np.zeros(2)
        best = min(useful, key=lambda c: (
            -len(set(c['view_ids']) & remaining),
            abs(c['nominal_angle'] / nominal - round(c['nominal_angle'] / nominal)),
            np.linalg.norm(c['base'][:2] - previous)))
        selected.append(best)
        pool = [c for c in pool if c is not best]
        remaining -= set(best['view_ids'])
    while pool and np.max(angular_gaps(selected)) > options['max_spacing_deg']:
        best = min(pool, key=lambda c: np.max(angular_gaps(selected + [c])))
        # Several insertions can be necessary before the largest tied gap falls.
        if any(_angular_difference(best['angle'], s['angle']) < 1e-6 for s in selected):
            pool = [c for c in pool if c is not best]
            continue
        selected.append(best)
        pool = [c for c in pool if c is not best]
    return sorted(selected, key=lambda c: c['angle']), sorted(remaining)


def optimize_station(candidate, views, model, structure, config):
    """Find feasible seeds and optimize/refine open scan tracts at a fixed base."""
    base, tracts, covered, rejected = candidate['base'], [], [], []
    scan, solver = config['scan'], config['solver']
    for band in sorted({v['band'] for v in views}):
        band_views = [v for v in views if v['band'] == band]
        initial, accepted = [], []
        seed = model.seed.copy()
        for view in band_views:
            position = view['target'] + scan['depth_target_m'] * view['normal']
            q = model.inverse_kinematics(position, look_at(position, view['target']), base,
                                         seed, solver['ik_attempts'],
                                         config['collision']['clearance_m'])
            if q is None:
                rejected.append({'view_id': view['id'], 'reason': 'no_collision_free_ik_found'})
                continue
            initial.append(q)
            accepted.append(view)
            seed = q
        if not accepted:
            continue
        joints = np.asarray(initial)
        targets = np.asarray([v['target'] for v in accepted])
        ids = [v['id'] for v in accepted]
        result = None
        for refinement in range(solver['max_refinements'] + 1):
            shape = joints.shape

            def constraints(flat):
                return build_constraints(flat.reshape(shape), model, base, targets, config)

            def visible(flat):
                return all(target_visible(structure['visual'], model.camera_pose(q, base)[0], t)
                           for q, t in zip(flat.reshape(shape), targets))

            result = optimize_scan(joints, make_tasks(joints, targets, model, base, config),
                                   constraints, solver, visible)
            if result['status'] != 'converged':
                break
            joints = result['joints']
            expanded_q, expanded_targets, expanded_ids, invalid = [], [], [], False
            checked = 0
            for index in range(len(joints)):
                expanded_q.append(joints[index])
                expanded_targets.append(targets[index])
                expanded_ids.append(ids[index])
                if index == len(joints) - 1:
                    continue
                steps = max(2, int(np.ceil(np.max(np.abs(joints[index + 1] - joints[index])) /
                                          solver['validation_step_rad'])))
                for fraction in np.arange(1, steps) / steps:
                    q = joints[index] * (1 - fraction) + joints[index + 1] * fraction
                    # Between required views, scan whichever real surface is
                    # first hit by the current optical axis. Do not interpolate
                    # target coordinates through voids in a nonconvex machine.
                    position, rotation = model.camera_pose(q, base)
                    hits, _, _ = structure['visual'].ray.intersects_location(
                        [position], [rotation[:, 2]], multiple_hits=True)
                    if len(hits):
                        target = hits[np.argmin(np.linalg.norm(hits - position, axis=1))]
                    else:
                        invalid = True
                        target = targets[index]  # Real target for IK repair.
                    v, lo, hi = build_constraints(q[None], model, base, target[None], config)
                    checked += 1
                    if (np.any(v < lo - solver['constraint_tolerance']) or
                            np.any(v > hi + solver['constraint_tolerance']) or
                            not target_visible(structure['visual'], model.camera_pose(q, base)[0], target)):
                        invalid = True
                    # Keep validation streaming even when a large IK branch
                    # jump would exceed the bounded optimization sample count.
                    if len(expanded_q) <= solver['max_samples_per_tract']:
                        expanded_q.append(q)
                        expanded_targets.append(target)
                        expanded_ids.append(None)
            if not invalid:
                result['validation_step_rad'] = solver['validation_step_rad']
                result['intermediate_samples_checked'] = checked
                result['targets'] = targets
                result['view_ids'] = ids
                tracts.append(result)
                covered.extend(v['id'] for v in accepted)
                break
            if len(expanded_q) > solver['max_samples_per_tract']:
                result['status'] = 'refinement_sample_limit'
                break
            # Refinement needs a feasible seed. Re-solve IK for newly inserted
            # samples; never ask the feasible-start HQP to accept an invalid one.
            if refinement < solver['max_refinements']:
                repaired = []
                for q, target in zip(expanded_q, expanded_targets):
                    position = model.camera_pose(q, base)[0]
                    direction = position - target
                    position = target + direction / np.linalg.norm(direction) * scan['depth_target_m']
                    fixed = model.inverse_kinematics(position, look_at(position, target), base, q,
                                                     solver['ik_attempts'],
                                                     config['collision']['clearance_m'])
                    if fixed is None:
                        break
                    repaired.append(fixed)
                if len(repaired) != len(expanded_q):
                    result['status'] = 'refinement_ik_failed'
                    break
                joints, targets, ids = np.asarray(repaired), np.asarray(expanded_targets), expanded_ids
            else:
                result['status'] = 'segment_validation_failed'
        if result and result['status'] != 'converged':
            rejected.append({'view_ids': [v['id'] for v in accepted], 'reason': result['status']})
    return {**candidate, 'tracts': tracts, 'view_ids': covered, 'rejected': rejected}


def plan_stations(structure, views, grid, model, config):
    """Search a bounded candidate set; never equate search failure with infeasibility."""
    successes, diagnostics = [], []
    inverse = np.linalg.inv(structure['transform'])
    center = np.mean(structure['bounds'], axis=0)
    count, exhausted = 0, True
    for candidate in station_candidates(structure, grid, config):
        if count >= config['stations']['max_candidates']:
            exhausted = False
            break
        count += 1
        nearby = []
        for view in views:
            local = inverse[:3, :3] @ view['target'] + inverse[:3, 3] - center
            angle = np.arctan2(local[1], local[0])
            if (view['face'] == candidate['face'] and
                    _angular_difference(angle, candidate['angle']) <=
                    np.deg2rad(config['stations']['view_window_deg'])):
                nearby.append(view)
        if not nearby:
            continue
        result = optimize_station(candidate, nearby, model, structure, config)
        diagnostics.extend(result['rejected'])
        if result['view_ids']:
            successes.append(result)
    selected, pending = select_stations(successes, [v['id'] for v in views], config['stations'])
    gaps = angular_gaps(selected)
    return {'stations': selected, 'pending_views': pending, 'angular_gaps_deg': gaps,
            'candidates_evaluated': count, 'search_exhausted': exhausted,
            'diagnostics': diagnostics,
            'status': 'complete' if selected and not pending and
            np.max(gaps) <= config['stations']['max_spacing_deg'] else 'partial',
            'base_transfers': 'not_planned'}
