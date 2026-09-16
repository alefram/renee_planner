"""YAML input, JSON prints and an optional one-shot ROS adapter."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from .scan_geometry import load_structure, resolve_path, surface_views


def load_config(path):
    """Return validated YAML with asset paths resolved relative to its directory.

    Parameters
    ----------
    path : str or Path
        Path to the documented hqp_scan.yaml configuration.

    Returns
    -------
    dict
        Validated configuration in SI units, with absolute asset paths.

    Raises
    ------
    ValueError
        Missing fields, invalid limits, unsupported policy or missing files.
    """
    path = Path(path).resolve()
    with path.open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    try:
        for section in ('structure', 'robot', 'map', 'scan', 'stations', 'collision', 'tasks', 'solver'):
            if not isinstance(config[section], dict):
                raise ValueError(f'{section} must be a mapping')
        for section, fields in [('structure', ['model']), ('robot', ['urdf', 'srdf']), ('map', ['yaml'])]:
            for field in fields:
                config[section][field] = resolve_path(config[section][field], path.parent)
        for section in ('structure', 'robot'):
            args = config[section]['xacro_args']
            if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
                raise ValueError(f'{section}.xacro_args must be a string list')
        positive = {
            'scan': ['depth_min_m', 'depth_target_m', 'depth_max_m', 'pointing_tolerance_deg', 'samples_per_face'],
            'stations': ['nominal_spacing_deg', 'max_spacing_deg', 'candidate_spacing_deg', 'view_window_deg', 'max_candidates'],
            'collision': ['clearance_m', 'map_obstacle_height_m'],
            'solver': ['constraint_tolerance', 'qp_max_iterations', 'max_iterations', 'trust_region_rad',
                       'finite_difference_step', 'step_tolerance_rad', 'line_search_steps', 'ik_attempts',
                       'validation_step_rad', 'max_samples_per_tract'],
        }
        for section, fields in positive.items():
            for field in fields:
                value = config[section][field]
                if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                    raise ValueError(f'{section}.{field} must be positive and finite')
        integers = {'scan': ['samples_per_face'], 'stations': ['max_candidates'],
                    'solver': ['qp_max_iterations', 'max_iterations', 'line_search_steps', 'ik_attempts',
                               'max_refinements', 'max_samples_per_tract']}
        for section, fields in integers.items():
            for field in fields:
                value = config[section][field]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f'{section}.{field} must be a nonnegative integer')
        scan, stations = config['scan'], config['stations']
        if not scan['depth_min_m'] < scan['depth_target_m'] < scan['depth_max_m']:
            raise ValueError('scan depths must satisfy min < target < max')
        if scan['pointing_tolerance_deg'] >= 90:
            raise ValueError('scan.pointing_tolerance_deg must be below 90')
        if not (stations['candidate_spacing_deg'] <= stations['nominal_spacing_deg'] <=
                stations['max_spacing_deg'] <= 40):
            raise ValueError('Station spacing must satisfy candidate <= nominal <= max <= 40')
        if stations['orientation'] != 'face_normal' or stations['fixed_during_scan'] is not True:
            raise ValueError('Stations require face_normal and fixed_during_scan: true')
        if config['collision']['unknown_map_cells_are_obstacles'] is not True:
            raise ValueError('Unknown map cells must remain obstacles')
        pose = np.asarray(config['structure']['pose'], dtype=float)
        if (pose.shape != (6,) or not np.all(np.isfinite(pose)) or
                np.any(np.abs(pose[3:5]) > 1e-9)):
            raise ValueError('structure.pose needs x y z roll pitch yaw with zero roll/pitch')
        for section, field, positive in [('scan', 'heights_from_bottom_m', True),
                                         ('stations', 'surface_offsets_m', True),
                                         ('stations', 'lateral_offsets_m', False)]:
            values = np.asarray(config[section][field], dtype=float)
            if (values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)) or
                    (positive and np.any(values <= 0))):
                raise ValueError(f'{section}.{field} needs a finite nonempty list')
        polygon = np.asarray(config['map']['footprint'], dtype=float)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3 or not np.all(np.isfinite(polygon)):
            raise ValueError('map.footprint requires at least three finite [x,y] vertices')
        edges = np.roll(polygon, -1, axis=0) - polygon
        cross = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
        if not (np.all(cross > 0) or np.all(cross < 0)):
            raise ValueError('map.footprint must be strictly convex and ordered')
        margin = config['map']['footprint_margin_m']
        if not np.isfinite(margin) or margin < 0:
            raise ValueError('map.footprint_margin_m must be finite and nonnegative')
        if sorted(config['tasks']['priority_order']) != ['observation', 'reference', 'smoothness']:
            raise ValueError('tasks.priority_order must contain each supported task exactly once')
        if config['frame_id'] != 'robot_map':
            raise ValueError('frame_id must match robot_map')
        names = config['robot']['joint_names']
        seed = np.asarray(config['robot']['initial_positions'], dtype=float)
        if len(names) != 6 or len(set(names)) != 6 or seed.shape != (6,) or not np.all(np.isfinite(seed)):
            raise ValueError('robot needs six distinct joint_names and finite initial_positions')
        for field in ('base_frame', 'camera_frame'):
            if not isinstance(config['robot'][field], str) or not config['robot'][field]:
                raise ValueError(f'robot.{field} must be a nonempty frame name')
    except (KeyError, TypeError) as error:
        raise ValueError(f'Missing or invalid configuration field: {error}') from error
    return config


def run(config, check_inputs=False):
    """Build a static scene and return a plan dictionary; never execute motion."""
    import coal
    from .robot_model import RobotModel, mesh_geometry
    from .station_planner import load_map, map_collision_objects, plan_stations

    structure = load_structure(config['structure']['model'], config['structure']['pose'],
                               config['structure']['xacro_args'])
    grid = load_map(config['map']['yaml'])
    world = [coal.CollisionObject(mesh_geometry(mesh)) for mesh in structure['collision']]
    world.extend(map_collision_objects(grid, config['collision']['map_obstacle_height_m']))
    model = RobotModel(config['robot'], world)
    views, missed = surface_views(structure, config['scan'])
    if check_inputs:
        return {'status': 'inputs_valid', 'surface_views': len(views), 'missed_rays': missed,
                'arm_joints': model.names, 'robot_geometry_count': len(model.geometry.geometryObjects),
                'world_geometry_count': len(world), 'planning_performed': False}
    result = plan_stations(structure, views, grid, model, config)
    result['missed_surface_rays'] = missed
    if missed:
        result['status'] = 'partial'
    result.update(frame_id=config['frame_id'], camera_frame=config['robot']['camera_frame'],
                  joint_names=model.names, execution='not_requested',
                  collision_model='URDF + SRDF exclusions; map cells extruded',
                  validation='discrete joint interpolation, not continuous certification')
    for station in result['stations']:
        for tract in station['tracts']:
            poses, depths, distances, pointing = [], [], [], []
            previous = None
            for q, target in zip(tract['joints'], tract['targets']):
                position, rotation = model.camera_pose(q, station['base'])
                quaternion = Rotation.from_matrix(rotation).as_quat()
                if previous is not None and quaternion @ previous < 0:
                    quaternion *= -1
                previous = quaternion
                poses.append({'position': position, 'orientation_xyzw': quaternion})
                offset = target - position
                depths.append(np.linalg.norm(offset))
                pointing.append(np.rad2deg(np.arccos(np.clip(rotation[:, 2] @ offset /
                                                             np.linalg.norm(offset), -1, 1))))
                distances.append(np.min(model.clearances(q, station['base'])))
            tract.update(camera_poses=poses, depth_min_m=min(depths), depth_max_m=max(depths),
                         clearance_min_m=min(distances), pointing_max_deg=max(pointing))
    return result


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Cannot serialize {type(value)}')


def main(argv=None):
    """Process one CLI/ROS input, print JSON and return 0 complete/2 partial/1 error."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='Path to hqp_scan.yaml')
    parser.add_argument('--ros', action='store_true', help='Read ROS parameter config_file')
    parser.add_argument('--check-inputs', action='store_true', help='Load models without searching')
    args, ros_args = parser.parse_known_args(argv)
    node, ros = None, None
    try:
        if args.ros:
            import rclpy as ros
            from rclpy.node import Node
            ros.init(args=ros_args)
            node = Node('hqp_scan_optimizer')
            path = node.declare_parameter('config_file', args.config or '').value
        else:
            if ros_args:
                parser.error(f'Unexpected arguments: {ros_args}')
            path = args.config
        if not path:
            raise ValueError('Provide --config or ROS parameter config_file')
        result = run(load_config(path), args.check_inputs)
        print(json.dumps(result, default=_json_default, indent=2, allow_nan=False))
        return 0 if result['status'] in ('complete', 'inputs_valid') else 2
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(json.dumps({'status': 'error', 'message': str(error)}), file=sys.stderr)
        return 1
    finally:
        if node is not None:
            node.destroy_node()
        if ros is not None and ros.ok():
            ros.shutdown()


if __name__ == '__main__':
    sys.exit(main())
