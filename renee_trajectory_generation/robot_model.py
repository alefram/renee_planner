"""Thin Pinocchio/Coal model adapter, independent of ROS communication."""

from functools import lru_cache
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .scan_geometry import read_urdf


def mesh_geometry(mesh):
    """Convert a triangle mesh in metres into a Coal collision geometry."""
    import coal

    vertices, triangles = coal.StdVec_Vec3s(), coal.StdVec_Triangle()
    for vertex in np.asarray(mesh.vertices):
        vertices.append(vertex)
    for triangle in np.asarray(mesh.faces):
        triangles.append(coal.Triangle(*[int(i) for i in triangle]))
    geometry = coal.BVHModelOBBRSS()
    geometry.beginModel(len(triangles), len(vertices))
    geometry.addSubModel(vertices, triangles)
    geometry.endModel()
    return geometry


class RobotModel:
    """Keep library model/data together; expose only small numerical methods.

    Parameters
    ----------
    config : dict
        Resolved URDF/SRDF paths, joint names, frames and initial positions.
    environment : sequence
        Coal CollisionObjects expressed in the map frame.
    """

    def __init__(self, config, environment):
        import coal
        import pinocchio as pin

        self.pin, self.coal = pin, coal
        xml = ET.tostring(read_urdf(config['urdf'], config['xacro_args']), encoding='unicode')
        self.model = pin.buildModelFromXML(xml)
        self.data = self.model.createData()
        self.geometry = pin.buildGeomFromUrdfString(self.model, xml, pin.GeometryType.COLLISION)
        self.geometry.addAllCollisionPairs()
        pin.removeCollisionPairs(self.model, self.geometry, str(config['srdf']))
        self.geometry_data = pin.GeometryData(self.geometry)
        self.full_seed = pin.neutral(self.model)
        # Freeze wheels/steering and other non-arm joints at their initial state.
        for name, value in config.get('other_joint_positions', {}).items():
            joint = self._joint(name)
            if joint.nq == 1:
                self.full_seed[joint.idx_q] = value
            elif joint.nq == 2 and joint.nv == 1:
                self.full_seed[joint.idx_q:joint.idx_q + 2] = [np.cos(value), np.sin(value)]
            else:
                raise ValueError(f'Unsupported non-arm joint configuration: {name}')
        self.names = config['joint_names']
        joints = [self._joint(name) for name in self.names]
        if len(joints) != 6 or any(j.nq != 1 or j.nv != 1 for j in joints):
            raise ValueError('Exactly six bounded one-DOF arm joints are required')
        self.indices = np.array([j.idx_q for j in joints])
        self.velocity_indices = np.array([j.idx_v for j in joints])
        arm_ids = {self.model.getJointId(name) for name in self.names}
        self.arm_geometry = []
        for index, geometry in enumerate(self.geometry.geometryObjects):
            joint = geometry.parentJoint
            while joint and joint not in arm_ids:
                joint = self.model.parents[joint]
            if joint in arm_ids:
                self.arm_geometry.append(index)
        self.lower = self.model.lowerPositionLimit[self.indices].copy()
        self.upper = self.model.upperPositionLimit[self.indices].copy()
        # Optional MoveIt position overrides may tighten, never widen, URDF limits.
        for i, name in enumerate(self.names):
            override = config.get('position_limits', {}).get(name, {})
            self.lower[i] = max(self.lower[i], override.get('min_position', -np.inf))
            self.upper[i] = min(self.upper[i], override.get('max_position', np.inf))
        if np.any(self.lower >= self.upper):
            raise ValueError('Invalid arm joint position limits')
        self.seed = np.asarray(config['initial_positions'], dtype=float)
        if self.seed.shape != (6,) or np.any(self.seed < self.lower) or np.any(self.seed > self.upper):
            raise ValueError('robot.initial_positions must satisfy six joint limits')
        self.camera_id = self._frame(config['camera_frame'])
        self.base_id = self._frame(config['base_frame'])
        pin.framesForwardKinematics(self.model, self.data, self.full_seed)
        self.root_from_base = self.data.oMf[self.base_id].copy()
        self.environment = list(environment)
        self.world_bounds = np.asarray([self._bounds(o) for o in self.environment])
        self.request = coal.DistanceRequest()
        self.request.enable_signed_distance = True
        if not len(self.geometry.geometryObjects):
            raise ValueError('Robot model has no collision geometry')

    def _joint(self, name):
        if not self.model.existJointName(name):
            raise ValueError(f'Unknown robot joint: {name}')
        return self.model.joints[self.model.getJointId(name)]

    def _frame(self, name):
        if not self.model.existFrame(name):
            raise ValueError(f'Unknown robot frame: {name}')
        return self.model.getFrameId(name)

    @staticmethod
    def _bounds(obj):
        bounds = obj.getAABB()
        return np.array([bounds.min_, bounds.max_])

    def _state(self, q, base):
        pin = self.pin
        full = self.full_seed.copy()
        full[self.indices] = q
        pin.framesForwardKinematics(self.model, self.data, full)
        rotation = Rotation.from_euler('z', base[2]).as_matrix()
        map_from_base = pin.SE3(rotation, np.array([base[0], base[1], 0.0]))
        return full, map_from_base * self.root_from_base.inverse()

    def camera_pose(self, q, base_pose):
        """Return camera position (m) and rotation matrix in the map frame."""
        _, map_from_root = self._state(q, base_pose)
        pose = map_from_root * self.data.oMf[self.camera_id]
        return pose.translation.copy(), pose.rotation.copy()

    def camera_jacobian(self, q, base_pose):
        """Return map-aligned [linear; angular] camera Jacobian, shape (6, 6)."""
        full, map_from_root = self._state(q, base_pose)
        jac = self.pin.computeFrameJacobian(self.model, self.data, full,
                                           self.camera_id, self.pin.LOCAL_WORLD_ALIGNED)
        jac = jac[:, self.velocity_indices].copy()
        jac[:3] = map_from_root.rotation @ jac[:3]
        jac[3:] = map_from_root.rotation @ jac[3:]
        return jac

    def clearances(self, q, base_pose):
        """Return minimum signed separation per robot geometry, in metres.

        AABB separation is used only to skip pairs already further apart than
        the current minimum. Narrow-phase Coal distance decides nearby pairs.
        This is a piecewise-smooth minimum, differentiated numerically by HQP.
        """
        return np.asarray(self._cached_clearances(tuple(q), tuple(base_pose)))

    @lru_cache(maxsize=512)
    def _cached_clearances(self, q, base):
        full, map_from_root = self._state(q, base)
        self.pin.updateGeometryPlacements(self.model, self.data, self.geometry,
                                          self.geometry_data, full)
        objects = []
        for geometry, pose in zip(self.geometry.geometryObjects, self.geometry_data.oMg):
            pose = map_from_root * pose
            objects.append(self.coal.CollisionObject(
                geometry.geometry, self.coal.Transform3s(pose.rotation, pose.translation)))
        bounds = np.asarray([self._bounds(o) for o in objects])
        distances = np.full(len(objects), 10.0)
        # Conservatively keep the arm, its camera and attached tools above the
        # floor at map z=0. Base wheels remain allowed to contact the floor.
        distances[self.arm_geometry] = bounds[self.arm_geometry, 0, 2]

        def evaluate(i, obj, bound, j=None):
            separation = np.maximum(np.maximum(bounds[i, 0] - bound[1],
                                                bound[0] - bounds[i, 1]), 0.0)
            if np.linalg.norm(separation) > distances[i] and (
                    j is None or np.linalg.norm(separation) > distances[j]):
                return
            result = self.coal.DistanceResult()
            distance = self.coal.distance(objects[i], obj, self.request, result)
            distances[i] = min(distances[i], distance)
            if j is not None:
                distances[j] = min(distances[j], distance)

        for pair in self.geometry.collisionPairs:
            evaluate(pair.first, objects[pair.second], bounds[pair.second], pair.second)
        for i in range(len(objects)):
            if not len(self.environment):
                continue
            separation = np.maximum(np.maximum(bounds[i, 0] - self.world_bounds[:, 1],
                                                self.world_bounds[:, 0] - bounds[i, 1]), 0.0)
            for j in np.argsort(np.linalg.norm(separation, axis=1)):
                if np.linalg.norm(separation[j]) > distances[i]:
                    break
                evaluate(i, self.environment[j], self.world_bounds[j])
        return tuple(distances)

    def inverse_kinematics(self, position, rotation, base_pose, seed, attempts, margin):
        """Find a collision-free camera IK seed; failure is not proof of unreachability."""
        def error(q):
            actual_position, actual_rotation = self.camera_pose(q, base_pose)
            return np.r_[actual_position - position,
                         Rotation.from_matrix(rotation.T @ actual_rotation).as_rotvec()]

        rng = np.random.default_rng(0)
        starts = [np.clip(seed, self.lower, self.upper)]
        starts += [rng.uniform(self.lower, self.upper) for _ in range(attempts - 1)]
        for start in starts:
            result = least_squares(error, start, bounds=(self.lower, self.upper),
                                   max_nfev=150, ftol=1e-8, xtol=1e-8, gtol=1e-8)
            if (np.linalg.norm(error(result.x)[:3]) < 0.001 and
                    np.linalg.norm(error(result.x)[3:]) < np.deg2rad(0.5) and
                    np.min(self.clearances(result.x, base_pose)) >= margin):
                return result.x
        return None
