#!/usr/bin/env python3
"""Arm motion through MoveIt: IK, collision checks and planned joint motions.

Wraps the move_group services and action as non-blocking calls (the node
polls the returned futures from its control timer, so it never blocks the
executor):

- ``apply_planning_scene``: add the YAML collision objects (e.g. the machine).
- ``compute_ik``: find a collision-free arm configuration that places the
  camera (or another frame, e.g. the nozzle) at a pose, for the given
  roll(s) about its axis.
- ``check_state_validity``: check an arm configuration against the planning
  scene (self-collision with the base, and the collision objects).
- ``move_action`` (MoveGroup): plan and execute a collision-free arm motion
  to a joint configuration (the travel posture before navigating, and the
  motion to each target's IK solution).

The base is fixed at its measured pose through the SRDF's floating
``virtual_joint`` (parent ``robot_map``), like
``renee_action_servers/src/camera_placement_action_server.cpp`` does.
"""

import numpy as np
from geometry_msgs.msg import Pose, PoseStamped, Transform
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    CollisionObject, Constraints, JointConstraint, MoveItErrorCodes, PlanningScene,
    PlanningSceneComponents, RobotState)
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene, GetPositionIK, GetStateValidity
from rclpy.action import ActionClient
from rclpy.duration import Duration
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive

from renee_trajectory_generation.hqp.tasks import camera_orientation


class ArmMotion:
    """Non-blocking move_group clients for IK, state validity, the scene and joint motions."""

    def __init__(self, node, arm_joint_names, *, namespace: str = "/robot", group: str = "arm",
                 ik_link: str = "robot_arm_tool0", virtual_joint: str = "virtual_joint",
                 planning_frame: str = "robot_map", ik_timeout_s: float = 0.1) -> None:
        """Create the service clients.

        Input:
            node: rclpy Node that owns the clients.
            arm_joint_names: the 6 arm joints, in HQP order.
            namespace: move_group namespace (services live under it).
            group: MoveIt planning group of the arm.
            ik_link: link whose pose the IK solves for (tool0).
            virtual_joint: SRDF floating joint between robot_map and the base.
            planning_frame: parent frame of virtual_joint (robot_map).
            ik_timeout_s: per-request IK solver timeout.

        Output:
            None.
        """
        namespace = "/" + namespace.strip("/") if namespace.strip("/") else ""
        self.node = node
        self.arm_joint_names = list(arm_joint_names)
        self.group = group
        self.ik_link = ik_link
        self.virtual_joint = virtual_joint
        self.planning_frame = planning_frame
        self.ik_timeout_s = ik_timeout_s
        self.ik_client = node.create_client(GetPositionIK, f"{namespace}/compute_ik")
        self.validity_client = node.create_client(
            GetStateValidity, f"{namespace}/check_state_validity")
        self.scene_client = node.create_client(
            ApplyPlanningScene, f"{namespace}/apply_planning_scene")
        self.get_scene_client = node.create_client(GetPlanningScene, f"{namespace}/get_planning_scene")
        self.move_client = ActionClient(node, MoveGroup, f"{namespace}/move_action")

    def ready(self) -> bool:
        """Output: bool, whether all three move_group services are available."""
        return (self.ik_client.service_is_ready() and self.validity_client.service_is_ready()
                and self.scene_client.service_is_ready())

    def apply_collision_objects(self, objects: list):
        """Add box collision objects to the planning scene.

        Input:
            objects: list of {"id", "frame", "box": [sx, sy, sz],
                "position": [x, y, z], optional "yaw"} dicts from the YAML.

        Output:
            rclpy Future of the ApplyPlanningScene call.
        """
        scene = PlanningScene()
        scene.is_diff = True
        for spec in objects:
            obj = CollisionObject()
            obj.header.frame_id = spec.get("frame", "robot_map")
            obj.id = spec["id"]
            primitive = SolidPrimitive()
            primitive.type = SolidPrimitive.BOX
            primitive.dimensions = [float(v) for v in spec["box"]]
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = (float(v) for v in spec["position"])
            quat = Rotation.from_euler("z", float(spec.get("yaw", 0.0))).as_quat()
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quat
            obj.primitives = [primitive]
            obj.primitive_poses = [pose]
            obj.operation = CollisionObject.ADD
            scene.world.collision_objects.append(obj)
        return self.scene_client.call_async(ApplyPlanningScene.Request(scene=scene))

    def _robot_state(self, q_arm, base_transform: Transform) -> RobotState:
        """Robot state with the given arm joints and base placement.

        Input:
            q_arm: 6 arm joint positions (rad).
            base_transform: robot_map -> robot_base_footprint transform.

        Output:
            RobotState: diff state for move_group requests.
        """
        state = RobotState()
        state.is_diff = True
        state.joint_state = JointState(name=self.arm_joint_names,
                                       position=[float(v) for v in q_arm])
        state.multi_dof_joint_state.header.frame_id = self.planning_frame
        state.multi_dof_joint_state.joint_names = [self.virtual_joint]
        state.multi_dof_joint_state.transforms = [base_transform]
        return state

    def request_camera_ik(self, position, look_at, frame: str, tool0_to_camera: np.ndarray,
                          base_transform: Transform, q_seed, rolls, down=None) -> list:
        """Ask for collision-free IK solutions for one camera pose.

        Input:
            position: camera position, (3,), in `frame`.
            look_at: point to aim at, (3,), in `frame`.
            frame: frame of position/look_at (known to move_group's TF).
            tool0_to_camera: (4, 4) pose of the camera optical frame in tool0.
            base_transform: robot_map -> robot_base_footprint (base is fixed).
            q_seed: 6 arm joints (rad) used as IK seed.
            rolls: rolls (rad) about the optical axis to try; [0.0] for a
                fixed upright image.
            down: image y reference of camera_orientation() (None = world down).

        Output:
            list: rclpy Futures of GetPositionIK, one per roll.
        """
        camera_to_tool0 = np.linalg.inv(tool0_to_camera)
        futures = []
        for roll in rolls:
            camera = np.eye(4)
            camera[:3, :3] = camera_orientation(np.asarray(position), np.asarray(look_at), roll, down)
            camera[:3, 3] = position
            tool0 = camera @ camera_to_tool0
            request = GetPositionIK.Request()
            ik = request.ik_request
            ik.group_name = self.group
            ik.ik_link_name = self.ik_link
            ik.avoid_collisions = True
            ik.timeout = Duration(seconds=self.ik_timeout_s).to_msg()
            ik.robot_state = self._robot_state(q_seed, base_transform)
            pose = PoseStamped()
            pose.header.frame_id = frame
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = tool0[:3, 3]
            quat = Rotation.from_matrix(tool0[:3, :3]).as_quat()
            pose.pose.orientation.x, pose.pose.orientation.y = quat[0], quat[1]
            pose.pose.orientation.z, pose.pose.orientation.w = quat[2], quat[3]
            ik.pose_stamped = pose
            futures.append(self.ik_client.call_async(request))
        return futures

    def pick_ik_solution(self, futures: list, q_seed, q_lower=None, q_upper=None):
        """Choose the successful IK solution closest to the seed.

        Each joint is taken as the 2*pi-equivalent angle closest to the seed
        that lies within [q_lower, q_upper] (e.g. the UR elbow is limited to
        +-pi, so an unwrapped 4.6 rad would be an invalid goal); a solution
        with a joint that fits no equivalent within the limits is discarded.

        Input:
            futures: completed futures returned by request_camera_ik().
            q_seed: 6 arm joints (rad) to compare against.
            q_lower, q_upper: optional arm joint limits (rad), (6,) each.

        Output:
            np.ndarray (6,) or None: closest valid arm configuration, None if
                every roll failed (no collision-free solution).
        """
        q_seed = np.asarray(q_seed, dtype=float)
        lower = np.full(q_seed.shape, -np.inf) if q_lower is None else np.asarray(q_lower, dtype=float)
        upper = np.full(q_seed.shape, np.inf) if q_upper is None else np.asarray(q_upper, dtype=float)
        best, best_distance = None, np.inf
        for future in futures:
            response = future.result()
            if response is None or response.error_code.val != MoveItErrorCodes.SUCCESS:
                continue
            by_name = dict(zip(response.solution.joint_state.name,
                               response.solution.joint_state.position))
            try:
                q = np.array([by_name[name] for name in self.arm_joint_names])
            except KeyError:
                continue
            # Per joint, the 2*pi-equivalent angle closest to the seed within
            # the joint limits (so equivalent wrist turns don't look far).
            candidates = q[:, None] + 2.0 * np.pi * np.arange(-2, 3)[None, :]
            valid = (candidates >= lower[:, None] - 1e-6) & (candidates <= upper[:, None] + 1e-6)
            if not valid.any(axis=1).all():
                continue  # some joint has no equivalent angle within its limits
            gaps = np.where(valid, np.abs(candidates - q_seed[:, None]), np.inf)
            q_near = candidates[np.arange(len(q)), np.argmin(gaps, axis=1)]
            distance = float(np.linalg.norm(q_near - q_seed))
            if distance < best_distance:
                best, best_distance = q_near, distance
        return best

    def request_joint_motion(self, q_goal, velocity_scaling: float = 0.3,
                             planning_time_s: float = 5.0,
                             pipeline_id: str = "pilz_industrial_motion_planner",
                             planner_id: str = "PTP", goal_tolerance_rad: float = 0.001,
                             locked_joints=()):
        """Plan and execute a collision-free arm motion to a joint goal.

        Input:
            q_goal: 6 arm joint positions (rad), in arm_joint_names order.
            velocity_scaling: MoveIt velocity/acceleration scaling (0-1].
            planning_time_s: planner time budget.
            pipeline_id: MoveIt planning pipeline.
            planner_id: planner within that pipeline (Pilz needs one
                explicitly; PTP for joint goals).
            goal_tolerance_rad: per-joint goal tolerance. A sampling planner
                (OMPL) may end anywhere within it: 0.01 rad per joint put the
                camera 4 cm off its target with the arm stretched out.
            locked_joints: arm joint names held at their goal (= start)
                value along the whole path. Pilz PTP interpolates in joint
                space, so a joint whose goal equals its start already stays
                still; other pipelines (OMPL) get a path constraint on it.

        Output:
            rclpy Future of the MoveGroup goal handle, or None if the
                move_action server is not available.
        """
        if not self.move_client.server_is_ready():
            return None
        goal = MoveGroup.Goal()
        request = goal.request
        request.group_name = self.group
        request.pipeline_id = pipeline_id
        request.planner_id = planner_id
        request.num_planning_attempts = 5
        request.allowed_planning_time = float(planning_time_s)
        request.max_velocity_scaling_factor = float(velocity_scaling)
        request.max_acceleration_scaling_factor = float(velocity_scaling)
        request.start_state.is_diff = True
        request.goal_constraints = [Constraints(joint_constraints=[
            JointConstraint(joint_name=name, position=float(q), tolerance_above=float(goal_tolerance_rad),
                            tolerance_below=float(goal_tolerance_rad), weight=1.0)
            for name, q in zip(self.arm_joint_names, q_goal)])]
        if locked_joints and pipeline_id != "pilz_industrial_motion_planner":
            goal_by_name = dict(zip(self.arm_joint_names, q_goal))
            request.path_constraints = Constraints(joint_constraints=[
                JointConstraint(joint_name=name, position=float(goal_by_name[name]), tolerance_above=0.01,
                                tolerance_below=0.01, weight=1.0)
                for name in locked_joints])
        goal.planning_options.plan_only = False
        return self.move_client.send_goal_async(goal)

    def request_base_pose(self):
        """Ask move_group for its current robot state (the base, virtual_joint).

        Output:
            rclpy Future of the GetPlanningScene call, or None if the service
                is not available; base_pose_from_scene() reads the reply.
        """
        if not self.get_scene_client.service_is_ready():
            return None
        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.ROBOT_STATE
        return self.get_scene_client.call_async(request)

    def base_pose_from_scene(self, response):
        """Output: (x, y, yaw) of virtual_joint in move_group's state, or None."""
        state = response.scene.robot_state.multi_dof_joint_state
        if self.virtual_joint not in state.joint_names:
            return None
        t = state.transforms[list(state.joint_names).index(self.virtual_joint)]
        q = t.rotation
        return (float(t.translation.x), float(t.translation.y),
                float(np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))))

    def request_validity(self, q_arm, base_transform: Transform):
        """Check one arm configuration against the planning scene.

        Input:
            q_arm: 6 arm joint positions (rad).
            base_transform: robot_map -> robot_base_footprint.

        Output:
            rclpy Future of GetStateValidity (response.valid, .contacts).
        """
        request = GetStateValidity.Request()
        request.group_name = self.group
        request.robot_state = self._robot_state(q_arm, base_transform)
        return self.validity_client.call_async(request)
