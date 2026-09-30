#!/usr/bin/env python3
"""Whole-body kinematic model of RB-VOGUI+ with UR5e, backed by Pinocchio.

Pinocchio is imported lazily inside the methods that need it, so this module
can be imported (e.g. from ``tasks.py`` for type hints) without requiring the
``pin`` wheel from ``requirements-hqp.txt`` to be installed.

The whole-body model has ``nv = 9``: a planar SE(2) base joint (``vx, vy,
wz`` expressed in the base's own frame, matching the convention expected by
``robotnik_base_control``'s ``TwistStamped``) followed by the six UR5e arm
joints. The eight wheel/steering joints of the base are locked at a fixed
reference configuration and do not appear in the decision vector.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

DEFAULT_ARM_JOINTS = [
    "robot_arm_shoulder_pan_joint",
    "robot_arm_shoulder_lift_joint",
    "robot_arm_elbow_joint",
    "robot_arm_wrist_1_joint",
    "robot_arm_wrist_2_joint",
    "robot_arm_wrist_3_joint",
]

DEFAULT_WHEEL_JOINTS = [
    f"robot_{corner}_{suffix}"
    for corner in ("front_right", "front_left", "back_left", "back_right")
    for suffix in ("wheel_joint", "steering_joint")
]

DEFAULT_EE_FRAME = "robot_arm_tool0"


@dataclass(frozen=True)
class JointLimits:
    joint_names: list
    v_slice: slice
    q_lower: np.ndarray
    q_upper: np.ndarray
    v_limit: np.ndarray


def load_urdf_from_xacro(xacro_path: str, mappings: dict) -> str:
    """Expand a xacro file to a plain URDF XML string, in memory.

    Input:
        xacro_path: path to the .urdf.xacro file to expand.
        mappings: xacro argument name -> value substitutions.

    Output:
        str: the fully expanded URDF as an XML string.
    """
    import xacro

    document = xacro.process_file(xacro_path, mappings={k: str(v) for k, v in mappings.items()})
    return document.toxml()


def resolve_default_xacro(wrist_camera: str = "none", ur_type: str = "ur5e"):
    """Locate the authoritative rbvogui_plus xacro and its default arguments.

    Input:
        wrist_camera: wrist camera type passed to the xacro
            ("none", "stereolabs_zed2i" or "realsense_d435i"); anything but
            "none" adds the camera's optical frames to the model.
        ur_type: Universal Robots arm model ("ur5e", "ur15", ...); must
            match the simulated/real robot (UR_TYPE in docker-compose).

    Output:
        tuple[str, dict]: (xacro_path, mappings) ready to pass to
            load_urdf_from_xacro() or RobotModel.from_xacro().
    """
    from ament_index_python.packages import get_package_share_directory

    share = get_package_share_directory("robotnik_description")
    xacro_path = f"{share}/robots/rbvogui/rbvogui_plus.urdf.xacro"
    mappings = {
        "namespace": "robot",
        "prefix": "robot_",
        "gazebo_ignition": "false",
        "end_effector": "none",
        "use_tool_changer": "false",
        "wrist_camera": wrist_camera,
        "ur_type": ur_type,
    }
    return xacro_path, mappings


def _contiguous_slice(indices: Sequence[int]) -> slice:
    """Turn a set of indices into a single contiguous slice.

    Input:
        indices: joint-vector indices that are expected to be contiguous
            once sorted.

    Output:
        slice: covering the indices, from the smallest to the largest + 1.

    Raises:
        ValueError: if the sorted indices are not actually contiguous.
    """
    ordered = sorted(indices)
    if not ordered:
        return slice(0, 0)
    if ordered != list(range(ordered[0], ordered[-1] + 1)):
        raise ValueError(f"Joint indices are not contiguous: {ordered}")
    return slice(ordered[0], ordered[-1] + 1)


class RobotModel:
    """Whole-body kinematics adapter, independent of ROS communication."""

    def __init__(self, urdf_xml: str, *,
                 arm_joint_names: Sequence[str] = tuple(DEFAULT_ARM_JOINTS),
                 locked_joint_names: Sequence[str] = tuple(DEFAULT_WHEEL_JOINTS),
                 ee_frame: str = DEFAULT_EE_FRAME) -> None:
        """Build the whole-body Pinocchio model from an expanded URDF.

        Input:
            urdf_xml: expanded URDF XML string (see load_urdf_from_xacro()).
            arm_joint_names: the 6 arm joints, in decision-vector order.
            locked_joint_names: joints to remove from the model (frozen at
                their neutral configuration), e.g. the 8 wheel joints.
            ee_frame: name of the end-effector frame that must exist in the
                resulting model.

        Output:
            None. Populates self.model/self.data plus the base/arm slices,
            and initializes self.q/self.dq to the neutral configuration.

        Raises:
            ValueError: if ee_frame is missing, the root joint isn't a
                planar SE(2) joint, or an arm joint is missing/not 1-DoF.
        """
        import pinocchio as pin

        self._pin = pin
        model_full = pin.buildModelFromXML(urdf_xml, pin.JointModelPlanar())
        q_ref = pin.neutral(model_full)
        joint_ids = [model_full.getJointId(name) for name in locked_joint_names
                     if model_full.existJointName(name)]
        self.model = pin.buildReducedModel(model_full, joint_ids, q_ref) if joint_ids else model_full
        self.data = self.model.createData()

        if not self.model.existFrame(ee_frame):
            raise ValueError(f"Unknown end-effector frame: {ee_frame}")
        self.ee_frame = ee_frame

        # Joint index 0 is Pinocchio's implicit 'universe'; index 1 is the
        # planar base joint we requested as root_joint.
        base_joint = self.model.joints[1]
        if base_joint.nq != 4 or base_joint.nv != 3:
            raise ValueError("Expected a planar SE(2) base joint (nq=4, nv=3)")
        self.base_q_slice = slice(base_joint.idx_q, base_joint.idx_q + base_joint.nq)
        self.base_v_slice = slice(base_joint.idx_v, base_joint.idx_v + base_joint.nv)

        self._arm_joint_names = list(arm_joint_names)
        for name in self._arm_joint_names:
            if not self.model.existJointName(name):
                raise ValueError(f"Unknown arm joint: {name}")
            joint = self.model.joints[self.model.getJointId(name)]
            if joint.nq != 1 or joint.nv != 1:
                raise ValueError(f"Arm joint {name} must be a single-DoF joint")
        self.arm_q_slice = _contiguous_slice(
            self.model.joints[self.model.getJointId(n)].idx_q for n in self._arm_joint_names)
        self.arm_v_slice = _contiguous_slice(
            self.model.joints[self.model.getJointId(n)].idx_v for n in self._arm_joint_names)

        self.q = pin.neutral(self.model)
        self.dq = np.zeros(self.model.nv)
        self.update()

    @classmethod
    def from_xacro(cls, xacro_path: str, mappings: dict, **kwargs) -> "RobotModel":
        """Build a RobotModel by expanding a xacro file first.

        Input:
            xacro_path: path to the .urdf.xacro file to expand.
            mappings: xacro argument name -> value substitutions.
            **kwargs: forwarded to RobotModel.__init__ (e.g. ee_frame).

        Output:
            RobotModel: constructed from the expanded URDF.
        """
        return cls(load_urdf_from_xacro(xacro_path, mappings), **kwargs)

    @property
    def nq(self) -> int:
        """Output: int, the configuration-vector size (10: 4 base + 6 arm)."""
        return self.model.nq

    @property
    def nv(self) -> int:
        """Output: int, the velocity-vector size (9: 3 base + 6 arm)."""
        return self.model.nv

    @property
    def joint_names(self) -> list:
        """Output: list[str], every joint name in this (reduced) model,
        including the implicit 'universe' and the planar base joint.
        """
        return list(self.model.names)

    @property
    def arm_joint_names(self) -> list:
        """Output: list[str], the 6 arm joint names in decision-vector order."""
        return list(self._arm_joint_names)

    def update(self, q: np.ndarray = None, dq: np.ndarray = None) -> None:
        """Recompute forward kinematics/Jacobians for the current state.

        Input:
            q: new configuration vector (shape (nq,)); keeps self.q if None.
            dq: new velocity vector (shape (nv,)); keeps self.dq if None.

        Output:
            None. Updates self.q/self.dq and refreshes self.data in place
            (joint placements, Jacobians, frame placements) so the next
            frame_pose()/frame_jacobian() call reflects this state.
        """
        pin = self._pin
        if q is not None:
            self.q = np.asarray(q, dtype=float).copy()
        if dq is not None:
            self.dq = np.asarray(dq, dtype=float).copy()
        pin.forwardKinematics(self.model, self.data, self.q, self.dq)
        pin.computeJointJacobians(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)

    def set_base_state(self, x: float, y: float, yaw: float,
                        vx: float, vy: float, wz: float) -> None:
        """Set the planar base pose/twist and refresh kinematics.

        Input:
            x, y, yaw: base pose in the odom frame (m, m, rad).
            vx, vy, wz: base twist in the base's own (body) frame,
                matching robotnik_base_control's TwistStamped convention.

        Output:
            None. Updates self.q/self.dq's base slice and calls update().
        """
        q = self.q.copy()
        q[self.base_q_slice] = [x, y, np.cos(yaw), np.sin(yaw)]
        dq = self.dq.copy()
        dq[self.base_v_slice] = [vx, vy, wz]
        self.update(q=q, dq=dq)

    def set_arm_state(self, q_arm: np.ndarray, dq_arm: np.ndarray = None) -> None:
        """Set the 6 arm joint positions/velocities and refresh kinematics.

        Input:
            q_arm: arm joint positions (rad), shape (6,), in
                arm_joint_names order.
            dq_arm: arm joint velocities (rad/s), shape (6,); keeps the
                current velocities for that slice if None.

        Output:
            None. Updates self.q/self.dq's arm slice and calls update().
        """
        q = self.q.copy()
        q[self.arm_q_slice] = q_arm
        dq = self.dq.copy()
        if dq_arm is not None:
            dq[self.arm_v_slice] = dq_arm
        self.update(q=q, dq=dq)

    def add_frame(self, name: str, parent_frame: str, translation: Sequence[float],
                  rpy: Sequence[float] = (0.0, 0.0, 0.0)) -> None:
        """Add a fixed frame (e.g. a tool tip not in the URDF) to the model.

        Input:
            name: new frame name; must not exist yet.
            parent_frame: existing frame it is rigidly attached to.
            translation: (3,) offset (m) in parent_frame.
            rpy: (3,) roll/pitch/yaw (rad) of the new frame in parent_frame.

        Output:
            None. Rebuilds self.data and refreshes kinematics.

        Raises:
            ValueError: if parent_frame is missing or name already exists.
        """
        pin = self._pin
        if not self.model.existFrame(parent_frame):
            raise ValueError(f"Unknown frame: {parent_frame}")
        if self.model.existFrame(name):
            raise ValueError(f"Frame already exists: {name}")
        parent_id = self.model.getFrameId(parent_frame)
        parent = self.model.frames[parent_id]
        offset = pin.SE3(pin.rpy.rpyToMatrix(*np.asarray(rpy, dtype=float)),
                         np.asarray(translation, dtype=float))
        self.model.addFrame(pin.Frame(name, parent.parentJoint, parent_id,
                                      parent.placement * offset, pin.FrameType.OP_FRAME))
        self.data = self.model.createData()
        self.update()

    def integrate(self, v: np.ndarray, dt: float) -> np.ndarray:
        """Integrate a whole-body velocity over dt into a new configuration.

        Input:
            v: whole-body velocity, shape (nv,) = (9,).
            dt: integration period (s).

        Output:
            np.ndarray: new configuration vector, shape (nq,) = (10,). Does
                not modify self.q; pass the result to update() to apply it.
        """
        return self._pin.integrate(self.model, self.q, np.asarray(v, dtype=float) * dt)

    def frame_jacobian(self, frame_name: str, reference_frame=None) -> np.ndarray:
        """Return a frame's Jacobian at the current state.

        Input:
            frame_name: name of an existing frame in the model.
            reference_frame: a pinocchio.ReferenceFrame; defaults to
                LOCAL_WORLD_ALIGNED (world-aligned axes, origin at the
                frame) when None.

        Output:
            np.ndarray: Jacobian, shape (6, nv) = (6, 9) -- rows 0:3 linear,
                3:6 angular velocity. Requires update() to have been called
                for the current state (done automatically by set_*_state()).

        Raises:
            ValueError: if frame_name does not exist in the model.
        """
        pin = self._pin
        if reference_frame is None:
            reference_frame = pin.LOCAL_WORLD_ALIGNED
        if not self.model.existFrame(frame_name):
            raise ValueError(f"Unknown frame: {frame_name}")
        frame_id = self.model.getFrameId(frame_name)
        return pin.getFrameJacobian(self.model, self.data, frame_id, reference_frame)

    def frame_pose(self, frame_name: str):
        """Return a frame's pose at the current state.

        Input:
            frame_name: name of an existing frame in the model.

        Output:
            pinocchio.SE3: pose of the frame in the world frame. Requires
                update() to have been called for the current state (done
                automatically by set_*_state()).

        Raises:
            ValueError: if frame_name does not exist in the model.
        """
        if not self.model.existFrame(frame_name):
            raise ValueError(f"Unknown frame: {frame_name}")
        return self.data.oMf[self.model.getFrameId(frame_name)]

    def velocity_slice(self, joint_names: Sequence[str]) -> slice:
        """Return the velocity-vector slice spanning a set of joints.

        Input:
            joint_names: joint names expected to occupy a contiguous range
                of self.dq.

        Output:
            slice: into the (nv,) velocity vector.

        Raises:
            ValueError: if a joint is unknown or the joints aren't
                contiguous in self.dq.
        """
        return _contiguous_slice(
            self.model.joints[self._joint_id(n)].idx_v for n in joint_names)

    def configuration_slice(self, joint_names: Sequence[str]) -> slice:
        """Return the configuration-vector slice spanning a set of joints.

        Input:
            joint_names: joint names expected to occupy a contiguous range
                of self.q.

        Output:
            slice: into the (nq,) configuration vector.

        Raises:
            ValueError: if a joint is unknown or the joints aren't
                contiguous in self.q.
        """
        return _contiguous_slice(
            self.model.joints[self._joint_id(n)].idx_q for n in joint_names)

    def joint_limits(self, joint_names: Sequence[str] = None) -> JointLimits:
        """Return position/velocity limits for a set of joints.

        Input:
            joint_names: joints to look up; defaults to arm_joint_names
                when None.

        Output:
            JointLimits: joint_names, the corresponding velocity_slice, and
                the URDF-declared q_lower/q_upper/v_limit arrays.
        """
        names = list(joint_names) if joint_names is not None else self._arm_joint_names
        v_slice = self.velocity_slice(names)
        q_slice = self.configuration_slice(names)
        return JointLimits(
            joint_names=names,
            v_slice=v_slice,
            q_lower=self.model.lowerPositionLimit[q_slice].copy(),
            q_upper=self.model.upperPositionLimit[q_slice].copy(),
            v_limit=self.model.velocityLimit[v_slice].copy(),
        )

    def _joint_id(self, name: str) -> int:
        """Look up a joint's Pinocchio index by name.

        Input:
            name: joint name to look up.

        Output:
            int: the joint's index in self.model.

        Raises:
            ValueError: if name does not exist in the model.
        """
        if not self.model.existJointName(name):
            raise ValueError(f"Unknown joint: {name}")
        return self.model.getJointId(name)
