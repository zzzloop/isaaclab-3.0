# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PICO controller retargeting for the AM-DP123 dual-arm robot.

The controller grip pose is rebased when XR execution enters ``RUNNING``. The
first valid controller sample therefore commands the URDF-derived hand home
pose exactly; subsequent controller deltas move the corresponding hand 1:1.
Stopping execution freezes both targets and re-arms the origins, which lets the
operator reposition the controllers before resuming without moving the robot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from amgg_robot_lab.contracts import (
    AM_DP123_HAND_JOINT_NAMES,
    AM_DP123_LEFT_HAND_JOINT_NAMES,
    AM_DP123_RIGHT_HAND_JOINT_NAMES,
)

if TYPE_CHECKING:
    from isaacteleop.retargeting_engine.interface import OutputCombiner

AM_DP123_LEFT_WRIST_ACTION_ELEMENTS: tuple[str, ...] = ("l_px", "l_py", "l_pz", "l_qx", "l_qy", "l_qz", "l_qw")
"""Element names of the left hand-base pose inside the action tensor."""

AM_DP123_RIGHT_WRIST_ACTION_ELEMENTS: tuple[str, ...] = (
    "r_px",
    "r_py",
    "r_pz",
    "r_qx",
    "r_qy",
    "r_qz",
    "r_qw",
)
"""Element names of the right hand-base pose inside the action tensor."""

AM_DP123_ACTION_LAYOUT: tuple[str, ...] = (
    AM_DP123_LEFT_WRIST_ACTION_ELEMENTS + AM_DP123_RIGHT_WRIST_ACTION_ELEMENTS + AM_DP123_HAND_JOINT_NAMES
)
"""Ordered element names of the 18-D AM-DP123 PICO action."""

# Kept for source compatibility with the first AM-DP123 bring-up. Absolute
# controller orientation offsets are no longer applied: orientation is rebased
# from the controller pose captured on Play.
AM_DP123_LEFT_WRIST_TARGET_OFFSET_DEG: tuple[float, float, float] = (90.0, 0.0, 0.0)
AM_DP123_RIGHT_WRIST_TARGET_OFFSET_DEG: tuple[float, float, float] = (-90.0, 0.0, 180.0)

AM_DP123_TRIGGER_DEADZONE = 0.05
"""Released-end PICO trigger deadzone before proportional hand closure begins."""

AM_DP123_CONTROLLER_POSITION_SCALE = 1.0
"""Controller translation gain applied to both hands [m/m]."""

# FK results for left_arm_hand_link/right_arm_hand_link at the contract home.
# Z includes AM_DP123_BASE_SPAWN_HEIGHT_M because Pink accepts world targets.
AM_DP123_LEFT_HAND_HOME_POSE: tuple[float, ...] = (
    0.4887442249962056,
    0.11600488672609706,
    0.8662781670126685,
    0.7350688652061255,
    -0.4826872891821262,
    0.4536493897228235,
    0.14453018878663146,
)
"""Left hand-base home pose [m, XYZW quaternion] in the simulation world frame."""

AM_DP123_RIGHT_HAND_HOME_POSE: tuple[float, ...] = (
    0.4508247721078661,
    -0.09157004412781035,
    0.8474318951986614,
    -0.5235562374930638,
    0.7311176702798987,
    0.0362851763415602,
    0.43593486252473773,
)
"""Right hand-base home pose [m, XYZW quaternion] in the simulation world frame."""

AM_DP123_IDLE_ACTION: tuple[float, ...] = (
    *AM_DP123_LEFT_HAND_HOME_POSE,
    *AM_DP123_RIGHT_HAND_HOME_POSE,
    1.0,
    1.0,
    1.0,
    1.0,
)
"""Reference 18-D world-frame hand pose action with both hands open."""

_MIN_QUATERNION_NORM = 1.0e-8


def _normalize_quaternion_xyzw(quaternion: np.ndarray) -> np.ndarray:
    """Return a normalized finite XYZW quaternion.

    Raises:
        ValueError: If the quaternion is not finite or has near-zero norm.
    """
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if quaternion.shape != (4,) or not np.isfinite(norm) or norm < _MIN_QUATERNION_NORM:
        raise ValueError("Expected a finite, non-zero XYZW quaternion.")
    return quaternion / norm


def _quaternion_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compose two XYZW quaternions as ``left * right``."""
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.array(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dtype=np.float64,
    )


def _quaternion_inverse_xyzw(quaternion: np.ndarray) -> np.ndarray:
    """Return the inverse of a unit XYZW quaternion."""
    quaternion = _normalize_quaternion_xyzw(quaternion)
    return np.array((-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]), dtype=np.float64)


def rebase_am_dp123_controller_pose(
    controller_pose: np.ndarray,
    controller_origin_pose: np.ndarray,
    hand_home_pose: np.ndarray,
    position_scale: float = AM_DP123_CONTROLLER_POSITION_SCALE,
) -> np.ndarray:
    """Map a controller pose delta onto an AM-DP123 hand home pose.

    All poses contain position [m] followed by an XYZW quaternion. Controller
    rotations are applied in the simulation world frame.

    Args:
        controller_pose: Current PICO controller pose [m, XYZW], shape ``(7,)``.
        controller_origin_pose: Pose captured when Play engaged [m, XYZW].
        hand_home_pose: Robot hand pose held at the engage instant [m, XYZW].
        position_scale: Dimensionless controller translation gain.

    Returns:
        Absolute robot hand target [m, XYZW], shape ``(7,)``.

    Raises:
        ValueError: If a pose is malformed, non-finite, or has an invalid quaternion.
    """
    controller_pose = np.asarray(controller_pose, dtype=np.float64)
    controller_origin_pose = np.asarray(controller_origin_pose, dtype=np.float64)
    hand_home_pose = np.asarray(hand_home_pose, dtype=np.float64)
    poses = (controller_pose, controller_origin_pose, hand_home_pose)
    if any(pose.shape != (7,) or not np.all(np.isfinite(pose)) for pose in poses):
        raise ValueError("Controller and hand poses must contain seven finite values.")
    if not np.isfinite(position_scale) or position_scale <= 0.0:
        raise ValueError("position_scale must be finite and positive.")

    controller_quaternion = _normalize_quaternion_xyzw(controller_pose[3:])
    origin_quaternion = _normalize_quaternion_xyzw(controller_origin_pose[3:])
    home_quaternion = _normalize_quaternion_xyzw(hand_home_pose[3:])
    rotation_delta = _quaternion_multiply_xyzw(controller_quaternion, _quaternion_inverse_xyzw(origin_quaternion))
    target_quaternion = _normalize_quaternion_xyzw(_quaternion_multiply_xyzw(rotation_delta, home_quaternion))
    target_position = hand_home_pose[:3] + position_scale * (controller_pose[:3] - controller_origin_pose[:3])
    return np.concatenate((target_position, target_quaternion))


class AmDp123ControllerClutch:
    """State machine for continuous, home-relative PICO pose control."""

    def __init__(self, reset_home_pose: tuple[float, ...], position_scale: float = 1.0) -> None:
        """Initialize the clutch from a world-frame hand home pose.

        Args:
            reset_home_pose: Reset hand pose [m, XYZW], shape ``(7,)``.
            position_scale: Dimensionless controller translation gain.
        """
        home = np.asarray(reset_home_pose, dtype=np.float64)
        self._reset_home = rebase_am_dp123_controller_pose(home, home, home, position_scale)
        self._position_scale = position_scale
        self._controller_origin: np.ndarray | None = None
        self._running_home: np.ndarray | None = None
        self._last_pose = self._reset_home.copy()

    @property
    def last_pose(self) -> np.ndarray:
        """Last valid hand target [m, XYZW]."""
        return self._last_pose.copy()

    def update(self, controller_pose: np.ndarray | None, *, running: bool, reset: bool = False) -> np.ndarray:
        """Advance the clutch and return an absolute hand target.

        Args:
            controller_pose: Controller grip pose [m, XYZW], or ``None`` when unavailable.
            running: Whether XR execution is currently in Play state.
            reset: Whether the simulation requested an episode reset.

        Returns:
            Absolute hand target [m, XYZW], shape ``(7,)``.
        """
        if reset:
            self._controller_origin = None
            self._running_home = None
            self._last_pose = self._reset_home.copy()
        elif not running:
            self._controller_origin = None

        if not running or controller_pose is None:
            return self.last_pose

        controller_pose = np.asarray(controller_pose, dtype=np.float64)
        try:
            candidate_origin = controller_pose.copy()
            home = self._reset_home if self._running_home is None else self._last_pose
            if self._controller_origin is None:
                rebase_am_dp123_controller_pose(candidate_origin, candidate_origin, home, self._position_scale)
                self._controller_origin = candidate_origin
                self._running_home = home.copy()
            target = rebase_am_dp123_controller_pose(
                controller_pose, self._controller_origin, self._running_home, self._position_scale
            )
        except ValueError:
            return self.last_pose

        if float(np.dot(target[3:], self._last_pose[3:])) < 0.0:
            target[3:] *= -1.0
        self._last_pose = target
        return self.last_pose


def am_dp123_trigger_to_gripper_command(trigger: float) -> float:
    """Map a PICO analog trigger to the AM-DP123 ``+1`` open / ``-1`` closed command."""
    closed_fraction = (float(trigger) - AM_DP123_TRIGGER_DEADZONE) / (1.0 - AM_DP123_TRIGGER_DEADZONE)
    closed_fraction = min(max(closed_fraction, 0.0), 1.0)
    return 1.0 - 2.0 * closed_fraction


def build_am_dp123_pico_pipeline() -> OutputCombiner:
    """Build the 18-D dual-hand home-relative PICO action graph."""
    from isaacteleop.retargeters import TensorReorderer
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
    from isaacteleop.retargeting_engine.interface import BaseRetargeter, OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.interface.execution_events import ExecutionState
    from isaacteleop.retargeting_engine.interface.tensor_group_type import OptionalType, TensorGroupType
    from isaacteleop.retargeting_engine.tensor_types import (
        ControllerInput,
        ControllerInputIndex,
        DLDataType,
        FloatType,
        NDArrayType,
        TransformMatrix,
    )

    class ControllerClutchRetargeter(BaseRetargeter):
        """Adapt a tracked IsaacTeleop controller to the AM-DP123 clutch."""

        def __init__(self, input_device: str, home_pose: tuple[float, ...], name: str) -> None:
            self._input_device = input_device
            self._clutch = AmDp123ControllerClutch(home_pose, AM_DP123_CONTROLLER_POSITION_SCALE)
            super().__init__(name=name)

        def input_spec(self):
            return {self._input_device: OptionalType(ControllerInput())}

        def output_spec(self):
            return {
                "ee_pose": TensorGroupType(
                    "ee_pose", [NDArrayType("pose", shape=(7,), dtype=DLDataType.FLOAT, dtype_bits=32)]
                )
            }

        def _compute_fn(self, inputs, outputs, context) -> None:
            running = context.execution_events.execution_state == ExecutionState.RUNNING
            controller = inputs[self._input_device]
            controller_pose = None
            if not controller.is_none and bool(controller[ControllerInputIndex.GRIP_IS_VALID]):
                position = np.from_dlpack(controller[ControllerInputIndex.GRIP_POSITION]).astype(np.float64)
                orientation = np.from_dlpack(controller[ControllerInputIndex.GRIP_ORIENTATION]).astype(np.float64)
                controller_pose = np.concatenate((position, orientation))
            outputs["ee_pose"][0] = self._clutch.update(
                controller_pose, running=running, reset=context.execution_events.reset
            ).astype(np.float32)

    class ControllerTriggerRetargeter(BaseRetargeter):
        """Controller-only proportional gripper input that requires no HandTracker."""

        def __init__(self, input_device: str, name: str) -> None:
            self._input_device = input_device
            self._last_command = 1.0
            super().__init__(name=name)

        def input_spec(self):
            return {self._input_device: OptionalType(ControllerInput())}

        def output_spec(self):
            return {"gripper_command": TensorGroupType("gripper_command", [FloatType("command")])}

        def _compute_fn(self, inputs, outputs, context) -> None:
            if context.execution_events.reset:
                self._last_command = 1.0
            controller = inputs[self._input_device]
            if not controller.is_none:
                self._last_command = am_dp123_trigger_to_gripper_command(
                    float(controller[ControllerInputIndex.TRIGGER_VALUE])
                )
            outputs["gripper_command"][0] = self._last_command

    controllers = ControllersSource(name="controllers")
    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_controllers = controllers.transformed(transform_input.output(ValueInput.VALUE))

    left_pose = ControllerClutchRetargeter(
        ControllersSource.LEFT, AM_DP123_LEFT_HAND_HOME_POSE, name="left_ee_pose"
    ).connect({ControllersSource.LEFT: transformed_controllers.output(ControllersSource.LEFT)})
    right_pose = ControllerClutchRetargeter(
        ControllersSource.RIGHT, AM_DP123_RIGHT_HAND_HOME_POSE, name="right_ee_pose"
    ).connect({ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT)})
    left_gripper = ControllerTriggerRetargeter(ControllersSource.LEFT, name="left_gripper").connect(
        {ControllersSource.LEFT: transformed_controllers.output(ControllersSource.LEFT)}
    )
    right_gripper = ControllerTriggerRetargeter(ControllersSource.RIGHT, name="right_gripper").connect(
        {ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT)}
    )

    reorderer = TensorReorderer(
        input_config={
            "left_pose": list(AM_DP123_LEFT_WRIST_ACTION_ELEMENTS),
            "right_pose": list(AM_DP123_RIGHT_WRIST_ACTION_ELEMENTS),
            "left_jaw_a": [AM_DP123_LEFT_HAND_JOINT_NAMES[0]],
            "left_jaw_b": [AM_DP123_LEFT_HAND_JOINT_NAMES[1]],
            "right_jaw_a": [AM_DP123_RIGHT_HAND_JOINT_NAMES[0]],
            "right_jaw_b": [AM_DP123_RIGHT_HAND_JOINT_NAMES[1]],
        },
        output_order=list(AM_DP123_ACTION_LAYOUT),
        name="am_dp123_action_reorderer",
        input_types={
            "left_pose": "array",
            "right_pose": "array",
            "left_jaw_a": "scalar",
            "left_jaw_b": "scalar",
            "right_jaw_a": "scalar",
            "right_jaw_b": "scalar",
        },
    )
    connected = reorderer.connect(
        {
            "left_pose": left_pose.output("ee_pose"),
            "right_pose": right_pose.output("ee_pose"),
            "left_jaw_a": left_gripper.output("gripper_command"),
            "left_jaw_b": left_gripper.output("gripper_command"),
            "right_jaw_a": right_gripper.output("gripper_command"),
            "right_jaw_b": right_gripper.output("gripper_command"),
        }
    )
    return OutputCombiner({"action": connected.output("output")})
