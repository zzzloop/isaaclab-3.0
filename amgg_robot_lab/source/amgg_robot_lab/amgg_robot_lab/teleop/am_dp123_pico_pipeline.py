# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IsaacTeleop controller-based PICO pipeline for AM-DP123.

The pipeline produces the 18-D action consumed by
:class:`~amgg_robot_lab.tasks.mdp.AmDp123PinkInverseKinematicsActionCfg`:

    [left wrist pose (xyz + xyzw, 7), right wrist pose (xyz + xyzw, 7),
     left_arm_hand_joint1_0, left_arm_hand_joint2_0,
     right_arm_hand_joint1_0, right_arm_hand_joint2_0]

Each hand receives the same ``GripperRetargeter`` scalar twice, once per jaw, so
both jaws of one hand close together. Wrist poses come from the PICO
controllers, which keeps headsets without hand tracking usable.

The per-side ``target_offset_roll`` / ``target_offset_yaw`` values are seeded from
the previous AM-DP123-style calibration and must be re-checked with a real
headset on the validation server; the retargeters are handed to
:attr:`IsaacTeleopCfg.retargeters_to_tune` so they can be adjusted live.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from amgg_robot_lab.contracts import (
    AM_DP123_HAND_JOINT_NAMES,
    AM_DP123_LEFT_HAND_JOINT_NAMES,
    AM_DP123_RIGHT_HAND_JOINT_NAMES,
)

if TYPE_CHECKING:
    from isaacteleop.retargeting_engine.interface import BaseRetargeter, OutputCombiner

AM_DP123_LEFT_WRIST_ACTION_ELEMENTS: tuple[str, ...] = ("l_px", "l_py", "l_pz", "l_qx", "l_qy", "l_qz", "l_qw")
"""Element names of the left wrist PICO pose inside the action tensor."""

AM_DP123_RIGHT_WRIST_ACTION_ELEMENTS: tuple[str, ...] = ("r_px", "r_py", "r_pz", "r_qx", "r_qy", "r_qz", "r_qw")
"""Element names of the right wrist PICO pose inside the action tensor."""

AM_DP123_ACTION_LAYOUT: tuple[str, ...] = (
    AM_DP123_LEFT_WRIST_ACTION_ELEMENTS + AM_DP123_RIGHT_WRIST_ACTION_ELEMENTS + AM_DP123_HAND_JOINT_NAMES
)
"""Ordered element names of the 18-D AM-DP123 PICO action."""

# PICO controller frame to AM-DP123 wrist frame offsets [deg].  These are the
# starting values from the previous AM-DP123 hand calibration and are tuned
# interactively through the IsaacTeleop retargeter tuning UI.
AM_DP123_LEFT_WRIST_TARGET_OFFSET_DEG: tuple[float, float, float] = (90.0, 0.0, 0.0)
AM_DP123_RIGHT_WRIST_TARGET_OFFSET_DEG: tuple[float, float, float] = (-90.0, 0.0, 180.0)

AM_DP123_TRIGGER_DEADZONE = 0.05
"""Released-end PICO trigger deadzone before proportional hand closure begins."""

AM_DP123_IDLE_ACTION: tuple[float, ...] = (
    0.36,
    0.2,
    0.9529,
    0.41757411,
    -0.66209021,
    -0.02053298,
    0.62197011,
    0.36,
    -0.2,
    0.9529,
    -0.06195791,
    0.54263546,
    -0.49132102,
    0.67846269,
    1.0,
    1.0,
    1.0,
    1.0,
)
"""Reference 18-D action that holds the contract home pose [m, quaternion, trigger].

Positions are the forward-kinematics wrist poses of
:data:`~amgg_robot_lab.contracts.AM_DP123_HOME_POSITIONS` in the world frame, that
is the ``base_link`` frame plus ``AM_DP123_BASE_SPAWN_HEIGHT_M``. Both wrist
quaternions are XYZW and both hands are open. ``scripts/tools/replay_demos.py``
uses this value when no recorded action buffer is available.
"""


def am_dp123_trigger_to_gripper_command(trigger: float) -> float:
    """Map a PICO analog trigger to the AM-DP123 ``+1`` open / ``-1`` closed command."""
    closed_fraction = (float(trigger) - AM_DP123_TRIGGER_DEADZONE) / (1.0 - AM_DP123_TRIGGER_DEADZONE)
    closed_fraction = min(max(closed_fraction, 0.0), 1.0)
    return 1.0 - 2.0 * closed_fraction


def build_am_dp123_pico_pipeline() -> tuple[OutputCombiner, list[BaseRetargeter]]:
    """Build the 18-D dual-wrist and two-jaw hand action graph.

    Returns:
        Tuple of the ``OutputCombiner`` producing the ``"action"`` tensor and the
        wrist retargeters that the tuning UI should expose.
    """
    from isaacteleop.retargeters import (
        Se3AbsRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
    from isaacteleop.retargeting_engine.interface import BaseRetargeter, OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.interface.tensor_group_type import OptionalType, TensorGroupType
    from isaacteleop.retargeting_engine.tensor_types import (
        ControllerInput,
        ControllerInputIndex,
        FloatType,
        TransformMatrix,
    )

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

    left_roll, left_pitch, left_yaw = AM_DP123_LEFT_WRIST_TARGET_OFFSET_DEG
    left_se3 = Se3AbsRetargeter(
        Se3RetargeterConfig(
            input_device=ControllersSource.LEFT,
            zero_out_xy_rotation=False,
            use_wrist_rotation=False,
            use_wrist_position=False,
            target_offset_roll=left_roll,
            target_offset_pitch=left_pitch,
            target_offset_yaw=left_yaw,
        ),
        name="left_ee_pose",
    )
    right_roll, right_pitch, right_yaw = AM_DP123_RIGHT_WRIST_TARGET_OFFSET_DEG
    right_se3 = Se3AbsRetargeter(
        Se3RetargeterConfig(
            input_device=ControllersSource.RIGHT,
            zero_out_xy_rotation=False,
            use_wrist_rotation=False,
            use_wrist_position=False,
            target_offset_roll=right_roll,
            target_offset_pitch=right_pitch,
            target_offset_yaw=right_yaw,
        ),
        name="right_ee_pose",
    )
    connected_left_se3 = left_se3.connect(
        {ControllersSource.LEFT: transformed_controllers.output(ControllersSource.LEFT)}
    )
    connected_right_se3 = right_se3.connect(
        {ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT)}
    )

    left_gripper = ControllerTriggerRetargeter(ControllersSource.LEFT, name="left_gripper")
    right_gripper = ControllerTriggerRetargeter(ControllersSource.RIGHT, name="right_gripper")
    connected_left_gripper = left_gripper.connect(
        {ControllersSource.LEFT: transformed_controllers.output(ControllersSource.LEFT)}
    )
    connected_right_gripper = right_gripper.connect(
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
            "left_pose": connected_left_se3.output("ee_pose"),
            "right_pose": connected_right_se3.output("ee_pose"),
            "left_jaw_a": connected_left_gripper.output("gripper_command"),
            "left_jaw_b": connected_left_gripper.output("gripper_command"),
            "right_jaw_a": connected_right_gripper.output("gripper_command"),
            "right_jaw_b": connected_right_gripper.output("gripper_command"),
        }
    )
    return OutputCombiner({"action": connected.output("output")}), [left_se3, right_se3]
