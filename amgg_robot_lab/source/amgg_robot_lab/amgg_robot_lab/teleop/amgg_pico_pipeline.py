# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IsaacTeleop controller-based PICO pipeline for AMGG."""


def build_amgg_pico_pipeline():
    """Build the 18-D dual-TCP and parallel-gripper action graph.

    Wrist inputs come from controllers, so PICO runtimes without hand tracking
    remain supported. Gripper triggers take priority over optional hand pinch.
    """
    from isaacteleop.retargeters import (
        GripperRetargeter,
        GripperRetargeterConfig,
        Se3AbsRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource, HandsSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    controllers = ControllersSource(name="controllers")
    hands = HandsSource(name="hands")
    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_controllers = controllers.transformed(transform_input.output(ValueInput.VALUE))
    left_se3 = Se3AbsRetargeter(
        Se3RetargeterConfig(
            input_device=ControllersSource.LEFT,
            zero_out_xy_rotation=False,
            use_wrist_rotation=False,
            use_wrist_position=False,
            target_offset_roll=90.0,
            target_offset_pitch=0.0,
            target_offset_yaw=0.0,
        ),
        name="left_ee_pose",
    )
    right_se3 = Se3AbsRetargeter(
        Se3RetargeterConfig(
            input_device=ControllersSource.RIGHT,
            zero_out_xy_rotation=False,
            use_wrist_rotation=False,
            use_wrist_position=False,
            target_offset_roll=-90.0,
            target_offset_pitch=0.0,
            target_offset_yaw=180.0,
        ),
        name="right_ee_pose",
    )
    connected_left_se3 = left_se3.connect(
        {ControllersSource.LEFT: transformed_controllers.output(ControllersSource.LEFT)}
    )
    connected_right_se3 = right_se3.connect(
        {ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT)}
    )
    left_gripper = GripperRetargeter(GripperRetargeterConfig(hand_side="left"), name="left_gripper")
    right_gripper = GripperRetargeter(GripperRetargeterConfig(hand_side="right"), name="right_gripper")
    connected_left_gripper = left_gripper.connect(
        {
            ControllersSource.LEFT: transformed_controllers.output(ControllersSource.LEFT),
            HandsSource.LEFT: hands.output(HandsSource.LEFT),
        }
    )
    connected_right_gripper = right_gripper.connect(
        {
            ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT),
            HandsSource.RIGHT: hands.output(HandsSource.RIGHT),
        }
    )
    left_pose = ["l_px", "l_py", "l_pz", "l_qx", "l_qy", "l_qz", "l_qw"]
    right_pose = ["r_px", "r_py", "r_pz", "r_qx", "r_qy", "r_qz", "r_qw"]
    gripper_elements = ["l_neg", "l_pos", "r_neg", "r_pos"]
    reorderer = TensorReorderer(
        input_config={
            "left_pose": left_pose,
            "right_pose": right_pose,
            "left_negative": ["l_neg"],
            "left_positive": ["l_pos"],
            "right_negative": ["r_neg"],
            "right_positive": ["r_pos"],
        },
        output_order=left_pose + right_pose + gripper_elements,
        name="amgg_action_reorderer",
        input_types={
            "left_pose": "array",
            "right_pose": "array",
            "left_negative": "scalar",
            "left_positive": "scalar",
            "right_negative": "scalar",
            "right_positive": "scalar",
        },
    )
    connected = reorderer.connect(
        {
            "left_pose": connected_left_se3.output("ee_pose"),
            "right_pose": connected_right_se3.output("ee_pose"),
            "left_negative": connected_left_gripper.output("gripper_command"),
            "left_positive": connected_left_gripper.output("gripper_command"),
            "right_negative": connected_right_gripper.output("gripper_command"),
            "right_positive": connected_right_gripper.output("gripper_command"),
        }
    )
    return OutputCombiner({"action": connected.output("output")}), [left_se3, right_se3]
