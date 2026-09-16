# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The AM-DP123 PICO action ABI must stay aligned with the joint contract."""

from __future__ import annotations

from math import isclose, isfinite, sqrt

from amgg_robot_lab.contracts import (
    AM_DP123_ABSOLUTE_IK_ACTION_DIM,
    AM_DP123_HAND_ACTION_SIDE_INDEX,
    AM_DP123_HAND_ACTION_TRIGGER_INDEX,
    AM_DP123_HAND_JOINT_NAMES,
    AM_DP123_HAND_TRIGGER_OPEN,
    AM_DP123_LEFT_HAND_JOINT_NAMES,
    AM_DP123_RIGHT_HAND_JOINT_NAMES,
)
from amgg_robot_lab.teleop import (
    AM_DP123_ACTION_LAYOUT,
    AM_DP123_IDLE_ACTION,
    AM_DP123_LEFT_WRIST_ACTION_ELEMENTS,
    AM_DP123_LEFT_WRIST_TARGET_OFFSET_DEG,
    AM_DP123_RIGHT_WRIST_ACTION_ELEMENTS,
    AM_DP123_RIGHT_WRIST_TARGET_OFFSET_DEG,
    AM_DP123_TRIGGER_DEADZONE,
    am_dp123_trigger_to_gripper_command,
)


def test_action_layout_is_the_documented_18d_order():
    """The 18-D order is left wrist, right wrist, then the four hand joints."""
    assert len(AM_DP123_ACTION_LAYOUT) == AM_DP123_ABSOLUTE_IK_ACTION_DIM == 18
    assert AM_DP123_ACTION_LAYOUT[:7] == AM_DP123_LEFT_WRIST_ACTION_ELEMENTS
    assert AM_DP123_ACTION_LAYOUT[7:14] == AM_DP123_RIGHT_WRIST_ACTION_ELEMENTS
    assert AM_DP123_ACTION_LAYOUT[14:] == AM_DP123_HAND_JOINT_NAMES
    assert len(set(AM_DP123_ACTION_LAYOUT)) == 18


def test_wrist_elements_describe_position_and_quaternion():
    """Each wrist contributes xyz plus an XYZW quaternion."""
    for elements, prefix in ((AM_DP123_LEFT_WRIST_ACTION_ELEMENTS, "l"), (AM_DP123_RIGHT_WRIST_ACTION_ELEMENTS, "r")):
        assert elements[:3] == (f"{prefix}_px", f"{prefix}_py", f"{prefix}_pz")
        assert elements[3:] == (f"{prefix}_qx", f"{prefix}_qy", f"{prefix}_qz", f"{prefix}_qw")


def test_hand_elements_pair_the_jaws_of_each_hand():
    """Entry order must match the action-side index used by the hand mapping."""
    hand_elements = AM_DP123_ACTION_LAYOUT[14:]
    side_index = AM_DP123_HAND_ACTION_SIDE_INDEX
    for index, element in enumerate(hand_elements):
        expected_side = 0 if element in AM_DP123_LEFT_HAND_JOINT_NAMES else 1
        assert element in AM_DP123_HAND_JOINT_NAMES
        assert side_index[index] == expected_side, element
    assert hand_elements[0] == AM_DP123_LEFT_HAND_JOINT_NAMES[0]
    assert hand_elements[1] == AM_DP123_LEFT_HAND_JOINT_NAMES[1]
    assert hand_elements[2] == AM_DP123_RIGHT_HAND_JOINT_NAMES[0]
    assert hand_elements[3] == AM_DP123_RIGHT_HAND_JOINT_NAMES[1]


def test_hand_trigger_indices_keep_left_and_right_independent():
    """A closed right trigger must never be replaced by the open left trigger."""
    raw_triggers = (AM_DP123_HAND_TRIGGER_OPEN,) * 2 + (-AM_DP123_HAND_TRIGGER_OPEN,) * 2
    mapped_triggers = tuple(raw_triggers[index] for index in AM_DP123_HAND_ACTION_TRIGGER_INDEX)
    assert mapped_triggers[:2] == (AM_DP123_HAND_TRIGGER_OPEN,) * 2
    assert mapped_triggers[2:] == (-AM_DP123_HAND_TRIGGER_OPEN,) * 2


def test_pico_trigger_maps_proportionally_without_hand_tracking():
    """Released and pressed controller triggers span the complete hand command."""
    assert am_dp123_trigger_to_gripper_command(0.0) == 1.0
    assert am_dp123_trigger_to_gripper_command(AM_DP123_TRIGGER_DEADZONE) == 1.0
    assert am_dp123_trigger_to_gripper_command(1.0) == -1.0
    assert isclose(am_dp123_trigger_to_gripper_command((1.0 + AM_DP123_TRIGGER_DEADZONE) / 2.0), 0.0, abs_tol=1e-12)
    assert am_dp123_trigger_to_gripper_command(-1.0) == 1.0
    assert am_dp123_trigger_to_gripper_command(2.0) == -1.0


def test_retargeter_offsets_are_finite_degrees():
    """The PICO wrist offsets are finite degree triples for both sides."""
    for offsets in (AM_DP123_LEFT_WRIST_TARGET_OFFSET_DEG, AM_DP123_RIGHT_WRIST_TARGET_OFFSET_DEG):
        assert len(offsets) == 3
        assert all(isfinite(value) for value in offsets)


def test_idle_action_is_a_valid_raw_pico_action():
    """The idle action keeps unit quaternions and open hands."""
    assert len(AM_DP123_IDLE_ACTION) == AM_DP123_ABSOLUTE_IK_ACTION_DIM
    for offset in (0, 7):
        quaternion = AM_DP123_IDLE_ACTION[offset + 3 : offset + 7]
        assert isclose(sqrt(sum(value * value for value in quaternion)), 1.0, abs_tol=1e-6)
        assert all(isfinite(value) for value in AM_DP123_IDLE_ACTION[offset : offset + 7])
    assert list(AM_DP123_IDLE_ACTION[14:]) == [AM_DP123_HAND_TRIGGER_OPEN] * 4
