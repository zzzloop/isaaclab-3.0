# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Public AM-DP123 joint ABI, home pose, and hand travel contract."""

from __future__ import annotations

from math import isclose

import pytest

from amgg_robot_lab.contracts import (
    AM_DP123_ABSOLUTE_IK_ACTION_DIM,
    AM_DP123_CONTROLLED_JOINT_NAMES,
    AM_DP123_HAND_ACTION_SIDE_INDEX,
    AM_DP123_HAND_CLOSED_POSITIONS,
    AM_DP123_HAND_JOINT_NAMES,
    AM_DP123_HAND_OPEN_POSITIONS,
    AM_DP123_HAND_TRIGGER_CLOSED,
    AM_DP123_HAND_TRIGGER_OPEN,
    AM_DP123_IK_JOINT_NAMES,
    AM_DP123_JOINT_POSITION_ACTION_DIM,
    AM_DP123_JOINT_SPECS,
    AM_DP123_LEFT_HAND_JOINT_NAMES,
    AM_DP123_RIGHT_HAND_JOINT_NAMES,
    AM_DP123_STATE_DIM,
    AM_DP123_STATE_JOINT_NAMES,
    am_dp123_hand_closed_fraction,
    am_dp123_hand_targets,
    require_am_dp123_joint_contract,
    validate_am_dp123_joint_names,
)
from amgg_robot_lab.contracts.am_dp123_joint_contract import AmDp123JointSpec


def test_contract_validates_and_keeps_declared_dimensions():
    """The 23-D state / 18-D command ABI is the public contract."""
    require_am_dp123_joint_contract()
    assert AM_DP123_STATE_DIM == 23
    assert len(AM_DP123_STATE_JOINT_NAMES) == 23
    assert AM_DP123_JOINT_POSITION_ACTION_DIM == 18
    assert AM_DP123_CONTROLLED_JOINT_NAMES == AM_DP123_IK_JOINT_NAMES + AM_DP123_HAND_JOINT_NAMES
    assert AM_DP123_ABSOLUTE_IK_ACTION_DIM == 7 * 2 + len(AM_DP123_HAND_JOINT_NAMES)


def test_joint_names_are_unique_and_non_empty():
    """Duplicate or empty joint names must be rejected before they reach a backend."""
    assert len(set(AM_DP123_STATE_JOINT_NAMES)) == 23
    assert all(AM_DP123_STATE_JOINT_NAMES)
    assert validate_am_dp123_joint_names(["a", "b"]) == ("a", "b")
    with pytest.raises(ValueError):
        validate_am_dp123_joint_names(["a", "a"])
    with pytest.raises(ValueError):
        validate_am_dp123_joint_names([""])


def test_home_pose_is_inside_limits_with_margin():
    """The shipped home pose is valid and leaves solver margin on the Pink joints."""
    for spec in AM_DP123_JOINT_SPECS:
        assert spec.lower_limit_rad <= spec.home_position_rad <= spec.upper_limit_rad
    for spec in (item for item in AM_DP123_JOINT_SPECS if item.group.endswith("_arm")):
        margin = min(spec.home_position_rad - spec.lower_limit_rad, spec.upper_limit_rad - spec.home_position_rad)
        assert margin > 0.05, f"{spec.name} home pose is too close to a limit"
    assert any(spec.home_position_rad != 0.0 for spec in AM_DP123_JOINT_SPECS)


def test_only_the_arms_leave_the_zero_pose():
    """Waist, head, and hands hold the URDF zero pose; the arms hold the ready pose."""
    for spec in AM_DP123_JOINT_SPECS:
        if spec.group in {"waist", "head", "left_hand", "right_hand"}:
            assert spec.home_position_rad == 0.0
    arm_homes = [spec.home_position_rad for spec in AM_DP123_JOINT_SPECS if spec.group.endswith("_arm")]
    assert sum(home != 0.0 for home in arm_homes) >= 10


def test_joint_spec_rejects_invalid_limits():
    """Limit validation guards every backend that imports the contract."""
    with pytest.raises(ValueError):
        AmDp123JointSpec("joint", "arm", 1.0, 0.0, 0.5, 1.0, 1.0).validate()
    with pytest.raises(ValueError):
        AmDp123JointSpec("joint", "arm", -1.0, 1.0, 2.0, 1.0, 1.0).validate()
    with pytest.raises(ValueError):
        AmDp123JointSpec("joint", "arm", -1.0, 1.0, 0.0, 0.0, 1.0).validate()
    with pytest.raises(ValueError):
        AmDp123JointSpec("joint", "arm", -1.0, 1.0, 0.0, 1.0, float("inf")).validate()


def test_hand_travel_spans_the_one_sided_limits():
    """Open is the URDF zero pose; closed reaches the one-sided limits."""
    for name in AM_DP123_HAND_JOINT_NAMES:
        assert AM_DP123_HAND_OPEN_POSITIONS[name] == 0.0
    spec_by_name = {spec.name: spec for spec in AM_DP123_JOINT_SPECS}
    for name in AM_DP123_HAND_JOINT_NAMES:
        closed = AM_DP123_HAND_CLOSED_POSITIONS[name]
        spec = spec_by_name[name]
        assert isclose(closed, spec.lower_limit_rad, abs_tol=1e-12) or isclose(
            closed, spec.upper_limit_rad, abs_tol=1e-12
        )
    assert AM_DP123_HAND_CLOSED_POSITIONS["left_arm_hand_joint1_0"] < 0.0
    assert AM_DP123_HAND_CLOSED_POSITIONS["left_arm_hand_joint2_0"] > 0.0


def test_hand_targets_interpolate_and_clamp():
    """Closure interpolation is monotonic, symmetric, and clamped to [0, 1]."""
    opened = am_dp123_hand_targets(0.0)
    closed = am_dp123_hand_targets(1.0)
    half = am_dp123_hand_targets(0.5)
    assert opened == {name: 0.0 for name in AM_DP123_HAND_JOINT_NAMES}
    assert closed == AM_DP123_HAND_CLOSED_POSITIONS
    for name in AM_DP123_HAND_JOINT_NAMES:
        assert isclose(half[name], closed[name] / 2.0, abs_tol=1e-12)
    assert am_dp123_hand_targets(-1.0) == opened
    assert am_dp123_hand_targets(4.0) == closed
    # Jaw pairs move symmetrically so that the mimic relation ``jaw2 = -jaw1`` holds.
    left_a, left_b = AM_DP123_LEFT_HAND_JOINT_NAMES
    right_a, right_b = AM_DP123_RIGHT_HAND_JOINT_NAMES
    for fraction in (0.0, 0.25, 0.75, 1.0):
        targets = am_dp123_hand_targets(fraction)
        assert isclose(targets[left_b], -targets[left_a], abs_tol=1e-12)
        assert isclose(targets[right_b], -targets[right_a], abs_tol=1e-12)
        assert isclose(targets[left_a], targets[right_a], abs_tol=1e-12)
    with pytest.raises(ValueError):
        am_dp123_hand_targets(float("nan"))


def test_trigger_convention_matches_the_gripper_retargeter():
    """``+1`` opens, ``-1`` closes, and out-of-range triggers clamp."""
    assert am_dp123_hand_closed_fraction(AM_DP123_HAND_TRIGGER_OPEN) == 0.0
    assert am_dp123_hand_closed_fraction(AM_DP123_HAND_TRIGGER_CLOSED) == 1.0
    assert isclose(am_dp123_hand_closed_fraction(0.0), 0.5, abs_tol=1e-12)
    assert am_dp123_hand_closed_fraction(2.0) == 0.0
    assert am_dp123_hand_closed_fraction(-2.0) == 1.0
    with pytest.raises(ValueError):
        am_dp123_hand_closed_fraction(float("inf"))
    assert am_dp123_hand_closed_fraction(AM_DP123_HAND_TRIGGER_OPEN) == 0.0
    assert AM_DP123_HAND_ACTION_SIDE_INDEX == (0, 0, 1, 1)
