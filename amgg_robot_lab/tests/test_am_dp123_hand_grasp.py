# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AM-DP123 hand grasp ABI: trigger, travel, and joint limits."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from math import isclose

import pytest

from amgg_robot_lab.assets import AM_DP123_URDF_PATH
from amgg_robot_lab.contracts import (
    AM_DP123_HAND_CLOSED_POSITIONS,
    AM_DP123_HAND_JOINT_NAMES,
    AM_DP123_HAND_OPEN_POSITIONS,
    AM_DP123_LEFT_HAND_JOINT_NAMES,
    AM_DP123_RIGHT_HAND_JOINT_NAMES,
    am_dp123_hand_closed_fraction,
    am_dp123_hand_targets,
)

URDF_JOINTS = {joint.get("name"): joint for joint in ET.parse(AM_DP123_URDF_PATH).getroot().findall("joint")}


def _limits(name: str) -> tuple[float, float]:
    limit = URDF_JOINTS[name].find("limit")
    return float(limit.get("lower")), float(limit.get("upper"))


def test_every_closure_stays_inside_the_urdf_limits():
    """Sweeping the trigger from open to closed never leaves the joint range."""
    for step in range(21):
        for name, target in am_dp123_hand_targets(step / 20.0).items():
            lower, upper = _limits(name)
            assert lower - 1e-12 <= target <= upper + 1e-12, name


def test_full_closure_reaches_the_urdf_limits():
    """A fully closed hand sits exactly on the one-sided URDF limits."""
    closed = am_dp123_hand_targets(1.0)
    for name, target in closed.items():
        lower, upper = _limits(name)
        assert isclose(target, lower, abs_tol=1e-12) or isclose(target, upper, abs_tol=1e-12), name
    assert set(closed) == set(AM_DP123_HAND_JOINT_NAMES)
    assert len(closed) == 4


def test_jaws_close_symmetrically_about_the_open_pose():
    """The two jaws of a hand mirror each other, matching the upstream mimic law."""
    for fraction in (0.0, 0.1, 0.5, 0.9, 1.0):
        targets = am_dp123_hand_targets(fraction)
        left_a, left_b = (targets[name] for name in AM_DP123_LEFT_HAND_JOINT_NAMES)
        right_a, right_b = (targets[name] for name in AM_DP123_RIGHT_HAND_JOINT_NAMES)
        assert isclose(left_b, -left_a, abs_tol=1e-12)
        assert isclose(right_b, -right_a, abs_tol=1e-12)
        assert isclose(left_a, right_a, abs_tol=1e-12)


def test_trigger_pipeline_maps_open_and_closed_hands():
    """The trigger-to-target pipeline is the same mapping the action term applies."""
    for trigger, fraction in ((1.0, 0.0), (0.5, 0.25), (0.0, 0.5), (-0.5, 0.75), (-1.0, 1.0)):
        assert isclose(am_dp123_hand_closed_fraction(trigger), fraction, abs_tol=1e-12)
        targets = am_dp123_hand_targets(am_dp123_hand_closed_fraction(trigger))
        for name in AM_DP123_HAND_JOINT_NAMES:
            expected = AM_DP123_HAND_OPEN_POSITIONS[name] + fraction * (
                AM_DP123_HAND_CLOSED_POSITIONS[name] - AM_DP123_HAND_OPEN_POSITIONS[name]
            )
            assert isclose(targets[name], expected, abs_tol=1e-12), name
    assert am_dp123_hand_targets(am_dp123_hand_closed_fraction(1.0)) == dict(AM_DP123_HAND_OPEN_POSITIONS)


def test_closure_only_touches_the_hand_joints():
    """The grasp mapping must never emit arm, waist, or head targets."""
    targets = am_dp123_hand_targets(0.5)
    assert set(targets) == set(AM_DP123_HAND_JOINT_NAMES)
    with pytest.raises(ValueError):
        am_dp123_hand_targets(float("-inf"))
