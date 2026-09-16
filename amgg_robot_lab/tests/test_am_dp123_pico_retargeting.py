# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline tests for PICO clutch retargeting and hand-frame IK targets."""

from __future__ import annotations

from math import cos, pi, sin

import numpy as np
import pytest

from amgg_robot_lab.assets import AM_DP123_BASE_SPAWN_HEIGHT_M
from amgg_robot_lab.contracts import AM_DP123_FRAMES, AM_DP123_HOME_POSITIONS, AM_DP123_IK_JOINT_NAMES
from amgg_robot_lab.kinematics import get_am_dp123_kinematics
from amgg_robot_lab.kinematics.am_dp123_urdf_kinematics import IkTarget, matrix_to_quaternion_xyzw
from amgg_robot_lab.teleop import (
    AM_DP123_LEFT_HAND_HOME_POSE,
    AM_DP123_RIGHT_HAND_HOME_POSE,
    AmDp123ControllerClutch,
    rebase_am_dp123_controller_pose,
)


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.array(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        )
    )


def test_hand_home_poses_are_derived_from_the_urdf():
    """Configured clutch homes exactly match URDF FK plus the base spawn height."""
    model = get_am_dp123_kinematics()
    joint_positions = dict(AM_DP123_HOME_POSITIONS)
    for frame, configured in (
        (AM_DP123_FRAMES.left_hand_base_link, AM_DP123_LEFT_HAND_HOME_POSE),
        (AM_DP123_FRAMES.right_hand_base_link, AM_DP123_RIGHT_HAND_HOME_POSE),
    ):
        transform = model.forward(frame, joint_positions)
        expected_position = transform[:3, 3] + np.array((0.0, 0.0, AM_DP123_BASE_SPAWN_HEIGHT_M))
        np.testing.assert_allclose(configured[:3], expected_position, atol=1.0e-10)
        np.testing.assert_allclose(configured[3:], matrix_to_quaternion_xyzw(transform[:3, :3]), atol=1.0e-10)


def test_controller_origin_maps_to_home_without_an_engage_jump():
    """An arbitrary controller room pose maps exactly to hand home on Play."""
    controller = np.array((1.3, -0.4, 1.1, 0.2, -0.3, 0.1, 0.9))
    target = rebase_am_dp123_controller_pose(controller, controller, np.asarray(AM_DP123_LEFT_HAND_HOME_POSE))
    np.testing.assert_allclose(target, AM_DP123_LEFT_HAND_HOME_POSE, atol=1.0e-12)


def test_controller_delta_maps_translation_and_world_rotation():
    """Controller translation and orientation deltas are applied to the hand home."""
    origin = np.array((0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0))
    half_angle = pi / 12.0
    delta_quaternion = np.array((0.0, 0.0, sin(half_angle), cos(half_angle)))
    controller = np.concatenate((origin[:3] + np.array((0.04, -0.03, 0.02)), delta_quaternion))
    home = np.asarray(AM_DP123_RIGHT_HAND_HOME_POSE)
    target = rebase_am_dp123_controller_pose(controller, origin, home)
    np.testing.assert_allclose(target[:3], home[:3] + (0.04, -0.03, 0.02), atol=1.0e-12)
    expected_quaternion = _quat_multiply(delta_quaternion, home[3:])
    if np.dot(target[3:], expected_quaternion) < 0.0:
        expected_quaternion *= -1.0
    np.testing.assert_allclose(target[3:], expected_quaternion, atol=1.0e-12)


def test_clutch_holds_dropouts_and_reengages_without_a_jump():
    """Stop, tracking loss, and re-Play preserve the last commanded hand pose."""
    home = np.asarray(AM_DP123_LEFT_HAND_HOME_POSE)
    clutch = AmDp123ControllerClutch(AM_DP123_LEFT_HAND_HOME_POSE)
    first = np.array((0.2, 0.1, 1.2, 0.0, 0.0, 0.0, 1.0))
    np.testing.assert_allclose(clutch.update(first, running=True), home)

    moved = first.copy()
    moved[:3] += (0.03, 0.02, -0.01)
    moved_target = clutch.update(moved, running=True)
    np.testing.assert_allclose(clutch.update(None, running=True), moved_target)
    np.testing.assert_allclose(clutch.update(moved, running=False), moved_target)

    repositioned = np.array((-0.3, 0.4, 0.8, 0.0, 0.0, 1.0, 0.0))
    np.testing.assert_allclose(clutch.update(repositioned, running=True), moved_target)
    continued = repositioned.copy()
    continued[0] += 0.02
    np.testing.assert_allclose(clutch.update(continued, running=True)[:3], moved_target[:3] + (0.02, 0.0, 0.0))

    np.testing.assert_allclose(clutch.update(first, running=True, reset=True), home)


@pytest.mark.parametrize("invalid_quaternion", [(0.0, 0.0, 0.0, 0.0), (np.nan, 0.0, 0.0, 1.0)])
def test_clutch_holds_invalid_controller_orientation(invalid_quaternion):
    """Invalid tracking data never reaches the IK action."""
    clutch = AmDp123ControllerClutch(AM_DP123_RIGHT_HAND_HOME_POSE)
    valid = np.array((0.1, -0.2, 1.0, 0.0, 0.0, 0.0, 1.0))
    held = clutch.update(valid, running=True)
    invalid = np.concatenate((valid[:3], invalid_quaternion))
    np.testing.assert_allclose(clutch.update(invalid, running=True), held)


def test_small_dual_hand_motion_is_reachable_from_home():
    """The URDF solver reaches translated and rotated targets of both hand-base frames."""
    model = get_am_dp123_kinematics()
    home = dict(AM_DP123_HOME_POSITIONS)
    targets = []
    angle = 0.05
    left_rotation_delta = np.array(((cos(angle), -sin(angle), 0.0), (sin(angle), cos(angle), 0.0), (0.0, 0.0, 1.0)))
    right_rotation_delta = np.array(
        ((cos(-angle), 0.0, sin(-angle)), (0.0, 1.0, 0.0), (-sin(-angle), 0.0, cos(-angle)))
    )
    for frame, delta, rotation_delta in (
        (AM_DP123_FRAMES.left_hand_base_link, (0.01, -0.005, 0.01), left_rotation_delta),
        (AM_DP123_FRAMES.right_hand_base_link, (0.01, 0.005, 0.01), right_rotation_delta),
    ):
        transform = model.forward(frame, home)
        transform[:3, 3] += delta
        transform[:3, :3] = rotation_delta @ transform[:3, :3]
        targets.append(IkTarget(frame, transform))
    result = model.solve(targets, AM_DP123_IK_JOINT_NAMES, home)
    assert result.converged, result
    for target in targets:
        achieved = model.forward(target.link_name, result.joint_positions)
        np.testing.assert_allclose(achieved[:3, 3], target.transform[:3, 3], atol=1.0e-4)
        np.testing.assert_allclose(achieved[:3, :3], target.transform[:3, :3], atol=1.0e-3)
