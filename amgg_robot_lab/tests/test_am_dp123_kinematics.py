# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline URDF FK, Jacobian, and dual-arm IK reference for AM-DP123."""

from __future__ import annotations

from math import atan2, isclose, sqrt

import numpy as np
import pytest

from amgg_robot_lab.assets import AM_DP123_BASE_SPAWN_HEIGHT_M
from amgg_robot_lab.contracts import (
    AM_DP123_FRAMES,
    AM_DP123_HAND_JOINT_NAMES,
    AM_DP123_HOME_POSITIONS,
    AM_DP123_IK_JOINT_NAMES,
    AM_DP123_STATE_JOINT_NAMES,
)
from amgg_robot_lab.kinematics import (
    AmDp123KinematicsError,
    AmDp123Pose,
    compute_am_dp123_forward_kinematics,
    get_am_dp123_kinematics,
    solve_am_dp123_inverse_kinematics,
)
from amgg_robot_lab.kinematics.am_dp123_urdf_kinematics import (
    IkTarget,
    matrix_to_quaternion_xyzw,
    quaternion_xyzw_to_matrix,
)
from amgg_robot_lab.teleop import AM_DP123_IDLE_ACTION

HOME_POSE = dict(AM_DP123_HOME_POSITIONS)
WRIST_TARGET_M = (0.36, 0.20, 0.88)


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """Return the axis-angle vector [rad] of a rotation matrix."""
    x, y, z, w = matrix_to_quaternion_xyzw(rotation)
    norm = sqrt(x * x + y * y + z * z)
    if norm < 1e-12:
        return np.zeros(3)
    angle = 2.0 * atan2(norm, abs(w))
    return (1.0 if w >= 0.0 else -1.0) * angle * np.array((x, y, z)) / norm


def test_model_topology_matches_the_asset():
    """The packaged model exposes the shipped 61 links and 31 movable joints."""
    model = get_am_dp123_kinematics()
    assert model.root_link == AM_DP123_FRAMES.base_link
    assert len(model.link_names) == 61
    assert len(model.joints) == 60
    assert len(model.actuated_joint_names) == 31
    assert set(AM_DP123_STATE_JOINT_NAMES).issubset(model.actuated_joint_names)
    with pytest.raises(AmDp123KinematicsError):
        model.forward("not_a_link", {})


def test_home_pose_puts_both_wrists_in_front_of_the_robot():
    """The shipped arm home pose solves to the documented wrist positions."""
    model = get_am_dp123_kinematics()
    for side, sign in (("left", 1.0), ("right", -1.0)):
        transform = model.forward(f"{side}_arm_link7", HOME_POSE)
        np.testing.assert_allclose(
            transform[:3, 3],
            (WRIST_TARGET_M[0], sign * WRIST_TARGET_M[1], WRIST_TARGET_M[2]),
            atol=1e-5,
            err_msg=side,
        )
        # The hand extends along +y of the left wrist frame and -y of the right one.
        hand_axis = (-1.0 if side == "left" else 1.0) * transform[:3, :3] @ np.array((0.0, 1.0, 0.0))
        assert hand_axis[2] < -0.3, f"{side} hand should point downwards at the home pose"
        assert hand_axis[0] > 0.3, f"{side} hand should point forwards at the home pose"


def test_forward_kinematics_api_validates_the_state_dimension():
    """The public FK entry point only accepts the 23-D state contract."""
    poses = compute_am_dp123_forward_kinematics([AM_DP123_HOME_POSITIONS[name] for name in AM_DP123_STATE_JOINT_NAMES])
    assert set(poses) == {AM_DP123_FRAMES.left_wrist_link, AM_DP123_FRAMES.right_wrist_link}
    for pose in poses.values():
        assert pose.reference_frame == AM_DP123_FRAMES.base_link
        assert isclose(sqrt(sum(value * value for value in pose.quaternion_xyzw)), 1.0, abs_tol=1e-9)
    with pytest.raises(AmDp123KinematicsError):
        compute_am_dp123_forward_kinematics([0.0] * 22)


def test_idle_action_holds_the_home_pose():
    """The published 18-D idle action is the world-frame home wrist pose plus open hands."""
    model = get_am_dp123_kinematics()
    assert len(AM_DP123_IDLE_ACTION) == 18
    for index, side in enumerate(("left", "right")):
        transform = model.forward(f"{side}_arm_link7", HOME_POSE)
        offset = index * 7
        expected_position = transform[:3, 3] + np.array((0.0, 0.0, AM_DP123_BASE_SPAWN_HEIGHT_M))
        np.testing.assert_allclose(
            AM_DP123_IDLE_ACTION[offset : offset + 3], expected_position, atol=1e-4, err_msg=side
        )
        np.testing.assert_allclose(
            AM_DP123_IDLE_ACTION[offset + 3 : offset + 7],
            matrix_to_quaternion_xyzw(transform[:3, :3]),
            atol=1e-6,
            err_msg=side,
        )
    assert list(AM_DP123_IDLE_ACTION[14:]) == [1.0, 1.0, 1.0, 1.0]


def test_geometric_jacobian_matches_numerical_differentiation():
    """Analytic Jacobian columns agree with central differences of the forward model."""
    model = get_am_dp123_kinematics()
    joint_names = list(AM_DP123_IK_JOINT_NAMES)
    epsilon = 1e-6
    for link_name in (AM_DP123_FRAMES.left_wrist_link, AM_DP123_FRAMES.right_wrist_link):
        jacobian = model.geometric_jacobian(link_name, joint_names, HOME_POSE)
        for index, name in enumerate(joint_names):
            forward = dict(HOME_POSE, **{name: HOME_POSE[name] + epsilon})
            backward = dict(HOME_POSE, **{name: HOME_POSE[name] - epsilon})
            forward_transform = model.forward(link_name, forward)
            backward_transform = model.forward(link_name, backward)
            linear = (forward_transform[:3, 3] - backward_transform[:3, 3]) / (2.0 * epsilon)
            angular = _rotation_vector(forward_transform[:3, :3] @ backward_transform[:3, :3].T) / (2.0 * epsilon)
            np.testing.assert_allclose(jacobian[:3, index], linear, atol=1e-5, err_msg=f"{link_name}/{name}")
            np.testing.assert_allclose(jacobian[3:, index], angular, atol=1e-5, err_msg=f"{link_name}/{name}")


def test_inverse_kinematics_recovers_a_perturbed_target():
    """The dual-arm DLS solver returns an in-limit configuration that reaches the target."""
    model = get_am_dp123_kinematics()
    seed = tuple(HOME_POSE[name] for name in AM_DP123_IK_JOINT_NAMES)
    poses = compute_am_dp123_forward_kinematics([AM_DP123_HOME_POSITIONS[name] for name in AM_DP123_STATE_JOINT_NAMES])
    left = poses[AM_DP123_FRAMES.left_wrist_link]
    right = poses[AM_DP123_FRAMES.right_wrist_link]
    left_target = AmDp123Pose(
        position_m=(left.position_m[0] + 0.01, left.position_m[1] - 0.01, left.position_m[2] + 0.01),
        quaternion_xyzw=left.quaternion_xyzw,
        reference_frame=AM_DP123_FRAMES.base_link,
    )
    solution = solve_am_dp123_inverse_kinematics(left_target, right, seed)
    assert len(solution) == len(AM_DP123_IK_JOINT_NAMES)
    positions = dict(zip(AM_DP123_IK_JOINT_NAMES, solution, strict=True))
    for name in AM_DP123_IK_JOINT_NAMES:
        joint = model.joint_by_name[name]
        assert joint.lower <= positions[name] <= joint.upper, name
    achieved = model.forward(AM_DP123_FRAMES.left_wrist_link, positions)
    np.testing.assert_allclose(achieved[:3, 3], left_target.position_m, atol=1e-4)
    np.testing.assert_allclose(achieved[:3, :3], quaternion_xyzw_to_matrix(left.quaternion_xyzw), atol=1e-3)


def test_inverse_kinematics_entry_point_validates_its_arguments():
    """Wrong seed length and foreign reference frames are rejected."""
    model = get_am_dp123_kinematics()
    zero_pose = AmDp123Pose((0.36, 0.20, 0.88), (0.0, 0.0, 0.0, 1.0), AM_DP123_FRAMES.base_link)
    with pytest.raises(AmDp123KinematicsError):
        solve_am_dp123_inverse_kinematics(zero_pose, zero_pose, (0.0,) * 13)
    foreign = AmDp123Pose((0.36, 0.20, 0.88), (0.0, 0.0, 0.0, 1.0), "left_arm_link7")
    with pytest.raises(AmDp123KinematicsError):
        solve_am_dp123_inverse_kinematics(foreign, zero_pose, (0.0,) * 14)
    with pytest.raises(AmDp123KinematicsError):
        model.solve([IkTarget(AM_DP123_FRAMES.left_wrist_link, np.eye(4))], ("not_a_joint",), {})


def test_hand_joint_names_are_not_part_of_the_ik_set():
    """Hand, waist, and head joints must stay out of the Pink IK set."""
    assert len(AM_DP123_IK_JOINT_NAMES) == 14
    assert set(AM_DP123_HAND_JOINT_NAMES).isdisjoint(AM_DP123_IK_JOINT_NAMES)
    assert set(AM_DP123_IK_JOINT_NAMES).issubset(AM_DP123_STATE_JOINT_NAMES)
