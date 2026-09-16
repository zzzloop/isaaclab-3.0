# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inverse-kinematics entry points for the AM-DP123 robot."""

from collections.abc import Sequence

import numpy as np

from amgg_robot_lab.contracts import AM_DP123_FRAMES, AM_DP123_IK_JOINT_NAMES

from .am_dp123_fk import get_am_dp123_kinematics
from .am_dp123_kinematics_model import AmDp123KinematicsError, AmDp123Pose
from .am_dp123_urdf_kinematics import IkTarget, quaternion_xyzw_to_matrix


def _pose_to_transform(pose: AmDp123Pose) -> np.ndarray:
    if pose.reference_frame != AM_DP123_FRAMES.base_link:
        raise AmDp123KinematicsError(
            f"AM-DP123 IK target must be expressed in '{AM_DP123_FRAMES.base_link}', got '{pose.reference_frame}'."
        )
    transform = np.eye(4)
    transform[:3, :3] = quaternion_xyzw_to_matrix(pose.quaternion_xyzw)
    transform[:3, 3] = pose.position_m
    return transform


def solve_am_dp123_inverse_kinematics(
    left_target: AmDp123Pose,
    right_target: AmDp123Pose,
    seed_joint_positions_rad: Sequence[float],
    *,
    allow_approximate: bool = False,
) -> tuple[float, ...]:
    """Solve limit-aware dual-arm IK from a seed configuration.

    The offline solver is the testable wrist-frame FK/IK reference for hardware.
    Isaac Lab runtime control uses Pink with the same URDF, joint order, and
    limits while targeting the fixed hand-base frames.

    Args:
        left_target: Desired left wrist pose in ``base_link``.
        right_target: Desired right wrist pose in ``base_link``.
        seed_joint_positions_rad: Initial positions [rad] in IK-joint order.
        allow_approximate: Return the closest iterate if strict tolerances fail.

    Returns:
        Joint-position solution [rad] in canonical IK-joint order.

    Raises:
        AmDp123KinematicsError: If the seed is invalid or IK does not converge.
    """
    if len(seed_joint_positions_rad) != len(AM_DP123_IK_JOINT_NAMES):
        raise AmDp123KinematicsError(f"Expected {len(AM_DP123_IK_JOINT_NAMES)} IK seed values.")
    seed = dict(zip(AM_DP123_IK_JOINT_NAMES, seed_joint_positions_rad, strict=True))
    result = get_am_dp123_kinematics().solve(
        (
            IkTarget(AM_DP123_FRAMES.left_wrist_link, _pose_to_transform(left_target)),
            IkTarget(AM_DP123_FRAMES.right_wrist_link, _pose_to_transform(right_target)),
        ),
        AM_DP123_IK_JOINT_NAMES,
        seed,
    )
    if not result.converged and not allow_approximate:
        raise AmDp123KinematicsError(
            "AM-DP123 IK did not converge: "
            f"position error={result.position_error_m:.6f} m, "
            f"orientation error={result.orientation_error_rad:.6f} rad."
        )
    return tuple(result.joint_positions[name] for name in AM_DP123_IK_JOINT_NAMES)
