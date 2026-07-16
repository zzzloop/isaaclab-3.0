# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inverse-kinematics entry points for the AMGG robot."""

from collections.abc import Sequence

import numpy as np

from amgg_robot_lab.contracts import AMGG_FRAMES, AMGG_IK_JOINT_NAMES

from .amgg_fk import get_amgg_kinematics
from .amgg_kinematics_model import AmggKinematicsError, AmggPose
from .amgg_urdf_kinematics import IkTarget, quaternion_xyzw_to_matrix


def _pose_to_transform(pose: AmggPose) -> np.ndarray:
    if pose.reference_frame != AMGG_FRAMES.base_link:
        raise AmggKinematicsError(
            f"AMGG IK target must be expressed in '{AMGG_FRAMES.base_link}', got '{pose.reference_frame}'."
        )
    transform = np.eye(4)
    transform[:3, :3] = quaternion_xyzw_to_matrix(pose.quaternion_xyzw)
    transform[:3, 3] = pose.position_m
    return transform


def solve_amgg_inverse_kinematics(
    left_target: AmggPose,
    right_target: AmggPose,
    seed_joint_positions_rad: Sequence[float],
    *,
    allow_approximate: bool = False,
) -> tuple[float, ...]:
    """Solve limit-aware dual-arm IK from a seed configuration.

    The offline solver is the testable FK/IK reference for hardware. Isaac Lab
    runtime control uses Pink with the same URDF, joint order, limits, and TCPs.

    Args:
        left_target: Desired left TCP pose in ``base_link``.
        right_target: Desired right TCP pose in ``base_link``.
        seed_joint_positions_rad: Initial positions [rad] in IK-joint order.
        allow_approximate: Return the closest iterate if strict tolerances fail.

    Returns:
        Joint-position solution [rad] in canonical IK-joint order.

    Raises:
        AmggKinematicsError: If the seed is invalid or IK does not converge.
    """
    if len(seed_joint_positions_rad) != len(AMGG_IK_JOINT_NAMES):
        raise AmggKinematicsError(f"Expected {len(AMGG_IK_JOINT_NAMES)} IK seed values.")
    seed = dict(zip(AMGG_IK_JOINT_NAMES, seed_joint_positions_rad, strict=True))
    result = get_amgg_kinematics().solve(
        (
            IkTarget(AMGG_FRAMES.left_tcp_link, _pose_to_transform(left_target)),
            IkTarget(AMGG_FRAMES.right_tcp_link, _pose_to_transform(right_target)),
        ),
        AMGG_IK_JOINT_NAMES,
        seed,
    )
    if not result.converged and not allow_approximate:
        raise AmggKinematicsError(
            "AMGG IK did not converge: "
            f"position error={result.position_error_m:.6f} m, "
            f"orientation error={result.orientation_error_rad:.6f} rad."
        )
    return tuple(result.joint_positions[name] for name in AMGG_IK_JOINT_NAMES)
