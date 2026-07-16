# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inverse-kinematics entry points for the AMGG robot."""

from collections.abc import Sequence

from .amgg_kinematics_model import AmggKinematicsError, AmggPose


def solve_amgg_inverse_kinematics(
    left_target: AmggPose,
    right_target: AmggPose,
    seed_joint_positions_rad: Sequence[float],
) -> tuple[float, ...]:
    """Solve collision-aware dual-arm IK from a seed configuration.

    Args:
        left_target: Desired left tool pose.
        right_target: Desired right tool pose.
        seed_joint_positions_rad: Initial joint positions [rad] in canonical controlled-joint order.

    Returns:
        Joint-position solution [rad] in canonical controlled-joint order.
    """
    del left_target, right_target, seed_joint_positions_rad
    raise AmggKinematicsError("AMGG IK is pending the canonical URDF, limits, and TCP frames.")
