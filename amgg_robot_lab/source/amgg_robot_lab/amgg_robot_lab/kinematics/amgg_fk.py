# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Forward-kinematics entry points for the AMGG robot."""

from collections.abc import Sequence

from .amgg_kinematics_model import AmggKinematicsError, AmggPose


def compute_amgg_forward_kinematics(joint_positions_rad: Sequence[float]) -> dict[str, AmggPose]:
    """Compute configured AMGG tool poses from joint positions.

    Args:
        joint_positions_rad: Joint positions [rad] in canonical observed-joint order.

    Returns:
        Mapping from configured tool-frame name to pose.
    """
    del joint_positions_rad
    raise AmggKinematicsError("AMGG FK is pending the canonical URDF and frame contract.")
