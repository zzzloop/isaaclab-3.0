# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Forward-kinematics entry points for the AMGG robot."""

from collections.abc import Sequence
from functools import lru_cache

from amgg_robot_lab.assets import AMGG_URDF_PATH
from amgg_robot_lab.contracts import AMGG_FRAMES, AMGG_OBSERVED_JOINT_NAMES

from .amgg_kinematics_model import AmggKinematicsError, AmggPose
from .amgg_urdf_kinematics import AmggUrdfKinematics, matrix_to_quaternion_xyzw


@lru_cache(maxsize=1)
def get_amgg_kinematics() -> AmggUrdfKinematics:
    """Return the cached normalized AMGG URDF model."""
    if not AMGG_URDF_PATH.is_file():
        raise AmggKinematicsError(f"AMGG URDF does not exist: {AMGG_URDF_PATH}")
    return AmggUrdfKinematics(AMGG_URDF_PATH)


def compute_amgg_forward_kinematics(joint_positions_rad: Sequence[float]) -> dict[str, AmggPose]:
    """Compute AMGG TCP poses from observed joint positions.

    Args:
        joint_positions_rad: Joint positions [rad] in canonical observed-joint order.

    Returns:
        Mapping from left and right TCP frame names to root-frame poses.
    """
    if len(joint_positions_rad) != len(AMGG_OBSERVED_JOINT_NAMES):
        raise AmggKinematicsError(
            f"Expected {len(AMGG_OBSERVED_JOINT_NAMES)} observed joints, got {len(joint_positions_rad)}."
        )
    positions = dict(zip(AMGG_OBSERVED_JOINT_NAMES, joint_positions_rad, strict=True))
    model = get_amgg_kinematics()
    result: dict[str, AmggPose] = {}
    for frame_name in (AMGG_FRAMES.left_tcp_link, AMGG_FRAMES.right_tcp_link):
        transform = model.forward(frame_name, positions)
        result[frame_name] = AmggPose(
            position_m=tuple(float(value) for value in transform[:3, 3]),
            quaternion_xyzw=matrix_to_quaternion_xyzw(transform[:3, :3]),
            reference_frame=AMGG_FRAMES.base_link,
        )
    return result
