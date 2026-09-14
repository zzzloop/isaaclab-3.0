# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Forward-kinematics entry points for the AM-DP123 robot."""

from collections.abc import Sequence
from functools import lru_cache

from amgg_robot_lab.assets import AM_DP123_URDF_PATH
from amgg_robot_lab.contracts import AM_DP123_FRAMES, AM_DP123_STATE_JOINT_NAMES

from .am_dp123_kinematics_model import AmDp123KinematicsError, AmDp123Pose
from .am_dp123_urdf_kinematics import AmDp123UrdfKinematics, matrix_to_quaternion_xyzw


@lru_cache(maxsize=1)
def get_am_dp123_kinematics() -> AmDp123UrdfKinematics:
    """Return the cached packaged AM-DP123 URDF model."""
    if not AM_DP123_URDF_PATH.is_file():
        raise AmDp123KinematicsError(f"AM-DP123 URDF does not exist: {AM_DP123_URDF_PATH}")
    return AmDp123UrdfKinematics(AM_DP123_URDF_PATH)


def compute_am_dp123_forward_kinematics(joint_positions_rad: Sequence[float]) -> dict[str, AmDp123Pose]:
    """Compute the wrist poses of the AM-DP123 for one joint configuration.

    Args:
        joint_positions_rad: Joint positions [rad] in canonical state-joint order.

    Returns:
        Mapping from left and right wrist frame names to ``base_link`` poses.
    """
    if len(joint_positions_rad) != len(AM_DP123_STATE_JOINT_NAMES):
        raise AmDp123KinematicsError(
            f"Expected {len(AM_DP123_STATE_JOINT_NAMES)} state joints, got {len(joint_positions_rad)}."
        )
    positions = dict(zip(AM_DP123_STATE_JOINT_NAMES, joint_positions_rad, strict=True))
    model = get_am_dp123_kinematics()
    result: dict[str, AmDp123Pose] = {}
    for frame_name in (AM_DP123_FRAMES.left_wrist_link, AM_DP123_FRAMES.right_wrist_link):
        transform = model.forward(frame_name, positions)
        result[frame_name] = AmDp123Pose(
            position_m=tuple(float(value) for value in transform[:3, 3]),
            quaternion_xyzw=matrix_to_quaternion_xyzw(transform[:3, :3]),
            reference_frame=AM_DP123_FRAMES.base_link,
        )
    return result
