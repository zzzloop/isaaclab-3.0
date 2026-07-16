# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Names of the built-in recorder terms expected in every AMGG episode."""

AMGG_RECORDED_HDF5_KEYS = (
    "actions",
    "processed_actions",
    "obs/robot_joint_pos",
    "obs/robot_joint_vel",
    "obs/left_tcp_pose",
    "obs/right_tcp_pose",
    "obs/object_state",
    "obs/goal",
    "obs/progress",
    "obs/image_head",
    "obs/image_left_wrist",
    "obs/image_right_wrist",
    "obs/image_overview",
)


def build_amgg_recorder_terms() -> tuple[str, ...]:
    """Return the HDF5 datasets produced by official AMGG recording."""
    return AMGG_RECORDED_HDF5_KEYS
