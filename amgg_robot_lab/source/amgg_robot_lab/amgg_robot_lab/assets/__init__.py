# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AM-DP123 robot asset configuration."""

from .am_dp123_robot_cfg import (
    AM_DP123_ASSET_DATA_DIR,
    AM_DP123_BASE_SPAWN_HEIGHT_M,
    AM_DP123_MESH_DIR,
    AM_DP123_ROS_PACKAGE_NAME,
    AM_DP123_URDF_PATH,
    get_am_dp123_robot_cfg,
)

__all__ = [
    "AM_DP123_ASSET_DATA_DIR",
    "AM_DP123_BASE_SPAWN_HEIGHT_M",
    "AM_DP123_MESH_DIR",
    "AM_DP123_ROS_PACKAGE_NAME",
    "AM_DP123_URDF_PATH",
    "get_am_dp123_robot_cfg",
]
