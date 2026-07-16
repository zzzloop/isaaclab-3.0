# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stable interfaces shared by AMGG simulation, datasets, and hardware."""

from .amgg_camera_contract import AMGG_CAMERA_BY_NAME, AMGG_CAMERAS, AmggCameraSpec, require_amgg_camera_contract
from .amgg_frame_contract import AMGG_FRAMES, AmggFrameContract
from .amgg_joint_contract import (
    AMGG_ABSOLUTE_IK_ACTION_DIM,
    AMGG_CONTROLLED_JOINT_NAMES,
    AMGG_GRIPPER_JOINT_NAMES,
    AMGG_HEAD_JOINT_NAMES,
    AMGG_HOME_POSITIONS,
    AMGG_IK_JOINT_NAMES,
    AMGG_JOINT_POSITION_ACTION_DIM,
    AMGG_JOINT_SPECS,
    AMGG_LEFT_ARM_JOINT_NAMES,
    AMGG_OBSERVED_JOINT_NAMES,
    AMGG_RIGHT_ARM_JOINT_NAMES,
    AMGG_STATE_DIM,
    AMGG_WAIST_JOINT_NAMES,
    AmggJointSpec,
    require_amgg_joint_contract,
    validate_amgg_joint_names,
)

__all__ = [
    "AMGG_CONTROLLED_JOINT_NAMES",
    "AMGG_ABSOLUTE_IK_ACTION_DIM",
    "AMGG_CAMERAS",
    "AMGG_CAMERA_BY_NAME",
    "AMGG_FRAMES",
    "AMGG_GRIPPER_JOINT_NAMES",
    "AMGG_HEAD_JOINT_NAMES",
    "AMGG_HOME_POSITIONS",
    "AMGG_IK_JOINT_NAMES",
    "AMGG_JOINT_SPECS",
    "AMGG_JOINT_POSITION_ACTION_DIM",
    "AMGG_LEFT_ARM_JOINT_NAMES",
    "AMGG_OBSERVED_JOINT_NAMES",
    "AMGG_RIGHT_ARM_JOINT_NAMES",
    "AMGG_STATE_DIM",
    "AMGG_WAIST_JOINT_NAMES",
    "AmggFrameContract",
    "AmggCameraSpec",
    "AmggJointSpec",
    "require_amgg_joint_contract",
    "require_amgg_camera_contract",
    "validate_amgg_joint_names",
]
