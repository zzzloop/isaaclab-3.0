# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-neutral AMGG teleoperation data and retargeting boundary."""

from dataclasses import dataclass

from amgg_robot_lab.kinematics.amgg_kinematics_model import AmggPose


@dataclass(frozen=True, slots=True)
class AmggPicoSample:
    """One calibrated PICO tracking sample."""

    timestamp_s: float
    left_wrist: AmggPose
    right_wrist: AmggPose
    left_hand: tuple[float, ...]
    right_hand: tuple[float, ...]
    tracking_valid: bool


@dataclass(frozen=True, slots=True)
class AmggJointCommand:
    """Joint-position command in canonical controlled-joint order."""

    timestamp_s: float
    joint_positions_rad: tuple[float, ...]


def retarget_amgg_pico_sample(sample: AmggPicoSample, seed_joint_positions_rad: tuple[float, ...]) -> AmggJointCommand:
    """Convert a calibrated PICO sample into a safe AMGG joint command."""
    del sample, seed_joint_positions_rad
    raise RuntimeError("AMGG retargeting is pending URDF-backed IK and hand mapping.")
