# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-neutral AMGG teleoperation data and retargeting boundary."""

from dataclasses import dataclass

from amgg_robot_lab.contracts import AMGG_CONTROLLED_JOINT_NAMES, AMGG_IK_JOINT_NAMES
from amgg_robot_lab.kinematics import AmggPose, solve_amgg_inverse_kinematics


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
    """Convert a calibrated PICO sample into a limit-aware joint command.

    The first hand value is normalized closure: 0 is open and 1 is closed.
    """
    if not sample.tracking_valid:
        raise ValueError("Cannot retarget an invalid PICO tracking sample.")
    valid_seed_lengths = {len(AMGG_IK_JOINT_NAMES), len(AMGG_CONTROLLED_JOINT_NAMES)}
    if len(seed_joint_positions_rad) not in valid_seed_lengths:
        raise ValueError("AMGG IK seed must use the 17-D IK or 21-D controlled-joint ABI.")
    solution = solve_amgg_inverse_kinematics(
        sample.left_wrist,
        sample.right_wrist,
        seed_joint_positions_rad[: len(AMGG_IK_JOINT_NAMES)],
    )
    left_closure = min(max(sample.left_hand[0] if sample.left_hand else 0.0, 0.0), 1.0)
    right_closure = min(max(sample.right_hand[0] if sample.right_hand else 0.0, 0.0), 1.0)
    grippers = (0.025 * left_closure,) * 2 + (0.025 * right_closure,) * 2
    return AmggJointCommand(sample.timestamp_s, solution + grippers)
