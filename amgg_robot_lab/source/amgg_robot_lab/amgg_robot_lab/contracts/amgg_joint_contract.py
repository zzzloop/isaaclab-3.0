# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical AMGG joint orders used as public simulation and hardware ABIs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class AmggJointSpec:
    """Physical and command limits for one AMGG joint.

    Attributes:
        name: Joint name matching the normalized description and hardware map.
        group: Logical group such as ``left_arm`` or ``waist``.
        lower_limit_rad: Lower position limit [rad].
        upper_limit_rad: Upper position limit [rad].
        home_position_rad: Home position [rad].
        max_velocity_rad_s: Maximum absolute velocity [rad/s].
        max_effort_nm: Maximum absolute effort [N*m].
        command_enabled: Whether the joint belongs to the joint command ABI.
    """

    name: str
    group: str
    lower_limit_rad: float
    upper_limit_rad: float
    home_position_rad: float
    max_velocity_rad_s: float
    max_effort_nm: float
    command_enabled: bool = True

    def validate(self) -> None:
        """Validate joint limits and finite physical values."""
        values = (
            self.lower_limit_rad,
            self.upper_limit_rad,
            self.home_position_rad,
            self.max_velocity_rad_s,
            self.max_effort_nm,
        )
        if not self.name:
            raise ValueError("AMGG joint names must be non-empty.")
        if not all(isfinite(value) for value in values):
            raise ValueError(f"Joint '{self.name}' contains a non-finite limit.")
        if self.lower_limit_rad >= self.upper_limit_rad:
            raise ValueError(f"Joint '{self.name}' has an invalid position interval.")
        if not self.lower_limit_rad <= self.home_position_rad <= self.upper_limit_rad:
            raise ValueError(f"Joint '{self.name}' home position is outside its limits.")
        if self.max_velocity_rad_s <= 0.0 or self.max_effort_nm <= 0.0:
            raise ValueError(f"Joint '{self.name}' velocity and effort limits must be positive.")


def _spec(
    name: str,
    group: str,
    lower: float,
    upper: float,
    velocity: float,
    effort: float,
    home: float = 0.0,
    *,
    command_enabled: bool = True,
) -> AmggJointSpec:
    return AmggJointSpec(name, group, lower, upper, home, velocity, effort, command_enabled)


AMGG_JOINT_SPECS: tuple[AmggJointSpec, ...] = (
    _spec("Waist01_Joint", "waist", 0.0, 2.09, 1.5, 376.0, 0.669381),
    _spec("Waist02_Joint", "waist", -2.09, 0.0, 1.5, 367.0, -0.835682),
    _spec("Body0422_Joint", "waist", -1.57, 1.57, 1.5, 367.0, -0.056214),
    _spec("ArmL02_Joint", "left_arm", -2.35, 2.35, 2.0, 40.0, -0.672166),
    _spec("AM_D02_J14_Joint", "left_arm", -2.26, 0.26, 2.0, 40.0, -0.425081),
    _spec("ArmL04_Joint", "left_arm", -1.65, 1.65, 2.0, 27.0, 0.126779),
    _spec("ArmL05_Joint", "left_arm", 0.0, 2.268, 2.0, 27.0, 0.372478),
    _spec("ArmL06_Joint", "left_arm", -1.57, 1.57, 2.0, 9.0, 0.030401),
    _spec("ArmL07_Joint", "left_arm", -0.724, 1.06, 2.0, 9.0, -0.310859),
    _spec("ArmL07Output_Joint", "left_arm", -1.37, 1.57, 2.0, 9.0, -0.905619),
    _spec("ArmR02_Joint", "right_arm", -2.35, 2.35, 2.0, 40.0, 0.645073),
    _spec("AM_D02R_J03_Joint", "right_arm", -0.26, 2.26, 2.0, 40.0, -0.226692),
    _spec("ArmR04_Joint", "right_arm", -1.65, 1.65, 2.0, 27.0, -0.150307),
    _spec("ArmR05_Joint", "right_arm", -2.268, 0.0, 2.0, 27.0, -0.078970),
    _spec("ArmR06_Joint", "right_arm", -1.57, 1.57, 2.0, 9.0, 0.171809),
    _spec("ArmR07_Joint", "right_arm", -0.724, 1.06, 2.0, 9.0, -0.362311),
    _spec("ArmR07Output_Joint", "right_arm", -1.37, 1.57, 2.0, 9.0, 0.788573),
    _spec("left_gripper_negative_finger_joint", "left_gripper", 0.0, 0.025, 0.2, 30.0),
    _spec("left_gripper_positive_finger_joint", "left_gripper", 0.0, 0.025, 0.2, 30.0),
    _spec("right_gripper_negative_finger_joint", "right_gripper", 0.0, 0.025, 0.2, 30.0),
    _spec("right_gripper_positive_finger_joint", "right_gripper", 0.0, 0.025, 0.2, 30.0),
    _spec("Head02_Joint", "head", -3.141592653589793, 3.141592653589793, 1.5, 2.9, command_enabled=False),
    _spec("Head03_Joint", "head", -0.5, 1.51, 1.5, 2.9, command_enabled=False),
)

AMGG_WAIST_JOINT_NAMES = tuple(spec.name for spec in AMGG_JOINT_SPECS if spec.group == "waist")
AMGG_LEFT_ARM_JOINT_NAMES = tuple(spec.name for spec in AMGG_JOINT_SPECS if spec.group == "left_arm")
AMGG_RIGHT_ARM_JOINT_NAMES = tuple(spec.name for spec in AMGG_JOINT_SPECS if spec.group == "right_arm")
AMGG_IK_JOINT_NAMES = AMGG_WAIST_JOINT_NAMES + AMGG_LEFT_ARM_JOINT_NAMES + AMGG_RIGHT_ARM_JOINT_NAMES
AMGG_GRIPPER_JOINT_NAMES = tuple(spec.name for spec in AMGG_JOINT_SPECS if "gripper" in spec.group)
AMGG_HEAD_JOINT_NAMES = tuple(spec.name for spec in AMGG_JOINT_SPECS if spec.group == "head")
AMGG_CONTROLLED_JOINT_NAMES = tuple(spec.name for spec in AMGG_JOINT_SPECS if spec.command_enabled)
AMGG_OBSERVED_JOINT_NAMES = tuple(spec.name for spec in AMGG_JOINT_SPECS)
AMGG_HOME_POSITIONS = {spec.name: spec.home_position_rad for spec in AMGG_JOINT_SPECS}

# Public controller action: left TCP pose, right TCP pose, then four finger targets.
AMGG_ABSOLUTE_IK_ACTION_DIM = 18
AMGG_JOINT_POSITION_ACTION_DIM = len(AMGG_CONTROLLED_JOINT_NAMES)
AMGG_STATE_DIM = len(AMGG_OBSERVED_JOINT_NAMES)


def validate_amgg_joint_names(names: Sequence[str]) -> tuple[str, ...]:
    """Validate and freeze an ordered joint-name sequence.

    Args:
        names: Ordered joint names.

    Returns:
        Joint names as an immutable tuple.
    """
    frozen_names = tuple(names)
    if any(not name for name in frozen_names):
        raise ValueError("AMGG joint names must be non-empty.")
    if len(set(frozen_names)) != len(frozen_names):
        raise ValueError("AMGG joint names must be unique.")
    return frozen_names


def require_amgg_joint_contract() -> None:
    """Validate the canonical AMGG state and command contracts."""
    for spec in AMGG_JOINT_SPECS:
        spec.validate()
    observed = validate_amgg_joint_names(AMGG_OBSERVED_JOINT_NAMES)
    controlled = validate_amgg_joint_names(AMGG_CONTROLLED_JOINT_NAMES)
    if not set(controlled).issubset(observed):
        raise ValueError("All commanded AMGG joints must also be observed.")
    if AMGG_STATE_DIM != 23 or AMGG_JOINT_POSITION_ACTION_DIM != 21:
        raise ValueError("The AMGG v1 state/action ABI must remain 23-D/21-D.")
