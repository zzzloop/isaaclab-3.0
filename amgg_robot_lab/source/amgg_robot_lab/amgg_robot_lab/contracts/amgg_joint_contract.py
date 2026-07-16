# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical AMGG joint order used as the public state/action ABI."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence


@dataclass(frozen=True, slots=True)
class AmggJointSpec:
    """Physical and command limits for one AMGG joint.

    Attributes:
        name: Joint name matching the robot description and hardware interface.
        group: Logical group such as ``left_arm`` or ``waist``.
        lower_limit_rad: Lower position limit [rad].
        upper_limit_rad: Upper position limit [rad].
        home_position_rad: Home position [rad].
        max_velocity_rad_s: Maximum absolute velocity [rad/s].
        max_effort_nm: Maximum absolute effort [N*m].
        command_enabled: Whether the public action ABI commands this joint.
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


# These tuples remain empty until the canonical URDF and hardware mapping are supplied.
AMGG_JOINT_SPECS: tuple[AmggJointSpec, ...] = ()
AMGG_CONTROLLED_JOINT_NAMES: tuple[str, ...] = ()
AMGG_OBSERVED_JOINT_NAMES: tuple[str, ...] = ()


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
    """Raise unless the canonical AMGG joint contract is ready for use."""
    if not AMGG_JOINT_SPECS or not AMGG_CONTROLLED_JOINT_NAMES or not AMGG_OBSERVED_JOINT_NAMES:
        raise RuntimeError("AMGG joint contract is not populated yet.")
    for spec in AMGG_JOINT_SPECS:
        spec.validate()
    validate_amgg_joint_names(AMGG_CONTROLLED_JOINT_NAMES)
    validate_amgg_joint_names(AMGG_OBSERVED_JOINT_NAMES)
