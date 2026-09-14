# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical AM-DP123 joint orders used as simulation and hardware ABIs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class AmDp123JointSpec:
    """Physical and command limits for one AM-DP123 joint.

    Attributes:
        name: Joint name as exported by ``AM-DP123.urdf``.
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
            raise ValueError("AM-DP123 joint names must be non-empty.")
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
) -> AmDp123JointSpec:
    return AmDp123JointSpec(name, group, lower, upper, home, velocity, effort, command_enabled)


# Arm homes are the offline dual-arm IK solution that places both wrist frames at
# (0.36 m, +/-0.20 m, 0.88 m) in ``base_link`` with the URDF's natural wrist
# orientation.  The solver reports a 0.385 rad margin to the nearest joint limit,
# so the pose is a reachable, in-limit starting configuration for PICO teleop.
AM_DP123_JOINT_SPECS: tuple[AmDp123JointSpec, ...] = (
    _spec("waist_joint1", "waist", -1.7453292519943295, 0.0, 2.62, 376.0, command_enabled=False),
    _spec("waist_joint2", "waist", -2.9670597283903604, 0.0, 2.62, 367.0, command_enabled=False),
    _spec("waist_joint3", "waist", -2.792526803190927, 1.0471975511965976, 2.62, 367.0, command_enabled=False),
    _spec("openarm_left_joint1", "left_arm", -1.25, 1.0, 3.14, 40.0, -0.195756),
    _spec("openarm_left_joint2", "left_arm", -2.2, 0.05, 3.14, 40.0, -0.380892),
    _spec("openarm_left_joint3", "left_arm", -1.59, 1.7, 3.14, 27.0, 0.888671),
    _spec("openarm_left_joint4", "left_arm", 0.0, 2.1, 3.14, 27.0, 1.155627),
    _spec("openarm_left_joint5", "left_arm", -1.57, 1.59, 3.14, 7.0, -0.04855),
    _spec("openarm_left_joint6", "left_arm", -0.4, 0.6, 3.14, 7.0, 0.215142),
    _spec("openarm_left_joint7", "left_arm", -0.44, 1.1, 3.14, 7.0, 0.0),
    _spec("openarm_right_joint1", "right_arm", -1.0, 1.25, 3.14, 40.0, 0.112077),
    _spec("openarm_right_joint2", "right_arm", -0.05, 2.2, 3.14, 40.0, 0.407696),
    _spec("openarm_right_joint3", "right_arm", -1.59, 1.7, 3.14, 27.0, -0.967329),
    _spec("openarm_right_joint4", "right_arm", 0.0, 2.1, 3.14, 27.0, 1.132665),
    _spec("openarm_right_joint5", "right_arm", -1.59, 1.57, 3.14, 7.0, -0.112517),
    _spec("openarm_right_joint6", "right_arm", -0.6, 0.4, 3.14, 7.0, 0.015152),
    _spec("openarm_right_joint7", "right_arm", -1.1, 0.44, 3.14, 7.0, 0.0),
    _spec("left_arm_hand_joint1_0", "left_hand", -0.32, 0.0, 1.0, 1.0),
    _spec("left_arm_hand_joint2_0", "left_hand", 0.0, 0.32, 1.0, 1.0),
    _spec("right_arm_hand_joint1_0", "right_hand", -0.32, 0.0, 1.0, 1.0),
    _spec("right_arm_hand_joint2_0", "right_hand", 0.0, 0.32, 1.0, 1.0),
    _spec("head_joint1", "head", -1.5707963267948966, 1.5707963267948966, 3.14, 2.9, command_enabled=False),
    _spec("head_joint2", "head", -0.6981317007977318, 0.6981317007977318, 3.14, 2.9, command_enabled=False),
)

AM_DP123_LOCOMOTION_JOINT_NAMES: tuple[str, ...] = (
    "left_steer_joint",
    "left_wheel_joint",
    "right_steer_joint",
    "right_wheel_joint",
    "b_left_steer_joint",
    "b_left_wheel_joint",
    "b_right_steer_joint",
    "b_right_wheel_joint",
)
"""Continuous mobile-base joints.

They are intentionally excluded from the state and command ABIs: the simulator
welds the base with ``fix_base=True`` and only damps these joints.
"""

AM_DP123_WAIST_JOINT_NAMES = tuple(spec.name for spec in AM_DP123_JOINT_SPECS if spec.group == "waist")
AM_DP123_HEAD_JOINT_NAMES = tuple(spec.name for spec in AM_DP123_JOINT_SPECS if spec.group == "head")
AM_DP123_LEFT_ARM_JOINT_NAMES = tuple(spec.name for spec in AM_DP123_JOINT_SPECS if spec.group == "left_arm")
AM_DP123_RIGHT_ARM_JOINT_NAMES = tuple(spec.name for spec in AM_DP123_JOINT_SPECS if spec.group == "right_arm")
AM_DP123_LEFT_HAND_JOINT_NAMES = tuple(spec.name for spec in AM_DP123_JOINT_SPECS if spec.group == "left_hand")
AM_DP123_RIGHT_HAND_JOINT_NAMES = tuple(spec.name for spec in AM_DP123_JOINT_SPECS if spec.group == "right_hand")
AM_DP123_HAND_JOINT_NAMES = AM_DP123_LEFT_HAND_JOINT_NAMES + AM_DP123_RIGHT_HAND_JOINT_NAMES
AM_DP123_ARM_JOINT_NAMES = AM_DP123_LEFT_ARM_JOINT_NAMES + AM_DP123_RIGHT_ARM_JOINT_NAMES

# Pink controls the 14 arm joints only, and the trigger mapping drives the four
# hand joints.  The waist and head stay at their home position, which keeps the
# operator's workspace fixed in front of the robot.
AM_DP123_IK_JOINT_NAMES = AM_DP123_ARM_JOINT_NAMES
AM_DP123_CONTROLLED_JOINT_NAMES = tuple(spec.name for spec in AM_DP123_JOINT_SPECS if spec.command_enabled)
AM_DP123_STATE_JOINT_NAMES = tuple(spec.name for spec in AM_DP123_JOINT_SPECS)
AM_DP123_HOME_POSITIONS = {spec.name: spec.home_position_rad for spec in AM_DP123_JOINT_SPECS}

# Finger travel: index 0 of each hand closes towards its lower limit, index 1
# towards its upper limit, so both jaws of one hand meet at the same time.
AM_DP123_HAND_OPEN_POSITIONS = dict.fromkeys(AM_DP123_HAND_JOINT_NAMES, 0.0)
AM_DP123_HAND_CLOSED_POSITIONS = {
    "left_arm_hand_joint1_0": -0.32,
    "left_arm_hand_joint2_0": 0.32,
    "right_arm_hand_joint1_0": -0.32,
    "right_arm_hand_joint2_0": 0.32,
}

# Public controller action: left wrist pose, right wrist pose, then four finger targets.
AM_DP123_ABSOLUTE_IK_ACTION_DIM = 18
AM_DP123_JOINT_POSITION_ACTION_DIM = len(AM_DP123_CONTROLLED_JOINT_NAMES)
AM_DP123_STATE_DIM = len(AM_DP123_STATE_JOINT_NAMES)

AM_DP123_HAND_ACTION_SIDE_INDEX: tuple[int, int, int, int] = (0, 0, 1, 1)
"""PICO hand action entry -> hand side index (``0`` left, ``1`` right).

IsaacTeleop's ``GripperRetargeter`` emits one scalar per hand and the retargeting
pipeline duplicates it for both jaws, so entries 0/1 (left hand) and 2/3 (right
hand) always carry the same trigger value.
"""

AM_DP123_HAND_TRIGGER_OPEN = 1.0
"""``GripperRetargeter`` output for a fully open hand."""

AM_DP123_HAND_TRIGGER_CLOSED = -1.0
"""``GripperRetargeter`` output for a fully closed fist."""


def am_dp123_hand_closed_fraction(trigger: float) -> float:
    """Convert a PICO hand trigger into a normalized closure ratio.

    Args:
        trigger: Hand trigger where :data:`AM_DP123_HAND_TRIGGER_OPEN` is fully
            open and :data:`AM_DP123_HAND_TRIGGER_CLOSED` is fully closed.

    Returns:
        Closure ratio in ``[0, 1]``, where ``0`` is fully open and ``1`` is fully
        closed. Values outside the trigger range are clamped.
    """
    if not isfinite(trigger):
        raise ValueError("AM-DP123 hand trigger must be finite.")
    span = AM_DP123_HAND_TRIGGER_OPEN - AM_DP123_HAND_TRIGGER_CLOSED
    return min(max((AM_DP123_HAND_TRIGGER_OPEN - trigger) / span, 0.0), 1.0)


def validate_am_dp123_joint_names(names: Sequence[str]) -> tuple[str, ...]:
    """Validate and freeze an ordered joint-name sequence.

    Args:
        names: Ordered joint names.

    Returns:
        Joint names as an immutable tuple.
    """
    frozen_names = tuple(names)
    if any(not name for name in frozen_names):
        raise ValueError("AM-DP123 joint names must be non-empty.")
    if len(set(frozen_names)) != len(frozen_names):
        raise ValueError("AM-DP123 joint names must be unique.")
    return frozen_names


def am_dp123_hand_targets(closed_fraction: float) -> dict[str, float]:
    """Interpolate hand joint targets between the open and closed positions.

    Args:
        closed_fraction: Normalized closure in ``[0, 1]``, where ``0`` is fully
            open and ``1`` is fully closed. Values outside the interval are clamped.

    Returns:
        Hand joint targets [rad] keyed by joint name.
    """
    if not isfinite(closed_fraction):
        raise ValueError("AM-DP123 hand closure must be finite.")
    fraction = min(max(closed_fraction, 0.0), 1.0)
    return {
        name: AM_DP123_HAND_OPEN_POSITIONS[name]
        + fraction * (AM_DP123_HAND_CLOSED_POSITIONS[name] - AM_DP123_HAND_OPEN_POSITIONS[name])
        for name in AM_DP123_HAND_JOINT_NAMES
    }


def require_am_dp123_joint_contract() -> None:
    """Validate the canonical AM-DP123 state and command contracts."""
    for spec in AM_DP123_JOINT_SPECS:
        spec.validate()
    observed = validate_am_dp123_joint_names(AM_DP123_STATE_JOINT_NAMES)
    controlled = validate_am_dp123_joint_names(AM_DP123_CONTROLLED_JOINT_NAMES)
    if not set(controlled).issubset(observed):
        raise ValueError("All commanded AM-DP123 joints must also be observed.")
    if set(AM_DP123_HAND_JOINT_NAMES) - set(controlled):
        raise ValueError("All AM-DP123 hand joints must be commandable.")
    if AM_DP123_STATE_DIM != 23 or AM_DP123_JOINT_POSITION_ACTION_DIM != 18:
        raise ValueError("The AM-DP123 v1 state/action ABI must remain 23-D/18-D.")
    if AM_DP123_ABSOLUTE_IK_ACTION_DIM != 18:
        raise ValueError("The AM-DP123 PICO action ABI must remain 18-D.")
    if len(AM_DP123_IK_JOINT_NAMES) != 14:
        raise ValueError("AM-DP123 Pink IK must control the 14 arm joints.")
    side_index = AM_DP123_HAND_ACTION_SIDE_INDEX
    if len(side_index) != len(AM_DP123_HAND_JOINT_NAMES):
        raise ValueError("AM-DP123 hand action layout must hold one entry per hand joint.")
    if list(side_index) != sorted(side_index) or set(side_index) != {0, 1}:
        raise ValueError("AM-DP123 hand action layout must pair the two jaws of each hand.")
