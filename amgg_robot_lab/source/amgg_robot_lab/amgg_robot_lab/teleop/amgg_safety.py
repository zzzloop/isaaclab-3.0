# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common AMGG safety checks used by simulation and real-robot backends."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import isfinite

from amgg_robot_lab.contracts import AMGG_CONTROLLED_JOINT_NAMES, AMGG_JOINT_SPECS


def validate_amgg_joint_command(command_rad: Sequence[float], expected_size: int) -> tuple[float, ...]:
    """Validate joint-command shape and finite values.

    Args:
        command_rad: Joint-position command [rad].
        expected_size: Required action dimension.

    Returns:
        Validated joint-position command [rad].
    """
    frozen_command = tuple(command_rad)
    if len(frozen_command) != expected_size:
        raise ValueError(f"Expected {expected_size} AMGG joint commands, received {len(frozen_command)}.")
    if not all(isfinite(value) for value in frozen_command):
        raise ValueError("AMGG joint command contains a non-finite value.")
    return frozen_command


@dataclass(slots=True)
class AmggCommandLimiter:
    """Stateful position, velocity, tracking-error, and watchdog limiter.

    Values use [rad] for revolute joints and [m] for prismatic gripper joints.
    """

    tracking_error_limit: float = 0.45
    gripper_tracking_error_limit_m: float = 0.03
    timeout_s: float = 0.20
    _last_command: tuple[float, ...] | None = field(default=None, init=False)
    _last_timestamp_s: float | None = field(default=None, init=False)

    def reset(self, measured_positions: Sequence[float], timestamp_s: float) -> None:
        """Initialize the limiter from a measured controlled-joint state."""
        self._last_command = validate_amgg_joint_command(measured_positions, len(AMGG_CONTROLLED_JOINT_NAMES))
        self._last_timestamp_s = timestamp_s

    def filter(
        self,
        requested_positions: Sequence[float],
        measured_positions: Sequence[float],
        timestamp_s: float,
    ) -> tuple[float, ...]:
        """Validate and rate-limit a requested joint-position command."""
        requested = validate_amgg_joint_command(requested_positions, len(AMGG_CONTROLLED_JOINT_NAMES))
        measured = validate_amgg_joint_command(measured_positions, len(AMGG_CONTROLLED_JOINT_NAMES))
        if self._last_command is None or self._last_timestamp_s is None:
            self.reset(measured, timestamp_s)
        assert self._last_command is not None and self._last_timestamp_s is not None
        dt = timestamp_s - self._last_timestamp_s
        if dt <= 0.0 or dt > self.timeout_s:
            raise ValueError(f"AMGG command timing violation: dt={dt:.6f} s.")
        specs = [spec for spec in AMGG_JOINT_SPECS if spec.command_enabled]
        safe_command = []
        for index, (target, state, previous, spec) in enumerate(
            zip(requested, measured, self._last_command, specs, strict=True)
        ):
            tracking_limit = (
                self.gripper_tracking_error_limit_m if "gripper" in spec.group else self.tracking_error_limit
            )
            if abs(previous - state) > tracking_limit:
                raise ValueError(f"AMGG joint '{spec.name}' tracking error exceeds the safety limit.")
            bounded = min(max(target, spec.lower_limit_rad), spec.upper_limit_rad)
            max_delta = spec.max_velocity_rad_s * dt
            bounded = min(max(bounded, previous - max_delta), previous + max_delta)
            if not isfinite(bounded):
                raise ValueError(f"AMGG joint command {index} became non-finite.")
            safe_command.append(bounded)
        self._last_command = tuple(safe_command)
        self._last_timestamp_s = timestamp_s
        return self._last_command

    def watchdog_expired(self, timestamp_s: float) -> bool:
        """Return whether command production exceeded the allowed interval."""
        return self._last_timestamp_s is None or timestamp_s - self._last_timestamp_s > self.timeout_s
