# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hardware-neutral AMGG command and state interface."""

from dataclasses import dataclass
from typing import Protocol

from amgg_robot_lab.contracts import AMGG_CONTROLLED_JOINT_NAMES, AMGG_HOME_POSITIONS, AMGG_OBSERVED_JOINT_NAMES


@dataclass(frozen=True, slots=True)
class AmggRobotState:
    """One measured real-robot state sample.

    Joint values use [rad] for revolute joints and [m] for prismatic joints.
    """

    timestamp_s: float
    joint_positions_rad: tuple[float, ...]
    joint_velocities_rad_s: tuple[float, ...]


class AmggRobotBackend(Protocol):
    """Interface required by real-robot teleoperation and recording."""

    def connect(self) -> None:
        """Connect to the robot without enabling motion."""

    def enable(self) -> None:
        """Enable command output after explicit safety checks."""

    def read_state(self) -> AmggRobotState:
        """Read the latest measured joint state."""

    def send_joint_position_targets(self, command_rad: tuple[float, ...], timestamp_s: float) -> None:
        """Send joint-position targets [rad] with a source timestamp [s]."""

    def stop(self) -> None:
        """Stop commanded motion and hold or disable according to hardware policy."""

    def disconnect(self) -> None:
        """Close the robot connection."""


class AmggDryRunBackend:
    """Deterministic no-motion backend for integration and safety tests."""

    def __init__(self) -> None:
        self._connected = False
        self._enabled = False
        self._timestamp_s = 0.0
        self._positions = tuple(AMGG_HOME_POSITIONS[name] for name in AMGG_OBSERVED_JOINT_NAMES)

    def connect(self) -> None:
        self._connected = True

    def enable(self) -> None:
        if not self._connected:
            raise RuntimeError("Connect the AMGG dry-run backend before enabling it.")
        self._enabled = True

    def read_state(self) -> AmggRobotState:
        if not self._connected:
            raise RuntimeError("AMGG dry-run backend is disconnected.")
        return AmggRobotState(self._timestamp_s, self._positions, (0.0,) * len(self._positions))

    def send_joint_position_targets(self, command_rad: tuple[float, ...], timestamp_s: float) -> None:
        if not self._enabled:
            raise RuntimeError("AMGG dry-run motion is not enabled.")
        if len(command_rad) != len(AMGG_CONTROLLED_JOINT_NAMES):
            raise ValueError("AMGG dry-run command does not follow the 21-D ABI.")
        self._positions = command_rad + self._positions[len(command_rad) :]
        self._timestamp_s = timestamp_s

    def stop(self) -> None:
        self._enabled = False

    def disconnect(self) -> None:
        self.stop()
        self._connected = False
