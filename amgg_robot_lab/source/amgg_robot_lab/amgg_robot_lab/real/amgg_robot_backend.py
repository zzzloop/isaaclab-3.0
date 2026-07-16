# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hardware-neutral AMGG command and state interface."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class AmggRobotState:
    """One measured real-robot state sample."""

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
