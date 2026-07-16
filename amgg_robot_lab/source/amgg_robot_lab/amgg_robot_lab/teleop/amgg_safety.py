# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common AMGG safety checks used by simulation and real-robot backends."""

from collections.abc import Sequence
from math import isfinite


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
