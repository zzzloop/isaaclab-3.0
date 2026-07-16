# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset and randomization events for the AMGG manager-based task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_amgg_task(env: ManagerBasedRLEnv, env_ids: Sequence[int] | None = None) -> None:
    """Reset AMGG robot and task objects to their configured initial states."""
    del env, env_ids
    raise NotImplementedError("Define AMGG robot and scene reset states before implementing events.")
