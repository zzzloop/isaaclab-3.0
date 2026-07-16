# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based environment configuration entry point for the AMGG task."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnvCfg

AMGG_PHYSICS_HZ = 120
AMGG_CONTROL_HZ = 30


def build_amgg_env_cfg() -> ManagerBasedRLEnvCfg:
    """Build the AMGG manager-based environment configuration.

    Returns:
        Complete manager-based environment configuration.

    Raises:
        RuntimeError: Until robot, scene, action, and observation contracts are populated.
    """
    raise RuntimeError("AMGG environment configuration is pending robot and scene assets.")
