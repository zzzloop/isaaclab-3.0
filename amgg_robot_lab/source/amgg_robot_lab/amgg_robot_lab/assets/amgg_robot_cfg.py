# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab asset configuration entry point for the AMGG robot."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.assets import ArticulationCfg

AMGG_ASSET_DATA_DIR = Path(__file__).resolve().parent / "data"
AMGG_URDF_PATH = AMGG_ASSET_DATA_DIR / "urdf" / "amgg_robot.urdf"


def get_amgg_robot_cfg() -> ArticulationCfg:
    """Build the AMGG robot articulation configuration.

    Returns:
        Isaac Lab articulation configuration for the AMGG robot.

    Raises:
        RuntimeError: If the robot model and joint contract have not been supplied yet.
    """
    raise RuntimeError(
        "AMGG robot configuration is pending. Add assets/data/urdf/amgg_robot.urdf and meshes, "
        "then populate the joint and frame contracts before building ArticulationCfg."
    )
