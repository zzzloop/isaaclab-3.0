# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Register AMGG tasks and delegate to Isaac Lab's official SE(3) teleop app."""

import runpy
from pathlib import Path

import amgg_robot_lab  # noqa: F401


def main() -> None:
    """Run the official teleop entry point after custom task registration."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "environments" / "teleoperation" / "teleop_se3_agent.py"
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
