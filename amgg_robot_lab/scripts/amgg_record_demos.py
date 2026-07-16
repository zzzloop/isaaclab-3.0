# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Register AMGG tasks and delegate to Isaac Lab's official demo recorder."""

import runpy
from pathlib import Path

import amgg_robot_lab  # noqa: F401


def main() -> None:
    """Run the official success-gated HDF5 recording entry point."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "tools" / "record_demos.py"
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
