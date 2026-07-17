# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Register AMGG tasks and delegate to Isaac Lab's official SE(3) teleop app."""

import runpy
import sys
from pathlib import Path

_AMGG_REGISTRATION_CALLBACK = "amgg_robot_lab.tasks.register_tasks"


def _inject_registration_callback() -> None:
    """Register AMGG tasks after the official launcher starts Kit."""
    has_callback = any(
        argument == "--external_callback" or argument.startswith("--external_callback=") for argument in sys.argv[1:]
    )
    if has_callback:
        raise SystemExit("amgg_teleop.py reserves --external_callback for AMGG task registration.")
    sys.argv.extend(["--external_callback", _AMGG_REGISTRATION_CALLBACK])


def main() -> None:
    """Run the official teleop entry point after custom task registration."""
    _inject_registration_callback()
    script = Path(__file__).resolve().parents[2] / "scripts" / "environments" / "teleoperation" / "teleop_se3_agent.py"
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
