# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Register AM-DP123 tasks and delegate to Isaac Lab's official SE(3) teleop app."""

import runpy
import sys
from pathlib import Path

_AM_DP123_REGISTRATION_CALLBACK = "amgg_robot_lab.tasks.register_tasks"
_EXTENSION_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "source" / "amgg_robot_lab"


def _ensure_extension_importable() -> None:
    """Prepend the extension source tree for checkout-based launches."""
    source_root = str(_EXTENSION_SOURCE_ROOT)
    if source_root not in sys.path:
        sys.path.insert(0, source_root)


def _inject_registration_callback() -> None:
    """Register the AM-DP123 task after the official launcher starts Kit."""
    has_callback = any(
        argument == "--external_callback" or argument.startswith("--external_callback=") for argument in sys.argv[1:]
    )
    if has_callback:
        raise SystemExit("amgg_teleop.py reserves --external_callback for AM-DP123 task registration.")
    sys.argv.extend(["--external_callback", _AM_DP123_REGISTRATION_CALLBACK])


def main() -> None:
    """Run the official teleop entry point after custom task registration."""
    _ensure_extension_importable()
    _inject_registration_callback()
    script = Path(__file__).resolve().parents[2] / "scripts" / "environments" / "teleoperation" / "teleop_se3_agent.py"
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
