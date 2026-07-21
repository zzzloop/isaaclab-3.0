# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Register AMGG tasks and delegate to Isaac Lab's official demo recorder."""

import runpy
import sys
from pathlib import Path

from amgg_gpu import configure_preferred_gpu

_AMGG_REGISTRATION_CALLBACK = "amgg_robot_lab.tasks.register_tasks"


def _inject_recording_defaults() -> None:
    """Auto-start AMGG recording unless the caller explicitly opts out."""
    option_names = ("--auto_start_recording", "--no-auto_start_recording")
    has_option = any(
        argument == option or argument.startswith(f"{option}=") for argument in sys.argv[1:] for option in option_names
    )
    if not has_option:
        sys.argv.append("--auto_start_recording")


def _inject_registration_callback() -> None:
    """Register AMGG tasks after the official launcher starts Kit."""
    has_callback = any(
        argument == "--external_callback" or argument.startswith("--external_callback=") for argument in sys.argv[1:]
    )
    if has_callback:
        raise SystemExit("amgg_record_demos.py reserves --external_callback for AMGG task registration.")
    sys.argv.extend(["--external_callback", _AMGG_REGISTRATION_CALLBACK])


def main() -> None:
    """Run the official success-gated HDF5 recording entry point."""
    configure_preferred_gpu()
    _inject_recording_defaults()
    _inject_registration_callback()
    script = Path(__file__).resolve().parents[2] / "scripts" / "tools" / "record_demos.py"
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
