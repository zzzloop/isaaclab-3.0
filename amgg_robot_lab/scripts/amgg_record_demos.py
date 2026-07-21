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
_AMGG_RECORDING_KIT_ARGS = (
    "--/renderer/multiGpu/enabled=false",
    "--/renderer/multiGpu/autoEnable=false",
    "--/renderer/multiGpu/maxGpuCount=1",
    "--/rtx/rendermode=Minimal",
    "--/rtx/minimal/mode=3",
    "--/omni/replicator/asyncRendering=false",
)


def _inject_recording_defaults() -> None:
    """Auto-start AMGG recording unless the caller explicitly opts out."""
    option_names = ("--auto_start_recording", "--no-auto_start_recording")
    has_option = any(
        argument == option or argument.startswith(f"{option}=") for argument in sys.argv[1:] for option in option_names
    )
    if not has_option:
        sys.argv.append("--auto_start_recording")


def _merge_kit_args(extra_args: tuple[str, ...]) -> None:
    """Append AMGG Kit settings while preserving caller-provided settings."""
    for index, argument in enumerate(sys.argv[1:], start=1):
        if argument == "--kit_args":
            if index + 1 >= len(sys.argv):
                raise SystemExit("--kit_args requires a value.")
            current_args = sys.argv[index + 1].split()
            merged_args = current_args + [arg for arg in extra_args if arg not in current_args]
            sys.argv[index + 1] = " ".join(merged_args)
            return
        if argument.startswith("--kit_args="):
            current_value = argument.split("=", maxsplit=1)[1]
            current_args = current_value.split()
            merged_args = current_args + [arg for arg in extra_args if arg not in current_args]
            sys.argv[index] = f"--kit_args={' '.join(merged_args)}"
            return

    sys.argv.extend(["--kit_args", " ".join(extra_args)])


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
    _merge_kit_args(_AMGG_RECORDING_KIT_ARGS)
    print(f"[AMGG] XR recording Kit args: {' '.join(_AMGG_RECORDING_KIT_ARGS)}", flush=True)
    _inject_recording_defaults()
    _inject_registration_callback()
    script = Path(__file__).resolve().parents[2] / "scripts" / "tools" / "record_demos.py"
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
