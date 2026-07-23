# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared Kit settings for AMGG XR entry points."""

from __future__ import annotations

import sys

AMGG_XR_KIT_ARGS = (
    "--/renderer/multiGpu/enabled=false",
    "--/renderer/multiGpu/autoEnable=false",
    "--/renderer/multiGpu/maxGpuCount=1",
    "--/exts/omni.kit.renderer.core/present/enabled=false",
    "--/app/updateOrder/checkForHydraRenderComplete=1000",
    "--/app/renderer/waitIdle=true",
    "--/app/hydraEngine/waitIdle=true",
    "--/app/asyncRendering=false",
    "--/app/asyncRenderingLowLatency=false",
    "--/omni/replicator/asyncRendering=false",
)


def merge_kit_args(extra_args: tuple[str, ...]) -> None:
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
