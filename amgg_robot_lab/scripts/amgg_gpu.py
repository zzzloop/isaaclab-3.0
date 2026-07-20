# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Select a stable physical GPU before Isaac Sim or PyTorch starts."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass
from io import StringIO


@dataclass(frozen=True)
class _GpuInfo:
    """GPU identity reported by ``nvidia-smi``."""

    physical_index: int
    uuid: str
    pci_bus_id: str

    @property
    def pci_sort_key(self) -> tuple[int, int, int, int]:
        """Return the numeric PCI address used by Vulkan enumeration."""
        domain_and_bus, device_and_function = self.pci_bus_id.rsplit(":", maxsplit=1)
        domain, bus = domain_and_bus.rsplit(":", maxsplit=1)
        device, function = device_and_function.split(".", maxsplit=1)
        return tuple(int(component, 16) for component in (domain, bus, device, function))


def _has_device_argument(arguments: Sequence[str]) -> bool:
    return any(argument == "--device" or argument.startswith("--device=") for argument in arguments[1:])


def _has_flag(arguments: Sequence[str], name: str) -> bool:
    return any(argument == name or argument.startswith(f"{name}=") for argument in arguments[1:])


def _is_headless(arguments: Sequence[str]) -> bool:
    return _has_flag(arguments, "--headless")


def _remove_explicit_visualizer(arguments: list[str]) -> bool:
    """Remove visualizer CLI options that conflict with deprecated headless mode."""
    cleaned = [arguments[0]]
    removed = False
    index = 1
    while index < len(arguments):
        argument = arguments[index]
        if argument in {"--visualizer", "--viz"}:
            removed = True
            index += 2 if index + 1 < len(arguments) else 1
            continue
        if argument.startswith("--visualizer=") or argument.startswith("--viz="):
            removed = True
            index += 1
            continue
        cleaned.append(argument)
        index += 1
    if removed:
        arguments[:] = cleaned
    return removed


def _parse_physical_indices(value: str, variable_name: str, *, allow_empty: bool = False) -> list[int]:
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not indices and not allow_empty:
        raise ValueError(f"{variable_name} must contain at least one physical GPU index.")
    if len(indices) != len(set(indices)):
        raise ValueError(f"{variable_name} must not contain duplicate physical GPU indices.")
    return indices


def _query_gpu_inventory() -> list[_GpuInfo]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    inventory = []
    for row in csv.reader(StringIO(result.stdout)):
        if len(row) != 3:
            raise RuntimeError(f"Unexpected nvidia-smi GPU row: {row!r}")
        inventory.append(_GpuInfo(int(row[0].strip()), row[1].strip(), row[2].strip()))
    if not inventory:
        raise RuntimeError("nvidia-smi returned no GPUs.")
    return inventory


def configure_preferred_gpu(
    arguments: list[str] | None = None,
    environment: MutableMapping[str, str] | None = None,
    inventory: Sequence[_GpuInfo] | None = None,
) -> int | None:
    """Pin CUDA, PhysX, RTX, and CloudXR to one physical GPU.

    Windowed launches default to physical GPU 0 because it owns the workstation
    Xorg presentation queue. Headless launches default to physical GPU 1.
    Physical GPUs 2 and 3 are quarantined by default because they have produced
    repeated driver-level Xid failures on the target workstation. Camera
    recording uses CUDA/RTX external-memory interop, so CUDA and Kit must use
    the same global PCI-order device namespace. The selected GPU is explicit
    for CUDA/PhysX, Kit/RTX, and CloudXR, while multi-GPU rendering is disabled.
    Passing ``--device`` opts out so an explicit operator choice is preserved.

    Args:
        arguments: Process argument vector to update. Defaults to :attr:`sys.argv`.
        environment: Process environment to update. Defaults to :attr:`os.environ`.
        inventory: Optional GPU inventory for deterministic testing.

    Returns:
        Selected PCI-order GPU index, or ``None`` when ``--device`` was
        explicitly provided.
    """
    arguments = sys.argv if arguments is None else arguments
    environment = os.environ if environment is None else environment
    if _is_headless(arguments) and _remove_explicit_visualizer(arguments):
        print(
            "[AMGG] Removed explicit --visualizer/--viz from deprecated --headless mode; "
            "XR will auto-inject KitVisualizer for app-update pumping.",
            flush=True,
        )
    if _has_device_argument(arguments):
        print("[AMGG] Explicit --device detected; automatic physical-GPU selection is disabled.", flush=True)
        return None

    launch_mode = "headless" if _is_headless(arguments) else "windowed"
    uses_xr_cameras = _has_flag(arguments, "--xr") and _has_flag(arguments, "--enable_cameras")
    xr_camera_rendering = "default"
    default_preferred_index = "1" if launch_mode == "headless" else "0"
    try:
        preferred_index = int(environment.get("AMGG_PREFERRED_GPU", default_preferred_index))
        allowed_indices = _parse_physical_indices(environment.get("AMGG_ALLOWED_GPUS", "0,1"), "AMGG_ALLOWED_GPUS")
        quarantined_indices = _parse_physical_indices(
            environment.get("AMGG_QUARANTINED_GPUS", "2,3"),
            "AMGG_QUARANTINED_GPUS",
            allow_empty=True,
        )
    except ValueError as error:
        raise SystemExit(f"Invalid AMGG GPU configuration: {error}") from error
    if preferred_index in quarantined_indices:
        raise SystemExit(
            f"AMGG_PREFERRED_GPU={preferred_index} is quarantined by "
            f"AMGG_QUARANTINED_GPUS={quarantined_indices}. Clear the quarantine only after the GPU has been reset "
            "and passes administrator diagnostics."
        )
    if preferred_index not in allowed_indices:
        raise SystemExit(f"AMGG_PREFERRED_GPU={preferred_index} is not present in AMGG_ALLOWED_GPUS={allowed_indices}.")

    try:
        detected = list(inventory) if inventory is not None else _query_gpu_inventory()
        by_physical_index = {gpu.physical_index: gpu for gpu in detected}
        missing_indices = [index for index in allowed_indices if index not in by_physical_index]
        if missing_indices:
            available = sorted(by_physical_index)
            raise SystemExit(
                f"AMGG allowed physical GPUs {missing_indices} were not found; available GPUs: {available}."
            )
        selected = by_physical_index[preferred_index]
        # Kit/Vulkan assigns renderer indices in PCI-bus order, independent
        # of CUDA_VISIBLE_DEVICES. Count every detected GPU here (including
        # disallowed devices) so the index still addresses the selected
        # physical card in Kit's global GPU table.
        renderer_gpus = sorted(detected, key=lambda gpu: gpu.pci_sort_key)
        selected_index = renderer_gpus.index(selected)
        renderer_mapping = ", ".join(
            f"Kit {index}=physical {gpu.physical_index}" for index, gpu in enumerate(renderer_gpus)
        )
        identity = f"UUID={selected.uuid}, PCI={selected.pci_bus_id}"
    except (FileNotFoundError, subprocess.SubprocessError, RuntimeError) as error:
        selected_index = preferred_index
        renderer_mapping = "unavailable; using physical ordinal fallback"
        identity = "UUID/PCI unavailable"
        print(f"[AMGG] Warning: GPU identity probe failed ({error}); using ordinal fallback.", flush=True)

    # Do not remap a non-zero physical GPU to cuda:0. RTX camera render
    # products are created in Kit's global Vulkan namespace, and remapping
    # only CUDA can make external-memory import address a different device.
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["NV_GPU_INDEX"] = str(selected_index)
    # AppLauncher accepts raw Kit settings through one parsed ``--kit_args``
    # value. Injecting the settings as top-level arguments makes applications
    # such as record_demos.py reject them after Kit starts.
    kit_settings = [
        f"--/renderer/activeGpu={selected_index}",
        "--/renderer/multiGpu/enabled=false",
        "--/renderer/multiGpu/autoEnable=false",
        "--/renderer/multiGpu/maxGpuCount=1",
    ]
    if uses_xr_cameras:
        # XR normally enables low-latency asynchronous rendering. Sensor
        # cameras create Replicator render products backed by CUDA/Vulkan
        # external memory, and toggling asynchronous rendering while those
        # products initialize has produced repeatable Xid 31 MMU faults across
        # multiple physical GPUs. Keep the recording path synchronous and stop
        # the throttling extension from turning async rendering back on.
        kit_settings.extend(
            [
                "--/exts/isaacsim.core.throttling/enable_async=false",
                "--/omni/replicator/asyncRendering=false",
            ]
        )
        if launch_mode == "headless":
            kit_settings.extend(
                [
                    "--/app/asyncRendering=false",
                    "--/app/asyncRenderingLowLatency=false",
                ]
            )
            xr_camera_rendering = "synchronous"
        else:
            # The window swapchain requires the XR app's asynchronous update
            # path. Keep it enabled while making only Replicator sensor
            # capture synchronous and preventing throttling from toggling it.
            xr_camera_rendering = "windowed-safe"
    kit_args = " ".join(kit_settings)
    arguments.extend(["--device", f"cuda:{selected_index}", "--kit_args", kit_args])
    print(
        f"[AMGG] Preferred physical GPU {preferred_index} ({identity}) -> "
        f"CUDA/PhysX/Kit/RTX/CloudXR PCI-order GPU {selected_index}; multi-GPU rendering disabled; "
        f"launch mode={launch_mode}; "
        f"XR camera rendering={xr_camera_rendering}; "
        f"allowed physical GPUs={allowed_indices}; quarantined physical GPUs={quarantined_indices}; "
        f"renderer map: {renderer_mapping}.",
        flush=True,
    )
    return selected_index
