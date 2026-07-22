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


def _parse_physical_indices(value: str) -> list[int]:
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not indices:
        raise ValueError("AMGG_ALLOWED_GPUS must contain at least one physical GPU index.")
    if len(indices) != len(set(indices)):
        raise ValueError("AMGG_ALLOWED_GPUS must not contain duplicate physical GPU indices.")
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
    """Map the preferred physical GPU to matching CUDA, Kit, and CloudXR indices.

    The AMGG defaults use physical GPU 1. All GPUs remain visible because
    Isaac Sim RTX/Vulkan device discovery can fail when ``CUDA_VISIBLE_DEVICES``
    hides GPUs from CUDA while Omniverse still enumerates them for graphics.
    Passing ``--device`` opts out so an explicit operator choice and environment
    are preserved.

    When the preferred physical GPU is not present in the queried inventory,
    the function falls back only to another explicitly allowed GPU.  If no
    allowed GPU is present, it aborts instead of silently selecting a hot or
    unavailable device.

    Args:
        arguments: Process argument vector to update. Defaults to :attr:`sys.argv`.
        environment: Process environment to update. Defaults to :attr:`os.environ`.
        inventory: Optional GPU inventory for deterministic testing.

    Returns:
        The selected logical GPU index, or ``None`` when ``--device`` was
        explicitly provided.
    """
    arguments = sys.argv if arguments is None else arguments
    environment = os.environ if environment is None else environment
    if _has_device_argument(arguments):
        print("[AMGG] Explicit --device detected; automatic physical-GPU selection is disabled.", flush=True)
        return None

    try:
        preferred_index = int(environment.get("AMGG_PREFERRED_GPU", "1"))
        allowed_indices = _parse_physical_indices(environment.get("AMGG_ALLOWED_GPUS", "1"))
    except ValueError as error:
        raise SystemExit(f"Invalid AMGG GPU configuration: {error}") from error
    if preferred_index not in allowed_indices:
        raise SystemExit(f"AMGG_PREFERRED_GPU={preferred_index} is not present in AMGG_ALLOWED_GPUS={allowed_indices}.")

    try:
        detected = list(inventory) if inventory is not None else _query_gpu_inventory()
        by_physical_index = {gpu.physical_index: gpu for gpu in detected}
        ordered_gpus = sorted(detected, key=lambda gpu: gpu.pci_sort_key)
        # Prefer the configured physical GPU; fall back to the first allowed
        # physical GPU that is actually present.  Do not fall back outside the
        # allow-list; GPU3 is known-bad on the AMGG server and GPU2 can run hot.
        candidate_indices = [preferred_index] + [index for index in allowed_indices if index != preferred_index]
        selected_physical = next((index for index in candidate_indices if index in by_physical_index), None)
        if selected_physical is None:
            detected_indices = [gpu.physical_index for gpu in detected]
            raise SystemExit(
                f"None of the allowed AMGG physical GPUs {allowed_indices} are present. "
                f"Detected physical GPUs: {detected_indices}. Set AMGG_ALLOWED_GPUS/AMGG_PREFERRED_GPU explicitly "
                "only if you intentionally want to use another card."
            )
        elif selected_physical != preferred_index:
            print(
                f"[AMGG] Warning: preferred physical GPU {preferred_index} is not available;"
                f" falling back to physical GPU {selected_physical}.",
                flush=True,
            )
        logical_index = next(index for index, gpu in enumerate(ordered_gpus) if gpu.physical_index == selected_physical)
        selected = ordered_gpus[logical_index]
        identity = f"UUID={selected.uuid}, PCI={selected.pci_bus_id}"
    except (FileNotFoundError, subprocess.SubprocessError, RuntimeError) as error:
        logical_index = preferred_index
        identity = "UUID/PCI unavailable"
        print(f"[AMGG] Warning: GPU identity probe failed ({error}); using ordinal fallback.", flush=True)

    removed_visible_devices = environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["NV_GPU_INDEX"] = str(logical_index)
    arguments.extend(["--device", f"cuda:{logical_index}"])
    visibility_note = ""
    if removed_visible_devices is not None:
        visibility_note = (
            f" Cleared CUDA_VISIBLE_DEVICES={removed_visible_devices!r} so RTX/Vulkan and CUDA enumerate"
            " the same GPUs."
        )
    print(
        f"[AMGG] Preferred physical GPU {preferred_index} ({identity}) -> "
        f"cuda:{logical_index}, Kit/CloudXR GPU {logical_index}; allowed physical GPUs={allowed_indices}."
        f"{visibility_note}",
        flush=True,
    )
    return logical_index
