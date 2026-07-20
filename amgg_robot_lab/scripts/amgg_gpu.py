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
    """Pin CUDA, PhysX, RTX, and CloudXR to one physical GPU.

    The AMGG default is physical GPU 2. Camera recording uses CUDA/RTX
    interop, so exposing several GPUs while physics and rendering select
    different devices can poison the CUDA context with error 700. Only the
    selected GPU is exposed to CUDA, while Kit receives the corresponding
    physical renderer index and multi-GPU rendering is disabled. Passing
    ``--device`` opts out so an explicit operator choice is preserved.

    Args:
        arguments: Process argument vector to update. Defaults to :attr:`sys.argv`.
        environment: Process environment to update. Defaults to :attr:`os.environ`.
        inventory: Optional GPU inventory for deterministic testing.

    Returns:
        Logical CUDA GPU index 0, or ``None`` when ``--device`` was explicitly
        provided.
    """
    arguments = sys.argv if arguments is None else arguments
    environment = os.environ if environment is None else environment
    if _has_device_argument(arguments):
        print("[AMGG] Explicit --device detected; automatic physical-GPU selection is disabled.", flush=True)
        return None

    try:
        preferred_index = int(environment.get("AMGG_PREFERRED_GPU", "2"))
        allowed_indices = _parse_physical_indices(environment.get("AMGG_ALLOWED_GPUS", "0,1,2"))
    except ValueError as error:
        raise SystemExit(f"Invalid AMGG GPU configuration: {error}") from error
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
        logical_index = 0
        environment["CUDA_VISIBLE_DEVICES"] = selected.uuid
        identity = f"UUID={selected.uuid}, PCI={selected.pci_bus_id}"
    except (FileNotFoundError, subprocess.SubprocessError, RuntimeError) as error:
        logical_index = 0
        environment["CUDA_VISIBLE_DEVICES"] = str(preferred_index)
        identity = "UUID/PCI unavailable"
        print(f"[AMGG] Warning: GPU identity probe failed ({error}); using ordinal fallback.", flush=True)

    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # CloudXR sees the same one-device CUDA namespace, hence logical index 0.
    environment["NV_GPU_INDEX"] = "0"
    arguments.extend(
        [
            "--device",
            "cuda:0",
            f"--/renderer/activeGpu={preferred_index}",
            "--/renderer/multiGpu/enabled=false",
            "--/renderer/multiGpu/autoEnable=false",
            "--/renderer/multiGpu/maxGpuCount=1",
        ]
    )
    print(
        f"[AMGG] Preferred physical GPU {preferred_index} ({identity}) -> CUDA/PhysX cuda:0, "
        f"RTX physical GPU {preferred_index}, CloudXR logical GPU 0; multi-GPU rendering disabled; "
        f"allowed physical GPUs={allowed_indices}.",
        flush=True,
    )
    return logical_index
