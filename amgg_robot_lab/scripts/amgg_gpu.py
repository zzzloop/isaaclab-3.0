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

_AMGG_DEFAULT_GPU = 0
_AMGG_DEFAULT_ALLOWED_GPUS = "0,1"
_AMGG_DEFAULT_DISPLAY_GPUS = "2"
_AMGG_BAD_GPUS = frozenset({3})
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


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


def _find_device_argument(arguments: Sequence[str]) -> str | None:
    for index, argument in enumerate(arguments[1:], start=1):
        if argument == "--device":
            if index + 1 >= len(arguments):
                raise SystemExit("--device requires a value.")
            return arguments[index + 1]
        if argument.startswith("--device="):
            return argument.split("=", maxsplit=1)[1]
    return None


def _parse_bool(value: str | None) -> bool:
    return value is not None and value.strip().lower() in _TRUE_VALUES


def _parse_physical_indices(value: str) -> list[int]:
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not indices:
        raise ValueError("AMGG_ALLOWED_GPUS must contain at least one physical GPU index.")
    if len(indices) != len(set(indices)):
        raise ValueError("AMGG_ALLOWED_GPUS must not contain duplicate physical GPU indices.")
    return indices


def _parse_gpu_index(name: str, value: str) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer physical GPU index.") from error


def _parse_optional_gpu_index(name: str, environment: MutableMapping[str, str]) -> int | None:
    value = environment.get(name)
    if value is None or value == "":
        return None
    return _parse_gpu_index(name, value)


def _parse_display_gpu_indices(environment: MutableMapping[str, str]) -> list[int]:
    value = environment.get("AMGG_DISPLAY_GPUS", _AMGG_DEFAULT_DISPLAY_GPUS)
    if value.strip().lower() in ("", "none"):
        return []
    if value.strip().lower() == "auto":
        return _query_display_gpu_indices()
    return _parse_physical_indices(value)


def _parse_cuda_device_ordinal(device: str) -> int | None:
    if device == "cpu":
        return None
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        try:
            return int(device.split(":", maxsplit=1)[1])
        except ValueError as error:
            raise SystemExit(f"Invalid --device value {device!r}; expected cuda:<index>.") from error
    raise SystemExit(f"Invalid --device value {device!r}; expected cpu, cuda, or cuda:<index>.")


def _query_display_gpu_indices() -> list[int]:
    """Best-effort detection of GPUs currently serving Xorg/GNOME/GDM."""
    result = subprocess.run(
        ["nvidia-smi", "pmon", "-c", "1", "-s", "u"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    display_gpus: set[int] = set()
    display_process_names = ("xorg", "gnome-shell", "gdm", "kwin", "wayland", "xwayland")
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        columns = stripped.split()
        if len(columns) < 5 or not columns[0].lstrip("-").isdigit():
            continue
        process_name = columns[-1].lower()
        process_type = columns[3].upper()
        if "G" in process_type or any(name in process_name for name in display_process_names):
            display_gpus.add(int(columns[0]))
    return sorted(display_gpus)


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


def _validate_index_is_safe(
    name: str,
    index: int,
    allowed_indices: Sequence[int],
    display_indices: Sequence[int],
    allow_display_gpu: bool,
) -> None:
    if index in _AMGG_BAD_GPUS:
        raise SystemExit(f"{name}={index} is blocked: AMGG treats physical GPU3 as a bad card.")
    if index in display_indices and not allow_display_gpu:
        raise SystemExit(
            f"{name}={index} targets DISPLAY GPU {index}. This card is reserved for Xorg/GNOME/GDM and can freeze "
            "the desktop if IsaacLab/RTX/CloudXR crashes. Set AMGG_ALLOW_DISPLAY_GPU=1 only if you intentionally "
            "accept that risk."
        )
    if index not in allowed_indices:
        raise SystemExit(f"{name}={index} is not present in AMGG_ALLOWED_GPUS={list(allowed_indices)}.")


def _physical_to_cuda_logical(physical_index: int, ordered_gpus: Sequence[_GpuInfo]) -> int:
    return next(index for index, gpu in enumerate(ordered_gpus) if gpu.physical_index == physical_index)


def _select_present_allowed_gpu(
    preferred_index: int,
    allowed_indices: Sequence[int],
    by_physical_index: dict[int, _GpuInfo],
) -> int:
    candidate_indices = [preferred_index] + [index for index in allowed_indices if index != preferred_index]
    selected_physical = next((index for index in candidate_indices if index in by_physical_index), None)
    if selected_physical is None:
        detected_indices = sorted(by_physical_index)
        raise SystemExit(
            f"None of the allowed AMGG physical GPUs {list(allowed_indices)} are present. "
            f"Detected physical GPUs: {detected_indices}. Refusing to fall back outside the allow-list."
        )
    if selected_physical != preferred_index:
        print(
            f"[AMGG] Warning: preferred physical GPU {preferred_index} is not available;"
            f" falling back to physical GPU {selected_physical}.",
            flush=True,
        )
    return selected_physical


def configure_preferred_gpu(
    arguments: list[str] | None = None,
    environment: MutableMapping[str, str] | None = None,
    inventory: Sequence[_GpuInfo] | None = None,
) -> int | None:
    """Map safe physical GPUs to CUDA, Kit/RTX, and CloudXR/OpenXR settings.

    The AMGG defaults use physical GPU 0 and allow fallback only to physical
    GPU 1. Physical GPU 2 is treated as the display/Xorg/GNOME card on the
    AMGG workstation, and physical GPU 3 is treated as a bad card. All GPUs
    remain visible because
    Isaac Sim RTX/Vulkan device discovery can fail when ``CUDA_VISIBLE_DEVICES``
    hides GPUs from CUDA while Omniverse still enumerates them for graphics.

    Isaac Lab maps ``--device cuda:X`` to both ``physics_gpu`` and
    ``active_gpu`` for Kit/RTX. For XR, CloudXR/OpenXR should use the same
    physical GPU as Kit/RTX. The environment variables are still separated so
    startup logs show the intended CUDA, Kit/RTX, and CloudXR/OpenXR choices,
    but by default the choices must agree to avoid cross-GPU deadlocks.

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
    allow_display_gpu = _parse_bool(environment.get("AMGG_ALLOW_DISPLAY_GPU"))
    try:
        display_indices = _parse_display_gpu_indices(environment)
        allowed_indices = _parse_physical_indices(environment.get("AMGG_ALLOWED_GPUS", _AMGG_DEFAULT_ALLOWED_GPUS))
        blocked_indices = sorted(set(display_indices if not allow_display_gpu else []) | _AMGG_BAD_GPUS)
        for index in allowed_indices:
            _validate_index_is_safe("AMGG_ALLOWED_GPUS", index, allowed_indices, display_indices, allow_display_gpu)
    except (subprocess.SubprocessError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Invalid AMGG GPU configuration: {error}") from error

    explicit_device = _find_device_argument(arguments)
    if explicit_device is not None:
        device_ordinal = _parse_cuda_device_ordinal(explicit_device)
        if device_ordinal is not None:
            _validate_index_is_safe("--device", device_ordinal, allowed_indices, display_indices, allow_display_gpu)
        print(
            f"[AMGG] Explicit --device={explicit_device} detected; automatic physical-GPU selection is disabled. "
            f"DISPLAY GPU detected={display_indices}; blocked GPUs={blocked_indices}; "
            f"allowed physical GPUs={allowed_indices}.",
            flush=True,
        )
        return None

    try:
        preferred_index = _parse_gpu_index(
            "AMGG_PREFERRED_GPU", environment.get("AMGG_PREFERRED_GPU", str(_AMGG_DEFAULT_GPU))
        )
        kit_index = _parse_optional_gpu_index("AMGG_KIT_GPU_INDEX", environment)
        cloudxr_index = _parse_optional_gpu_index("AMGG_CLOUDXR_GPU_INDEX", environment)
    except ValueError as error:
        raise SystemExit(f"Invalid AMGG GPU configuration: {error}") from error
    _validate_index_is_safe("AMGG_PREFERRED_GPU", preferred_index, allowed_indices, display_indices, allow_display_gpu)

    try:
        detected = list(inventory) if inventory is not None else _query_gpu_inventory()
        by_physical_index = {gpu.physical_index: gpu for gpu in detected}
        ordered_gpus = sorted(detected, key=lambda gpu: gpu.pci_sort_key)
        selected_physical = _select_present_allowed_gpu(preferred_index, allowed_indices, by_physical_index)
        if kit_index is None:
            kit_index = selected_physical
        if cloudxr_index is None:
            cloudxr_index = kit_index
        _validate_index_is_safe("AMGG_KIT_GPU_INDEX", kit_index, allowed_indices, display_indices, allow_display_gpu)
        _validate_index_is_safe(
            "AMGG_CLOUDXR_GPU_INDEX", cloudxr_index, allowed_indices, display_indices, allow_display_gpu
        )
        if kit_index != selected_physical:
            raise SystemExit(
                f"AMGG_KIT_GPU_INDEX={kit_index} does not match CUDA compute physical GPU {selected_physical}. "
                "Isaac Lab maps --device to Kit/RTX active_gpu, so AMGG keeps compute and Kit/RTX on the same "
                "physical GPU for XR stability."
            )
        if cloudxr_index != kit_index:
            raise SystemExit(
                f"AMGG_CLOUDXR_GPU_INDEX={cloudxr_index} does not match Kit/RTX physical GPU {kit_index}. "
                "CloudXR/OpenXR must stream from the same GPU that Kit/RTX renders on."
            )
        logical_index = _physical_to_cuda_logical(selected_physical, ordered_gpus)
        selected = ordered_gpus[logical_index]
        identity = f"UUID={selected.uuid}, PCI={selected.pci_bus_id}"
    except (FileNotFoundError, subprocess.SubprocessError, RuntimeError) as error:
        logical_index = preferred_index
        selected_physical = preferred_index
        kit_index = selected_physical if kit_index is None else kit_index
        cloudxr_index = kit_index if cloudxr_index is None else cloudxr_index
        identity = "UUID/PCI unavailable"
        print(f"[AMGG] Warning: GPU identity probe failed ({error}); using ordinal fallback.", flush=True)

    removed_visible_devices = environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["NV_GPU_INDEX"] = str(cloudxr_index)
    environment["AMGG_SELECTED_CUDA_PHYSICAL_GPU"] = str(selected_physical)
    environment["AMGG_SELECTED_KIT_PHYSICAL_GPU"] = str(kit_index)
    environment["AMGG_SELECTED_CLOUDXR_PHYSICAL_GPU"] = str(cloudxr_index)
    environment["AMGG_SELECTED_CUDA_LOGICAL_INDEX"] = str(logical_index)
    arguments.extend(["--device", f"cuda:{logical_index}"])
    visibility_note = ""
    if removed_visible_devices is not None:
        visibility_note = (
            f" Cleared CUDA_VISIBLE_DEVICES={removed_visible_devices!r} so RTX/Vulkan and CUDA enumerate"
            " the same GPUs."
        )
    if allow_display_gpu and any(index in display_indices for index in (selected_physical, kit_index, cloudxr_index)):
        print(
            "[AMGG] WARNING: AMGG_ALLOW_DISPLAY_GPU=1 allows IsaacLab/RTX/CloudXR to use a DISPLAY GPU. "
            "A GPU crash on this card can freeze Xorg/GNOME/GDM.",
            flush=True,
        )
    print(
        f"[AMGG] GPU selection: CUDA compute physical GPU {selected_physical} ({identity}) -> cuda:{logical_index}; "
        f"Kit/RTX physical GPU {kit_index}; CloudXR/OpenXR physical GPU {cloudxr_index}; "
        f"NV_GPU_INDEX={environment['NV_GPU_INDEX']}; DISPLAY GPU detected={display_indices}; "
        f"blocked GPUs={blocked_indices}; allowed physical GPUs={allowed_indices}."
        f"{visibility_note}",
        flush=True,
    )
    return logical_index
