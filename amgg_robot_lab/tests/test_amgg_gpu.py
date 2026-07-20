# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for AMGG physical-to-logical GPU selection."""

import importlib.util
import sys
import unittest
from pathlib import Path


def _load_gpu_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "amgg_gpu.py"
    spec = importlib.util.spec_from_file_location("amgg_gpu", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


amgg_gpu = _load_gpu_module()


class TestAmggGpu(unittest.TestCase):
    """Validate stable GPU mapping before Isaac Sim starts."""

    def test_preferred_physical_gpu_is_mapped_by_pci_order(self) -> None:
        inventory = [
            amgg_gpu._GpuInfo(0, "GPU-zero", "00000000:B1:00.0"),
            amgg_gpu._GpuInfo(1, "GPU-one", "00000000:31:00.0"),
            amgg_gpu._GpuInfo(2, "GPU-two", "00000000:4B:00.0"),
        ]
        arguments = ["amgg_teleop.py", "--xr"]
        environment = {}

        logical_index = amgg_gpu.configure_preferred_gpu(arguments, environment, inventory)

        self.assertEqual(logical_index, 0)
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "GPU-two")
        self.assertEqual(environment["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual(environment["NV_GPU_INDEX"], "0")
        self.assertIn("--device", arguments)
        self.assertIn("cuda:0", arguments)
        self.assertIn("--/renderer/activeGpu=2", arguments)
        self.assertIn("--/renderer/multiGpu/enabled=false", arguments)
        self.assertIn("--/renderer/multiGpu/maxGpuCount=1", arguments)

    def test_explicit_device_preserves_environment(self) -> None:
        arguments = ["amgg_teleop.py", "--device", "cuda:1"]
        environment = {"CUDA_VISIBLE_DEVICES": "1"}

        logical_index = amgg_gpu.configure_preferred_gpu(arguments, environment, inventory=[])

        self.assertIsNone(logical_index)
        self.assertEqual(environment, {"CUDA_VISIBLE_DEVICES": "1"})
        self.assertEqual(arguments, ["amgg_teleop.py", "--device", "cuda:1"])

    def test_physical_gpu_override(self) -> None:
        inventory = [
            amgg_gpu._GpuInfo(0, "GPU-zero", "00000000:31:00.0"),
            amgg_gpu._GpuInfo(1, "GPU-one", "00000000:4B:00.0"),
            amgg_gpu._GpuInfo(2, "GPU-two", "00000000:B1:00.0"),
        ]
        arguments = ["amgg_smoke_test.py"]
        environment = {"AMGG_PREFERRED_GPU": "1", "AMGG_ALLOWED_GPUS": "0,1,2"}

        logical_index = amgg_gpu.configure_preferred_gpu(arguments, environment, inventory)

        self.assertEqual(logical_index, 0)
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "GPU-one")
        self.assertEqual(environment["NV_GPU_INDEX"], "0")
        self.assertIn("--/renderer/activeGpu=1", arguments)


if __name__ == "__main__":
    unittest.main()
