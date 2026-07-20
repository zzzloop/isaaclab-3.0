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

        self.assertEqual(logical_index, 1)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", environment)
        self.assertEqual(environment["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual(environment["NV_GPU_INDEX"], "1")
        self.assertIn("--device", arguments)
        self.assertIn("cuda:1", arguments)
        kit_args = arguments[arguments.index("--kit_args") + 1]
        self.assertIn("--/renderer/activeGpu=1", kit_args)
        self.assertIn("--/renderer/multiGpu/enabled=false", kit_args)
        self.assertIn("--/renderer/multiGpu/maxGpuCount=1", kit_args)

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

        self.assertEqual(logical_index, 1)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", environment)
        self.assertEqual(environment["NV_GPU_INDEX"], "1")
        kit_args = arguments[arguments.index("--kit_args") + 1]
        self.assertIn("--/renderer/activeGpu=1", kit_args)

    def test_renderer_index_counts_disallowed_gpus(self) -> None:
        inventory = [
            amgg_gpu._GpuInfo(0, "GPU-zero", "00000000:B1:00.0"),
            amgg_gpu._GpuInfo(1, "GPU-one", "00000000:31:00.0"),
            amgg_gpu._GpuInfo(2, "GPU-two", "00000000:4B:00.0"),
            amgg_gpu._GpuInfo(3, "GPU-three", "00000000:21:00.0"),
        ]
        arguments = ["amgg_record_demos.py", "--xr", "--enable_cameras"]
        environment = {}

        amgg_gpu.configure_preferred_gpu(arguments, environment, inventory)

        self.assertNotIn("CUDA_VISIBLE_DEVICES", environment)
        self.assertEqual(environment["NV_GPU_INDEX"], "2")
        self.assertIn("cuda:2", arguments)
        kit_args = arguments[arguments.index("--kit_args") + 1]
        self.assertIn("--/renderer/activeGpu=2", kit_args)


if __name__ == "__main__":
    unittest.main()
