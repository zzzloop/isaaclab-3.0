# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for AMGG physical-to-logical GPU selection."""

import importlib.util
import sys
import unittest
from pathlib import Path

_AMGG_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_AMGG_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_AMGG_SCRIPTS_DIR))


def _load_gpu_module():
    module_path = _AMGG_SCRIPTS_DIR / "amgg_gpu.py"
    spec = importlib.util.spec_from_file_location("amgg_gpu", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


amgg_gpu = _load_gpu_module()


def _load_record_module():
    module_path = _AMGG_SCRIPTS_DIR / "amgg_record_demos.py"
    spec = importlib.util.spec_from_file_location("amgg_record_demos", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


amgg_record_demos = _load_record_module()


def _load_teleop_module():
    module_path = _AMGG_SCRIPTS_DIR / "amgg_teleop.py"
    spec = importlib.util.spec_from_file_location("amgg_teleop", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


amgg_teleop = _load_teleop_module()


class TestAmggGpu(unittest.TestCase):
    """Validate stable GPU mapping before Isaac Sim starts."""

    def test_default_physical_gpu_zero_is_mapped_without_hiding_devices(self) -> None:
        inventory = [
            amgg_gpu._GpuInfo(0, "GPU-zero", "00000000:31:00.0"),
            amgg_gpu._GpuInfo(1, "GPU-one", "00000000:4B:00.0"),
            amgg_gpu._GpuInfo(2, "GPU-two", "00000000:B1:00.0"),
            amgg_gpu._GpuInfo(3, "GPU-three", "00000000:CA:00.0"),
        ]
        arguments = ["amgg_teleop.py", "--xr"]
        environment = {"CUDA_VISIBLE_DEVICES": "0,1,2"}

        logical_index = amgg_gpu.configure_preferred_gpu(arguments, environment, inventory)

        self.assertEqual(logical_index, 0)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", environment)
        self.assertEqual(environment["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual(environment["NV_GPU_INDEX"], "0")
        self.assertEqual(environment["AMGG_SELECTED_CUDA_PHYSICAL_GPU"], "0")
        self.assertEqual(environment["AMGG_SELECTED_KIT_PHYSICAL_GPU"], "0")
        self.assertEqual(environment["AMGG_SELECTED_CLOUDXR_PHYSICAL_GPU"], "0")
        self.assertEqual(environment["AMGG_SELECTED_CUDA_LOGICAL_INDEX"], "0")
        self.assertEqual(arguments[-2:], ["--device", "cuda:0"])

    def test_explicit_device_preserves_environment(self) -> None:
        arguments = ["amgg_teleop.py", "--device", "cuda:1"]
        environment = {"CUDA_VISIBLE_DEVICES": "1"}

        logical_index = amgg_gpu.configure_preferred_gpu(arguments, environment, inventory=[])

        self.assertIsNone(logical_index)
        self.assertEqual(environment, {"CUDA_VISIBLE_DEVICES": "1"})
        self.assertEqual(arguments, ["amgg_teleop.py", "--device", "cuda:1"])

    def test_explicit_device_rejects_display_gpu(self) -> None:
        arguments = ["amgg_teleop.py", "--device", "cuda:2"]
        environment = {}

        with self.assertRaises(SystemExit):
            amgg_gpu.configure_preferred_gpu(arguments, environment, inventory=[])

    def test_physical_gpu_override(self) -> None:
        inventory = [
            amgg_gpu._GpuInfo(0, "GPU-zero", "00000000:31:00.0"),
            amgg_gpu._GpuInfo(1, "GPU-one", "00000000:4B:00.0"),
            amgg_gpu._GpuInfo(2, "GPU-two", "00000000:B1:00.0"),
        ]
        arguments = ["amgg_smoke_test.py"]
        environment = {"AMGG_PREFERRED_GPU": "1", "AMGG_ALLOWED_GPUS": "0,1"}

        logical_index = amgg_gpu.configure_preferred_gpu(arguments, environment, inventory)

        self.assertEqual(logical_index, 1)
        self.assertEqual(environment["NV_GPU_INDEX"], "1")
        self.assertEqual(arguments[-1], "cuda:1")

    def test_default_physical_gpu_zero_supports_single_gpu_hosts(self) -> None:
        inventory = [amgg_gpu._GpuInfo(0, "GPU-zero", "00000000:01:00.0")]
        arguments = ["amgg_record_demos.py", "--xr"]
        environment = {}

        logical_index = amgg_gpu.configure_preferred_gpu(arguments, environment, inventory)

        self.assertEqual(logical_index, 0)
        self.assertEqual(environment["NV_GPU_INDEX"], "0")
        self.assertEqual(arguments[-1], "cuda:0")
        self.assertNotIn("CUDA_VISIBLE_DEVICES", environment)

    def test_preferred_physical_gpu_falls_back_to_next_allowed(self) -> None:
        # Preferred physical 1 is absent, but allowed physical 0 is present.
        # Should fall back to physical 0 instead of touching display GPU2.
        inventory = [
            amgg_gpu._GpuInfo(0, "GPU-zero", "00000000:31:00.0"),
            amgg_gpu._GpuInfo(2, "GPU-two", "00000000:B1:00.0"),
        ]
        arguments = ["amgg_record_demos.py", "--xr"]
        environment = {"AMGG_PREFERRED_GPU": "1", "AMGG_ALLOWED_GPUS": "0,1"}

        logical_index = amgg_gpu.configure_preferred_gpu(arguments, environment, inventory)

        self.assertEqual(logical_index, 0)
        self.assertEqual(environment["NV_GPU_INDEX"], "0")

    def test_display_gpu_two_requires_explicit_override(self) -> None:
        inventory = [
            amgg_gpu._GpuInfo(0, "GPU-zero", "00000000:31:00.0"),
            amgg_gpu._GpuInfo(2, "GPU-two", "00000000:B1:00.0"),
        ]
        for environment in (
            {"AMGG_ALLOWED_GPUS": "0,1,2"},
            {"AMGG_PREFERRED_GPU": "2", "AMGG_ALLOWED_GPUS": "0,1,2"},
            {"AMGG_KIT_GPU_INDEX": "2", "AMGG_ALLOWED_GPUS": "0,1,2"},
            {"AMGG_CLOUDXR_GPU_INDEX": "2", "AMGG_ALLOWED_GPUS": "0,1,2"},
        ):
            with self.subTest(environment=environment):
                arguments = ["amgg_record_demos.py", "--xr"]
                with self.assertRaises(SystemExit):
                    amgg_gpu.configure_preferred_gpu(arguments, environment, inventory)
                self.assertNotIn("--device", arguments)

    def test_display_gpu_two_can_be_allowed_explicitly(self) -> None:
        inventory = [
            amgg_gpu._GpuInfo(0, "GPU-zero", "00000000:31:00.0"),
            amgg_gpu._GpuInfo(2, "GPU-two", "00000000:B1:00.0"),
        ]
        arguments = ["amgg_record_demos.py", "--xr"]
        environment = {
            "AMGG_ALLOW_DISPLAY_GPU": "1",
            "AMGG_ALLOWED_GPUS": "0,1,2",
            "AMGG_PREFERRED_GPU": "2",
        }

        logical_index = amgg_gpu.configure_preferred_gpu(arguments, environment, inventory)

        self.assertEqual(logical_index, 1)
        self.assertEqual(environment["NV_GPU_INDEX"], "2")
        self.assertEqual(environment["AMGG_SELECTED_CUDA_PHYSICAL_GPU"], "2")
        self.assertEqual(arguments[-1], "cuda:1")

    def test_gpu_three_is_blocked_even_when_requested(self) -> None:
        inventory = [
            amgg_gpu._GpuInfo(0, "GPU-zero", "00000000:31:00.0"),
            amgg_gpu._GpuInfo(3, "GPU-three", "00000000:CA:00.0"),
        ]
        for environment in (
            {"AMGG_ALLOWED_GPUS": "0,1,3"},
            {"AMGG_PREFERRED_GPU": "3", "AMGG_ALLOWED_GPUS": "0,1"},
            {"AMGG_KIT_GPU_INDEX": "3", "AMGG_ALLOWED_GPUS": "0,1"},
            {"AMGG_CLOUDXR_GPU_INDEX": "3", "AMGG_ALLOWED_GPUS": "0,1"},
        ):
            with self.subTest(environment=environment):
                arguments = ["amgg_record_demos.py", "--xr"]
                with self.assertRaises(SystemExit):
                    amgg_gpu.configure_preferred_gpu(arguments, environment, inventory)
                self.assertNotIn("--device", arguments)

    def test_split_compute_kit_cloudxr_gpu_is_rejected(self) -> None:
        inventory = [
            amgg_gpu._GpuInfo(0, "GPU-zero", "00000000:31:00.0"),
            amgg_gpu._GpuInfo(1, "GPU-one", "00000000:4B:00.0"),
        ]
        for environment in (
            {"AMGG_PREFERRED_GPU": "0", "AMGG_KIT_GPU_INDEX": "1", "AMGG_ALLOWED_GPUS": "0,1"},
            {"AMGG_PREFERRED_GPU": "0", "AMGG_CLOUDXR_GPU_INDEX": "1", "AMGG_ALLOWED_GPUS": "0,1"},
        ):
            with self.subTest(environment=environment):
                arguments = ["amgg_record_demos.py", "--xr"]
                with self.assertRaises(SystemExit):
                    amgg_gpu.configure_preferred_gpu(arguments, environment, inventory)
                self.assertNotIn("--device", arguments)

    def test_recorder_limits_xr_kit_to_single_gpu(self) -> None:
        original_argv = sys.argv[:]
        try:
            sys.argv = ["amgg_record_demos.py", "--xr", "--kit_args", "--/app/window/width=1280"]

            amgg_record_demos.merge_kit_args(amgg_record_demos.AMGG_XR_KIT_ARGS)

            self.assertTrue(sys.argv[3].startswith("--/app/window/width=1280 "))
            self.assertIn("--/renderer/multiGpu/enabled=false", sys.argv[3])
            self.assertIn("--/renderer/multiGpu/autoEnable=false", sys.argv[3])
            self.assertIn("--/renderer/multiGpu/maxGpuCount=1", sys.argv[3])
            self.assertIn("--/exts/omni.kit.renderer.core/present/enabled=false", sys.argv[3])
            self.assertIn("--/app/updateOrder/checkForHydraRenderComplete=1000", sys.argv[3])
            self.assertIn("--/app/renderer/waitIdle=true", sys.argv[3])
            self.assertIn("--/app/hydraEngine/waitIdle=true", sys.argv[3])
            self.assertIn("--/app/asyncRendering=false", sys.argv[3])
            self.assertIn("--/app/asyncRenderingLowLatency=false", sys.argv[3])
            self.assertIn("--/omni/replicator/asyncRendering=false", sys.argv[3])
        finally:
            sys.argv = original_argv

    def test_recorder_adds_xr_kit_args_when_missing(self) -> None:
        original_argv = sys.argv[:]
        try:
            sys.argv = ["amgg_record_demos.py", "--xr"]

            amgg_record_demos.merge_kit_args(amgg_record_demos.AMGG_XR_KIT_ARGS)

            self.assertEqual(sys.argv[-2], "--kit_args")
            self.assertEqual(
                sys.argv[-1],
                " ".join(amgg_record_demos.AMGG_XR_KIT_ARGS),
            )
        finally:
            sys.argv = original_argv

    def test_teleop_adds_xr_kit_args_when_missing(self) -> None:
        original_argv = sys.argv[:]
        try:
            sys.argv = ["amgg_teleop.py", "--xr"]

            amgg_teleop.merge_kit_args(amgg_teleop.AMGG_XR_KIT_ARGS)

            self.assertEqual(sys.argv[-2], "--kit_args")
            self.assertIn("--/renderer/multiGpu/enabled=false", sys.argv[-1])
            self.assertIn("--/exts/omni.kit.renderer.core/present/enabled=false", sys.argv[-1])
            self.assertIn("--/app/asyncRendering=false", sys.argv[-1])
        finally:
            sys.argv = original_argv

    def test_recorder_defaults_to_continuous_collection(self) -> None:
        original_argv = sys.argv[:]
        try:
            sys.argv = ["amgg_record_demos.py", "--xr"]

            amgg_record_demos._inject_continuous_demo_default()

            self.assertEqual(sys.argv[-2:], ["--num_demos", "0"])
        finally:
            sys.argv = original_argv

    def test_recorder_treats_single_demo_as_continuous_collection(self) -> None:
        original_argv = sys.argv[:]
        try:
            sys.argv = ["amgg_record_demos.py", "--xr", "--num_demos", "1"]

            amgg_record_demos._inject_continuous_demo_default()

            self.assertEqual(sys.argv[-2:], ["--num_demos", "0"])
        finally:
            sys.argv = original_argv

    def test_recorder_preserves_finite_multi_demo_count(self) -> None:
        original_argv = sys.argv[:]
        try:
            sys.argv = ["amgg_record_demos.py", "--xr", "--num_demos=3"]

            amgg_record_demos._inject_continuous_demo_default()

            self.assertEqual(sys.argv[-1], "--num_demos=3")
        finally:
            sys.argv = original_argv


if __name__ == "__main__":
    unittest.main()
