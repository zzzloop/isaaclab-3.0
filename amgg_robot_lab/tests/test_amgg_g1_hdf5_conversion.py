# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for G1 recorder HDF5 validation before LeRobot conversion."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


def _load_converter_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "amgg_convert_g1_hdf5_to_lerobot.py"
    spec = importlib.util.spec_from_file_location("amgg_g1_converter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to import AMGG G1 converter.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_episode(path: Path, *, success: bool = True) -> None:
    with h5py.File(path, "w") as file:
        episode = file.create_group("data/demo_0")
        episode.attrs["success"] = success
        episode.create_dataset("actions", data=np.zeros((4, 38), dtype=np.float32))
        episode.create_dataset("processed_actions", data=np.zeros((4, 38), dtype=np.float32))
        obs = episode.create_group("obs")
        obs.create_dataset("robot_joint_pos", data=np.zeros((4, 53), dtype=np.float32))
        obs.create_dataset("rh56dfx_motor_proxy", data=np.zeros((4, 12), dtype=np.float32))
        obs.create_dataset("tactile", data=np.zeros((4, 12), dtype=np.float32))
        obs.create_dataset("left_eef_pos", data=np.zeros((4, 3), dtype=np.float32))
        obs.create_dataset("left_eef_quat", data=np.zeros((4, 4), dtype=np.float32))
        obs.create_dataset("right_eef_pos", data=np.zeros((4, 3), dtype=np.float32))
        obs.create_dataset("right_eef_quat", data=np.zeros((4, 4), dtype=np.float32))
        obs.create_dataset("object_state", data=np.zeros((4, 13), dtype=np.float32))
        obs.create_dataset("goal", data=np.zeros((4, 4), dtype=np.float32))
        obs.create_dataset("progress", data=np.zeros((4, 1), dtype=np.float32))


class TestAmggG1Hdf5Conversion(unittest.TestCase):
    """Validate success filtering and fixed dimensions."""

    def test_loads_hardware_annotated_sim_episode(self) -> None:
        converter = _load_converter_module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "g1.hdf5"
            _write_episode(path)
            episodes = converter.load_g1_episodes(
                path,
                action_source="raw",
                include_images=False,
                only_successful=True,
            )
            self.assertEqual(len(episodes), 1)
            self.assertEqual(episodes[0].state.shape, (4, 53))
            self.assertEqual(episodes[0].tactile.shape, (4, 12))
            self.assertEqual(episodes[0].tcp_pose.shape, (4, 14))

    def test_filters_failed_episode(self) -> None:
        converter = _load_converter_module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "g1_failed.hdf5"
            _write_episode(path, success=False)
            with self.assertRaisesRegex(ValueError, "No matching successful"):
                converter.load_g1_episodes(
                    path,
                    action_source="raw",
                    include_images=False,
                    only_successful=True,
                )

    def test_downsamples_all_streams_with_one_synchronized_stride(self) -> None:
        converter = _load_converter_module()
        frames = np.arange(6, dtype=np.float32)
        episode = converter.G1EpisodeArrays(
            name="demo_0",
            state=np.repeat(frames[:, None], 53, axis=1),
            action=np.repeat(frames[:, None], 38, axis=1),
            motor_proxy=np.repeat(frames[:, None], 12, axis=1),
            tactile=np.repeat(frames[:, None], 12, axis=1),
            tcp_pose=np.repeat(frames[:, None], 14, axis=1),
            environment_state=np.repeat(frames[:, None], 18, axis=1),
            images={"front": np.repeat(frames[:, None, None, None], 3, axis=3).astype(np.uint8)},
        )

        downsampled = converter.downsample_g1_episode(episode, stride=2)

        self.assertEqual(downsampled.length, 3)
        np.testing.assert_array_equal(downsampled.state[:, 0], [0.0, 2.0, 4.0])
        np.testing.assert_array_equal(downsampled.action[:, 0], [0.0, 2.0, 4.0])
        np.testing.assert_array_equal(downsampled.images["front"][:, 0, 0, 0], [0, 2, 4])

    def test_rejects_non_integer_rate_conversion_before_lerobot_import(self) -> None:
        converter = _load_converter_module()
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "g1.hdf5"
            output_path = Path(directory) / "output"
            _write_episode(input_path)
            with self.assertRaisesRegex(ValueError, "integer multiple"):
                converter.convert_g1_dataset(
                    input_path,
                    output_path,
                    task_id="Isaac-AMGG-G1-ClutterTransfer-v0",
                    repo_id="local/test",
                    fps=30,
                    source_fps=50,
                    action_source="raw",
                    include_images=False,
                    only_successful=True,
                )


if __name__ == "__main__":
    unittest.main()
