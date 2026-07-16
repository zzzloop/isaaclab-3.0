# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for HDF5 validation before LeRobot conversion."""

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np


def _load_converter_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "amgg_convert_hdf5_to_lerobot.py"
    spec = importlib.util.spec_from_file_location("amgg_converter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to import AMGG converter.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestAmggHdf5Conversion(unittest.TestCase):
    """Validate episode filtering, dimensions, and synchronization."""

    @staticmethod
    def _write_episode(path: Path) -> None:
        with h5py.File(path, "w") as file:
            episode = file.create_group("data/demo_0")
            episode.attrs["success"] = True
            episode.create_dataset("processed_actions", data=np.zeros((3, 21), dtype=np.float32))
            obs = episode.create_group("obs")
            obs.create_dataset("robot_joint_pos", data=np.zeros((3, 23), dtype=np.float32))
            obs.create_dataset("left_tcp_pose", data=np.zeros((3, 7), dtype=np.float32))
            obs.create_dataset("right_tcp_pose", data=np.zeros((3, 7), dtype=np.float32))
            obs.create_dataset("object_state", data=np.zeros((3, 13), dtype=np.float32))
            obs.create_dataset("goal", data=np.zeros((3, 4), dtype=np.float32))
            obs.create_dataset("progress", data=np.zeros((3, 1), dtype=np.float32))

    def test_loads_successful_processed_action_episode(self) -> None:
        converter = _load_converter_module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.hdf5"
            self._write_episode(path)
            episodes = converter.load_episodes(
                path,
                action_source="processed",
                include_images=False,
                only_successful=True,
            )
            self.assertEqual(len(episodes), 1)
            self.assertEqual(episodes[0].length, 3)
            self.assertEqual(episodes[0].environment_state.shape, (3, 18))

    def test_full_adapter_finalizes_lerobot_dataset(self) -> None:
        converter = _load_converter_module()

        class FakeLeRobotDataset:
            created = None

            def __init__(self, root: Path):
                self.root = root
                self.frames = []
                self.saved_episodes = 0
                self.finalized = False

            @classmethod
            def create(cls, *, root, **_kwargs):
                root = Path(root)
                (root / "meta").mkdir(parents=True)
                cls.created = cls(root)
                return cls.created

            def add_frame(self, frame, *, task):
                self.frames.append((frame, task))

            def save_episode(self):
                self.saved_episodes += 1

            def finalize(self):
                self.finalized = True

        datasets_module = types.ModuleType("lerobot.datasets")
        datasets_module.LeRobotDataset = FakeLeRobotDataset
        lerobot_module = types.ModuleType("lerobot")
        lerobot_module.datasets = datasets_module

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "demo.hdf5"
            output_path = root / "lerobot"
            self._write_episode(input_path)
            with patch.dict(sys.modules, {"lerobot": lerobot_module, "lerobot.datasets": datasets_module}):
                episode_count, frame_count = converter.convert_dataset(
                    input_path,
                    output_path,
                    task_id="Isaac-AMGG-PickPlace-v0",
                    repo_id="local/test",
                    fps=30,
                    action_source="processed",
                    include_images=False,
                    only_successful=True,
                )
            dataset = FakeLeRobotDataset.created
            self.assertIsNotNone(dataset)
            self.assertEqual((episode_count, frame_count), (1, 3))
            self.assertEqual(dataset.saved_episodes, 1)
            self.assertTrue(dataset.finalized)
            self.assertEqual(len(dataset.frames), 3)
            self.assertIn("Pick up the orange cube", dataset.frames[0][1])
            metadata = json.loads((output_path / "meta" / "amgg_schema.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["action_source"], "processed")
            self.assertEqual(len(metadata["state_names"]), 23)


if __name__ == "__main__":
    unittest.main()
