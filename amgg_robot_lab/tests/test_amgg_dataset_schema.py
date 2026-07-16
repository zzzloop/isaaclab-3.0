# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the backend-neutral AMGG dataset schema."""

import unittest

from amgg_robot_lab.recording.amgg_dataset_schema import AmggDatasetSpec, make_amgg_dataset_spec


class TestAmggDatasetSchema(unittest.TestCase):
    """Validate schema invariants without requiring Isaac Sim or LeRobot."""

    def test_valid_low_dimensional_schema(self) -> None:
        spec = make_amgg_dataset_spec("Isaac-AMGG-PickPlace-v0")
        spec.validate()
        self.assertEqual(len(spec.observation_joint_names), 23)
        self.assertEqual(len(spec.action_joint_names), 21)
        self.assertEqual(len(spec.camera_names), 4)

    def test_rejects_missing_task(self) -> None:
        spec = AmggDatasetSpec(
            fps=30,
            task_id="not-registered",
            task="",
            observation_joint_names=("joint_a",),
            action_joint_names=("joint_a",),
            camera_names=(),
        )
        with self.assertRaises(ValueError):
            spec.validate()


if __name__ == "__main__":
    unittest.main()
