# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the backend-neutral AMGG dataset schema."""

import unittest

from amgg_robot_lab.recording.amgg_dataset_schema import AmggDatasetSpec


class TestAmggDatasetSchema(unittest.TestCase):
    """Validate schema invariants without requiring Isaac Sim or LeRobot."""

    def test_valid_low_dimensional_schema(self) -> None:
        spec = AmggDatasetSpec(
            fps=30,
            task="AMGG pick and place",
            observation_joint_names=("joint_a",),
            action_joint_names=("joint_a",),
        )
        spec.validate()

    def test_rejects_missing_task(self) -> None:
        spec = AmggDatasetSpec(
            fps=30,
            observation_joint_names=("joint_a",),
            action_joint_names=("joint_a",),
        )
        with self.assertRaises(ValueError):
            spec.validate()


if __name__ == "__main__":
    unittest.main()
