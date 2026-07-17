# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Unitree G1 with RH56DFX data contract."""

import unittest

from amgg_robot_lab.contracts.amgg_g1_contract import (
    AMGG_G1_HAND_MOTOR_NAMES,
    AMGG_G1_HARDWARE_ACTION_NAMES,
    AMGG_G1_OBSERVATION_JOINT_NAMES,
    AMGG_G1_SIM_OBSERVATION_JOINT_NAMES,
    AMGG_G1_SIM_RAW_ACTION_NAMES,
    AMGG_G1_TACTILE_NAMES,
    validate_amgg_g1_contract,
)
from amgg_robot_lab.recording.amgg_g1_dataset_schema import make_amgg_g1_dataset_spec


class TestAmggG1Contract(unittest.TestCase):
    """Keep the simulation and real-hand ABIs explicit and versioned."""

    def test_dimensions(self) -> None:
        validate_amgg_g1_contract()
        self.assertEqual(len(AMGG_G1_SIM_OBSERVATION_JOINT_NAMES), 53)
        self.assertEqual(len(AMGG_G1_OBSERVATION_JOINT_NAMES), 41)
        self.assertEqual(len(AMGG_G1_SIM_RAW_ACTION_NAMES), 38)
        self.assertEqual(len(AMGG_G1_HARDWARE_ACTION_NAMES), 26)
        self.assertEqual(len(AMGG_G1_TACTILE_NAMES), 12)

    def test_unitree_motor_order(self) -> None:
        self.assertEqual(
            AMGG_G1_HAND_MOTOR_NAMES[:6],
            (
                "right_pinky",
                "right_ring",
                "right_middle",
                "right_index",
                "right_thumb_bend",
                "right_thumb_rotation",
            ),
        )

    def test_dataset_task(self) -> None:
        spec = make_amgg_g1_dataset_spec("Isaac-AMGG-G1-PrecisionInsert-v0", fps=30)
        self.assertEqual(len(spec.action_names("raw")), 38)
        self.assertIn("insert", spec.instruction)


if __name__ == "__main__":
    unittest.main()
