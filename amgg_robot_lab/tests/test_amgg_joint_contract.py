# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for AMGG joint-contract validation."""

import unittest

from amgg_robot_lab.contracts.amgg_joint_contract import validate_amgg_joint_names


class TestAmggJointContract(unittest.TestCase):
    """Validate stable ordered joint-name rules without requiring Isaac Sim."""

    def test_accepts_unique_ordered_names(self) -> None:
        self.assertEqual(validate_amgg_joint_names(("joint_a", "joint_b")), ("joint_a", "joint_b"))

    def test_rejects_duplicate_names(self) -> None:
        with self.assertRaises(ValueError):
            validate_amgg_joint_names(("joint_a", "joint_a"))


if __name__ == "__main__":
    unittest.main()
