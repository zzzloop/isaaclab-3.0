# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for hardware-boundary safety behavior."""

import unittest

from amgg_robot_lab.contracts import AMGG_CONTROLLED_JOINT_NAMES, AMGG_HOME_POSITIONS
from amgg_robot_lab.real import AmggDryRunBackend
from amgg_robot_lab.teleop.amgg_safety import AmggCommandLimiter


class TestAmggSafety(unittest.TestCase):
    def test_limiter_rate_limits_large_step(self) -> None:
        home = tuple(AMGG_HOME_POSITIONS[name] for name in AMGG_CONTROLLED_JOINT_NAMES)
        limiter = AmggCommandLimiter()
        limiter.reset(home, 1.0)
        request = tuple(value + 1.0 for value in home)
        command = limiter.filter(request, home, 1.01)
        self.assertTrue(all(abs(value - old) <= 0.02 + 1e-9 for value, old in zip(command, home, strict=True)))

    def test_dry_run_backend_requires_enable(self) -> None:
        backend = AmggDryRunBackend()
        backend.connect()
        command = tuple(AMGG_HOME_POSITIONS[name] for name in AMGG_CONTROLLED_JOINT_NAMES)
        with self.assertRaises(RuntimeError):
            backend.send_joint_position_targets(command, 1.0)
        backend.enable()
        backend.send_joint_position_targets(command, 1.0)
        self.assertEqual(backend.read_state().joint_positions_rad[: len(command)], command)


if __name__ == "__main__":
    unittest.main()
