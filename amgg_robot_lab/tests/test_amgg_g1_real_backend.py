# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the guarded Unitree G1 real-robot backend boundary."""

from pathlib import Path
import unittest

from amgg_robot_lab.contracts import AMGG_G1_CONTROLLED_ARM_JOINT_NAMES, AMGG_G1_HAND_MOTOR_NAMES
from amgg_robot_lab.real import AMGG_G1_HARDWARE_COMMAND_NAMES, G1HardwareCommandLimiter, UnitreeG1DryRunBackend


class TestAmggG1RealBackend(unittest.TestCase):
    def test_command_contract_is_arms_then_inspire_hands(self) -> None:
        self.assertEqual(AMGG_G1_HARDWARE_COMMAND_NAMES[:14], AMGG_G1_CONTROLLED_ARM_JOINT_NAMES)
        self.assertEqual(AMGG_G1_HARDWARE_COMMAND_NAMES[14:], AMGG_G1_HAND_MOTOR_NAMES)
        self.assertEqual(len(AMGG_G1_HARDWARE_COMMAND_NAMES), 26)

    def test_dry_run_backend_uses_g1_dimensions(self) -> None:
        backend = UnitreeG1DryRunBackend()
        backend.connect()
        state = backend.read_state()
        self.assertEqual(len(state.joint_positions_rad), 41)
        with self.assertRaises(RuntimeError):
            backend.send_joint_position_targets((0.0,) * 26, 1.0)
        backend.enable()
        backend.send_joint_position_targets((0.1,) * 26, 1.0)
        state = backend.read_state()
        self.assertEqual(state.joint_positions_rad[15:29], (0.1,) * 14)
        self.assertEqual(state.joint_positions_rad[29:], (0.1,) * 12)

    def test_g1_limiter_rate_limits_large_step(self) -> None:
        limiter = G1HardwareCommandLimiter()
        measured = (0.0,) * 26
        limiter.reset(measured, 1.0)
        command = limiter.filter((1.0,) * 26, measured, 1.01)
        self.assertLessEqual(max(abs(value) for value in command[:14]), 0.008 + 1e-9)
        self.assertLessEqual(max(abs(value) for value in command[14:]), 0.015 + 1e-9)

    def test_real_cli_requires_explicit_motion_safety_flags(self) -> None:
        cli_source = (Path(__file__).resolve().parents[1] / "scripts" / "amgg_teleop_real.py").read_text(
            encoding="utf-8"
        )
        backend_source = (
            Path(__file__).resolve().parents[1]
            / "source"
            / "amgg_robot_lab"
            / "amgg_robot_lab"
            / "real"
            / "amgg_g1_unitree_backend.py"
        ).read_text(encoding="utf-8")
        self.assertIn("--enable_motion", cli_source)
        self.assertIn("--physical_estop_ready", cli_source)
        self.assertIn("--robot_supported", cli_source)
        self.assertIn("--operator_clear", cli_source)
        self.assertIn("rt/arm_sdk", backend_source)
        self.assertIn("rt/inspire/cmd", backend_source)


if __name__ == "__main__":
    unittest.main()
