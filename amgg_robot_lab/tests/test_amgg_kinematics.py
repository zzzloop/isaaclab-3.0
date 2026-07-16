# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Numerical tests for the normalized AMGG URDF kinematics."""

import unittest

import numpy as np
from amgg_robot_lab.contracts import AMGG_FRAMES, AMGG_HOME_POSITIONS, AMGG_IK_JOINT_NAMES
from amgg_robot_lab.kinematics import IkTarget, get_amgg_kinematics


class TestAmggKinematics(unittest.TestCase):
    """Validate analytical derivatives and the dual-arm IK solver."""

    def test_analytic_jacobian_matches_finite_difference(self) -> None:
        model = get_amgg_kinematics()
        names = AMGG_IK_JOINT_NAMES
        positions = dict(AMGG_HOME_POSITIONS)
        analytic = model.geometric_jacobian(AMGG_FRAMES.left_tcp_link, names, positions)[:3]
        numeric = np.zeros_like(analytic)
        epsilon = 1e-7
        nominal = model.forward(AMGG_FRAMES.left_tcp_link, positions)[:3, 3]
        for index, name in enumerate(names):
            shifted = dict(positions)
            shifted[name] += epsilon
            numeric[:, index] = (model.forward(AMGG_FRAMES.left_tcp_link, shifted)[:3, 3] - nominal) / epsilon
        self.assertTrue(np.allclose(analytic, numeric, atol=1e-6))

    def test_dual_arm_ik_round_trip(self) -> None:
        model = get_amgg_kinematics()
        target_positions = dict(AMGG_HOME_POSITIONS)
        target_positions.update(
            {
                "Waist01_Joint": 0.12,
                "Waist02_Joint": -0.10,
                "Body0422_Joint": 0.08,
                "ArmL02_Joint": 0.16,
                "AM_D02_J14_Joint": -0.48,
                "ArmL04_Joint": 0.12,
                "ArmL05_Joint": 0.40,
                "ArmL06_Joint": -0.15,
                "ArmR02_Joint": -0.14,
                "AM_D02R_J03_Joint": 0.46,
                "ArmR04_Joint": -0.10,
                "ArmR05_Joint": -0.38,
                "ArmR06_Joint": 0.13,
            }
        )
        targets = (
            IkTarget(AMGG_FRAMES.left_tcp_link, model.forward(AMGG_FRAMES.left_tcp_link, target_positions)),
            IkTarget(AMGG_FRAMES.right_tcp_link, model.forward(AMGG_FRAMES.right_tcp_link, target_positions)),
        )
        seed = {name: AMGG_HOME_POSITIONS[name] for name in AMGG_IK_JOINT_NAMES}
        result = model.solve(targets, AMGG_IK_JOINT_NAMES, seed)
        self.assertTrue(result.converged)
        self.assertLess(result.position_error_m, 1e-4)
        self.assertLess(result.orientation_error_rad, 2e-3)
        for name, value in result.joint_positions.items():
            joint = model.joint_by_name[name]
            self.assertLessEqual(joint.lower, value)
            self.assertLessEqual(value, joint.upper)


if __name__ == "__main__":
    unittest.main()
