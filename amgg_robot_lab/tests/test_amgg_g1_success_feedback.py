# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression contracts for practical G1 success detection and feedback."""

import ast
import unittest
from pathlib import Path


def _function_defaults(source: str, function_name: str) -> dict[str, object]:
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name)
    argument_names = [argument.arg for argument in function.args.args]
    default_names = argument_names[-len(function.args.defaults) :]
    return {name: ast.literal_eval(value) for name, value in zip(default_names, function.args.defaults)}


class TestAmggG1SuccessFeedback(unittest.TestCase):
    """Keep success thresholds operable and termination feedback visible."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.terms_source = (
            cls.repo_root
            / "amgg_robot_lab"
            / "source"
            / "amgg_robot_lab"
            / "amgg_robot_lab"
            / "tasks"
            / "mdp"
            / "amgg_g1_terms.py"
        ).read_text(encoding="utf-8")

    def test_practical_success_thresholds(self) -> None:
        expected = {
            "clutter_transfer_success": {
                "xy_tolerance": 0.09,
                "z_tolerance": 0.055,
                "max_speed": 0.15,
            },
            "bimanual_reorient_success": {
                "position_tolerance": 0.08,
                "alignment_cosine": 0.92,
                "level_cosine": 0.92,
                "max_speed": 0.15,
            },
            "precision_insert_success": {
                "xy_tolerance": 0.025,
                "z_tolerance": 0.04,
                "upright_cosine": 0.96,
                "max_speed": 0.10,
            },
        }
        for function_name, expected_defaults in expected.items():
            with self.subTest(function=function_name):
                defaults = _function_defaults(self.terms_source, function_name)
                for name, value in expected_defaults.items():
                    self.assertEqual(defaults[name], value)

    def test_plain_teleop_reports_termination_reason(self) -> None:
        teleop_source = (
            self.repo_root / "scripts" / "environments" / "teleoperation" / "teleop_se3_agent.py"
        ).read_text(encoding="utf-8")
        self.assertIn("def _report_termination", teleop_source)
        self.assertIn("[SUCCESS] Task completed", teleop_source)
        self.assertIn("env.termination_manager.get_term", teleop_source)
        self.assertIn("_report_termination(env, teleop_interface)", teleop_source)

    def test_recorder_supports_unlimited_episodes(self) -> None:
        recorder_source = (self.repo_root / "scripts" / "tools" / "record_demos.py").read_text(encoding="utf-8")
        self.assertIn('"--num_demos", type=int, default=0', recorder_source)
        self.assertIn("Set to 0 for infinite", recorder_source)
        self.assertIn("exported_successful_episode_count >= args_cli.num_demos", recorder_source)
        self.assertIn("Episode exported; resetting for the next demonstration", recorder_source)


if __name__ == "__main__":
    unittest.main()
