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


def _load_standalone_function(source: str, function_name: str):
    """Load a pure top-level function without importing the Kit application script."""
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name)
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<recorder-helper>", "exec"), namespace)
    return namespace[function_name]


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
            "random_clutter_transfer_success": {
                "xy_tolerance": 0.095,
                "z_tolerance": 0.055,
                "max_speed": 0.15,
            },
            "random_cube_bucket_success": {
                "x_tolerance": 0.060,
                "y_tolerance": 0.050,
                "minimum_z": 1.015,
                "maximum_z": 1.070,
                "max_speed": 0.15,
            },
            "bimanual_reorient_success": {
                "xy_tolerance": 0.10,
                "z_tolerance": 0.075,
                "alignment_cosine": 0.85,
                "max_long_axis_z_component": 0.25,
                "max_speed": 0.20,
            },
            "precision_insert_success": {
                "xy_tolerance": 0.04,
                "z_tolerance": 0.07,
                "vertical_axis_cosine": 0.88,
                "max_speed": 0.15,
            },
            "random_precision_insert_success": {
                "xy_tolerance": 0.045,
                "z_tolerance": 0.07,
                "vertical_axis_cosine": 0.86,
                "max_speed": 0.15,
            },
        }
        for function_name, expected_defaults in expected.items():
            with self.subTest(function=function_name):
                defaults = _function_defaults(self.terms_source, function_name)
                for name, value in expected_defaults.items():
                    self.assertEqual(defaults[name], value)

    def test_symmetric_objects_do_not_require_a_specific_face_up(self) -> None:
        self.assertIn("torch.abs(long_axis_w[:, 0]) > alignment_cosine", self.terms_source)
        self.assertIn("torch.abs(long_axis_w[:, 2]) < max_long_axis_z_component", self.terms_source)
        self.assertIn("torch.abs(long_axis_w[:, 2]) > vertical_axis_cosine", self.terms_source)
        self.assertNotIn("world_z[:, 2] > level_cosine", self.terms_source)
        self.assertNotIn("] > upright_cosine", self.terms_source)

    def test_random_cube_bucket_uses_container_bounds(self) -> None:
        self.assertIn('_GOALS["random_cube_bucket"]', self.terms_source)
        self.assertIn("x_ok = torch.abs(position[:, 0] - goal[0]) < x_tolerance", self.terms_source)
        self.assertIn("y_ok = torch.abs(position[:, 1] - goal[1]) < y_tolerance", self.terms_source)
        self.assertIn("(position[:, 2] > minimum_z) & (position[:, 2] < maximum_z)", self.terms_source)

    def test_randomized_transfer_and_insertion_have_task_specific_success(self) -> None:
        self.assertIn('_GOALS["random_clutter_transfer"]', self.terms_source)
        self.assertIn('_GOALS["random_precision_insert"]', self.terms_source)
        self.assertIn("def random_clutter_transfer_success", self.terms_source)
        self.assertIn("def random_precision_insert_success", self.terms_source)

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
        self.assertIn("SUCCESS! Demo {current_recorded_demo_count} saved. Resetting...", recorder_source)

    def test_amgg_recorder_auto_starts_without_pico_start_event(self) -> None:
        recorder_source = (self.repo_root / "scripts" / "tools" / "record_demos.py").read_text(encoding="utf-8")
        wrapper_source = (self.repo_root / "amgg_robot_lab" / "scripts" / "amgg_record_demos.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"--auto_start_recording"', recorder_source)
        self.assertIn('"--auto_start_recording"', wrapper_source)

        update_state = _load_standalone_function(recorder_source, "_update_recording_active_state")

        # PICO/CloudXR initially reports STOPPED when no START message exists.
        # AMGG auto-start recording must ignore that initial inactive state.
        self.assertEqual(update_state(True, False, True, False), (True, False))
        # Once a real START edge was observed, subsequent STOP remains meaningful.
        self.assertEqual(update_state(True, True, True, False), (True, True))
        self.assertEqual(update_state(True, False, True, True), (False, True))
        # The official opt-in behavior remains unchanged for other callers.
        self.assertEqual(update_state(True, False, False, False), (False, False))

    def test_readme_documents_all_g1_recording_and_conversion_commands(self) -> None:
        readme = (self.repo_root / "amgg_robot_lab" / "README_CN.md").read_text(encoding="utf-8")
        for task_name in (
            "ClutterTransfer",
            "RandomClutterTransfer",
            "RandomCubeBucket",
            "BimanualReorient",
            "PrecisionInsert",
            "RandomPrecisionInsert",
        ):
            self.assertIn(f"Isaac-AMGG-G1-{task_name}-XR-v0", readme)
            self.assertIn(f"Isaac-AMGG-G1-{task_name}-v0", readme)
        self.assertIn("amgg_convert_g1_hdf5_to_lerobot.py", readme)
        self.assertIn("--source_fps 60", readme)
        self.assertIn("--fps 30", readme)


if __name__ == "__main__":
    unittest.main()
