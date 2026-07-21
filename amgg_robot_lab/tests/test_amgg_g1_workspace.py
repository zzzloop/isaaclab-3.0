# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for the reachable G1 manipulation workspace."""

import math
import runpy
import unittest
from pathlib import Path


class TestAmggG1Workspace(unittest.TestCase):
    """Keep every task layout inside the demonstrated forward reach."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        workspace_path = (
            cls.project_root / "source" / "amgg_robot_lab" / "amgg_robot_lab" / "tasks" / "amgg_g1_workspace.py"
        )
        cls.workspace = runpy.run_path(str(workspace_path))

    def test_all_task_positions_have_forward_reach_margin(self) -> None:
        layouts = self.workspace["AMGG_G1_TASK_LAYOUTS"]
        reach_limit = self.workspace["AMGG_G1_REACH_LIMIT_M"]
        reach_margin = self.workspace["AMGG_G1_REACH_MARGIN_M"]
        reset_ranges = self.workspace["AMGG_G1_TASK_OBJECT_RESET_RANGES"]

        self.assertEqual(
            set(layouts),
            {
                "clutter_transfer",
                "random_clutter_transfer",
                "random_cube_bucket",
                "bimanual_reorient",
                "precision_insert",
                "random_precision_insert",
            },
        )
        for task_slug, layout in layouts.items():
            for name, position in layout.items():
                with self.subTest(task=task_slug, entity=name):
                    self.assertLessEqual(math.hypot(position[0], position[1]), reach_limit - reach_margin)
            with self.subTest(task=task_slug, entity="randomized_object"):
                reset_x_half_range = max(abs(value) for value in reset_ranges[task_slug]["x"])
                reset_y_half_range = max(abs(value) for value in reset_ranges[task_slug]["y"])
                randomized_radius = math.hypot(
                    abs(layout["object"][0]) + reset_x_half_range,
                    layout["object"][1] + reset_y_half_range,
                )
                self.assertLessEqual(randomized_radius, reach_limit - reach_margin)

    def test_scene_and_evaluation_share_workspace_layouts(self) -> None:
        scene_source = (
            self.project_root
            / "source"
            / "amgg_robot_lab"
            / "amgg_robot_lab"
            / "tasks"
            / "amgg_g1_manipulation_env_cfg.py"
        ).read_text(encoding="utf-8")
        terms_source = (
            self.project_root / "source" / "amgg_robot_lab" / "amgg_robot_lab" / "tasks" / "mdp" / "amgg_g1_terms.py"
        ).read_text(encoding="utf-8")

        self.assertIn('AMGG_G1_TASK_LAYOUTS["clutter_transfer"]', scene_source)
        self.assertIn('AMGG_G1_TASK_LAYOUTS["random_clutter_transfer"]', scene_source)
        self.assertIn('AMGG_G1_TASK_LAYOUTS["random_cube_bucket"]', scene_source)
        self.assertIn('AMGG_G1_TASK_LAYOUTS["bimanual_reorient"]', scene_source)
        self.assertIn('AMGG_G1_TASK_LAYOUTS["precision_insert"]', scene_source)
        self.assertIn('AMGG_G1_TASK_LAYOUTS["random_precision_insert"]', scene_source)
        self.assertIn('layout["goal"]', terms_source)

    def test_bimanual_spawn_is_separated_from_supports(self) -> None:
        layout = self.workspace["AMGG_G1_TASK_LAYOUTS"]["bimanual_reorient"]
        reset_ranges = self.workspace["AMGG_G1_TASK_OBJECT_RESET_RANGES"]["bimanual_reorient"]
        rotation = self.workspace["AMGG_G1_TASK_OBJECT_ROTATIONS"]["bimanual_reorient"]

        # The 90-degree spawn makes the bar only 55 mm wide across the hands.
        self.assertAlmostEqual(abs(rotation[0]), math.sqrt(0.5), places=6)
        self.assertAlmostEqual(abs(rotation[3]), math.sqrt(0.5), places=6)
        yaw_error = reset_ranges["yaw"][1]
        bar_x_half_extent = 0.30 / 2 * math.sin(yaw_error) + 0.055 / 2 * math.cos(yaw_error)
        bar_right_edge = layout["object"][0] + reset_ranges["x"][1] + bar_x_half_extent
        right_support_left_edge = layout["right_support"][0] - 0.070 / 2
        self.assertGreaterEqual(right_support_left_edge - bar_right_edge, 0.02)

        bar_y_half_extent = 0.30 / 2 * math.cos(yaw_error) + 0.055 / 2 * math.sin(yaw_error)
        bar_near_edge = layout["object"][1] + reset_ranges["y"][0] - bar_y_half_extent
        table_near_edge = 0.14
        table_far_edge = 0.96
        bar_far_edge = layout["object"][1] + reset_ranges["y"][1] + bar_y_half_extent
        self.assertGreaterEqual(bar_near_edge, table_near_edge)
        self.assertLessEqual(bar_far_edge, table_far_edge)

    def test_large_fixtures_are_beyond_startup_hand_band(self) -> None:
        layouts = self.workspace["AMGG_G1_TASK_LAYOUTS"]
        self.assertGreaterEqual(layouts["bimanual_reorient"]["left_support"][1], 0.38)
        self.assertGreaterEqual(layouts["bimanual_reorient"]["right_support"][1], 0.38)
        self.assertGreaterEqual(layouts["precision_insert"]["goal"][1], 0.36)
        self.assertGreaterEqual(layouts["random_precision_insert"]["goal"][1], 0.36)
        self.assertLessEqual(layouts["random_cube_bucket"]["bucket_far"][1], 0.407)

    def test_bimanual_action_widens_both_wrists_only_for_task_two(self) -> None:
        offset = self.workspace["AMGG_G1_BIMANUAL_WRIST_X_OFFSET_M"]
        self.assertEqual(offset, 0.035)

        action_source = (
            self.project_root / "source" / "amgg_robot_lab" / "amgg_robot_lab" / "tasks" / "mdp" / "amgg_actions.py"
        ).read_text(encoding="utf-8")
        env_source = (
            self.project_root
            / "source"
            / "amgg_robot_lab"
            / "amgg_robot_lab"
            / "tasks"
            / "amgg_g1_manipulation_env_cfg.py"
        ).read_text(encoding="utf-8")

        self.assertIn("_LEFT_WRIST_X_ACTION_INDEX] -= AMGG_G1_BIMANUAL_WRIST_X_OFFSET_M", action_source)
        self.assertIn("_RIGHT_WRIST_X_ACTION_INDEX] += AMGG_G1_BIMANUAL_WRIST_X_OFFSET_M", action_source)
        self.assertEqual(env_source.count("mdp.AmggG1BimanualPinkInverseKinematicsAction"), 1)

    def test_precision_spawn_and_fixture_have_clearance(self) -> None:
        layout = self.workspace["AMGG_G1_TASK_LAYOUTS"]["precision_insert"]
        table_top = 1.0
        key_bottom = layout["object"][2] - 0.14 / 2
        self.assertGreaterEqual(key_bottom - table_top, 0.005)

        left_inner_edge = layout["guide_left"][0] + 0.025 / 2
        right_inner_edge = layout["guide_right"][0] - 0.025 / 2
        cross_wall_left_edge = layout["guide_near"][0] - 0.060 / 2
        cross_wall_right_edge = layout["guide_near"][0] + 0.060 / 2
        self.assertGreaterEqual(cross_wall_left_edge - left_inner_edge, 0.002)
        self.assertGreaterEqual(right_inner_edge - cross_wall_right_edge, 0.002)

        reset_x_half_range = self.workspace["AMGG_G1_TASK_OBJECT_RESET_RANGES"]["precision_insert"]["x"][1]
        key_right_edge = layout["object"][0] + reset_x_half_range + 0.045 / 2
        nearest_fixture_edge = min(left_inner_edge, cross_wall_left_edge)
        self.assertGreaterEqual(nearest_fixture_edge - key_right_edge, 0.02)

    def test_random_cube_bucket_has_generalized_reachable_reset(self) -> None:
        layout = self.workspace["AMGG_G1_TASK_LAYOUTS"]["random_cube_bucket"]
        reset_ranges = self.workspace["AMGG_G1_TASK_OBJECT_RESET_RANGES"]["random_cube_bucket"]

        self.assertLessEqual(layout["bucket_far"][1], 0.407)
        self.assertGreaterEqual(reset_ranges["x"][1] - reset_ranges["x"][0], 0.16)
        self.assertGreaterEqual(reset_ranges["y"][1] - reset_ranges["y"][0], 0.08)
        self.assertGreater(layout["goal"][0], layout["object"][0])
        self.assertGreater(layout["goal"][1], layout["object"][1])
        for name in ("distractor_a", "distractor_b", "distractor_c"):
            self.assertIn(name, layout)

    def test_randomized_variants_have_wider_reachable_resets(self) -> None:
        reset_ranges = self.workspace["AMGG_G1_TASK_OBJECT_RESET_RANGES"]
        self.assertGreater(
            reset_ranges["random_clutter_transfer"]["x"][1] - reset_ranges["random_clutter_transfer"]["x"][0],
            reset_ranges["clutter_transfer"]["x"][1] - reset_ranges["clutter_transfer"]["x"][0],
        )
        self.assertGreater(
            reset_ranges["random_precision_insert"]["yaw"][1] - reset_ranges["random_precision_insert"]["yaw"][0],
            reset_ranges["precision_insert"]["yaw"][1] - reset_ranges["precision_insert"]["yaw"][0],
        )


if __name__ == "__main__":
    unittest.main()
