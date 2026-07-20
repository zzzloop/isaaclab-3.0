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
        reset_x_half_range = self.workspace["AMGG_G1_OBJECT_RESET_X_HALF_RANGE_M"]
        reset_y_half_range = self.workspace["AMGG_G1_OBJECT_RESET_Y_HALF_RANGE_M"]

        self.assertEqual(set(layouts), {"clutter_transfer", "bimanual_reorient", "precision_insert"})
        for task_slug, layout in layouts.items():
            for name, position in layout.items():
                with self.subTest(task=task_slug, entity=name):
                    self.assertLessEqual(math.hypot(position[0], position[1]), reach_limit - reach_margin)
            with self.subTest(task=task_slug, entity="randomized_object"):
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
        self.assertIn('AMGG_G1_TASK_LAYOUTS["bimanual_reorient"]', scene_source)
        self.assertIn('AMGG_G1_TASK_LAYOUTS["precision_insert"]', scene_source)
        self.assertIn('layout["goal"]', terms_source)

    def test_bimanual_spawn_is_separated_from_supports(self) -> None:
        layout = self.workspace["AMGG_G1_TASK_LAYOUTS"]["bimanual_reorient"]
        reset_y_half_range = self.workspace["AMGG_G1_OBJECT_RESET_Y_HALF_RANGE_M"]
        bar_near_edge = layout["object"][1] - reset_y_half_range - 0.055 / 2
        support_far_edge = max(layout["left_support"][1], layout["right_support"][1]) + 0.13 / 2
        self.assertGreaterEqual(bar_near_edge - support_far_edge, 0.01)

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

        reset_x_half_range = self.workspace["AMGG_G1_OBJECT_RESET_X_HALF_RANGE_M"]
        key_right_edge = layout["object"][0] + reset_x_half_range + 0.045 / 2
        nearest_fixture_edge = min(left_inner_edge, cross_wall_left_edge)
        self.assertGreaterEqual(nearest_fixture_edge - key_right_edge, 0.02)


if __name__ == "__main__":
    unittest.main()
