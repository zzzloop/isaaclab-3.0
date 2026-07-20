# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared reachable-workspace layouts for the AMGG G1 task suite."""

from __future__ import annotations

# The original bar center at y=0.42 m was observed at the arm's forward limit.
# Conservatively treat that value as a radial tabletop limit so lateral
# offsets cannot silently place an entity outside the demonstrated reach.
AMGG_G1_REACH_LIMIT_M = 0.42
AMGG_G1_REACH_MARGIN_M = 0.01
AMGG_G1_OBJECT_RESET_X_HALF_RANGE_M = 0.035
AMGG_G1_OBJECT_RESET_Y_HALF_RANGE_M = 0.03


AMGG_G1_TASK_LAYOUTS: dict[str, dict[str, tuple[float, float, float]]] = {
    "clutter_transfer": {
        "object": (-0.18, 0.25, 1.035),
        "distractor_a": (-0.02, 0.30, 1.035),
        "distractor_b": (0.08, 0.24, 1.040),
        "goal": (0.18, 0.34, 1.035),
        "goal_marker": (0.18, 0.34, 1.003),
    },
    "bimanual_reorient": {
        "object": (0.00, 0.37, 1.035),
        "left_support": (-0.18, 0.23, 1.055),
        "right_support": (0.18, 0.23, 1.055),
        "goal": (0.00, 0.23, 1.115),
        "goal_marker": (0.00, 0.23, 1.115),
    },
    "precision_insert": {
        "object": (-0.08, 0.35, 1.076),
        "guide_left": (0.115, 0.320, 1.055),
        "guide_right": (0.205, 0.320, 1.055),
        "guide_near": (0.160, 0.278, 1.055),
        "guide_far": (0.160, 0.362, 1.055),
        "goal": (0.160, 0.320, 1.070),
        "goal_marker": (0.160, 0.320, 1.003),
    },
}
