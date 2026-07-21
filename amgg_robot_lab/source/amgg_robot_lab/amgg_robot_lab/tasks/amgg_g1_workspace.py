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

# Task-two uses absolute PICO wrist targets.  Move each wrist outward by this
# amount so the official +/-0.1487 m idle spread clears the support fixture.
AMGG_G1_BIMANUAL_WRIST_X_OFFSET_M = 0.035

# Task-specific reset ranges keep the larger task-two and task-three assets
# out of the G1 hands' observed startup occupancy.  Task one keeps the wider
# visual-domain randomization used for data collection.
AMGG_G1_TASK_OBJECT_RESET_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "clutter_transfer": {
        "x": (-0.035, 0.035),
        "y": (-0.030, 0.030),
        "yaw": (-0.25, 0.25),
    },
    "random_clutter_transfer": {
        "x": (-0.070, 0.070),
        "y": (-0.050, 0.050),
        "yaw": (-0.60, 0.60),
    },
    "bimanual_reorient": {
        "x": (-0.008, 0.008),
        "y": (-0.008, 0.008),
        "yaw": (-0.06, 0.06),
    },
    "precision_insert": {
        "x": (-0.010, 0.010),
        "y": (-0.008, 0.008),
        "yaw": (-0.12, 0.12),
    },
    "random_precision_insert": {
        "x": (-0.030, 0.030),
        "y": (-0.020, 0.020),
        "yaw": (-0.45, 0.45),
    },
    "random_cube_bucket": {
        "x": (-0.090, 0.090),
        "y": (-0.045, 0.045),
        "yaw": (-0.785, 0.785),
    },
}

# Isaac Lab quaternion convention is (w, x, y, z).  Task two starts with
# the bar's long axis along world y so it fits in the gap between the hands;
# the goal requires rotating it back onto world x.
AMGG_G1_TASK_OBJECT_ROTATIONS: dict[str, tuple[float, float, float, float]] = {
    "clutter_transfer": (1.0, 0.0, 0.0, 0.0),
    "random_clutter_transfer": (1.0, 0.0, 0.0, 0.0),
    "bimanual_reorient": (0.70710678, 0.0, 0.0, 0.70710678),
    "precision_insert": (1.0, 0.0, 0.0, 0.0),
    "random_precision_insert": (1.0, 0.0, 0.0, 0.0),
    "random_cube_bucket": (1.0, 0.0, 0.0, 0.0),
}


AMGG_G1_TASK_LAYOUTS: dict[str, dict[str, tuple[float, float, float]]] = {
    "clutter_transfer": {
        "object": (-0.18, 0.25, 1.035),
        "distractor_a": (-0.02, 0.30, 1.035),
        "distractor_b": (0.08, 0.24, 1.040),
        "goal": (0.18, 0.34, 1.035),
        "goal_marker": (0.18, 0.34, 1.003),
    },
    "random_clutter_transfer": {
        # Wider tabletop clutter variant for testing visual and initial-state
        # generalization while preserving the same orange-target semantics.
        "object": (-0.16, 0.245, 1.035),
        "distractor_a": (-0.03, 0.285, 1.035),
        "distractor_b": (0.055, 0.235, 1.040),
        "distractor_c": (-0.19, 0.335, 1.035),
        "distractor_d": (0.130, 0.280, 1.035),
        "goal": (0.18, 0.345, 1.035),
        "goal_marker": (0.18, 0.345, 1.003),
    },
    "bimanual_reorient": {
        # The narrow, longitudinal spawn lies between the default hands.
        # The support fixture is in the forward band, beyond the fingertips.
        "object": (0.00, 0.300, 1.035),
        "left_support": (-0.10, 0.390, 1.050),
        "right_support": (0.10, 0.390, 1.050),
        "goal": (0.00, 0.390, 1.105),
        "goal_marker": (0.00, 0.390, 1.105),
    },
    "precision_insert": {
        # Both the key and socket are moved beyond the startup fingertips.
        # Their lateral separation preserves a collision-free reset.
        "object": (-0.06, 0.390, 1.076),
        "guide_left": (-0.005, 0.370, 1.055),
        "guide_right": (0.085, 0.370, 1.055),
        "guide_near": (0.040, 0.3335, 1.055),
        "guide_far": (0.040, 0.4065, 1.055),
        "goal": (0.040, 0.370, 1.070),
        "goal_marker": (0.040, 0.370, 1.003),
    },
    "random_precision_insert": {
        # The guide remains fixed for mechanical consistency; the key starts
        # from a broader reachable patch with randomized yaw.
        "object": (-0.070, 0.365, 1.076),
        "guide_left": (-0.005, 0.370, 1.055),
        "guide_right": (0.085, 0.370, 1.055),
        "guide_near": (0.040, 0.3335, 1.055),
        "guide_far": (0.040, 0.4065, 1.055),
        "goal": (0.040, 0.370, 1.070),
        "goal_marker": (0.040, 0.370, 1.003),
    },
    "random_cube_bucket": {
        # The cube reset range intentionally spans a larger tabletop patch for
        # generalization while the bucket fixture stays inside the observed
        # forward reach limit.
        "object": (-0.12, 0.255, 1.030),
        "distractor_a": (-0.030, 0.255, 1.030),
        "distractor_b": (0.020, 0.210, 1.030),
        "distractor_c": (-0.185, 0.330, 1.030),
        "bucket": (0.080, 0.345, 1.003),
        "bucket_collision_left": (0.009, 0.345, 1.050),
        "bucket_collision_right": (0.151, 0.345, 1.050),
        "bucket_collision_near": (0.080, 0.294, 1.050),
        "bucket_collision_far": (0.080, 0.396, 1.050),
        "goal": (0.080, 0.345, 1.026),
        "goal_marker": (0.080, 0.345, 1.003),
    },
}
