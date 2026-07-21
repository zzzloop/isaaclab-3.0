# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Versioned research specifications for the Unitree G1 task suite."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AmggG1TaskSpec:
    """Stable task identity, instruction, and evaluation contract."""

    task_id: str
    slug: str
    instruction: str
    research_axis: str
    success_hold_steps: int
    max_episode_seconds: float


AMGG_G1_TASK_SPECS = (
    AmggG1TaskSpec(
        task_id="Isaac-AMGG-G1-ClutterTransfer-v0",
        slug="clutter_transfer",
        instruction="Pick the orange block from the clutter and place it stably in the green goal zone.",
        research_axis="clutter robustness and spatial generalization",
        success_hold_steps=12,
        max_episode_seconds=45.0,
    ),
    AmggG1TaskSpec(
        task_id="Isaac-AMGG-G1-RandomCubeBucket-v0",
        slug="random_cube_bucket",
        instruction="Pick the randomized orange cube and drop it stably inside the green bucket.",
        research_axis="initial-state generalization and container placement",
        success_hold_steps=12,
        max_episode_seconds=45.0,
    ),
    AmggG1TaskSpec(
        task_id="Isaac-AMGG-G1-BimanualReorient-v0",
        slug="bimanual_reorient",
        instruction="Use both hands to reorient the blue bar and place it level across the two supports.",
        research_axis="bimanual coordination and object reorientation",
        success_hold_steps=15,
        max_episode_seconds=55.0,
    ),
    AmggG1TaskSpec(
        task_id="Isaac-AMGG-G1-PrecisionInsert-v0",
        slug="precision_insert",
        instruction="Pick the yellow key and insert it upright into the narrow purple guide socket.",
        research_axis="contact-rich precision and tight-tolerance placement",
        success_hold_steps=15,
        max_episode_seconds=50.0,
    ),
)

AMGG_G1_TASK_SPEC_BY_ID = {spec.task_id: spec for spec in AMGG_G1_TASK_SPECS}
AMGG_G1_TASK_SPEC_BY_SLUG = {spec.slug: spec for spec in AMGG_G1_TASK_SPECS}
