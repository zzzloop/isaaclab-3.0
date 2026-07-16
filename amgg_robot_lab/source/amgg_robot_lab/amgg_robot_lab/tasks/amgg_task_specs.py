# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Versioned task definitions shared by simulation and dataset metadata."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AmggTaskSpec:
    """Stable research-task identity and evaluation contract."""

    task_id: str
    slug: str
    instruction: str
    object_names: tuple[str, ...]
    success_hold_steps: int
    max_episode_seconds: float


AMGG_TASK_SPECS: tuple[AmggTaskSpec, ...] = (
    AmggTaskSpec(
        "Isaac-AMGG-PickPlace-v0",
        "pick_place",
        "Pick up the orange cube and place it stably inside the green target zone.",
        ("object",),
        10,
        35.0,
    ),
    AmggTaskSpec(
        "Isaac-AMGG-BimanualLift-v0",
        "bimanual_lift",
        "Use both grippers to lift the blue bar above the goal height and hold it level.",
        ("bar",),
        15,
        40.0,
    ),
    AmggTaskSpec(
        "Isaac-AMGG-Handover-v0",
        "handover",
        "Transfer the yellow cylinder from the left workspace to the right target zone.",
        ("handover_object",),
        10,
        40.0,
    ),
    AmggTaskSpec(
        "Isaac-AMGG-Sort-v0",
        "sort",
        "Place the red and blue cubes into their matching target zones.",
        ("red_object", "blue_object"),
        15,
        55.0,
    ),
)

AMGG_TASK_SPEC_BY_ID = {spec.task_id: spec for spec in AMGG_TASK_SPECS}
AMGG_TASK_SPEC_BY_SLUG = {spec.slug: spec for spec in AMGG_TASK_SPECS}
