# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LeRobot feature contract for Unitree G1 with RH56DFX hands."""

from dataclasses import dataclass

from amgg_robot_lab.contracts.amgg_g1_contract import (
    AMGG_G1_CAMERA_NAMES,
    AMGG_G1_EMBODIMENT,
    AMGG_G1_HARDWARE_ACTION_NAMES,
    AMGG_G1_SCHEMA_VERSION,
    AMGG_G1_SIM_OBSERVATION_JOINT_NAMES,
    AMGG_G1_SIM_PROCESSED_ACTION_NAMES,
    AMGG_G1_SIM_RAW_ACTION_NAMES,
    AMGG_G1_TACTILE_NAMES,
)
from amgg_robot_lab.tasks.amgg_g1_task_specs import AMGG_G1_TASK_SPEC_BY_ID


@dataclass(frozen=True, slots=True)
class AmggG1DatasetSpec:
    """Versioned G1 simulation dataset description."""

    task_id: str
    instruction: str
    fps: int
    embodiment: str = AMGG_G1_EMBODIMENT
    schema_version: str = AMGG_G1_SCHEMA_VERSION
    observation_joint_names: tuple[str, ...] = AMGG_G1_SIM_OBSERVATION_JOINT_NAMES
    tactile_names: tuple[str, ...] = AMGG_G1_TACTILE_NAMES
    camera_names: tuple[str, ...] = AMGG_G1_CAMERA_NAMES

    def action_names(self, source: str) -> tuple[str, ...]:
        """Return names for the selected simulation action representation."""
        if source == "raw":
            return AMGG_G1_SIM_RAW_ACTION_NAMES
        if source == "processed":
            return AMGG_G1_SIM_PROCESSED_ACTION_NAMES
        if source == "hardware":
            return AMGG_G1_HARDWARE_ACTION_NAMES
        raise ValueError(f"Unsupported G1 action source: {source}")

    def validate(self) -> None:
        """Validate task identity, rate, and fixed dimensions."""
        if self.task_id not in AMGG_G1_TASK_SPEC_BY_ID:
            raise ValueError(f"Unknown AMGG G1 task: {self.task_id}")
        if self.fps <= 0:
            raise ValueError("G1 dataset FPS must be positive.")
        if len(self.observation_joint_names) != 53 or len(self.tactile_names) != 12:
            raise ValueError("G1 simulation state/tactile dimensions must be 53/12.")


def make_amgg_g1_dataset_spec(task_id: str, fps: int = 30) -> AmggG1DatasetSpec:
    """Create and validate the dataset contract for one G1 task."""
    task = AMGG_G1_TASK_SPEC_BY_ID[task_id]
    spec = AmggG1DatasetSpec(task_id=task_id, instruction=task.instruction, fps=fps)
    spec.validate()
    return spec
