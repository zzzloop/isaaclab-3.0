# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Versioned AMGG HDF5-to-LeRobot feature contract."""

from dataclasses import dataclass

from amgg_robot_lab.contracts import (
    AMGG_CAMERAS,
    AMGG_CONTROLLED_JOINT_NAMES,
    AMGG_OBSERVED_JOINT_NAMES,
)
from amgg_robot_lab.tasks.amgg_task_specs import AMGG_TASK_SPEC_BY_ID

AMGG_SCHEMA_VERSION = "1.0.0"
AMGG_OBSERVATION_STATE_KEY = "observation.state"
AMGG_ENVIRONMENT_STATE_KEY = "observation.environment_state"
AMGG_TCP_STATE_KEY = "observation.tcp_pose"
AMGG_ACTION_KEY = "action"
AMGG_TIMESTAMP_KEY = "timestamp"
AMGG_TASK_KEY = "task"
AMGG_IMAGE_PREFIX = "observation.images."
AMGG_RAW_ACTION_NAMES = (
    "left_tcp.x",
    "left_tcp.y",
    "left_tcp.z",
    "left_tcp.qx",
    "left_tcp.qy",
    "left_tcp.qz",
    "left_tcp.qw",
    "right_tcp.x",
    "right_tcp.y",
    "right_tcp.z",
    "right_tcp.qx",
    "right_tcp.qy",
    "right_tcp.qz",
    "right_tcp.qw",
    "left_gripper.negative",
    "left_gripper.positive",
    "right_gripper.negative",
    "right_gripper.positive",
)


@dataclass(frozen=True, slots=True)
class AmggDatasetSpec:
    """Dataset-level configuration independent of storage backend."""

    fps: int
    task_id: str
    task: str
    observation_joint_names: tuple[str, ...]
    action_joint_names: tuple[str, ...]
    camera_names: tuple[str, ...]
    schema_version: str = AMGG_SCHEMA_VERSION

    def validate(self) -> None:
        """Validate rate, task identity, dimensions, and cameras."""
        if self.fps <= 0:
            raise ValueError("AMGG dataset FPS must be positive.")
        if self.task_id not in AMGG_TASK_SPEC_BY_ID or not self.task:
            raise ValueError(f"Unknown or empty AMGG task: {self.task_id}")
        if not self.observation_joint_names or not self.action_joint_names:
            raise ValueError("AMGG dataset joint names must be populated.")
        if len(set(self.camera_names)) != len(self.camera_names):
            raise ValueError("AMGG camera names must be unique.")
        if tuple(self.camera_names) != tuple(camera.name for camera in AMGG_CAMERAS):
            raise ValueError("AMGG camera ABI does not match the exact four-camera contract.")


def make_amgg_dataset_spec(task_id: str, fps: int = 30) -> AmggDatasetSpec:
    """Create the canonical dataset spec for a registered AMGG task."""
    task = AMGG_TASK_SPEC_BY_ID[task_id]
    spec = AmggDatasetSpec(
        fps=fps,
        task_id=task_id,
        task=task.instruction,
        observation_joint_names=AMGG_OBSERVED_JOINT_NAMES,
        action_joint_names=AMGG_CONTROLLED_JOINT_NAMES,
        camera_names=tuple(camera.name for camera in AMGG_CAMERAS),
    )
    spec.validate()
    return spec
