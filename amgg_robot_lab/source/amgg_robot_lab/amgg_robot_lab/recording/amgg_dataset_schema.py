# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stable AMGG dataset keys shared by HDF5 and LeRobot conversion."""

from dataclasses import dataclass

AMGG_SCHEMA_VERSION = "0.1.0"
AMGG_OBSERVATION_STATE_KEY = "observation.state"
AMGG_ACTION_KEY = "action"
AMGG_TIMESTAMP_KEY = "timestamp"
AMGG_TASK_KEY = "task"
AMGG_IMAGE_PREFIX = "observation.images."


@dataclass(frozen=True, slots=True)
class AmggDatasetSpec:
    """Dataset-level configuration independent of storage backend."""

    fps: int = 30
    task: str = ""
    observation_joint_names: tuple[str, ...] = ()
    action_joint_names: tuple[str, ...] = ()
    camera_names: tuple[str, ...] = ()

    def validate(self) -> None:
        """Validate rate, task text, feature dimensions, and camera names."""
        if self.fps <= 0:
            raise ValueError("AMGG dataset FPS must be positive.")
        if not self.task:
            raise ValueError("AMGG dataset task text must be non-empty.")
        if not self.observation_joint_names or not self.action_joint_names:
            raise ValueError("AMGG dataset joint names must be populated.")
        if len(set(self.camera_names)) != len(self.camera_names):
            raise ValueError("AMGG camera names must be unique.")
