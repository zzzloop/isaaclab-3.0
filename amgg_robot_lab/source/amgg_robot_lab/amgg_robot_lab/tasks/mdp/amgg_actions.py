# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMGG action terms."""

from __future__ import annotations

import torch
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.pink_task_space_actions import PinkInverseKinematicsAction
from isaaclab.utils.configclass import configclass

from ..amgg_g1_workspace import AMGG_G1_BIMANUAL_WRIST_X_OFFSET_M

_LEFT_WRIST_X_ACTION_INDEX = 0
_RIGHT_WRIST_X_ACTION_INDEX = 7


class AmggG1BimanualPinkInverseKinematicsAction(PinkInverseKinematicsAction):
    """Pink action with a symmetric task-two wrist-clearance offset."""

    def process_actions(self, actions: torch.Tensor) -> None:
        """Move both absolute wrist targets outward before solving IK."""
        widened_actions = actions.clone()
        widened_actions[:, _LEFT_WRIST_X_ACTION_INDEX] -= AMGG_G1_BIMANUAL_WRIST_X_OFFSET_M
        widened_actions[:, _RIGHT_WRIST_X_ACTION_INDEX] += AMGG_G1_BIMANUAL_WRIST_X_OFFSET_M
        super().process_actions(widened_actions)
        # Preserve the public PICO action ABI. Replaying the same raw action in
        # this task passes through the same deterministic offset.
        self._raw_actions[:] = actions


class AmggPinkInverseKinematicsAction(PinkInverseKinematicsAction):
    """Pink action with metric parallel-gripper target conversion."""

    def process_actions(self, actions: torch.Tensor) -> None:
        """Map trigger commands to finger travel before solving IK."""
        # IsaacTeleop GripperRetargeter: +1 open, -1 closed.
        # AMGG research gripper: 0 m open, 0.025 m closed.
        mapped_actions = actions.clone()
        mapped_actions[:, -4:] = (1.0 - mapped_actions[:, -4:].clamp(-1.0, 1.0)) * 0.0125
        super().process_actions(mapped_actions)
        # Keep the public raw action ABI for diagnostics and HDF5 recording.
        self._raw_actions[:] = actions


@configclass
class AmggPinkInverseKinematicsActionCfg(PinkInverseKinematicsActionCfg):
    """Pink action with a physical PICO-gripper command mapping."""

    class_type: type = AmggPinkInverseKinematicsAction
