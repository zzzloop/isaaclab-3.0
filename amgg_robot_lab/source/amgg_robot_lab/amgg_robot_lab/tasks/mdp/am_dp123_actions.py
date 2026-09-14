# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AM-DP123 action terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.pink_task_space_actions import PinkInverseKinematicsAction
from isaaclab.utils.configclass import configclass

from amgg_robot_lab.contracts import (
    AM_DP123_HAND_ACTION_SIDE_INDEX,
    AM_DP123_HAND_CLOSED_POSITIONS,
    AM_DP123_HAND_JOINT_NAMES,
    AM_DP123_HAND_OPEN_POSITIONS,
    AM_DP123_HAND_TRIGGER_CLOSED,
    AM_DP123_HAND_TRIGGER_OPEN,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class AmDp123PinkInverseKinematicsAction(PinkInverseKinematicsAction):
    """Pink action with trigger-driven AM-DP123 hand targets."""

    def __init__(self, cfg: AmDp123PinkInverseKinematicsActionCfg, env: ManagerBasedEnv):
        """Initialize the hand mapping tensors for the trailing action entries."""
        super().__init__(cfg, env)
        self._hand_joint_count = len(AM_DP123_HAND_JOINT_NAMES)
        side_index = torch.tensor(AM_DP123_HAND_ACTION_SIDE_INDEX, device=self.device)
        self._hand_side_index = side_index
        self._hand_open_positions = torch.tensor(
            [AM_DP123_HAND_OPEN_POSITIONS[name] for name in AM_DP123_HAND_JOINT_NAMES], device=self.device
        )
        self._hand_travel = torch.tensor(
            [
                AM_DP123_HAND_CLOSED_POSITIONS[name] - AM_DP123_HAND_OPEN_POSITIONS[name]
                for name in AM_DP123_HAND_JOINT_NAMES
            ],
            device=self.device,
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        """Map trigger commands to hand joint targets before solving IK.

        The trigger follows
        :data:`~amgg_robot_lab.contracts.AM_DP123_HAND_TRIGGER_OPEN` /
        :data:`~amgg_robot_lab.contracts.AM_DP123_HAND_TRIGGER_CLOSED`, so the
        closure ratio mirrors
        :func:`~amgg_robot_lab.contracts.am_dp123_hand_closed_fraction`. Both jaws
        of one hand travel from the same scalar, which is why the four trailing
        action entries are two duplicated pairs.
        """
        span = AM_DP123_HAND_TRIGGER_OPEN - AM_DP123_HAND_TRIGGER_CLOSED
        hand_targets = actions[:, -self._hand_joint_count :].clamp(
            AM_DP123_HAND_TRIGGER_CLOSED, AM_DP123_HAND_TRIGGER_OPEN
        )
        closed_fraction = (AM_DP123_HAND_TRIGGER_OPEN - hand_targets) / span
        mapped_actions = actions.clone()
        mapped_actions[:, -self._hand_joint_count :] = (
            self._hand_open_positions + self._hand_travel * (closed_fraction[:, self._hand_side_index])
        )
        super().process_actions(mapped_actions)
        # Restore the public PICO action ABI for diagnostics and demonstration recording.
        self._raw_actions[:] = actions


@configclass
class AmDp123PinkInverseKinematicsActionCfg(PinkInverseKinematicsActionCfg):
    """Pink action with a physical PICO hand-trigger mapping."""

    class_type: type = AmDp123PinkInverseKinematicsAction
