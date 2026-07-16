# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the AMGG manager-based task."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from isaaclab.envs import ManagerBasedRLEnv


def amgg_observation_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return measured joint positions [rad] in canonical observed-joint order."""
    del env
    raise NotImplementedError("Populate the AMGG joint contract before implementing observations.")
