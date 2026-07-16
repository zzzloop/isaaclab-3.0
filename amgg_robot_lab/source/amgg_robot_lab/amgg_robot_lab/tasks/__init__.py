# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration for AMGG manager-based tasks."""

try:
    import gymnasium as gym
except ModuleNotFoundError:
    gym = None


def _register(task_id: str, config_class: str) -> None:
    if gym is None:
        return
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": f"{__name__}.amgg_manipulation_env_cfg:{config_class}"},
        disable_env_checker=True,
    )


_register("Isaac-AMGG-PickPlace-v0", "AmggPickPlaceEnvCfg")
_register("Isaac-AMGG-BimanualLift-v0", "AmggBimanualLiftEnvCfg")
_register("Isaac-AMGG-Handover-v0", "AmggHandoverEnvCfg")
_register("Isaac-AMGG-Sort-v0", "AmggSortEnvCfg")

from .amgg_task_specs import AMGG_TASK_SPEC_BY_ID, AMGG_TASK_SPEC_BY_SLUG, AMGG_TASK_SPECS, AmggTaskSpec  # noqa: E402

__all__ = ["AMGG_TASK_SPEC_BY_ID", "AMGG_TASK_SPEC_BY_SLUG", "AMGG_TASK_SPECS", "AmggTaskSpec"]
