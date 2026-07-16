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


def register_tasks() -> None:
    """Provide a post-AppLauncher callback for AMGG task registration.

    Importing this module performs the Gym registration. The callable exists so
    official Isaac Lab applications can load the module through their
    ``--external_callback`` argument after Kit has initialized USD schemas.
    """


from .amgg_task_specs import AMGG_TASK_SPEC_BY_ID, AMGG_TASK_SPEC_BY_SLUG, AMGG_TASK_SPECS, AmggTaskSpec  # noqa: E402

__all__ = ["AMGG_TASK_SPEC_BY_ID", "AMGG_TASK_SPEC_BY_SLUG", "AMGG_TASK_SPECS", "AmggTaskSpec", "register_tasks"]
