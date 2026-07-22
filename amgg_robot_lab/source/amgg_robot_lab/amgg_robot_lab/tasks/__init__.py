# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration for AMGG manager-based tasks."""

try:
    import gymnasium as gym
except ModuleNotFoundError:
    gym = None


def _register(task_id: str, config_class: str, module_name: str = "amgg_manipulation_env_cfg") -> None:
    if gym is None:
        return
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": f"{__name__}.{module_name}:{config_class}"},
        disable_env_checker=True,
    )


_register("Isaac-AMGG-PickPlace-v0", "AmggPickPlaceEnvCfg")
_register("Isaac-AMGG-BimanualLift-v0", "AmggBimanualLiftEnvCfg")
_register("Isaac-AMGG-Handover-v0", "AmggHandoverEnvCfg")
_register("Isaac-AMGG-Sort-v0", "AmggSortEnvCfg")
_register(
    "Isaac-AMGG-G1-ClutterTransfer-v0",
    "AmggG1ClutterTransferEnvCfg",
    module_name="amgg_g1_manipulation_env_cfg",
)
_register(
    "Isaac-AMGG-G1-RandomClutterTransfer-v0",
    "AmggG1RandomClutterTransferEnvCfg",
    module_name="amgg_g1_manipulation_env_cfg",
)
_register(
    "Isaac-AMGG-G1-RandomCubeBucket-v0",
    "AmggG1RandomCubeBucketEnvCfg",
    module_name="amgg_g1_manipulation_env_cfg",
)
_register(
    "Isaac-AMGG-G1-BimanualReorient-v0",
    "AmggG1BimanualReorientEnvCfg",
    module_name="amgg_g1_manipulation_env_cfg",
)
_register(
    "Isaac-AMGG-G1-PrecisionInsert-v0",
    "AmggG1PrecisionInsertEnvCfg",
    module_name="amgg_g1_manipulation_env_cfg",
)
_register(
    "Isaac-AMGG-G1-RandomPrecisionInsert-v0",
    "AmggG1RandomPrecisionInsertEnvCfg",
    module_name="amgg_g1_manipulation_env_cfg",
)
_register(
    "Isaac-AMGG-G1-ClutterTransfer-XR-v0",
    "AmggG1ClutterTransferXrEnvCfg",
    module_name="amgg_g1_manipulation_env_cfg",
)
_register(
    "Isaac-AMGG-G1-RandomClutterTransfer-XR-v0",
    "AmggG1RandomClutterTransferXrEnvCfg",
    module_name="amgg_g1_manipulation_env_cfg",
)
_register(
    "Isaac-AMGG-G1-RandomCubeBucket-XR-v0",
    "AmggG1RandomCubeBucketXrEnvCfg",
    module_name="amgg_g1_manipulation_env_cfg",
)
_register(
    "Isaac-AMGG-G1-BimanualReorient-XR-v0",
    "AmggG1BimanualReorientXrEnvCfg",
    module_name="amgg_g1_manipulation_env_cfg",
)
_register(
    "Isaac-AMGG-G1-PrecisionInsert-XR-v0",
    "AmggG1PrecisionInsertXrEnvCfg",
    module_name="amgg_g1_manipulation_env_cfg",
)
_register(
    "Isaac-AMGG-G1-RandomPrecisionInsert-XR-v0",
    "AmggG1RandomPrecisionInsertXrEnvCfg",
    module_name="amgg_g1_manipulation_env_cfg",
)


def register_tasks() -> None:
    """Provide a post-AppLauncher callback for AMGG task registration.

    Importing this module performs the Gym registration. The callable exists so
    official Isaac Lab applications can load the module through their
    ``--external_callback`` argument after Kit has initialized USD schemas.
    """


from .amgg_g1_task_specs import (  # noqa: E402
    AMGG_G1_TASK_SPEC_BY_ID,
    AMGG_G1_TASK_SPEC_BY_SLUG,
    AMGG_G1_TASK_SPECS,
    AmggG1TaskSpec,
)
from .amgg_task_specs import AMGG_TASK_SPEC_BY_ID, AMGG_TASK_SPEC_BY_SLUG, AMGG_TASK_SPECS, AmggTaskSpec  # noqa: E402

__all__ = [
    "AMGG_G1_TASK_SPEC_BY_ID",
    "AMGG_G1_TASK_SPEC_BY_SLUG",
    "AMGG_G1_TASK_SPECS",
    "AMGG_TASK_SPEC_BY_ID",
    "AMGG_TASK_SPEC_BY_SLUG",
    "AMGG_TASK_SPECS",
    "AmggG1TaskSpec",
    "AmggTaskSpec",
    "register_tasks",
]
