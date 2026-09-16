# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration for the AM-DP123 manager-based teleoperation task."""

try:
    import gymnasium as gym
except ModuleNotFoundError:
    gym = None

AM_DP123_PICO_XR_TASK_ID = "Isaac-AM-DP123-Pico-XR-v0"
"""Registered task id of the AM-DP123 PICO XR teleoperation environment."""

AM_DP123_PICO_XR_ENV_CFG = "AmDp123PicoXrEnvCfg"
AM_DP123_PICO_XR_ENV_CFG_MODULE = "am_dp123_pico_xr_env_cfg"

AM_DP123_PI05_EVAL_TASK_ID = "Isaac-AM-DP123-Pi05-Eval-v0"
"""Registered task id of the AM-DP123 PI0.5 policy evaluation environment."""

AM_DP123_PI05_EVAL_ENV_CFG = "AmDp123Pi05EvalEnvCfg"
AM_DP123_PI05_EVAL_ENV_CFG_MODULE = "am_dp123_pi05_env_cfg"

if gym is not None:
    gym.register(
        id=AM_DP123_PICO_XR_TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": f"{__name__}.{AM_DP123_PICO_XR_ENV_CFG_MODULE}:{AM_DP123_PICO_XR_ENV_CFG}"},
        disable_env_checker=True,
    )
    gym.register(
        id=AM_DP123_PI05_EVAL_TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": f"{__name__}.{AM_DP123_PI05_EVAL_ENV_CFG_MODULE}:{AM_DP123_PI05_EVAL_ENV_CFG}"},
        disable_env_checker=True,
    )


def register_tasks() -> None:
    """Provide a post-AppLauncher callback for AM-DP123 task registration.

    Importing this module performs the Gym registration. The callable exists so
    official Isaac Lab applications can load the module through their
    ``--external_callback`` argument after Kit has initialized USD schemas.
    """


__all__ = [
    "AM_DP123_PI05_EVAL_ENV_CFG",
    "AM_DP123_PI05_EVAL_ENV_CFG_MODULE",
    "AM_DP123_PI05_EVAL_TASK_ID",
    "AM_DP123_PICO_XR_ENV_CFG",
    "AM_DP123_PICO_XR_ENV_CFG_MODULE",
    "AM_DP123_PICO_XR_TASK_ID",
    "register_tasks",
]
