# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch an AMGG task and hold the FK-derived idle action for finite steps."""

import argparse
import sys

import amgg_robot_lab  # noqa: F401
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-AMGG-PickPlace-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=240)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra_args]


def main() -> None:
    """Run finite idle-action physics and observation checks."""
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env = gym.make(args_cli.task, cfg=env_cfg)
        observation, _ = env.reset()
        action = torch.tensor(env_cfg.idle_action, device=env.unwrapped.device, dtype=torch.float32)
        action = action.repeat(args_cli.num_envs, 1)
        if tuple(action.shape) != tuple(env.action_space.shape):
            raise RuntimeError(f"Idle action {tuple(action.shape)} does not match {env.action_space.shape}.")
        for _ in range(args_cli.num_steps):
            observation, _, _, _, _ = env.step(action)
            policy = observation["policy"]
            for name, value in policy.items():
                if torch.is_floating_point(value) and not torch.isfinite(value).all():
                    raise RuntimeError(f"Observation '{name}' contains non-finite values.")
        print(f"AMGG smoke test passed: task={args_cli.task}, steps={args_cli.num_steps}")
        print(f"action_shape={env.action_space.shape}, policy_keys={sorted(observation['policy'])}")
        env.close()


if __name__ == "__main__":
    main()
