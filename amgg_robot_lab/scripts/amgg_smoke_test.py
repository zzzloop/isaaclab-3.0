# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch an AMGG task and hold the FK-derived idle action for finite steps."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-AMGG-PickPlace-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=240)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac Sim and USD-dependent modules must be imported only after Kit starts.
import amgg_robot_lab  # noqa: E402, F401
import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: E402, F401
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> None:
    """Run finite idle-action physics and observation checks."""
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        observation, _ = env.reset()
        action = torch.tensor(env_cfg.idle_action, device=env.unwrapped.device, dtype=torch.float32)
        action = action.repeat(args_cli.num_envs, 1)
        if tuple(action.shape) != tuple(env.action_space.shape):
            raise RuntimeError(f"Idle action {tuple(action.shape)} does not match {env.action_space.shape}.")
        print(f"Starting finite-step validation: task={args_cli.task}, steps={args_cli.num_steps}", flush=True)
        progress_interval = max(1, min(60, args_cli.num_steps // 4))
        for step in range(1, args_cli.num_steps + 1):
            observation, _, _, _, _ = env.step(action)
            policy = observation["policy"]
            for name, value in policy.items():
                if torch.is_floating_point(value) and not torch.isfinite(value).all():
                    raise RuntimeError(f"Observation '{name}' contains non-finite values.")
            if step % progress_interval == 0 or step == args_cli.num_steps:
                print(f"AMGG smoke progress: {step}/{args_cli.num_steps}", flush=True)
        print(f"AMGG smoke test passed: task={args_cli.task}, steps={args_cli.num_steps}")
        print(f"action_shape={env.action_space.shape}, policy_keys={sorted(observation['policy'])}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
