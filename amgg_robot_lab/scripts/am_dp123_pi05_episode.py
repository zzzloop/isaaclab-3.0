# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate or replay a recorded AM-DP123 PI0.5 episode."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

_EXTENSION_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "source" / "amgg_robot_lab"
if str(_EXTENSION_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTENSION_SOURCE_ROOT))

from amgg_robot_lab.policy import (  # noqa: E402
    AM_DP123_PI05_CONTROLLED_JOINT_NAMES,
    AM_DP123_PI05_STATE_JOINT_NAMES,
    Pi05EpisodeValidationSummary,
    load_pi05_episode,
)

DEFAULT_TASK_ID = "Isaac-AM-DP123-Pi05-Eval-v0"


def _nonnegative_int(value: str) -> int:
    """Parse a nonnegative integer CLI value."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    """Build arguments shared by offline validation and simulator replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate", "replay"), default="validate")
    parser.add_argument("--episode", required=True, help="Path to episode_XXXXXX.npz.")
    parser.add_argument("--metadata", default=None, help="Metadata JSON; defaults to the NPZ path with .json suffix.")
    parser.add_argument("--task", default=DEFAULT_TASK_ID, help="Registered task used by replay mode.")
    parser.add_argument(
        "--max_steps", type=_nonnegative_int, default=0, help="Replay at most N rows; 0 replays the full episode."
    )
    return parser


def _load_metadata(path: Path, override: str | None) -> dict[str, Any]:
    """Load and validate the metadata JSON adjacent to an episode."""
    metadata_path = Path(override) if override is not None else path.with_suffix(".json")
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Episode metadata '{metadata_path}' must contain a JSON object.")
    return data


def _validate_metadata(metadata: Mapping[str, Any], summary: Pi05EpisodeValidationSummary) -> None:
    """Check metadata fields that must agree with the NPZ payload."""
    if int(metadata.get("format_version", 0)) != 2:
        raise ValueError("Episode metadata format_version must be 2.")
    if int(metadata.get("steps", -1)) != summary.steps:
        raise ValueError("Episode metadata step count does not match the NPZ payload.")
    if int(metadata.get("model_action_dim", -1)) != summary.model_action_dim:
        raise ValueError("Episode metadata model_action_dim does not match the NPZ payload.")
    if int(metadata.get("inference_failure_steps", -1)) != summary.inference_failure_steps:
        raise ValueError("Episode metadata inference_failure_steps does not match the NPZ payload.")
    if tuple(metadata.get("state_joint_names", ())) != AM_DP123_PI05_STATE_JOINT_NAMES:
        raise ValueError("Episode metadata state_joint_names does not match the canonical state ABI.")
    if tuple(metadata.get("controlled_joint_names", ())) != AM_DP123_PI05_CONTROLLED_JOINT_NAMES:
        raise ValueError("Episode metadata controlled_joint_names does not match the canonical action ABI.")


def _print_summary(path: Path, summary: Pi05EpisodeValidationSummary) -> None:
    """Print a stable one-line acceptance result."""
    print(
        f"[pi05-episode] valid path={path} steps={summary.steps} "
        f"model_action_dim={summary.model_action_dim} failures={summary.inference_failure_steps} "
        f"terminated={summary.terminated_steps} truncated={summary.truncated_steps}",
        flush=True,
    )


def _replay(args_cli: argparse.Namespace, targets: np.ndarray) -> None:
    """Apply recorded absolute joint targets [rad] to a fresh simulation."""
    import gymnasium as gym
    import torch

    import amgg_robot_lab.tasks  # noqa: F401
    from amgg_robot_lab.tasks.am_dp123_pi05_env_cfg import AmDp123Pi05EvalEnvCfg

    env_cfg = AmDp123Pi05EvalEnvCfg()
    env_cfg.sim.device = args_cli.device
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    try:
        env.reset()
        replay_targets = targets if args_cli.max_steps == 0 else targets[: args_cli.max_steps]
        for index, target in enumerate(replay_targets):
            action = torch.tensor(target, dtype=torch.float32, device=env.device).unsqueeze(0)
            _, _, terminated, truncated, _ = env.step(action)
            if bool(terminated[0]) or bool(truncated[0]):
                print(f"[pi05-episode] replay stopped at environment boundary after step {index + 1}", flush=True)
                break
        else:
            print(f"[pi05-episode] replayed {len(replay_targets)} steps", flush=True)
    finally:
        env.close()


def main() -> None:
    """Validate an episode offline or replay it in Isaac Sim."""
    parser = _build_parser()
    known_args, _ = parser.parse_known_args()
    episode_path = Path(known_args.episode)
    arrays, summary = load_pi05_episode(episode_path)
    metadata = _load_metadata(episode_path, known_args.metadata)
    _validate_metadata(metadata, summary)
    _print_summary(episode_path, summary)

    if known_args.mode == "validate":
        parser.parse_args()
        return

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    app_launcher = AppLauncher(args_cli, enable_cameras=True)
    simulation_app = app_launcher.app
    try:
        _replay(args_cli, arrays["applied_joint_target"])
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
