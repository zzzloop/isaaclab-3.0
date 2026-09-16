# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the AM-DP123 PI0.5 evaluation task with a mock or a remote OpenPI policy.

The script parses its CLI before starting Isaac Sim, then steps the
``Isaac-AM-DP123-Pi05-Eval-v0`` task at 30 Hz. Model actions are mapped onto the 18-D
simulation ABI by the external action-layout JSON and the :class:`Pi05ActionAdapter`
safety adapter. ``mock_hold`` and ``mock_sine`` validate the full observation, camera,
stepping, and recording path without any model server; ``remote`` lazily depends on
``openpi_client`` and keeps the last bounded target whenever inference fails.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_EXTENSION_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "source" / "amgg_robot_lab"
_REPO_ROOT = Path(__file__).resolve().parents[2]

if str(_EXTENSION_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTENSION_SOURCE_ROOT))

from amgg_robot_lab.policy import (  # noqa: E402
    AM_DP123_PI05_CONTROLLED_JOINT_NAMES,
    AM_DP123_PI05_REQUIRED_CAMERAS,
    AM_DP123_PI05_STATE_JOINT_NAMES,
    Pi05ActionAdapter,
    as_numpy,
    build_pi05_observation,
    controlled_state_indices,
    load_action_layout,
    load_default_action_layout,
    to_uint8_rgb,
)

DEFAULT_TASK_ID = "Isaac-AM-DP123-Pi05-Eval-v0"
MAX_CONSECUTIVE_INFERENCE_FAILURES = 5
MAX_RECORDED_IMAGE_FRAMES = 120

# mock_sine only perturbs a few shoulder and elbow joints inside their limits. Every
# other joint, including the fingers, holds its reset position so the motion is easy to
# inspect and cannot leave the safe workspace.
_MOCK_SINE_JOINTS = (0, 1, 2, 3, 7, 8, 9, 10)
_MOCK_SINE_AMPLITUDE_RAD = (0.10, 0.08, 0.08, 0.05, 0.10, 0.08, 0.08, 0.05)
_MOCK_SINE_FREQUENCY_HZ = 0.25


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser without any Isaac Sim import."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default=DEFAULT_TASK_ID, help="Registered PI0.5 evaluation task id.")
    parser.add_argument(
        "--policy",
        default="mock_hold",
        choices=("mock_hold", "mock_sine", "remote"),
        help="Policy source: hold, small sine, or a remote OpenPI WebSocket server.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="OpenPI WebSocket host for --policy remote.")
    parser.add_argument("--port", type=int, default=8000, help="OpenPI WebSocket port for --policy remote.")
    parser.add_argument(
        "--prompt",
        default="pick up the orange cube and place it on the green target",
        help="Task instruction sent to OpenPI.",
    )
    parser.add_argument(
        "--action_layout",
        default=None,
        help="Action-layout JSON. Defaults to the identity 18-D layout.",
    )
    parser.add_argument(
        "--action_horizon",
        type=int,
        default=0,
        help="Maximum chunk rows to execute per inference. 0 uses the whole returned chunk.",
    )
    parser.add_argument("--max_steps", type=int, default=120, help="Total control steps to run across all episodes.")
    parser.add_argument("--record_dir", default=None, help="Override the episode output directory.")
    parser.add_argument("--record_images", action="store_true", help="Also record the four cameras (bounded).")
    parser.add_argument("--no_record", action="store_true", help="Disable episode recording entirely.")
    parser.add_argument("--image_stride", type=int, default=15, help="Record one image frame every N steps.")
    return parser


def _run_evaluation(args_cli: argparse.Namespace) -> None:
    """Create the environment and run the requested policy."""
    import gymnasium as gym
    import torch

    import amgg_robot_lab.tasks  # noqa: F401  (registers the task ids)
    from amgg_robot_lab.tasks.am_dp123_pi05_env_cfg import AM_DP123_PI05_CONTROL_DT, AmDp123Pi05EvalEnvCfg

    print(
        f"[pi05] starting task={args_cli.task} policy={args_cli.policy} max_steps={args_cli.max_steps} "
        f"device={args_cli.device} viz={args_cli.visualizer}",
        flush=True,
    )
    env_cfg = AmDp123Pi05EvalEnvCfg()
    env_cfg.sim.device = args_cli.device
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    print("[pi05] environment ready", flush=True)

    layout = (
        load_default_action_layout() if args_cli.action_layout is None else load_action_layout(args_cli.action_layout)
    )
    adapter = Pi05ActionAdapter(layout, AM_DP123_PI05_CONTROL_DT)
    state_indices = controlled_state_indices()

    policy = None
    image_transform = to_uint8_rgb
    if args_cli.policy == "remote":
        from openpi_client import websocket_client_policy

        policy = websocket_client_policy.WebsocketClientPolicy(host=args_cli.host, port=args_cli.port)
        image_transform = _make_remote_image_transform()

    record_root = _record_root(args_cli) if not args_cli.no_record else None
    print(
        f"[pi05] task={args_cli.task} policy={args_cli.policy} layout={layout.path} "
        f"model_action_dim={layout.model_action_dim} control_dt={AM_DP123_PI05_CONTROL_DT:.4f}s "
        f"record_root={record_root}",
        flush=True,
    )

    total_steps = 0
    episode_index = 0
    exit_reason = "max_steps_reached"
    while total_steps < args_cli.max_steps:
        obs, _ = env.reset()
        policy_obs = obs["policy"]
        state = as_numpy(policy_obs["robot_joint_pos"][0])
        adapter.reset(state[list(state_indices)])
        sine_base = adapter.current_target

        if episode_index == 0:
            _log_payload(build_pi05_observation(*_observation_inputs(policy_obs), args_cli.prompt, image_transform))

        recorder = _EpisodeRecorder(
            record_images=args_cli.record_images and record_root is not None,
            image_stride=args_cli.image_stride,
        )
        pending: list[tuple[np.ndarray, np.ndarray]] = []
        consecutive_failures = 0
        episode_reason = "max_steps_reached"

        while True:
            if not pending:
                if policy is None:
                    model_action = _mock_action(
                        args_cli.policy, sine_base, len(recorder), state_indices, policy_obs, AM_DP123_PI05_CONTROL_DT
                    )
                    target = adapter.adapt(model_action)[0]
                    pending.append((model_action, target))
                else:
                    payload = build_pi05_observation(*_observation_inputs(policy_obs), args_cli.prompt, image_transform)
                    try:
                        chunk = as_numpy(policy.infer(payload)["actions"]).astype(np.float64, copy=False)
                        if args_cli.action_horizon > 0:
                            chunk = chunk[: args_cli.action_horizon]
                        targets = adapter.adapt(chunk)
                    except Exception as exc:
                        consecutive_failures += 1
                        print(f"[pi05] inference failure {consecutive_failures}: {exc}", flush=True)
                        if consecutive_failures >= MAX_CONSECUTIVE_INFERENCE_FAILURES:
                            exit_reason = "inference_failed"
                            episode_reason = "inference_failed"
                            break
                        hold_action = np.zeros(layout.model_action_dim, dtype=np.float64)
                        pending.append((hold_action, adapter.current_target.copy()))
                    else:
                        consecutive_failures = 0
                        pending.extend(zip(chunk, targets, strict=True))

            raw_action, target = pending.pop(0)
            action_tensor = torch.tensor(target, dtype=torch.float32, device=env.device).unsqueeze(0)
            obs, _, terminated, truncated, _ = env.step(action_tensor)
            policy_obs = obs["policy"]
            total_steps += 1

            done_terminated = bool(as_numpy(terminated)[0])
            done_truncated = bool(as_numpy(truncated)[0])
            images = _observation_images(policy_obs) if recorder.records_images else None
            recorder.append(
                joint_pos=as_numpy(policy_obs["robot_joint_pos"][0]),
                joint_vel=as_numpy(policy_obs["robot_joint_vel"][0]),
                object_position=as_numpy(policy_obs["object_position"][0]),
                model_action=raw_action,
                target=target,
                terminated=done_terminated,
                truncated=done_truncated,
                images=images,
            )

            if done_terminated or done_truncated:
                episode_reason = _termination_reason(env, done_truncated)
                break
            if total_steps >= args_cli.max_steps:
                episode_reason = "max_steps_reached"
                break

        if record_root is not None and len(recorder) > 0:
            recorder.save(record_root, episode_index, _episode_metadata(args_cli, layout, env_cfg, episode_reason))
        episode_index += 1
        if exit_reason == "inference_failed" or total_steps >= args_cli.max_steps:
            break

    print(
        f"[pi05] finished after {total_steps} steps, {episode_index} episode(s), reason={exit_reason}",
        flush=True,
    )
    env.close()


def _observation_inputs(policy_obs: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return the 23-D state, 23-D velocity, and four camera frames from a policy obs."""
    return (
        as_numpy(policy_obs["robot_joint_pos"][0]),
        as_numpy(policy_obs["robot_joint_vel"][0]),
        _observation_images(policy_obs),
    )


def _observation_images(policy_obs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return the four camera observations keyed by contract camera name."""
    return {name: as_numpy(policy_obs[f"image_{name}"][0]) for name in AM_DP123_PI05_REQUIRED_CAMERAS}


def _mock_action(
    mode: str,
    sine_base: np.ndarray,
    step: int,
    state_indices: tuple[int, ...],
    policy_obs: Mapping[str, Any],
    control_dt: float,
) -> np.ndarray:
    """Build the raw model action for a mock policy."""
    if mode == "mock_sine":
        action = sine_base.copy()
        phase = 2.0 * math.pi * _MOCK_SINE_FREQUENCY_HZ * step * control_dt
        for offset, index in enumerate(_MOCK_SINE_JOINTS):
            action[index] = sine_base[index] + _MOCK_SINE_AMPLITUDE_RAD[offset] * math.sin(phase)
        return action
    state = as_numpy(policy_obs["robot_joint_pos"][0])
    return state[list(state_indices)].astype(np.float64)


def _make_remote_image_transform():
    """Build the OpenPI image transform with a deferred ``openpi_client`` import."""
    from openpi_client import image_tools

    def transform(image: object) -> np.ndarray:
        resized = image_tools.resize_with_pad(to_uint8_rgb(image), 224, 224)
        return image_tools.convert_to_uint8(resized)

    return transform


def _log_payload(payload: Mapping[str, Any]) -> None:
    """Print the OpenPI payload keys and shapes for the first episode."""
    summary = {key: getattr(value, "shape", type(value).__name__) for key, value in payload.items()}
    print(f"[pi05] OpenPI payload: {summary}", flush=True)


def _record_root(args_cli: argparse.Namespace) -> Path:
    """Return (and create) the episode output directory."""
    root = (
        Path(args_cli.record_dir)
        if args_cli.record_dir is not None
        else Path("outputs") / "am_dp123_pi05_eval" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _git_commit() -> str | None:
    """Return the current Git commit, or ``None`` when unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _termination_reason(env: Any, truncated: bool) -> str:
    """Return the active termination term names, or ``time_out`` on truncation."""
    if truncated:
        return "time_out"
    manager = getattr(env, "termination_manager", None)
    if manager is None:
        return "terminated"
    active = [term for term in manager.active_terms if bool(as_numpy(manager.get_term(term))[0])]
    return ",".join(active) if active else "terminated"


def _episode_metadata(args_cli: argparse.Namespace, layout: Any, env_cfg: Any, reason: str) -> dict[str, Any]:
    """Assemble the JSON metadata saved next to every episode."""
    return {
        "task_id": args_cli.task,
        "prompt": args_cli.prompt,
        "policy_mode": args_cli.policy,
        "remote_host": args_cli.host if args_cli.policy == "remote" else None,
        "remote_port": args_cli.port if args_cli.policy == "remote" else None,
        "action_layout_path": str(layout.path) if layout.path is not None else None,
        "action_layout_sha256": layout.sha256(),
        "model_action_dim": layout.model_action_dim,
        "state_joint_names": list(AM_DP123_PI05_STATE_JOINT_NAMES),
        "controlled_joint_names": list(AM_DP123_PI05_CONTROLLED_JOINT_NAMES),
        "control_dt": env_cfg.decimation * env_cfg.sim.dt,
        "git_commit": _git_commit(),
        "termination_reason": reason,
    }


class _EpisodeRecorder:
    """Bounded in-memory episode buffer that writes one NPZ and one JSON per episode."""

    def __init__(self, record_images: bool, image_stride: int):
        self._record_images = record_images
        self._image_stride = max(1, image_stride)
        self._joint_pos: list[np.ndarray] = []
        self._joint_vel: list[np.ndarray] = []
        self._object_position: list[np.ndarray] = []
        self._model_action: list[np.ndarray] = []
        self._target: list[np.ndarray] = []
        self._terminated: list[bool] = []
        self._truncated: list[bool] = []
        self._images: list[tuple[int, dict[str, np.ndarray]]] = []
        self._step = 0

    @property
    def records_images(self) -> bool:
        """Whether image recording is enabled for this episode."""
        return self._record_images

    def __len__(self) -> int:
        """Number of control steps recorded so far."""
        return self._step

    def append(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        object_position: np.ndarray,
        model_action: np.ndarray,
        target: np.ndarray,
        terminated: bool,
        truncated: bool,
        images: Mapping[str, np.ndarray] | None,
    ) -> None:
        """Record one control step, bounding image memory with a stride and a hard cap."""
        self._joint_pos.append(np.array(joint_pos, dtype=np.float32, copy=True))
        self._joint_vel.append(np.array(joint_vel, dtype=np.float32, copy=True))
        self._object_position.append(np.array(object_position, dtype=np.float32, copy=True))
        self._model_action.append(np.array(model_action, dtype=np.float32, copy=True))
        self._target.append(np.array(target, dtype=np.float32, copy=True))
        self._terminated.append(terminated)
        self._truncated.append(truncated)
        if (
            self._record_images
            and images is not None
            and self._step % self._image_stride == 0
            and len(self._images) < MAX_RECORDED_IMAGE_FRAMES
        ):
            self._images.append((self._step, {name: to_uint8_rgb(frame) for name, frame in images.items()}))
        self._step += 1

    def save(self, directory: Path, episode_index: int, metadata: Mapping[str, Any]) -> None:
        """Write ``episode_XXXXXX.npz`` and ``episode_XXXXXX.json``."""
        stem = f"episode_{episode_index:06d}"
        arrays: dict[str, np.ndarray] = {
            "joint_pos": np.stack(self._joint_pos),
            "joint_vel": np.stack(self._joint_vel),
            "object_position": np.stack(self._object_position),
            "model_action": np.stack(self._model_action),
            "applied_joint_target": np.stack(self._target),
            "terminated": np.asarray(self._terminated, dtype=bool),
            "truncated": np.asarray(self._truncated, dtype=bool),
        }
        np.savez_compressed(directory / f"{stem}.npz", **arrays)

        meta = dict(metadata)
        meta["steps"] = self._step
        meta["terminated_steps"] = int(arrays["terminated"].sum())
        meta["truncated_steps"] = int(arrays["truncated"].sum())
        meta["recorded_image_frames"] = len(self._images)
        (directory / f"{stem}.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

        if self._images:
            image_arrays: dict[str, np.ndarray] = {
                "step": np.asarray([step for step, _ in self._images], dtype=np.int64)
            }
            for name in AM_DP123_PI05_REQUIRED_CAMERAS:
                image_arrays[f"image_{name}"] = np.stack([frame[name] for _, frame in self._images])
            np.savez_compressed(directory / f"{stem}_images.npz", **image_arrays)
        print(f"[pi05] wrote {stem} with {self._step} steps to {directory}", flush=True)


def main() -> None:
    """Parse the CLI, start Isaac Sim, and run the evaluation."""
    from isaaclab.app import AppLauncher

    parser = _build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    app_launcher = AppLauncher(args_cli, enable_cameras=True)
    simulation_app = app_launcher.app
    try:
        _run_evaluation(args_cli)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
