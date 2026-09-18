# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the AM-DP123 PI0.5 evaluation task with a mock or a remote OpenPI policy.

The script parses its CLI before starting Isaac Sim, then steps the
``Isaac-AM-DP123-Pi05-Eval-v0`` task at 30 Hz. Requests contain 18 physical controls;
OpenPI pads these to the fixed 32-D model width and returns raw 23-D BPX actions. The client
collapses them back to 18 physical controls, which expand
to 20 URDF targets through the action-layout JSON and :class:`Pi05ActionAdapter`
safety adapter. ``mock_hold`` and ``mock_sine`` validate the full observation, camera,
stepping, and recording path without any model server; ``remote`` lazily depends on
``openpi_client`` and exits with an error instead of continuing after inference fails.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import subprocess
import sys
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_EXTENSION_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "source" / "amgg_robot_lab"
_REPO_ROOT = Path(__file__).resolve().parents[2]

if str(_EXTENSION_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTENSION_SOURCE_ROOT))

from amgg_robot_lab.policy import (  # noqa: E402
    AM_DP123_PI05_CAPTURE_CAMERAS,
    AM_DP123_PI05_CONTROLLED_JOINT_NAMES,
    AM_DP123_PI05_MODEL_JOINT_NAMES,
    AM_DP123_PI05_REQUIRED_CAMERAS,
    AM_DP123_PI05_STATE_JOINT_NAMES,
    Pi05ActionAdapter,
    as_numpy,
    build_pi05_observation,
    controlled_state_indices,
    from_bpx_server_actions,
    load_action_layout,
    load_default_action_layout,
    to_pi05_policy_state,
    to_uint8_rgb,
)

DEFAULT_TASK_ID = "Isaac-AM-DP123-Pi05-Eval-v0"
MAX_CONSECUTIVE_INFERENCE_FAILURES = 5
MAX_RECORDED_IMAGE_FRAMES = 120

# mock_sine only perturbs a few shoulder and elbow joints inside their limits. Every
# other joint, including the fingers, holds its reset position so the motion is easy to
# inspect and cannot leave the safe workspace.
_MOCK_SINE_JOINTS = (0, 1, 2, 3, 8, 9, 10, 11)
_MOCK_SINE_AMPLITUDE_RAD = (0.10, 0.08, 0.08, 0.05, 0.10, 0.08, 0.08, 0.05)
_MOCK_SINE_FREQUENCY_HZ = 0.25


@dataclass(frozen=True, slots=True)
class _PendingAction:
    """One model action and the bounded target that will be applied next."""

    model_action: np.ndarray
    target: np.ndarray
    inference_valid: bool
    inference_error: str = ""


@dataclass(frozen=True, slots=True)
class _EpisodeStep:
    """All data associated with one applied control step."""

    joint_pos: np.ndarray
    joint_vel: np.ndarray
    object_position: np.ndarray
    model_action: np.ndarray
    target: np.ndarray
    terminated: bool
    truncated: bool
    inference_valid: bool
    inference_error: str
    images: Mapping[str, np.ndarray] | None = None


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer CLI value."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    """Parse a nonnegative integer CLI value."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def _positive_float(value: str) -> float:
    """Parse a strictly positive floating-point CLI value."""
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be a positive finite number")
    return parsed


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
        "--connect_timeout",
        type=_positive_float,
        default=30.0,
        help="Seconds to wait for the OpenPI TCP endpoint before starting Isaac Sim.",
    )
    parser.add_argument(
        "--prompt",
        default="pick up the orange cube and place it on the green target",
        help="Task instruction sent to OpenPI.",
    )
    parser.add_argument(
        "--action_layout",
        default=None,
        help="Action-layout JSON. Defaults to the confirmed AM-DP123 32-D layout.",
    )
    parser.add_argument(
        "--action_horizon",
        type=_nonnegative_int,
        default=0,
        help="Maximum chunk rows to execute per inference. 0 uses the whole returned chunk.",
    )
    parser.add_argument(
        "--max_steps", type=_positive_int, default=120, help="Total control steps to run across all episodes."
    )
    parser.add_argument("--record_dir", default=None, help="Override the episode output directory.")
    parser.add_argument("--record_images", action="store_true", help="Also record the four cameras (bounded).")
    parser.add_argument("--no_record", action="store_true", help="Disable episode recording entirely.")
    parser.add_argument("--image_stride", type=_positive_int, default=15, help="Record one image frame every N steps.")
    return parser


def _preflight_remote_client(host: str, port: int, timeout: float) -> None:
    """Validate the lightweight OpenPI client and wait for its TCP endpoint."""
    print("[pi05] checking OpenPI client dependencies", flush=True)
    try:
        import msgpack  # noqa: F401
        import websockets.sync.client  # noqa: F401
        from openpi_client import (
            image_tools,  # noqa: F401
            websocket_client_policy,  # noqa: F401
        )
        from PIL import Image  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "OpenPI client import failed. Install dm-tree, msgpack, pillow, and websockets>=11 "
            "in isaaclab30, then expose packages/openpi-client/src through PYTHONPATH."
        ) from exc

    deadline = time.monotonic() + timeout
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=min(2.0, timeout)):
                print(f"[pi05] OpenPI TCP endpoint ready at {host}:{port}", flush=True)
                return
        except OSError as exc:
            last_error = exc
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    raise ConnectionError(f"OpenPI server {host}:{port} was unreachable for {timeout:.1f}s") from last_error


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
        print(f"[pi05] importing OpenPI client for ws://{args_cli.host}:{args_cli.port}", flush=True)
        from openpi_client import websocket_client_policy

        print(f"[pi05] connecting to ws://{args_cli.host}:{args_cli.port}", flush=True)
        policy = websocket_client_policy.WebsocketClientPolicy(host=args_cli.host, port=args_cli.port)
        print("[pi05] OpenPI WebSocket connected", flush=True)
        image_transform = _make_remote_image_transform()

    record_root = _record_root(args_cli) if not args_cli.no_record else None
    print(
        f"[pi05] task={args_cli.task} policy={args_cli.policy} layout={layout.path} "
        f"policy_action_dim={layout.policy_action_dim} model_action_dim={layout.model_action_dim} "
        f"control_dt={AM_DP123_PI05_CONTROL_DT:.4f}s "
        f"record_root={record_root}",
        flush=True,
    )

    total_steps = 0
    episode_index = 0
    exit_reason = "max_steps_reached"
    payload_logged = False
    action_chunk_logged = False
    obs, _ = env.reset()
    while total_steps < args_cli.max_steps:
        policy_obs = obs["policy"]
        state = as_numpy(policy_obs["robot_joint_pos"][0])
        adapter.reset(state[list(state_indices)])
        sine_base = to_pi05_policy_state(state).astype(np.float64)

        recorder = _EpisodeRecorder(
            record_images=args_cli.record_images and record_root is not None,
            image_stride=args_cli.image_stride,
        )
        pending: deque[_PendingAction] = deque()
        consecutive_failures = 0
        episode_reason = "max_steps_reached"

        while True:
            if not pending:
                try:
                    payload = build_pi05_observation(*_observation_inputs(policy_obs), args_cli.prompt, image_transform)
                    if not payload_logged:
                        _log_payload(payload)
                        payload_logged = True
                    if policy is None:
                        chunk = _mock_action(
                            args_cli.policy,
                            sine_base,
                            len(recorder),
                            policy_obs,
                            AM_DP123_PI05_CONTROL_DT,
                        )[None, :]
                    else:
                        response = policy.infer(payload)
                        chunk = _extract_policy_action_chunk(response)
                        if args_cli.action_horizon > 0:
                            chunk = chunk[: args_cli.action_horizon]
                    if not action_chunk_logged:
                        print(f"[pi05] policy action chunk: shape={chunk.shape}", flush=True)
                        action_chunk_logged = True
                    targets = adapter.adapt(chunk)
                except Exception as exc:
                    consecutive_failures += 1
                    error = f"{type(exc).__name__}: {exc}"
                    print(f"[pi05] protocol/inference failure {consecutive_failures}: {error}", flush=True)
                    if policy is not None or consecutive_failures >= MAX_CONSECUTIVE_INFERENCE_FAILURES:
                        exit_reason = "inference_failed"
                        episode_reason = "inference_failed"
                        break
                    pending.append(
                        _PendingAction(
                            model_action=np.zeros(layout.policy_action_dim, dtype=np.float64),
                            target=adapter.current_target,
                            inference_valid=False,
                            inference_error=error,
                        )
                    )
                else:
                    consecutive_failures = 0
                    pending.extend(
                        _PendingAction(model_action=raw, target=target, inference_valid=True)
                        for raw, target in zip(chunk, targets, strict=True)
                    )

            pending_action = pending.popleft()
            action_tensor = torch.tensor(pending_action.target, dtype=torch.float32, device=env.device).unsqueeze(0)
            obs, _, terminated, truncated, extras = env.step(action_tensor)
            policy_obs = obs["policy"]
            total_steps += 1

            done_terminated = bool(as_numpy(terminated)[0])
            done_truncated = bool(as_numpy(truncated)[0])
            record_policy_obs = _record_policy_observation(policy_obs, extras, done_terminated or done_truncated)
            recorder.append(
                _EpisodeStep(
                    joint_pos=as_numpy(record_policy_obs["robot_joint_pos"][0]),
                    joint_vel=as_numpy(record_policy_obs["robot_joint_vel"][0]),
                    object_position=as_numpy(record_policy_obs["object_position"][0]),
                    model_action=pending_action.model_action,
                    target=pending_action.target,
                    terminated=done_terminated,
                    truncated=done_truncated,
                    inference_valid=pending_action.inference_valid,
                    inference_error=pending_action.inference_error,
                    images=_observation_images(record_policy_obs) if recorder.records_images else None,
                )
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
    if exit_reason == "inference_failed":
        raise RuntimeError("PI0.5 evaluation stopped because remote inference failed; see the error above.")


def _observation_inputs(policy_obs: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return raw 23-D state/velocity and the three BPX policy frames."""
    return (
        as_numpy(policy_obs["robot_joint_pos"][0]),
        as_numpy(policy_obs["robot_joint_vel"][0]),
        _observation_images(policy_obs, AM_DP123_PI05_REQUIRED_CAMERAS),
    )


def _observation_images(
    policy_obs: Mapping[str, Any], camera_names: tuple[str, ...] = AM_DP123_PI05_CAPTURE_CAMERAS
) -> dict[str, np.ndarray]:
    """Return selected camera observations keyed by contract camera name."""
    return {name: as_numpy(policy_obs[f"image_{name}"][0]) for name in camera_names}


def _record_policy_observation(
    policy_obs: Mapping[str, Any], extras: Mapping[str, Any], done: bool
) -> Mapping[str, Any]:
    """Return the terminal policy observation on done, otherwise the regular observation."""
    if not done:
        return policy_obs
    final_obs = extras.get("final_obs")
    if not isinstance(final_obs, Mapping) or "policy" not in final_obs:
        raise RuntimeError("The PI0.5 environment terminated without extras['final_obs']; recording would be invalid.")
    final_policy_obs = final_obs["policy"]
    if not isinstance(final_policy_obs, Mapping):
        raise RuntimeError("extras['final_obs']['policy'] is not an observation mapping.")
    return final_policy_obs


def _mock_action(
    mode: str,
    sine_base: np.ndarray,
    step: int,
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
    return to_pi05_policy_state(state).astype(np.float64)


def _make_remote_image_transform():
    """Build the BPX center-crop transform with a deferred Pillow import."""
    from PIL import Image

    def transform(image: object) -> np.ndarray:
        rgb = to_uint8_rgb(image)
        height, width = rgb.shape[:2]
        side = min(height, width)
        top = (height - side) // 2
        left = (width - side) // 2
        cropped = rgb[top : top + side, left : left + side]
        resized = Image.fromarray(cropped).resize((224, 224), resample=Image.Resampling.BILINEAR)
        return np.ascontiguousarray(np.asarray(resized, dtype=np.uint8))

    return transform


def _extract_policy_action_chunk(response: object) -> np.ndarray:
    """Validate an OpenPI response and return an effective 18-D action chunk."""
    if not isinstance(response, Mapping) or "actions" not in response:
        raise ValueError("OpenPI response must contain an 'actions' array.")
    chunk = from_bpx_server_actions(response["actions"]).astype(np.float64, copy=False)
    if chunk.ndim == 1:
        chunk = chunk[None, :]
    if chunk.ndim != 2:
        raise ValueError(f"OpenPI actions must be a vector or matrix, got shape {chunk.shape}.")
    return chunk


def _log_payload(payload: Mapping[str, Any]) -> None:
    """Print the OpenPI payload keys and shapes for the first episode."""
    summary = {
        key: (
            {
                nested_key: getattr(nested_value, "shape", type(nested_value).__name__)
                for nested_key, nested_value in value.items()
            }
            if isinstance(value, Mapping)
            else getattr(value, "shape", type(value).__name__)
        )
        for key, value in payload.items()
    }
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
        "policy_action_dim": layout.policy_action_dim,
        "state_joint_names": list(AM_DP123_PI05_STATE_JOINT_NAMES),
        "model_joint_names": list(AM_DP123_PI05_MODEL_JOINT_NAMES),
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
        self._steps: list[_EpisodeStep] = []
        self._recorded_image_frames = 0

    @property
    def records_images(self) -> bool:
        """Whether image recording is enabled for this episode."""
        return self._record_images

    def __len__(self) -> int:
        """Number of control steps recorded so far."""
        return len(self._steps)

    def append(self, step: _EpisodeStep) -> None:
        """Record one control step, bounding image memory with a stride and a hard cap."""
        images = None
        if (
            self._record_images
            and step.images is not None
            and len(self._steps) % self._image_stride == 0
            and self._recorded_image_frames < MAX_RECORDED_IMAGE_FRAMES
        ):
            images = {name: to_uint8_rgb(frame) for name, frame in step.images.items()}
            self._recorded_image_frames += 1
        self._steps.append(
            replace(
                step,
                joint_pos=np.array(step.joint_pos, dtype=np.float32, copy=True),
                joint_vel=np.array(step.joint_vel, dtype=np.float32, copy=True),
                object_position=np.array(step.object_position, dtype=np.float32, copy=True),
                model_action=np.array(step.model_action, dtype=np.float32, copy=True),
                target=np.array(step.target, dtype=np.float32, copy=True),
                images=images,
            )
        )

    def save(self, directory: Path, episode_index: int, metadata: Mapping[str, Any]) -> None:
        """Write ``episode_XXXXXX.npz`` and ``episode_XXXXXX.json``."""
        stem = f"episode_{episode_index:06d}"
        arrays: dict[str, np.ndarray] = {
            "joint_pos": np.stack([step.joint_pos for step in self._steps]),
            "joint_vel": np.stack([step.joint_vel for step in self._steps]),
            "object_position": np.stack([step.object_position for step in self._steps]),
            "model_action": np.stack([step.model_action for step in self._steps]),
            "applied_joint_target": np.stack([step.target for step in self._steps]),
            "terminated": np.asarray([step.terminated for step in self._steps], dtype=bool),
            "truncated": np.asarray([step.truncated for step in self._steps], dtype=bool),
            "inference_valid": np.asarray([step.inference_valid for step in self._steps], dtype=bool),
            "inference_error": np.asarray([step.inference_error for step in self._steps], dtype=np.str_),
        }
        np.savez_compressed(directory / f"{stem}.npz", **arrays)

        meta = dict(metadata)
        meta["format_version"] = 2
        meta["steps"] = len(self._steps)
        meta["terminated_steps"] = int(arrays["terminated"].sum())
        meta["truncated_steps"] = int(arrays["truncated"].sum())
        meta["inference_failure_steps"] = int((~arrays["inference_valid"]).sum())
        meta["recorded_image_frames"] = self._recorded_image_frames
        (directory / f"{stem}.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

        recorded_images = [(index, step.images) for index, step in enumerate(self._steps) if step.images is not None]
        if recorded_images:
            image_arrays: dict[str, np.ndarray] = {"step": np.asarray([index for index, _ in recorded_images])}
            for name in AM_DP123_PI05_CAPTURE_CAMERAS:
                image_arrays[f"image_{name}"] = np.stack([images[name] for _, images in recorded_images])
            np.savez_compressed(directory / f"{stem}_images.npz", **image_arrays)
        print(f"[pi05] wrote {stem} with {len(self._steps)} steps to {directory}", flush=True)


def main() -> None:
    """Parse the CLI, start Isaac Sim, and run the evaluation."""
    from isaaclab.app import AppLauncher

    parser = _build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    if args_cli.policy == "remote":
        _preflight_remote_client(args_cli.host, args_cli.port, args_cli.connect_timeout)

    app_launcher = AppLauncher(args_cli, enable_cameras=True)
    simulation_app = app_launcher.app
    try:
        _run_evaluation(args_cli)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
