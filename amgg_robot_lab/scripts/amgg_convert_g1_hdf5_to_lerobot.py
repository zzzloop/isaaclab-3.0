# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert AMGG Unitree G1 recorder HDF5 episodes into LeRobot Dataset v3."""

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from amgg_robot_lab.contracts.amgg_g1_contract import (
    AMGG_G1_CAMERA_NAMES,
    AMGG_G1_EMBODIMENT,
    AMGG_G1_SCHEMA_VERSION,
    AMGG_G1_SIM_HAND_TO_MOTOR_INDICES,
)
from amgg_robot_lab.recording.amgg_g1_dataset_schema import make_amgg_g1_dataset_spec

CAMERA_HDF5_KEYS = {
    "front": "obs/image_front",
    "overview": "obs/image_overview",
}


@dataclass(frozen=True, slots=True)
class G1EpisodeArrays:
    """Validated synchronized arrays for one G1 demonstration."""

    name: str
    state: np.ndarray
    action: np.ndarray
    motor_proxy: np.ndarray
    tactile: np.ndarray
    tcp_pose: np.ndarray
    environment_state: np.ndarray
    images: dict[str, np.ndarray]

    @property
    def length(self) -> int:
        """Return the number of synchronized frames."""
        return self.state.shape[0]


def downsample_g1_episode(episode: G1EpisodeArrays, stride: int) -> G1EpisodeArrays:
    """Downsample every synchronized stream by an integer temporal stride."""
    if stride < 1:
        raise ValueError("G1 downsampling stride must be at least one.")
    return G1EpisodeArrays(
        name=episode.name,
        state=episode.state[::stride],
        action=episode.action[::stride],
        motor_proxy=episode.motor_proxy[::stride],
        tactile=episode.tactile[::stride],
        tcp_pose=episode.tcp_pose[::stride],
        environment_state=episode.environment_state[::stride],
        images={name: frames[::stride] for name, frames in episode.images.items()},
    )


def _float_array(group: h5py.Group, key: str) -> np.ndarray:
    if key not in group:
        raise KeyError(f"Episode {group.name} is missing '{key}'.")
    value = np.asarray(group[key], dtype=np.float32)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError(f"{group.name}/{key} must be a finite 2-D array, got {value.shape}.")
    return value


def _rgb_array(group: h5py.Group, key: str) -> np.ndarray:
    if key not in group:
        raise KeyError(f"Episode {group.name} is missing '{key}'. Record with --enable_cameras.")
    value = np.asarray(group[key])
    if value.ndim != 4 or value.shape[-1] not in {3, 4}:
        raise ValueError(f"{group.name}/{key} must have shape T,H,W,3/4, got {value.shape}.")
    value = value[..., :3]
    if np.issubdtype(value.dtype, np.floating):
        scale = 255.0 if float(value.max(initial=0.0)) <= 1.0 else 1.0
        value = np.clip(value * scale, 0.0, 255.0).astype(np.uint8)
    elif value.dtype != np.uint8:
        value = np.clip(value, 0, 255).astype(np.uint8)
    return value


def load_g1_episode(group: h5py.Group, *, action_source: str, include_images: bool) -> G1EpisodeArrays:
    """Load one G1 episode and enforce the simulation ABI."""
    state = _float_array(group, "obs/robot_joint_pos")
    action = _float_array(group, "actions" if action_source == "raw" else "processed_actions")
    motor_proxy = _float_array(group, "obs/rh56dfx_motor_proxy")
    tactile = _float_array(group, "obs/tactile")
    left_tcp = np.concatenate(
        (_float_array(group, "obs/left_eef_pos"), _float_array(group, "obs/left_eef_quat")), axis=1
    )
    right_tcp = np.concatenate(
        (_float_array(group, "obs/right_eef_pos"), _float_array(group, "obs/right_eef_quat")), axis=1
    )
    tcp_pose = np.concatenate((left_tcp, right_tcp), axis=1)
    environment_state = np.concatenate(
        (
            _float_array(group, "obs/object_state"),
            _float_array(group, "obs/goal"),
            _float_array(group, "obs/progress"),
        ),
        axis=1,
    )
    images = {name: _rgb_array(group, key) for name, key in CAMERA_HDF5_KEYS.items()} if include_images else {}
    arrays = [state, action, motor_proxy, tactile, tcp_pose, environment_state, *images.values()]
    if len({array.shape[0] for array in arrays}) != 1:
        raise ValueError(f"Episode {group.name} streams are not synchronized.")
    expected = {"state": 53, "action": 38, "motor_proxy": 12, "tactile": 12, "tcp_pose": 14}
    actual = {
        "state": state.shape[1],
        "action": action.shape[1],
        "motor_proxy": motor_proxy.shape[1],
        "tactile": tactile.shape[1],
        "tcp_pose": tcp_pose.shape[1],
    }
    if actual != expected:
        raise ValueError(f"Episode {group.name} violates the G1 simulation ABI: expected={expected}, actual={actual}")
    return G1EpisodeArrays(
        name=group.name.rsplit("/", 1)[-1],
        state=state,
        action=action,
        motor_proxy=motor_proxy,
        tactile=tactile,
        tcp_pose=tcp_pose,
        environment_state=environment_state,
        images=images,
    )


def load_g1_episodes(
    input_path: Path,
    *,
    action_source: str,
    include_images: bool,
    only_successful: bool,
) -> list[G1EpisodeArrays]:
    """Load all matching episodes and check cross-episode consistency."""
    episodes = []
    with h5py.File(input_path, "r") as file:
        if "data" not in file:
            raise KeyError("HDF5 root is missing the 'data' group.")
        for name in sorted(file["data"].keys()):
            group = file["data"][name]
            if only_successful and not bool(group.attrs.get("success", False)):
                continue
            episodes.append(load_g1_episode(group, action_source=action_source, include_images=include_images))
    if not episodes:
        raise ValueError("No matching successful G1 episodes were found.")
    reference = episodes[0]
    for episode in episodes[1:]:
        if (
            episode.state.shape[1:] != reference.state.shape[1:]
            or episode.action.shape[1:] != reference.action.shape[1:]
        ):
            raise ValueError("G1 episodes use inconsistent state/action schemas.")
        for camera_name in reference.images:
            if episode.images[camera_name].shape[1:] != reference.images[camera_name].shape[1:]:
                raise ValueError(f"Camera '{camera_name}' shape changes between episodes.")
    return episodes


def _features(episodes: list[G1EpisodeArrays], state_names, action_names) -> dict:
    first = episodes[0]
    features = {
        "observation.state": {"dtype": "float32", "shape": (53,), "names": list(state_names)},
        "observation.rh56dfx_motor_proxy": {
            "dtype": "float32",
            "shape": (12,),
            "names": [f"motor_proxy_{index}" for index in range(12)],
        },
        "observation.tactile": {
            "dtype": "float32",
            "shape": (12,),
            "names": [f"force_{index}" for index in range(12)],
        },
        "observation.tcp_pose": {
            "dtype": "float32",
            "shape": (14,),
            "names": [f"tcp_pose_{index}" for index in range(14)],
        },
        "observation.environment_state": {
            "dtype": "float32",
            "shape": (first.environment_state.shape[1],),
            "names": [f"environment_{index}" for index in range(first.environment_state.shape[1])],
        },
        "action": {"dtype": "float32", "shape": (38,), "names": list(action_names)},
    }
    for camera_name, images in first.images.items():
        features[f"observation.images.{camera_name}"] = {
            "dtype": "video",
            "shape": images.shape[1:],
            "names": ["height", "width", "channel"],
        }
    return features


def convert_g1_dataset(
    input_path: Path,
    output_dir: Path,
    *,
    task_id: str,
    repo_id: str,
    fps: int,
    source_fps: int,
    action_source: str,
    include_images: bool,
    only_successful: bool,
) -> tuple[int, int]:
    """Convert G1 simulation demonstrations into a local LeRobot dataset."""
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    if source_fps < fps or source_fps % fps != 0:
        raise ValueError(f"--source_fps ({source_fps}) must be an integer multiple of --fps ({fps}).")
    temporal_stride = source_fps // fps
    spec = make_amgg_g1_dataset_spec(task_id, fps)
    episodes = load_g1_episodes(
        input_path,
        action_source=action_source,
        include_images=include_images,
        only_successful=only_successful,
    )
    episodes = [downsample_g1_episode(episode, temporal_stride) for episode in episodes]
    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError as error:
        raise RuntimeError("Install LeRobot in the separate conversion environment before conversion.") from error
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=_features(episodes, spec.observation_joint_names, spec.action_names(action_source)),
        root=output_dir,
        robot_type=AMGG_G1_EMBODIMENT,
        use_videos=include_images,
        image_writer_threads=4 if include_images else 0,
    )
    add_frame_parameters = inspect.signature(dataset.add_frame).parameters
    total_frames = 0
    for episode in episodes:
        for frame_index in range(episode.length):
            frame = {
                "observation.state": episode.state[frame_index],
                "observation.rh56dfx_motor_proxy": episode.motor_proxy[frame_index],
                "observation.tactile": episode.tactile[frame_index],
                "observation.tcp_pose": episode.tcp_pose[frame_index],
                "observation.environment_state": episode.environment_state[frame_index],
                "action": episode.action[frame_index],
            }
            for camera_name, images in episode.images.items():
                frame[f"observation.images.{camera_name}"] = images[frame_index]
            if "task" in add_frame_parameters:
                dataset.add_frame(frame, task=spec.instruction)
            else:
                frame["task"] = spec.instruction
                dataset.add_frame(frame)
            total_frames += 1
        dataset.save_episode()
    dataset.finalize()
    metadata = {
        "schema_version": AMGG_G1_SCHEMA_VERSION,
        "task_id": task_id,
        "instruction": spec.instruction,
        "embodiment": AMGG_G1_EMBODIMENT,
        "fps": fps,
        "source_fps": source_fps,
        "temporal_stride": temporal_stride,
        "action_source": action_source,
        "action_space": "official_isaaclab_g1_inspire_sim_38d",
        "rh56dfx_hardware_action_dim": 26,
        "sim_hand_to_motor_indices": list(AMGG_G1_SIM_HAND_TO_MOTOR_INDICES),
        "hardware_calibration_required": True,
        "camera_names": list(AMGG_G1_CAMERA_NAMES) if include_images else [],
        "source_hdf5": input_path.name,
    }
    metadata_path = output_dir / "meta" / "amgg_g1_schema.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(episodes), total_frames


def main() -> None:
    """Parse CLI arguments and convert a local dataset without uploading it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_hdf5", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--task", required=True, dest="task_id")
    parser.add_argument("--repo_id", default="local/amgg_g1_dataset")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--source_fps",
        type=int,
        default=None,
        help="Source HDF5 rate before synchronized downsampling; defaults to --fps for backward compatibility.",
    )
    parser.add_argument("--action_source", choices=("raw", "processed"), default="raw")
    parser.add_argument("--include_images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--only_successful", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    episodes, frames = convert_g1_dataset(
        args.input_hdf5.resolve(),
        args.output_dir.resolve(),
        task_id=args.task_id,
        repo_id=args.repo_id,
        fps=args.fps,
        source_fps=args.source_fps if args.source_fps is not None else args.fps,
        action_source=args.action_source,
        include_images=args.include_images,
        only_successful=args.only_successful,
    )
    print(f"Converted {episodes} G1 episodes / {frames} frames to {args.output_dir}")


if __name__ == "__main__":
    main()
