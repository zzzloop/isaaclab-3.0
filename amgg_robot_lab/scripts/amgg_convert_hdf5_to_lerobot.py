# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert successful AMGG recorder HDF5 episodes into LeRobot Dataset v3."""

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from amgg_robot_lab.recording.amgg_dataset_schema import (
    AMGG_ACTION_KEY,
    AMGG_ENVIRONMENT_STATE_KEY,
    AMGG_IMAGE_PREFIX,
    AMGG_OBSERVATION_STATE_KEY,
    AMGG_RAW_ACTION_NAMES,
    AMGG_SCHEMA_VERSION,
    AMGG_TCP_STATE_KEY,
    make_amgg_dataset_spec,
)

CAMERA_HDF5_KEYS = {
    "head": "obs/image_head",
    "left_wrist": "obs/image_left_wrist",
    "right_wrist": "obs/image_right_wrist",
    "overview": "obs/image_overview",
}


@dataclass(frozen=True, slots=True)
class EpisodeArrays:
    """Validated arrays for one demonstration."""

    name: str
    state: np.ndarray
    action: np.ndarray
    tcp_pose: np.ndarray
    environment_state: np.ndarray
    images: dict[str, np.ndarray]

    @property
    def length(self) -> int:
        """Number of synchronized frames."""
        return self.state.shape[0]


def _float_array(group: h5py.Group, key: str) -> np.ndarray:
    if key not in group:
        raise KeyError(f"Episode {group.name} is missing '{key}'.")
    array = np.asarray(group[key], dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{group.name}/{key} must be a finite 2-D array, got {array.shape}.")
    return array


def _rgb_array(group: h5py.Group, key: str) -> np.ndarray:
    if key not in group:
        raise KeyError(f"Episode {group.name} is missing '{key}'. Record with cameras enabled.")
    array = np.asarray(group[key])
    if array.ndim != 4 or array.shape[-1] not in {3, 4}:
        raise ValueError(f"{group.name}/{key} must be T,H,W,3/4 RGB(A), got {array.shape}.")
    array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        scale = 255.0 if float(array.max(initial=0.0)) <= 1.0 else 1.0
        array = np.clip(array * scale, 0.0, 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def load_episode(
    group: h5py.Group,
    *,
    action_source: str,
    include_images: bool,
) -> EpisodeArrays:
    """Load and validate one AMGG HDF5 episode."""
    state = _float_array(group, "obs/robot_joint_pos")
    action_key = "processed_actions" if action_source == "processed" else "actions"
    action = _float_array(group, action_key)
    left_tcp = _float_array(group, "obs/left_tcp_pose")
    right_tcp = _float_array(group, "obs/right_tcp_pose")
    tcp_pose = np.concatenate((left_tcp, right_tcp), axis=1)
    environment_parts = [
        _float_array(group, "obs/object_state"),
        _float_array(group, "obs/goal"),
        _float_array(group, "obs/progress"),
    ]
    environment_state = np.concatenate(environment_parts, axis=1)
    images = {name: _rgb_array(group, key) for name, key in CAMERA_HDF5_KEYS.items()} if include_images else {}
    arrays = [state, action, tcp_pose, environment_state, *images.values()]
    lengths = {array.shape[0] for array in arrays}
    if len(lengths) != 1:
        raise ValueError(f"Episode {group.name} streams are not synchronized: lengths={sorted(lengths)}")
    if state.shape[1] != 23:
        raise ValueError(f"Episode {group.name} state must follow the 23-D AMGG ABI, got {state.shape[1]}.")
    expected_action_dim = 21 if action_source == "processed" else 18
    if action.shape[1] != expected_action_dim:
        raise ValueError(
            f"Episode {group.name} {action_source} action must be {expected_action_dim}-D, got {action.shape[1]}."
        )
    if tcp_pose.shape[1] != 14:
        raise ValueError(f"Episode {group.name} TCP pose must be 14-D.")
    return EpisodeArrays(group.name.rsplit("/", 1)[-1], state, action, tcp_pose, environment_state, images)


def load_episodes(
    input_path: Path,
    *,
    action_source: str,
    include_images: bool,
    only_successful: bool,
) -> list[EpisodeArrays]:
    """Load all selected demonstrations from an Isaac Lab HDF5 file."""
    episodes: list[EpisodeArrays] = []
    with h5py.File(input_path, "r") as file:
        if "data" not in file:
            raise KeyError("HDF5 root is missing the 'data' group.")
        for name in sorted(file["data"].keys()):
            group = file["data"][name]
            if only_successful and not bool(group.attrs.get("success", False)):
                continue
            episodes.append(load_episode(group, action_source=action_source, include_images=include_images))
    if not episodes:
        raise ValueError("No matching AMGG episodes were found.")
    reference = episodes[0]
    for episode in episodes[1:]:
        if (
            episode.state.shape[1:] != reference.state.shape[1:]
            or episode.action.shape[1:] != reference.action.shape[1:]
        ):
            raise ValueError("AMGG episodes use inconsistent state/action schemas.")
        for camera_name in reference.images:
            if episode.images[camera_name].shape[1:] != reference.images[camera_name].shape[1:]:
                raise ValueError(f"Camera '{camera_name}' shape changes between episodes.")
    return episodes


def _features(episodes: list[EpisodeArrays], state_names: tuple[str, ...], action_names: tuple[str, ...]) -> dict:
    first = episodes[0]
    features = {
        AMGG_OBSERVATION_STATE_KEY: {
            "dtype": "float32",
            "shape": (first.state.shape[1],),
            "names": list(state_names),
        },
        AMGG_ACTION_KEY: {
            "dtype": "float32",
            "shape": (first.action.shape[1],),
            "names": list(action_names),
        },
        AMGG_TCP_STATE_KEY: {
            "dtype": "float32",
            "shape": (first.tcp_pose.shape[1],),
            "names": [f"tcp_pose_{index}" for index in range(first.tcp_pose.shape[1])],
        },
        AMGG_ENVIRONMENT_STATE_KEY: {
            "dtype": "float32",
            "shape": (first.environment_state.shape[1],),
            "names": [f"environment_{index}" for index in range(first.environment_state.shape[1])],
        },
    }
    for camera_name, images in first.images.items():
        features[f"{AMGG_IMAGE_PREFIX}{camera_name}"] = {
            "dtype": "video",
            "shape": images.shape[1:],
            "names": ["height", "width", "channel"],
        }
    return features


def convert_dataset(
    input_path: Path,
    output_dir: Path,
    *,
    task_id: str,
    repo_id: str,
    fps: int,
    action_source: str,
    include_images: bool,
    only_successful: bool,
) -> tuple[int, int]:
    """Convert AMGG HDF5 demonstrations into a local LeRobot dataset."""
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}. Choose a new directory.")
    spec = make_amgg_dataset_spec(task_id, fps)
    episodes = load_episodes(
        input_path,
        action_source=action_source,
        include_images=include_images,
        only_successful=only_successful,
    )
    action_names = spec.action_joint_names if action_source == "processed" else AMGG_RAW_ACTION_NAMES
    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError as error:
        raise RuntimeError(
            "LeRobot is not installed. Install the current LeRobot package in a separate conversion environment."
        ) from error
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=_features(episodes, spec.observation_joint_names, action_names),
        root=output_dir,
        robot_type="amgg",
        use_videos=include_images,
        image_writer_threads=4 if include_images else 0,
    )
    add_frame_parameters = inspect.signature(dataset.add_frame).parameters
    total_frames = 0
    for episode in episodes:
        for frame_index in range(episode.length):
            frame = {
                AMGG_OBSERVATION_STATE_KEY: episode.state[frame_index],
                AMGG_ACTION_KEY: episode.action[frame_index],
                AMGG_TCP_STATE_KEY: episode.tcp_pose[frame_index],
                AMGG_ENVIRONMENT_STATE_KEY: episode.environment_state[frame_index],
            }
            for camera_name, images in episode.images.items():
                frame[f"{AMGG_IMAGE_PREFIX}{camera_name}"] = images[frame_index]
            if "task" in add_frame_parameters:
                dataset.add_frame(frame, task=spec.task)
            else:
                frame["task"] = spec.task
                dataset.add_frame(frame)
            total_frames += 1
        dataset.save_episode()
    dataset.finalize()
    metadata = {
        "schema_version": AMGG_SCHEMA_VERSION,
        "task_id": task_id,
        "instruction": spec.task,
        "fps": fps,
        "state_names": list(spec.observation_joint_names),
        "action_source": action_source,
        "action_names": list(action_names),
        "camera_names": list(spec.camera_names) if include_images else [],
        "source_hdf5": input_path.name,
    }
    metadata_path = output_dir / "meta" / "amgg_schema.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(episodes), total_frames


def main() -> None:
    """Parse arguments and convert a local dataset without uploading it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_hdf5", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--task", required=True, dest="task_id")
    parser.add_argument("--repo_id", default="local/amgg_dataset")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--action_source", choices=("processed", "raw"), default="processed")
    parser.add_argument("--include_images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--only_successful", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    episodes, frames = convert_dataset(
        args.input_hdf5.resolve(),
        args.output_dir.resolve(),
        task_id=args.task_id,
        repo_id=args.repo_id,
        fps=args.fps,
        action_source=args.action_source,
        include_images=args.include_images,
        only_successful=args.only_successful,
    )
    print(f"Converted {episodes} episodes / {frames} frames to {args.output_dir}")


if __name__ == "__main__":
    main()
