# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline validation helpers for recorded AM-DP123 PI0.5 episodes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from amgg_robot_lab.contracts import (
    AM_DP123_CONTROLLED_JOINT_NAMES,
    AM_DP123_JOINT_SPECS,
    AM_DP123_STATE_DIM,
)

AM_DP123_PI05_EPISODE_REQUIRED_FIELDS = (
    "joint_pos",
    "joint_vel",
    "object_position",
    "model_action",
    "applied_joint_target",
    "terminated",
    "truncated",
    "inference_valid",
    "inference_error",
)
"""Fields required by version 2 of the PI0.5 episode format."""


@dataclass(frozen=True, slots=True)
class Pi05EpisodeValidationSummary:
    """Summary of a successfully validated episode.

    Attributes:
        steps: Number of recorded control steps.
        model_action_dim: Width of every recorded model action.
        inference_failure_steps: Number of steps that held the previous target because
            observation construction, inference, or action adaptation failed.
        terminated_steps: Number of steps marked as terminated.
        truncated_steps: Number of steps marked as truncated.
    """

    steps: int
    model_action_dim: int
    inference_failure_steps: int
    terminated_steps: int
    truncated_steps: int


def validate_pi05_episode(arrays: Mapping[str, np.ndarray]) -> Pi05EpisodeValidationSummary:
    """Validate shapes, finite values, and joint limits in a recorded episode.

    Args:
        arrays: Mapping loaded from an episode NPZ file. Joint positions and applied
            targets use [rad], joint velocities use [rad/s], and object positions use [m].

    Returns:
        A compact validation summary.

    Raises:
        ValueError: If a required field is missing or the episode violates its data
            contract.
    """
    missing = [field for field in AM_DP123_PI05_EPISODE_REQUIRED_FIELDS if field not in arrays]
    if missing:
        raise ValueError(f"Episode is missing required fields: {missing}.")

    joint_pos = np.asarray(arrays["joint_pos"])
    if joint_pos.ndim != 2 or joint_pos.shape[1] != AM_DP123_STATE_DIM or joint_pos.shape[0] == 0:
        raise ValueError(f"joint_pos must have shape (N, {AM_DP123_STATE_DIM}) with N > 0, got {joint_pos.shape}.")
    steps = joint_pos.shape[0]

    expected_shapes = {
        "joint_vel": (steps, AM_DP123_STATE_DIM),
        "object_position": (steps, 3),
        "applied_joint_target": (steps, len(AM_DP123_CONTROLLED_JOINT_NAMES)),
        "terminated": (steps,),
        "truncated": (steps,),
        "inference_valid": (steps,),
        "inference_error": (steps,),
    }
    for field, expected in expected_shapes.items():
        actual = np.asarray(arrays[field]).shape
        if actual != expected:
            raise ValueError(f"{field} must have shape {expected}, got {actual}.")

    model_action = np.asarray(arrays["model_action"])
    if model_action.ndim != 2 or model_action.shape[0] != steps or model_action.shape[1] == 0:
        raise ValueError(f"model_action must have shape (N, model_action_dim), got {model_action.shape}.")

    for field in ("joint_pos", "joint_vel", "object_position", "model_action", "applied_joint_target"):
        if not np.all(np.isfinite(np.asarray(arrays[field]))):
            raise ValueError(f"{field} contains NaN or Inf.")

    specs = {spec.name: spec for spec in AM_DP123_JOINT_SPECS}
    lower = np.asarray([specs[name].lower_limit_rad for name in AM_DP123_CONTROLLED_JOINT_NAMES])
    upper = np.asarray([specs[name].upper_limit_rad for name in AM_DP123_CONTROLLED_JOINT_NAMES])
    target = np.asarray(arrays["applied_joint_target"])
    tolerance = 1.0e-6
    if np.any(target < lower - tolerance) or np.any(target > upper + tolerance):
        raise ValueError("applied_joint_target exceeds the AM-DP123 joint position limits [rad].")

    inference_valid = np.asarray(arrays["inference_valid"], dtype=bool)
    inference_error = np.asarray(arrays["inference_error"], dtype=str)
    if np.any(inference_valid & (inference_error != "")):
        raise ValueError("Successful inference steps must have an empty inference_error.")
    if np.any(~inference_valid & (inference_error == "")):
        raise ValueError("Failed inference steps must describe the failure in inference_error.")

    terminated = np.asarray(arrays["terminated"], dtype=bool)
    truncated = np.asarray(arrays["truncated"], dtype=bool)
    return Pi05EpisodeValidationSummary(
        steps=steps,
        model_action_dim=model_action.shape[1],
        inference_failure_steps=int((~inference_valid).sum()),
        terminated_steps=int(terminated.sum()),
        truncated_steps=int(truncated.sum()),
    )


def load_pi05_episode(path: str | Path) -> tuple[dict[str, np.ndarray], Pi05EpisodeValidationSummary]:
    """Load and validate one PI0.5 episode NPZ file.

    Args:
        path: Path to ``episode_XXXXXX.npz``.

    Returns:
        The materialized arrays and their validation summary.
    """
    resolved = Path(path)
    with np.load(resolved, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    return arrays, validate_pi05_episode(arrays)


__all__ = [
    "AM_DP123_PI05_EPISODE_REQUIRED_FIELDS",
    "Pi05EpisodeValidationSummary",
    "load_pi05_episode",
    "validate_pi05_episode",
]
