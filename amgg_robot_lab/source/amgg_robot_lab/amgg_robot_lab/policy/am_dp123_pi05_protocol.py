# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OpenPI observation payloads and the PI0.5 action-layout safety adapter for AM-DP123.

This module deliberately depends only on NumPy and :mod:`amgg_robot_lab.contracts`. The
observation conversion, the action-layout JSON schema, and the safety adapter can be
tested offline on Windows without Isaac Sim, Torch, OpenPI, or a WebSocket client.

The model output dimension is never hard-coded: it comes from the external action-layout
JSON, which selects the 18 entries of the canonical simulation ABI and scales them into
absolute joint targets.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import numpy as np

from amgg_robot_lab.contracts import (
    AM_DP123_CONTROLLED_JOINT_NAMES,
    AM_DP123_JOINT_SPECS,
    AM_DP123_STATE_DIM,
    AM_DP123_STATE_JOINT_NAMES,
    AmDp123JointSpec,
)

AM_DP123_PI05_STATE_JOINT_NAMES = AM_DP123_STATE_JOINT_NAMES
"""Canonical 23-D state order sent as ``observation/state``."""

AM_DP123_PI05_CONTROLLED_JOINT_NAMES = AM_DP123_CONTROLLED_JOINT_NAMES
"""Canonical 18-D simulation command order produced by the action adapter."""

AM_DP123_PI05_REQUIRED_CAMERAS: tuple[str, str, str, str] = (
    "head_left",
    "head_right",
    "left_wrist",
    "right_wrist",
)
"""Camera names the OpenPI payload requires; the order fixes the payload keys."""

AM_DP123_PI05_OBSERVATION_KEYS: dict[str, str] = {
    "image": "observation/image",
    "image_right": "observation/image_right",
    "wrist_image": "observation/wrist_image",
    "wrist_image_right": "observation/wrist_image_right",
    "state": "observation/state",
    "joint_velocity": "observation/joint_velocity",
    "prompt": "prompt",
}
"""Stable OpenPI payload keys, including the left/right extension keys."""

AM_DP123_PI05_ABSOLUTE_MODE = "absolute_joint_position"
AM_DP123_PI05_DELTA_MODE = "delta_joint_position"
AM_DP123_PI05_ACTION_MODES: tuple[str, str] = (AM_DP123_PI05_ABSOLUTE_MODE, AM_DP123_PI05_DELTA_MODE)
AM_DP123_PI05_LAYOUT_SCHEMA_VERSION = 1

_LAYOUT_DIR = Path(__file__).resolve().parent / "layouts"
AM_DP123_PI05_DEFAULT_LAYOUT_PATH = _LAYOUT_DIR / "am_dp123_joint_position_18.json"
AM_DP123_PI05_32_TEMPLATE_PATH = _LAYOUT_DIR / "am_dp123_pi05_32_template.json"


def as_numpy(value: object) -> np.ndarray:
    """Return a NumPy view of an array-like, accepting CPU Torch tensors.

    Args:
        value: NumPy array, Torch tensor, or anything :func:`numpy.asarray` accepts.

    Returns:
        The value as a NumPy array. Tensor-likes are detached and moved to CPU first.
    """
    if isinstance(value, np.ndarray):
        return value
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach().cpu().numpy()
    return np.asarray(value)


def controlled_state_indices() -> tuple[int, ...]:
    """Return where the 18 commanded joints sit inside the 23-D state vector."""
    state_positions = {name: index for index, name in enumerate(AM_DP123_STATE_JOINT_NAMES)}
    return tuple(state_positions[name] for name in AM_DP123_CONTROLLED_JOINT_NAMES)


def to_uint8_rgb(image: object) -> np.ndarray:
    """Convert a camera frame to a contiguous ``uint8`` RGB array.

    Args:
        image: ``HxWx3`` or ``HxWx4`` array-like. Floating-point input is interpreted as
            the ``[0, 1]`` range used by Isaac Sim; integer input is clipped to ``[0, 255]``.

    Returns:
        Contiguous ``uint8`` RGB image with shape ``HxWx3``.

    Raises:
        ValueError: If the image is not a finite three- or four-channel frame.
    """
    array = np.asarray(as_numpy(image))
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError(f"Expected an HxWx3 or HxWx4 image, got shape {array.shape}.")
    if array.shape[2] == 4:
        array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        if not np.all(np.isfinite(array)):
            raise ValueError("Image contains non-finite values.")
        array = np.clip(array, 0.0, 1.0) * 255.0
    elif np.issubdtype(array.dtype, np.integer):
        array = np.clip(array, 0, 255)
    else:
        raise ValueError(f"Unsupported image dtype '{array.dtype}'.")
    return np.ascontiguousarray(array.round().astype(np.uint8))


def build_pi05_observation(
    joint_pos: object,
    joint_vel: object,
    images: Mapping[str, object],
    prompt: str,
    image_transform: Callable[[object], np.ndarray] | None = None,
) -> dict[str, object]:
    """Build the OpenPI remote-inference payload from Isaac Lab observations.

    Args:
        joint_pos: 23-D joint positions ordered like
            :data:`AM_DP123_PI05_STATE_JOINT_NAMES`.
        joint_vel: 23-D joint velocities in the same order.
        images: Mapping keyed by :data:`AM_DP123_PI05_REQUIRED_CAMERAS`.
        prompt: Task instruction sent verbatim to OpenPI.
        image_transform: Optional per-image conversion. Defaults to
            :func:`to_uint8_rgb`. Remote runs pass an OpenPI-backed transform that
            resizes with padding to 224x224.

    Returns:
        Payload dict with the canonical OpenPI keys.

    Raises:
        ValueError: If a camera is missing, a state vector has the wrong shape, or a
            state value is not finite.
    """
    state = as_numpy(joint_pos)
    if state.shape != (AM_DP123_STATE_DIM,):
        raise ValueError(f"Expected a {AM_DP123_STATE_DIM}-D joint position state, got shape {state.shape}.")
    if not np.all(np.isfinite(state)):
        raise ValueError("Joint position state contains non-finite values.")

    velocity = as_numpy(joint_vel)
    if velocity.shape != (AM_DP123_STATE_DIM,):
        raise ValueError(f"Expected a {AM_DP123_STATE_DIM}-D joint velocity state, got shape {velocity.shape}.")
    if not np.all(np.isfinite(velocity)):
        raise ValueError("Joint velocity state contains non-finite values.")

    if not isinstance(prompt, str):
        raise ValueError("OpenPI prompt must be a string.")

    transform = image_transform or to_uint8_rgb
    keys = AM_DP123_PI05_OBSERVATION_KEYS
    converted: dict[str, np.ndarray] = {}
    for camera in AM_DP123_PI05_REQUIRED_CAMERAS:
        if camera not in images or images[camera] is None:
            raise ValueError(f"Missing required camera '{camera}' in the observation payload.")
        converted[camera] = transform(images[camera])

    return {
        keys["image"]: converted["head_left"],
        keys["image_right"]: converted["head_right"],
        keys["wrist_image"]: converted["left_wrist"],
        keys["wrist_image_right"]: converted["right_wrist"],
        keys["state"]: np.ascontiguousarray(state.astype(np.float32)),
        keys["joint_velocity"]: np.ascontiguousarray(velocity.astype(np.float32)),
        keys["prompt"]: prompt,
    }


def _as_float_vector(value: object, size: int, field: str) -> tuple[float, ...]:
    """Broadcast a scalar or length-``size`` sequence to a finite float tuple."""
    if isinstance(value, bool):
        raise ValueError(f"Action layout field '{field}' must be numeric.")
    if isinstance(value, (int, float)):
        values = (float(value),) * size
    elif isinstance(value, Sequence) and not isinstance(value, str):
        values = tuple(float(entry) for entry in value)
        if len(values) != size:
            raise ValueError(f"Action layout field '{field}' must hold {size} values, got {len(values)}.")
    else:
        raise ValueError(f"Action layout field '{field}' must be a scalar or a {size}-value list.")
    if not all(isfinite(entry) for entry in values):
        raise ValueError(f"Action layout field '{field}' contains non-finite values.")
    return values


def action_layout_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of an action-layout JSON file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class AmDp123ActionLayout:
    """External mapping from a model action vector to the 18-D simulation ABI.

    Attributes:
        schema_version: Layout schema version. Only version ``1`` is supported.
        confirmed: ``False`` marks a template whose model semantics are unverified.
        model_action_dim: Width of the model action chunk.
        mode: ``absolute_joint_position`` or ``delta_joint_position``.
        sim_joint_names: Simulation joint order the selected entries command. It must
            equal :data:`AM_DP123_PI05_CONTROLLED_JOINT_NAMES`.
        source_indices: One model-action index per simulation joint. ``None`` only while
            the layout is unconfirmed.
        scale: Per-joint multiplier applied after selection.
        offset: Per-joint offset applied after scaling.
        velocity_scale: Fraction of each joint's ``max_velocity_rad_s`` allowed per step.
        notes: Free-form human notes.
        path: Source file, when the layout was loaded from disk.
    """

    schema_version: int
    confirmed: bool
    model_action_dim: int
    mode: str
    sim_joint_names: tuple[str, ...]
    source_indices: tuple[int, ...] | None
    scale: tuple[float, ...]
    offset: tuple[float, ...]
    velocity_scale: float
    notes: str = ""
    path: Path | None = None

    @property
    def sim_action_dim(self) -> int:
        """Number of simulated joints the layout commands."""
        return len(self.sim_joint_names)

    def require_confirmed(self) -> None:
        """Reject a template whose model-action semantics are still unverified."""
        if not self.confirmed or self.source_indices is None:
            raise ValueError(
                "Action layout is unconfirmed: its model action semantics must be filled in and "
                "reviewed before it can drive the robot."
            )

    def scale_array(self) -> np.ndarray:
        """Return ``scale`` as a float array of shape ``(sim_action_dim,)``."""
        return np.asarray(self.scale, dtype=np.float64)

    def offset_array(self) -> np.ndarray:
        """Return ``offset`` as a float array of shape ``(sim_action_dim,)``."""
        return np.asarray(self.offset, dtype=np.float64)

    def sha256(self) -> str | None:
        """Return the digest of the source file, or ``None`` for in-memory layouts."""
        return action_layout_sha256(self.path) if self.path is not None else None

    @classmethod
    def from_dict(cls, data: Mapping[str, object], path: str | Path | None = None) -> AmDp123ActionLayout:
        """Validate and build a layout from a decoded JSON object.

        Args:
            data: Decoded layout dict.
            path: Source file used for the digest metadata.

        Returns:
            The validated layout.

        Raises:
            ValueError: If any field is missing, malformed, or inconsistent with the ABI.
        """
        schema_version = int(data.get("schema_version", 0))
        if schema_version != AM_DP123_PI05_LAYOUT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported action layout schema version {schema_version}.")

        mode = str(data.get("mode", ""))
        if mode not in AM_DP123_PI05_ACTION_MODES:
            raise ValueError(f"Unsupported action layout mode '{mode}'.")

        model_action_dim = int(data.get("model_action_dim", 0))
        if model_action_dim < 1:
            raise ValueError("Action layout model_action_dim must be positive.")

        raw_names = data.get("sim_joint_names")
        if not isinstance(raw_names, Sequence) or isinstance(raw_names, str):
            raise ValueError("Action layout sim_joint_names must be a list.")
        sim_joint_names = tuple(str(name) for name in raw_names)
        if sim_joint_names != AM_DP123_PI05_CONTROLLED_JOINT_NAMES:
            raise ValueError(
                "Action layout sim_joint_names must equal AM_DP123_CONTROLLED_JOINT_NAMES in canonical order."
            )

        confirmed = bool(data.get("confirmed", True))
        raw_indices = data.get("source_indices")
        source_indices: tuple[int, ...] | None
        if raw_indices is None:
            if confirmed:
                raise ValueError("A confirmed action layout must define source_indices.")
            source_indices = None
        else:
            if not isinstance(raw_indices, Sequence) or isinstance(raw_indices, str):
                raise ValueError("Action layout source_indices must be a list or null.")
            source_indices = tuple(int(index) for index in raw_indices)
            if len(source_indices) != len(sim_joint_names):
                raise ValueError(
                    f"Action layout source_indices must hold {len(sim_joint_names)} entries, got {len(source_indices)}."
                )
            if len(set(source_indices)) != len(source_indices):
                raise ValueError("Action layout source_indices must be unique.")
            if any(index < 0 or index >= model_action_dim for index in source_indices):
                raise ValueError("Action layout source_indices must fall inside the model action vector.")

        size = len(sim_joint_names)
        velocity_scale = float(data.get("velocity_scale", 1.0))
        if not isfinite(velocity_scale) or velocity_scale <= 0.0:
            raise ValueError("Action layout velocity_scale must be a positive finite number.")

        return cls(
            schema_version=schema_version,
            confirmed=confirmed,
            model_action_dim=model_action_dim,
            mode=mode,
            sim_joint_names=sim_joint_names,
            source_indices=source_indices,
            scale=_as_float_vector(data.get("scale", 1.0), size, "scale"),
            offset=_as_float_vector(data.get("offset", 0.0), size, "offset"),
            velocity_scale=velocity_scale,
            notes=str(data.get("notes", "")),
            path=Path(path) if path is not None else None,
        )


def load_action_layout(path: str | Path) -> AmDp123ActionLayout:
    """Load, validate, and freeze an action layout from JSON.

    Args:
        path: JSON file written against the documented layout schema.

    Returns:
        The validated layout with its source path recorded.
    """
    resolved = Path(path)
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"Action layout '{resolved}' must decode to a JSON object.")
    return AmDp123ActionLayout.from_dict(data, path=resolved)


def load_default_action_layout() -> AmDp123ActionLayout:
    """Load the identity 18-D layout used by the mock policies."""
    return load_action_layout(AM_DP123_PI05_DEFAULT_LAYOUT_PATH)


class Pi05ActionAdapter:
    """Map raw model action chunks onto bounded absolute joint targets.

    The adapter selects the 18 simulated joints with the layout's ``source_indices``,
    applies ``scale`` and ``offset``, converts to absolute or delta targets, and then
    clamps both position and per-step velocity using the joint contract. It depends only
    on NumPy and the contracts so it can be unit-tested without a simulator.
    """

    def __init__(
        self,
        layout: AmDp123ActionLayout,
        control_dt: float,
        joint_specs: Sequence[AmDp123JointSpec] = AM_DP123_JOINT_SPECS,
    ):
        """Build an adapter for one control period.

        Args:
            layout: Confirmed action layout.
            control_dt: Seconds between two control steps, i.e. ``sim.dt * decimation``.
            joint_specs: Joint contract providing the position and velocity limits.

        Raises:
            ValueError: If the layout is unconfirmed, the control period is invalid, or a
                commanded joint is missing from the joint contract.
        """
        layout.require_confirmed()
        if not isfinite(control_dt) or control_dt <= 0.0:
            raise ValueError("control_dt must be a positive finite number.")

        spec_by_name = {spec.name: spec for spec in joint_specs}
        missing = [name for name in layout.sim_joint_names if name not in spec_by_name]
        if missing:
            raise ValueError(f"Action layout joints missing from the joint contract: {missing}.")

        self._layout = layout
        self._control_dt = float(control_dt)
        self._lower = np.array([spec_by_name[name].lower_limit_rad for name in layout.sim_joint_names])
        self._upper = np.array([spec_by_name[name].upper_limit_rad for name in layout.sim_joint_names])
        self._max_step = (
            np.array([spec_by_name[name].max_velocity_rad_s for name in layout.sim_joint_names])
            * self._control_dt
            * layout.velocity_scale
        )
        selected = layout.source_indices
        assert selected is not None  # require_confirmed() guarantees this.
        self._source_indices = np.asarray(selected, dtype=np.intp)
        self._scale = layout.scale_array()
        self._offset = layout.offset_array()
        self._last_target: np.ndarray | None = None

    @property
    def layout(self) -> AmDp123ActionLayout:
        """The action layout backing this adapter."""
        return self._layout

    @property
    def action_dim(self) -> int:
        """Width of the simulation command produced by :meth:`adapt`."""
        return self._layout.sim_action_dim

    @property
    def current_target(self) -> np.ndarray:
        """Last bounded target, or the reset pose before the first chunk."""
        if self._last_target is None:
            raise RuntimeError("Pi05ActionAdapter.reset() must be called before reading the target.")
        return self._last_target.copy()

    def reset(self, current_joint_pos: object) -> np.ndarray:
        """Rebase velocity limiting at the measured joint positions after a reset.

        Args:
            current_joint_pos: Current commanded joints in the layout's order.

        Returns:
            The clamped reset target that the adapter will use as its starting point.
        """
        positions = np.asarray(as_numpy(current_joint_pos), dtype=np.float64)
        if positions.shape != (self.action_dim,):
            raise ValueError(f"reset expects {self.action_dim}-D joint positions, got shape {positions.shape}.")
        if not np.all(np.isfinite(positions)):
            raise ValueError("reset joint positions must be finite.")
        self._last_target = np.clip(positions, self._lower, self._upper).copy()
        return self._last_target.copy()

    def adapt(self, actions: object) -> np.ndarray:
        """Convert a raw model action or chunk into a bounded ``(T, 18)`` target chunk.

        Args:
            actions: ``(model_action_dim,)`` single action or ``(T, model_action_dim)`` chunk.

        Returns:
            Float64 array of shape ``(T, 18)`` with position- and velocity-limited targets.

        Raises:
            RuntimeError: If :meth:`reset` has not been called yet.
            ValueError: If the chunk is empty, non-finite, or has the wrong width.
        """
        values = np.asarray(as_numpy(actions), dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2:
            raise ValueError(f"Model actions must be 1-D or 2-D, got shape {values.shape}.")
        if values.shape[0] == 0:
            raise ValueError("Model action chunk is empty.")
        if values.shape[1] != self._layout.model_action_dim:
            raise ValueError(
                f"Model action width {values.shape[1]} does not match layout model_action_dim "
                f"{self._layout.model_action_dim}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("Model actions contain NaN or Inf.")
        if self._last_target is None:
            raise RuntimeError("Pi05ActionAdapter.reset() must be called before adapt().")

        selected = values[:, self._source_indices]
        transformed = selected * self._scale + self._offset

        targets = np.empty((transformed.shape[0], self.action_dim), dtype=np.float64)
        previous = self._last_target
        for row, candidate in enumerate(transformed):
            target = candidate if self._layout.mode == AM_DP123_PI05_ABSOLUTE_MODE else previous + candidate
            target = np.clip(target, self._lower, self._upper)
            target = np.clip(target, previous - self._max_step, previous + self._max_step)
            targets[row] = target
            previous = target
        self._last_target = previous
        return targets


__all__ = [
    "AM_DP123_PI05_32_TEMPLATE_PATH",
    "AM_DP123_PI05_ABSOLUTE_MODE",
    "AM_DP123_PI05_ACTION_MODES",
    "AM_DP123_PI05_CONTROLLED_JOINT_NAMES",
    "AM_DP123_PI05_DEFAULT_LAYOUT_PATH",
    "AM_DP123_PI05_DELTA_MODE",
    "AM_DP123_PI05_LAYOUT_SCHEMA_VERSION",
    "AM_DP123_PI05_OBSERVATION_KEYS",
    "AM_DP123_PI05_REQUIRED_CAMERAS",
    "AM_DP123_PI05_STATE_JOINT_NAMES",
    "AmDp123ActionLayout",
    "Pi05ActionAdapter",
    "action_layout_sha256",
    "as_numpy",
    "build_pi05_observation",
    "controlled_state_indices",
    "load_action_layout",
    "load_default_action_layout",
    "to_uint8_rgb",
]
