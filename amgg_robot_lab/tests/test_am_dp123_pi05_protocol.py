# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OpenPI observation payloads and the PI0.5 action-layout safety adapter."""

from __future__ import annotations

import numpy as np
import pytest

from amgg_robot_lab.contracts import AM_DP123_CONTROLLED_JOINT_NAMES, AM_DP123_JOINT_SPECS, AM_DP123_STATE_JOINT_NAMES
from amgg_robot_lab.policy import (
    AM_DP123_PI05_32_TEMPLATE_PATH,
    AM_DP123_PI05_ABSOLUTE_MODE,
    AM_DP123_PI05_DELTA_MODE,
    AM_DP123_PI05_OBSERVATION_KEYS,
    AM_DP123_PI05_REQUIRED_CAMERAS,
    AmDp123ActionLayout,
    Pi05ActionAdapter,
    build_pi05_observation,
    controlled_state_indices,
    load_action_layout,
    load_default_action_layout,
    to_uint8_rgb,
)

_CONTROL_DT = 1.0 / 30.0
_SPECS = {spec.name: spec for spec in AM_DP123_JOINT_SPECS}
_SPECS_IN_ORDER = [_SPECS[name] for name in AM_DP123_CONTROLLED_JOINT_NAMES]
_LOWER = np.array([spec.lower_limit_rad for spec in _SPECS_IN_ORDER])
_UPPER = np.array([spec.upper_limit_rad for spec in _SPECS_IN_ORDER])
_HOME = np.array([spec.home_position_rad for spec in _SPECS_IN_ORDER])
_MID = (_LOWER + _UPPER) / 2.0


def _layout_dict(**overrides: object) -> dict[str, object]:
    layout: dict[str, object] = {
        "schema_version": 1,
        "confirmed": True,
        "model_action_dim": 18,
        "mode": AM_DP123_PI05_ABSOLUTE_MODE,
        "sim_joint_names": list(AM_DP123_CONTROLLED_JOINT_NAMES),
        "source_indices": list(range(18)),
        "scale": 1.0,
        "offset": 0.0,
        "velocity_scale": 0.5,
    }
    layout.update(overrides)
    return layout


def test_default_layout_is_the_identity_18d_abi():
    """The shipped default layout maps every model entry to one commanded joint."""
    layout = load_default_action_layout()
    assert layout.schema_version == 1
    assert layout.confirmed is True
    assert layout.model_action_dim == 18
    assert layout.sim_joint_names == AM_DP123_CONTROLLED_JOINT_NAMES
    assert layout.source_indices == tuple(range(18))
    assert layout.mode == AM_DP123_PI05_ABSOLUTE_MODE
    assert layout.sha256() is not None


def test_layout_joint_names_must_equal_the_canonical_order():
    """Reordered or renamed simulation joints are rejected."""
    with pytest.raises(ValueError):
        AmDp123ActionLayout.from_dict(_layout_dict(sim_joint_names=list(reversed(AM_DP123_CONTROLLED_JOINT_NAMES))))
    with pytest.raises(ValueError):
        AmDp123ActionLayout.from_dict(_layout_dict(sim_joint_names=list(AM_DP123_CONTROLLED_JOINT_NAMES[:-1])))


def test_layout_rejects_duplicate_and_out_of_range_indices():
    """A layout may not select one model entry twice or read past the action vector."""
    with pytest.raises(ValueError):
        AmDp123ActionLayout.from_dict(_layout_dict(source_indices=[0] * 18))
    with pytest.raises(ValueError):
        AmDp123ActionLayout.from_dict(_layout_dict(source_indices=[*range(17), 18]))
    with pytest.raises(ValueError):
        AmDp123ActionLayout.from_dict(_layout_dict(source_indices=list(range(17))))


def test_confirmed_layout_requires_source_indices():
    """A confirmed layout without a mapping is invalid."""
    with pytest.raises(ValueError):
        AmDp123ActionLayout.from_dict(_layout_dict(source_indices=None))


def test_unconfirmed_template_cannot_drive_the_robot():
    """The 32-D template stays unconfirmed, so it never builds an adapter."""
    template = load_action_layout(AM_DP123_PI05_32_TEMPLATE_PATH)
    assert template.model_action_dim == 32
    assert template.confirmed is False
    assert template.source_indices is None
    with pytest.raises(ValueError):
        template.require_confirmed()
    with pytest.raises(ValueError):
        Pi05ActionAdapter(template, _CONTROL_DT)


def test_any_model_dimension_maps_onto_the_18d_abi():
    """A 24-D model vector maps its selected entries onto the commanded joints."""
    source_indices = tuple(range(5, 23))
    layout = AmDp123ActionLayout.from_dict(_layout_dict(model_action_dim=24, source_indices=list(source_indices)))
    adapter = Pi05ActionAdapter(layout, _CONTROL_DT)
    adapter.reset(_HOME)

    model_action = np.zeros(24)
    model_action[list(source_indices)] = _HOME
    targets = adapter.adapt(model_action)
    assert targets.shape == (1, 18)
    np.testing.assert_allclose(targets[0], _HOME, atol=1e-9)


def test_chunk_and_single_action_shapes():
    """A 1-D action becomes one row; a chunk keeps its row count."""
    adapter = Pi05ActionAdapter(AmDp123ActionLayout.from_dict(_layout_dict()), _CONTROL_DT)
    adapter.reset(np.zeros(18))
    assert adapter.adapt(np.full(18, 0.01)).shape == (1, 18)
    assert adapter.adapt(np.full((5, 18), 0.01)).shape == (5, 18)


def test_invalid_actions_are_rejected():
    """Wrong widths, empty chunks, and non-finite values never reach the simulator."""
    adapter = Pi05ActionAdapter(AmDp123ActionLayout.from_dict(_layout_dict()), _CONTROL_DT)
    adapter.reset(np.zeros(18))
    with pytest.raises(ValueError):
        adapter.adapt(np.zeros(17))
    with pytest.raises(ValueError):
        adapter.adapt(np.zeros((0, 18)))
    with pytest.raises(ValueError):
        adapter.adapt(np.full(18, np.nan))
    with pytest.raises(ValueError):
        adapter.adapt(np.full((3, 18), np.inf))


def test_adapt_before_reset_is_rejected():
    """Velocity limiting needs a measured starting point."""
    adapter = Pi05ActionAdapter(AmDp123ActionLayout.from_dict(_layout_dict()), _CONTROL_DT)
    with pytest.raises(RuntimeError):
        adapter.adapt(np.zeros(18))


def test_scale_and_offset_broadcast_scalar_and_vector():
    """Scalar and per-joint scale/offset are applied before limiting."""
    scalar = Pi05ActionAdapter(
        AmDp123ActionLayout.from_dict(_layout_dict(scale=1.5, offset=0.0, velocity_scale=1e6)), _CONTROL_DT
    )
    scalar.reset(_MID)
    np.testing.assert_allclose(scalar.adapt(_MID / 1.5)[0], _MID, atol=1e-9)

    shifted = Pi05ActionAdapter(
        AmDp123ActionLayout.from_dict(_layout_dict(scale=1.0, offset=0.1, velocity_scale=1e6)), _CONTROL_DT
    )
    shifted.reset(_MID)
    np.testing.assert_allclose(shifted.adapt(_MID - 0.1)[0], _MID, atol=1e-9)

    vector = Pi05ActionAdapter(
        AmDp123ActionLayout.from_dict(_layout_dict(scale=[2.0] * 18, offset=[-0.5] * 18, velocity_scale=1e6)),
        _CONTROL_DT,
    )
    vector.reset(_MID)
    np.testing.assert_allclose(vector.adapt((_MID + 0.5) / 2.0)[0], _MID, atol=1e-9)


def test_absolute_and_delta_modes():
    """Absolute targets follow the model; delta targets accumulate from the last target."""
    delta_value = 0.001
    absolute = Pi05ActionAdapter(AmDp123ActionLayout.from_dict(_layout_dict(velocity_scale=1e6)), _CONTROL_DT)
    absolute.reset(_MID)
    np.testing.assert_allclose(absolute.adapt(_MID + delta_value)[0], _MID + delta_value, atol=1e-12)

    delta = Pi05ActionAdapter(
        AmDp123ActionLayout.from_dict(_layout_dict(mode=AM_DP123_PI05_DELTA_MODE, velocity_scale=1e6)), _CONTROL_DT
    )
    delta.reset(_MID)
    np.testing.assert_allclose(delta.adapt(np.full(18, delta_value))[0], _MID + delta_value, atol=1e-12)
    np.testing.assert_allclose(delta.adapt(np.full(18, delta_value))[0], _MID + 2.0 * delta_value, atol=1e-12)


def test_position_limits_clamp_every_joint():
    """Unbounded model output is clamped to the joint contract limits."""
    adapter = Pi05ActionAdapter(AmDp123ActionLayout.from_dict(_layout_dict(velocity_scale=1e6)), _CONTROL_DT)
    adapter.reset(np.zeros(18))
    np.testing.assert_allclose(adapter.adapt(np.full(18, 100.0))[0], _UPPER, atol=1e-12)
    np.testing.assert_allclose(adapter.adapt(np.full(18, -100.0))[0], _LOWER, atol=1e-12)


def test_velocity_limit_uses_contract_velocity_and_control_period():
    """One step may not move a joint faster than ``max_velocity * control_dt * scale``."""
    adapter = Pi05ActionAdapter(AmDp123ActionLayout.from_dict(_layout_dict()), _CONTROL_DT)
    adapter.reset(np.zeros(18))
    step = np.array([spec.max_velocity_rad_s for spec in _SPECS_IN_ORDER]) * _CONTROL_DT * 0.5
    np.testing.assert_allclose(adapter.adapt(np.full(18, 100.0))[0], np.minimum(_UPPER, step), atol=1e-12)


def test_reset_rebases_at_the_measured_position():
    """Reset starts from the measured joints instead of zero and clamps bad input."""
    adapter = Pi05ActionAdapter(AmDp123ActionLayout.from_dict(_layout_dict()), _CONTROL_DT)
    adapter.reset(_HOME)
    np.testing.assert_allclose(adapter.current_target, _HOME, atol=1e-9)
    np.testing.assert_allclose(adapter.adapt(_HOME)[0], _HOME, atol=1e-9)

    adapter.reset(np.full(18, 100.0))
    np.testing.assert_allclose(adapter.current_target, _UPPER, atol=1e-12)


def test_layout_rejects_unknown_mode_and_schema():
    """Unknown modes and schema versions fail validation."""
    with pytest.raises(ValueError):
        AmDp123ActionLayout.from_dict(_layout_dict(mode="cartesian_pose"))
    with pytest.raises(ValueError):
        AmDp123ActionLayout.from_dict(_layout_dict(schema_version=2))


def test_controlled_state_indices_align_with_state_names():
    """The 18 controlled joints are a stable subset of the 23-D state order."""
    indices = controlled_state_indices()
    assert len(indices) == 18
    assert tuple(AM_DP123_STATE_JOINT_NAMES[index] for index in indices) == AM_DP123_CONTROLLED_JOINT_NAMES


def test_to_uint8_rgb_converts_float_rgba():
    """Float RGBA input becomes contiguous uint8 RGB, dropping alpha."""
    rgba = np.zeros((2, 2, 4), dtype=np.float32)
    rgba[..., 0] = 1.0
    rgba[..., 3] = 1.0
    converted = to_uint8_rgb(rgba)
    assert converted.dtype == np.uint8
    assert converted.shape == (2, 2, 3)
    assert converted.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(converted[..., 0], np.full((2, 2), 255, dtype=np.uint8))
    np.testing.assert_array_equal(converted[..., 1:], np.zeros((2, 2, 2), dtype=np.uint8))


def test_to_uint8_rgb_rejects_bad_shape_and_nonfinite():
    """Only finite three- or four-channel frames are accepted."""
    with pytest.raises(ValueError):
        to_uint8_rgb(np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        to_uint8_rgb(np.full((4, 4, 3), np.nan, dtype=np.float32))


def _images() -> dict[str, np.ndarray]:
    return {name: np.zeros((4, 5, 3), dtype=np.uint8) for name in AM_DP123_PI05_REQUIRED_CAMERAS}


def test_observation_payload_keys_and_dtypes():
    """The payload carries the four cameras, the 23-D state pair, and the prompt."""
    payload = build_pi05_observation(
        np.arange(23, dtype=np.float32),
        np.ones(23, dtype=np.float32),
        _images(),
        "pick the cube",
    )
    keys = AM_DP123_PI05_OBSERVATION_KEYS
    assert keys["state"] in payload and keys["image"] in payload
    assert keys["image_right"] in payload and keys["wrist_image_right"] in payload
    assert payload[keys["prompt"]] == "pick the cube"
    assert payload[keys["state"]].shape == (23,)
    assert payload[keys["state"]].dtype == np.float32
    assert payload[keys["joint_velocity"]].dtype == np.float32
    for key in ("image", "image_right", "wrist_image", "wrist_image_right"):
        assert payload[keys[key]].dtype == np.uint8
    assert payload[keys["image"]].shape == (4, 5, 3)


def test_observation_payload_uses_the_image_transform():
    """Remote callers can substitute the OpenPI resize transform."""
    calls: list[str] = []

    def transform(image: np.ndarray) -> np.ndarray:
        calls.append("called")
        return np.zeros((224, 224, 3), dtype=np.uint8)

    payload = build_pi05_observation(
        np.zeros(23, dtype=np.float32), np.zeros(23, dtype=np.float32), _images(), "prompt", transform
    )
    assert len(calls) == 4
    assert payload[AM_DP123_PI05_OBSERVATION_KEYS["image"]].shape == (224, 224, 3)


def test_observation_rejects_missing_camera_and_bad_state():
    """Missing cameras, wrong state widths, and NaN state are rejected."""
    images = _images()
    del images["left_wrist"]
    with pytest.raises(ValueError):
        build_pi05_observation(np.zeros(23), np.zeros(23), images, "prompt")
    with pytest.raises(ValueError):
        build_pi05_observation(np.zeros(22), np.zeros(23), _images(), "prompt")
    bad_state = np.zeros(23, dtype=np.float32)
    bad_state[0] = np.nan
    with pytest.raises(ValueError):
        build_pi05_observation(bad_state, np.zeros(23), _images(), "prompt")
