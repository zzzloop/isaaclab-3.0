# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AM-DP123 PI0.5 32-D observation and action contract tests."""

from __future__ import annotations

import numpy as np
import pytest

from amgg_robot_lab.contracts import AM_DP123_JOINT_SPECS, AM_DP123_STATE_JOINT_NAMES
from amgg_robot_lab.policy import (
    AM_DP123_PI05_32_TEMPLATE_PATH,
    AM_DP123_PI05_ABSOLUTE_MODE,
    AM_DP123_PI05_CONTROLLED_JOINT_NAMES,
    AM_DP123_PI05_MODEL_DIM,
    AM_DP123_PI05_MODEL_JOINT_NAMES,
    AM_DP123_PI05_OBSERVATION_KEYS,
    AM_DP123_PI05_POLICY_DIM,
    AM_DP123_PI05_REQUIRED_CAMERAS,
    AM_DP123_PI05_SOURCE_INDICES,
    AM_DP123_PI05_ZERO_PADDING_INDICES,
    AmDp123ActionLayout,
    Pi05ActionAdapter,
    build_pi05_observation,
    controlled_state_indices,
    from_bpx_server_actions,
    load_default_action_layout,
    to_pi05_model_state,
    to_pi05_policy_state,
    to_uint8_rgb,
    validate_pi05_episode,
)

_CONTROL_DT = 1.0 / 30.0
_SPECS = {spec.name: spec for spec in AM_DP123_JOINT_SPECS}
_SPECS_IN_ORDER = [_SPECS[name] for name in AM_DP123_PI05_CONTROLLED_JOINT_NAMES]
_LOWER = np.array([spec.lower_limit_rad for spec in _SPECS_IN_ORDER])
_UPPER = np.array([spec.upper_limit_rad for spec in _SPECS_IN_ORDER])
_HOME = np.array([spec.home_position_rad for spec in _SPECS_IN_ORDER])
_SCALE = [1.0] * 14 + [1.0, -1.0, 1.0, -1.0, 1.0, 1.0]


def _layout_dict(**overrides: object) -> dict[str, object]:
    layout: dict[str, object] = {
        "schema_version": 1,
        "confirmed": True,
        "model_action_dim": AM_DP123_PI05_MODEL_DIM,
        "policy_action_dim": AM_DP123_PI05_POLICY_DIM,
        "mode": AM_DP123_PI05_ABSOLUTE_MODE,
        "sim_joint_names": list(AM_DP123_PI05_CONTROLLED_JOINT_NAMES),
        "source_indices": list(AM_DP123_PI05_SOURCE_INDICES),
        "zero_padding_indices": list(AM_DP123_PI05_ZERO_PADDING_INDICES),
        "scale": _SCALE,
        "offset": 0.0,
        "velocity_scale": 0.5,
    }
    layout.update(overrides)
    return layout


def _model_action_from_target(target: np.ndarray) -> np.ndarray:
    action = np.zeros(AM_DP123_PI05_POLICY_DIM, dtype=np.float64)
    action[0:7] = target[0:7]
    action[8:15] = target[7:14]
    action[7] = target[14]
    action[15] = target[16]
    action[16:18] = target[18:20]
    return action


def _images() -> dict[str, np.ndarray]:
    return {name: np.zeros((4, 5, 3), dtype=np.uint8) for name in AM_DP123_PI05_REQUIRED_CAMERAS}


def test_default_layout_is_confirmed_32d_contract():
    layout = load_default_action_layout()
    assert layout.path == AM_DP123_PI05_32_TEMPLATE_PATH
    assert layout.confirmed is True
    assert layout.model_action_dim == 32
    assert layout.policy_action_dim == 18
    assert layout.sim_joint_names == AM_DP123_PI05_CONTROLLED_JOINT_NAMES
    assert layout.source_indices == AM_DP123_PI05_SOURCE_INDICES
    assert layout.zero_padding_indices == AM_DP123_PI05_ZERO_PADDING_INDICES
    assert layout.sha256() is not None


def test_layout_rejects_wrong_width_mapping_and_padding():
    with pytest.raises(ValueError, match="must be 32"):
        AmDp123ActionLayout.from_dict(_layout_dict(model_action_dim=18))
    with pytest.raises(ValueError, match="must be 18"):
        AmDp123ActionLayout.from_dict(_layout_dict(policy_action_dim=32))
    with pytest.raises(ValueError, match="canonical AM-DP123 PI0.5 mapping"):
        AmDp123ActionLayout.from_dict(_layout_dict(source_indices=[0] * 20))
    with pytest.raises(ValueError, match="18 through 31"):
        AmDp123ActionLayout.from_dict(_layout_dict(zero_padding_indices=list(range(18, 31))))


def test_layout_joint_names_must_equal_internal_urdf_order():
    with pytest.raises(ValueError):
        AmDp123ActionLayout.from_dict(
            _layout_dict(sim_joint_names=list(reversed(AM_DP123_PI05_CONTROLLED_JOINT_NAMES)))
        )


def test_adapter_expands_each_gripper_scalar_to_mimic_pair():
    adapter = Pi05ActionAdapter(AmDp123ActionLayout.from_dict(_layout_dict(velocity_scale=1.0e6)), _CONTROL_DT)
    adapter.reset(_HOME)
    action = _model_action_from_target(_HOME)
    action[7] = -0.20
    action[15] = -0.10
    target = adapter.adapt(action)[0]
    np.testing.assert_allclose(target[14:18], [-0.20, 0.20, -0.10, 0.10], atol=1.0e-9)
    np.testing.assert_allclose(target[18:20], action[16:18], atol=1.0e-9)


def test_adapter_rejects_internal_model_width_from_websocket():
    adapter = Pi05ActionAdapter(load_default_action_layout(), _CONTROL_DT)
    adapter.reset(_HOME)
    with pytest.raises(ValueError, match="Policy action width 32"):
        adapter.adapt(np.zeros(32))


def test_adapter_rejects_invalid_actions_and_clamps_limits():
    adapter = Pi05ActionAdapter(load_default_action_layout(), _CONTROL_DT)
    adapter.reset(_HOME)
    with pytest.raises(ValueError):
        adapter.adapt(np.zeros(17))
    with pytest.raises(ValueError):
        adapter.adapt(np.zeros((0, 18)))
    with pytest.raises(ValueError):
        adapter.adapt(np.full(18, np.nan))

    unbounded = np.full(18, 100.0)
    fast = Pi05ActionAdapter(AmDp123ActionLayout.from_dict(_layout_dict(velocity_scale=1.0e6)), _CONTROL_DT)
    fast.reset(_HOME)
    result = fast.adapt(unbounded)[0]
    assert np.all(result >= _LOWER)
    assert np.all(result <= _UPPER)


def test_controlled_state_indices_cover_arms_hands_and_head_only():
    indices = controlled_state_indices()
    assert len(indices) == 20
    assert tuple(AM_DP123_STATE_JOINT_NAMES[index] for index in indices) == AM_DP123_PI05_CONTROLLED_JOINT_NAMES
    assert all(not AM_DP123_STATE_JOINT_NAMES[index].startswith("waist_") for index in indices)


def test_raw_state_converts_to_exact_18_plus_14_pi05_layout():
    raw = np.arange(23, dtype=np.float32)
    policy_state = to_pi05_policy_state(raw)
    state = to_pi05_model_state(raw)
    raw_index = {name: index for index, name in enumerate(AM_DP123_STATE_JOINT_NAMES)}
    expected_names = (
        *AM_DP123_PI05_MODEL_JOINT_NAMES[0:7],
        "left_arm_hand_joint1_0",
        *AM_DP123_PI05_MODEL_JOINT_NAMES[8:15],
        "right_arm_hand_joint1_0",
        "head_joint1",
        "head_joint2",
    )
    expected = np.array([raw[raw_index[name]] for name in expected_names], dtype=np.float32)
    assert policy_state.shape == (18,)
    np.testing.assert_array_equal(policy_state, expected)
    assert state.shape == (32,)
    np.testing.assert_array_equal(state[:18], expected)
    np.testing.assert_array_equal(state[18:], np.zeros(14, dtype=np.float32))


def test_bpx_raw_server_actions_collapse_to_effective_order():
    raw = np.arange(23, dtype=np.float32)
    effective = from_bpx_server_actions(raw)
    expected = np.concatenate((raw[3:10], [raw[10] + raw[11]], raw[12:19], [raw[19] + raw[20]], raw[21:23]))
    assert effective.shape == (18,)
    np.testing.assert_array_equal(effective, expected)

    chunk = from_bpx_server_actions(np.stack((raw, raw + 100)))
    assert chunk.shape == (2, 18)
    np.testing.assert_array_equal(chunk[0], expected)


def test_bpx_server_action_conversion_validates_width_and_values():
    effective = np.arange(18, dtype=np.float32)
    assert from_bpx_server_actions(effective) is effective
    with pytest.raises(ValueError, match="width 23"):
        from_bpx_server_actions(np.zeros(22))
    with pytest.raises(ValueError, match="non-finite"):
        from_bpx_server_actions(np.full(23, np.nan))


def test_observation_payload_matches_18d_norm_stats_contract():
    images = _images()
    images["head_right"] = np.zeros((4, 5, 3), dtype=np.uint8)
    images["head_left"].fill(11)
    images["head_right"].fill(22)
    images["left_wrist"].fill(33)
    images["right_wrist"].fill(44)
    payload = build_pi05_observation(
        np.arange(23, dtype=np.float32),
        np.ones(23, dtype=np.float32),
        images,
        "pick the cube",
    )
    keys = AM_DP123_PI05_OBSERVATION_KEYS
    assert payload[keys["state"]].shape == (18,)
    assert payload[keys["state"]].dtype == np.float32
    assert set(payload) == {"qpos", "images", "prompt"}
    assert set(payload["images"]) == {"left_eye", "left_wrist", "right_wrist"}
    assert payload["images"]["left_eye"].shape == (4, 5, 3)
    assert int(payload["images"]["left_eye"][0, 0, 0]) == 11
    assert int(payload["images"]["left_wrist"][0, 0, 0]) == 33
    assert int(payload["images"]["right_wrist"][0, 0, 0]) == 44
    assert AM_DP123_PI05_REQUIRED_CAMERAS == ("head_left", "left_wrist", "right_wrist")


def test_observation_rejects_missing_camera_and_bad_raw_state():
    images = _images()
    del images["left_wrist"]
    with pytest.raises(ValueError):
        build_pi05_observation(np.zeros(23), np.zeros(23), images, "prompt")
    with pytest.raises(ValueError):
        build_pi05_observation(np.zeros(22), np.zeros(23), _images(), "prompt")
    bad_state = np.zeros(23)
    bad_state[0] = np.nan
    with pytest.raises(ValueError):
        build_pi05_observation(bad_state, np.zeros(23), _images(), "prompt")


def test_to_uint8_rgb_converts_float_rgba():
    rgba = np.zeros((2, 2, 4), dtype=np.float32)
    rgba[..., 0] = 1.0
    rgba[..., 3] = 1.0
    converted = to_uint8_rgb(rgba)
    assert converted.dtype == np.uint8
    assert converted.shape == (2, 2, 3)
    np.testing.assert_array_equal(converted[..., 0], np.full((2, 2), 255, dtype=np.uint8))


def _episode_arrays(steps: int = 3) -> dict[str, np.ndarray]:
    return {
        "joint_pos": np.zeros((steps, 23), dtype=np.float32),
        "joint_vel": np.zeros((steps, 23), dtype=np.float32),
        "object_position": np.zeros((steps, 3), dtype=np.float32),
        "model_action": np.zeros((steps, 18), dtype=np.float32),
        "applied_joint_target": np.repeat(_HOME[None, :], steps, axis=0).astype(np.float32),
        "terminated": np.zeros(steps, dtype=bool),
        "truncated": np.zeros(steps, dtype=bool),
        "inference_valid": np.ones(steps, dtype=bool),
        "inference_error": np.full(steps, "", dtype=np.str_),
    }


def test_episode_validation_accepts_18d_policy_and_20d_execution():
    arrays = _episode_arrays()
    arrays["inference_valid"][1] = False
    arrays["inference_error"][1] = "TimeoutError: server unavailable"
    summary = validate_pi05_episode(arrays)
    assert summary.model_action_dim == 18
    assert summary.inference_failure_steps == 1


def test_episode_validation_rejects_bad_execution_width_and_limits():
    arrays = _episode_arrays()
    arrays["applied_joint_target"] = np.zeros((3, 18), dtype=np.float32)
    with pytest.raises(ValueError, match="applied_joint_target"):
        validate_pi05_episode(arrays)

    arrays = _episode_arrays()
    arrays["applied_joint_target"][0, 0] = 100.0
    with pytest.raises(ValueError, match="joint position limits"):
        validate_pi05_episode(arrays)
