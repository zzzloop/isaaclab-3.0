# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The PI0.5 evaluation task stays registered and decoupled from the PICO teleop task.

The task configuration imports Isaac Lab, so these checks inspect the source contract
instead of importing it; they run on any machine that has the checkout.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "source" / "amgg_robot_lab" / "amgg_robot_lab"
TASKS_SOURCE = (PACKAGE_ROOT / "tasks" / "__init__.py").read_text(encoding="utf-8")
ENV_SOURCE = (PACKAGE_ROOT / "tasks" / "am_dp123_pi05_env_cfg.py").read_text(encoding="utf-8")
SHARED_ENV_SOURCE = (PACKAGE_ROOT / "tasks" / "am_dp123_pico_xr_env_cfg.py").read_text(encoding="utf-8")
SCRIPT_SOURCE = (PROJECT_ROOT / "scripts" / "am_dp123_pi05_eval.py").read_text(encoding="utf-8")


def test_pi05_task_is_registered():
    """Gym registration exposes the PI0.5 task next to the PICO task."""
    assert "AM_DP123_PI05_EVAL_TASK_ID" in TASKS_SOURCE
    assert '"Isaac-AM-DP123-Pi05-Eval-v0"' in TASKS_SOURCE
    assert "AM_DP123_PI05_EVAL_ENV_CFG_MODULE" in TASKS_SOURCE
    assert "AM_DP123_PI05_EVAL_ENV_CFG" in TASKS_SOURCE
    assert "AM_DP123_PICO_XR_TASK_ID" in TASKS_SOURCE


def test_env_cfg_uses_direct_joint_position_action():
    """The policy task commands joints directly with the official action term."""
    assert "JointPositionActionCfg" in ENV_SOURCE
    assert "joint_names=list(AM_DP123_CONTROLLED_JOINT_NAMES)" in ENV_SOURCE
    assert "preserve_order=True" in ENV_SOURCE
    assert "use_default_offset=False" in ENV_SOURCE
    assert "clip=AM_DP123_PI05_CLIP" in ENV_SOURCE
    assert "AM_DP123_PI05_DECIMATION = 4" in ENV_SOURCE
    assert "AM_DP123_PI05_SIM_DT = 1.0 / 120.0" in ENV_SOURCE


def test_env_cfg_has_no_pink_ik_or_teleop_pipeline():
    """The policy task must not build the PICO teleoperation pipeline."""
    for token in (
        "PinkIKControllerCfg",
        "AmDp123PinkInverseKinematicsActionCfg",
        "build_am_dp123_pico_pipeline",
        "IsaacTeleopCfg",
        "XrCameraFeedCfg",
        "isaacteleop",
        "isaac_teleop",
    ):
        assert token not in ENV_SOURCE, token


def test_env_cfg_reuses_the_camera_scene_and_observations():
    """The four-camera scene and observation group are shared, not duplicated."""
    for token in ("AmDp123SceneCfg", "ObservationsCfg", "EventCfg", "TerminationsCfg"):
        assert token in ENV_SOURCE
    for camera in ("head_left", "head_right", "left_wrist", "right_wrist"):
        assert f"image_{camera}" in SHARED_ENV_SOURCE
        assert f'_camera("{camera}"' in SHARED_ENV_SOURCE


def test_package_data_ships_the_action_layouts():
    """The layout JSON files are declared as package data."""
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = project["tool"]["setuptools"]["package-data"]["amgg_robot_lab"]
    assert "policy/layouts/*.json" in package_data


def test_shipped_layouts_decode_and_mark_confirmation():
    """The identity layout is confirmed; the 32-D template is explicitly unconfirmed."""
    layouts = PACKAGE_ROOT / "policy" / "layouts"
    default = json.loads((layouts / "am_dp123_joint_position_18.json").read_text(encoding="utf-8"))
    template = json.loads((layouts / "am_dp123_pi05_32_template.json").read_text(encoding="utf-8"))
    assert default["confirmed"] is True and default["model_action_dim"] == 18
    assert template["confirmed"] is False and template["model_action_dim"] == 32
    assert template["source_indices"] is None


def test_script_defers_openpi_and_mock_policies():
    """The run script parses the CLI first and imports OpenPI only for remote mode."""
    module_level = [
        line for line in SCRIPT_SOURCE.splitlines() if line.startswith(("import openpi_client", "from openpi_client"))
    ]
    assert module_level == []
    assert "from openpi_client import image_tools" in SCRIPT_SOURCE
    assert "from openpi_client import websocket_client_policy" in SCRIPT_SOURCE
    assert "AppLauncher(args_cli, enable_cameras=True)" in SCRIPT_SOURCE
    for policy in ("mock_hold", "mock_sine", "remote"):
        assert f'"{policy}"' in SCRIPT_SOURCE
    for flag in (
        "--policy",
        "--host",
        "--port",
        "--prompt",
        "--action_layout",
        "--action_horizon",
        "--max_steps",
        "--record_dir",
        "--record_images",
        "--no_record",
    ):
        assert flag in SCRIPT_SOURCE
    assert "AM_DP123_PI05_CONTROL_DT" in SCRIPT_SOURCE
    assert "_make_remote_image_transform" in SCRIPT_SOURCE
    assert "MAX_CONSECUTIVE_INFERENCE_FAILURES" in SCRIPT_SOURCE
