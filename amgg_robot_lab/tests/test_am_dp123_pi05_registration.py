# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Registration and dependency-boundary tests for the PI0.5 evaluation task."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import sys
import tomllib
from pathlib import Path

import pytest

from amgg_robot_lab import tasks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "source" / "amgg_robot_lab" / "amgg_robot_lab"
PI05_ENV_PATH = PACKAGE_ROOT / "tasks" / "am_dp123_pi05_env_cfg.py"
SHARED_ENV_PATH = PACKAGE_ROOT / "tasks" / "am_dp123_scene_cfg.py"
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "am_dp123_pi05_eval.py"


def _parsed(path: Path) -> ast.Module:
    """Parse a Python source file into an AST."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imported_modules(path: Path) -> set[str]:
    """Return direct module names imported by a Python file."""
    modules: set[str] = set()
    for node in ast.walk(_parsed(path)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def _load_eval_module():
    """Load the evaluation script without starting Isaac Sim."""
    spec = importlib.util.spec_from_file_location("am_dp123_pi05_eval", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pi05_task_id_is_public_and_stable():
    """The package exposes the registered task id as a public contract."""
    assert tasks.AM_DP123_PI05_EVAL_TASK_ID == "Isaac-AM-DP123-Pi05-Eval-v0"
    assert tasks.AM_DP123_PICO_XR_TASK_ID == "Isaac-AM-DP123-Pico-XR-v0"


def test_pi05_config_has_an_independent_dependency_boundary():
    """The PI0.5 config reaches shared scene code without importing the PICO stack."""
    pi05_imports = _imported_modules(PI05_ENV_PATH)
    shared_imports = _imported_modules(SHARED_ENV_PATH)
    assert "am_dp123_scene_cfg" in pi05_imports
    assert "am_dp123_pico_xr_env_cfg" not in pi05_imports
    forbidden = {"isaaclab_teleop", "amgg_robot_lab.teleop", "isaaclab.controllers.pink_ik"}
    assert pi05_imports.isdisjoint(forbidden)
    assert shared_imports.isdisjoint(forbidden)


def test_shared_scene_exposes_four_camera_observations():
    """The controller-neutral scene owns all four camera sensors and observations."""
    assigned_names = {
        target.id
        for node in ast.walk(_parsed(SHARED_ENV_PATH))
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in ([node.target] if isinstance(node, ast.AnnAssign) else node.targets)
        if isinstance(target, ast.Name)
    }
    for camera in ("head_left", "head_right", "left_wrist", "right_wrist"):
        assert camera in assigned_names
        assert f"image_{camera}" in assigned_names


@pytest.mark.skipif(importlib.util.find_spec("isaaclab") is None, reason="Isaac Lab is unavailable")
def test_pi05_config_imports_when_teleop_is_unavailable(monkeypatch: pytest.MonkeyPatch):
    """A runtime import of the PI0.5 config does not require IsaacTeleop."""
    monkeypatch.setitem(sys.modules, "isaaclab_teleop", None)
    sys.modules.pop("amgg_robot_lab.tasks.am_dp123_pi05_env_cfg", None)
    module = importlib.import_module("amgg_robot_lab.tasks.am_dp123_pi05_env_cfg")
    cfg = module.AmDp123Pi05EvalEnvCfg()
    assert cfg.compute_final_obs is True
    assert cfg.actions.joint_positions.preserve_order is True


def test_package_data_ships_the_action_layouts():
    """The layout JSON files are declared as package data."""
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = project["tool"]["setuptools"]["package-data"]["amgg_robot_lab"]
    assert "policy/layouts/*.json" in package_data


def test_shipped_layouts_decode_and_mark_confirmation():
    """The canonical 32-D layout is confirmed and carries the reserved padding."""
    layouts = PACKAGE_ROOT / "policy" / "layouts"
    template = json.loads((layouts / "am_dp123_pi05_32_template.json").read_text(encoding="utf-8"))
    assert template["confirmed"] is True and template["model_action_dim"] == 32
    assert template["policy_action_dim"] == 18
    assert template["source_indices"] == [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 7, 7, 15, 15, 16, 17]
    assert template["zero_padding_indices"] == list(range(18, 32))


def test_evaluation_script_help_lists_the_policy_contract():
    """The runnable entry point advertises every required policy and CLI option."""
    module = _load_eval_module()
    help_text = module._build_parser().format_help()
    for policy in ("mock_hold", "mock_sine", "remote"):
        assert policy in help_text
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
        assert flag in help_text


def test_terminal_recording_uses_final_observation():
    """A terminal transition records the pre-reset observation exposed by Isaac Lab."""
    module = _load_eval_module()
    reset_obs = {"robot_joint_pos": "reset"}
    final_obs = {"robot_joint_pos": "terminal"}
    assert module._record_policy_observation(reset_obs, {"final_obs": {"policy": final_obs}}, True) is final_obs
    assert module._record_policy_observation(reset_obs, {}, False) is reset_obs
    with pytest.raises(RuntimeError, match="final_obs"):
        module._record_policy_observation(reset_obs, {}, True)
