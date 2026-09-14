# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The extension must stay importable and auditable without Isaac Sim."""

from __future__ import annotations

import hashlib
import importlib
import sys

from amgg_robot_lab.assets import AM_DP123_ASSET_DATA_DIR, AM_DP123_MESH_DIR, AM_DP123_URDF_PATH

# SHA-256 of the upstream AM-DP123.urdf. Any edit to the shipped model breaks the
# audit trail and this hash, so the extension keeps the URDF byte-identical.
AM_DP123_URDF_SHA256 = "d97141a34c626418197a8f76b51d58541dd14f5f4afe23d79b518885fa67aa5c"


def test_urdf_is_byte_identical_to_upstream():
    """The packaged URDF must not be rewritten, renamed, or re-exported."""
    digest = hashlib.sha256(AM_DP123_URDF_PATH.read_bytes()).hexdigest()
    assert digest == AM_DP123_URDF_SHA256


def test_asset_tree_ships_the_full_mesh_set():
    """All 61 STL meshes travel with the package."""
    meshes = sorted(path.name for path in AM_DP123_MESH_DIR.glob("*.STL"))
    assert len(meshes) == 61
    assert "base_link.STL" in meshes
    assert "head_camera_link.STL" in meshes
    assert AM_DP123_URDF_PATH.parent.parent == AM_DP123_ASSET_DATA_DIR


def test_offline_modules_import_without_the_teleop_runtime():
    """Contracts, assets, kinematics, and the action ABI stay dependency-light."""
    for module in (
        "amgg_robot_lab.contracts",
        "amgg_robot_lab.assets",
        "amgg_robot_lab.kinematics",
        "amgg_robot_lab.teleop",
    ):
        importlib.import_module(module)
    # ``isaacteleop`` and ``pink`` are Linux-only runtime dependencies; the ABI
    # data above must remain importable without them.
    assert "isaacteleop" not in sys.modules
    assert "pink" not in sys.modules


def test_pipeline_builder_defers_its_runtime_imports():
    """The PICO pipeline builder must import ``isaacteleop`` lazily."""
    import inspect

    from amgg_robot_lab.teleop import am_dp123_pico_pipeline

    source = inspect.getsource(am_dp123_pico_pipeline.build_am_dp123_pico_pipeline)
    assert "from isaacteleop" in source
    module_source = inspect.getsource(am_dp123_pico_pipeline)
    assert module_source.index("def build_am_dp123_pico_pipeline") < module_source.index("from isaacteleop")


def test_task_registration_module_exposes_the_callback():
    """``amgg_robot_lab.tasks`` keeps the ``--external_callback`` entry point.

    Importing the task package needs Isaac Lab, so only the source contract is
    checked here.
    """
    from pathlib import Path

    package_dir = Path(__file__).resolve().parents[1] / "source" / "amgg_robot_lab" / "amgg_robot_lab"
    tasks_source = (package_dir / "tasks" / "__init__.py").read_text(encoding="utf-8")
    assert "def register_tasks() -> None:" in tasks_source
    assert '"Isaac-AM-DP123-Pico-XR-v0"' in tasks_source or "AM_DP123_PICO_XR_TASK_ID" in tasks_source
    env_source = (package_dir / "tasks" / "am_dp123_pico_xr_env_cfg.py").read_text(encoding="utf-8")
    assert "XrCameraFeedCfg" in env_source
    assert "disable_external_cameras" not in env_source
