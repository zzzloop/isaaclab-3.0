# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for AMGG CloudXR startup hygiene."""

import importlib.util
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from pathlib import Path

_AMGG_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_AMGG_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_AMGG_SCRIPTS_DIR))


def _load_cloudxr_module():
    module_path = _AMGG_SCRIPTS_DIR / "amgg_cloudxr.py"
    spec = importlib.util.spec_from_file_location("amgg_cloudxr", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


amgg_cloudxr = _load_cloudxr_module()


class TestAmggCloudxr(unittest.TestCase):
    """Validate cleanup of stale CloudXR IPC sockets."""

    def _make_ipc_path(self, install_dir: Path) -> Path:
        run_dir = install_dir / "run"
        run_dir.mkdir(parents=True)
        ipc_path = run_dir / "ipc_cloudxr"
        ipc_path.touch()
        return ipc_path

    def test_removes_stale_socket_for_xr_autolaunch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            install_dir = Path(tmp_dir) / ".cloudxr"
            ipc_path = self._make_ipc_path(install_dir)

            with patch.object(amgg_cloudxr.stat, "S_ISSOCK", return_value=True):
                removed = amgg_cloudxr.cleanup_stale_cloudxr_ipc(
                    ["amgg_record_demos.py", "--xr"], {}, install_dir=install_dir
                )

            self.assertTrue(removed)
            self.assertFalse(ipc_path.exists())

    def test_skips_cleanup_when_xr_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            install_dir = Path(tmp_dir) / ".cloudxr"
            ipc_path = self._make_ipc_path(install_dir)

            removed = amgg_cloudxr.cleanup_stale_cloudxr_ipc(["amgg_record_demos.py"], {}, install_dir=install_dir)

            self.assertFalse(removed)
            self.assertTrue(ipc_path.exists())

    def test_skips_cleanup_when_auto_launch_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            install_dir = Path(tmp_dir) / ".cloudxr"
            ipc_path = self._make_ipc_path(install_dir)

            removed = amgg_cloudxr.cleanup_stale_cloudxr_ipc(
                ["amgg_record_demos.py", "--xr", "--no-auto_launch_cloudxr"], {}, install_dir=install_dir
            )

            self.assertFalse(removed)
            self.assertTrue(ipc_path.exists())

    def test_terminates_stale_runtime_process_for_xr_autolaunch(self) -> None:
        fake_run_result = SimpleNamespace(stdout="123\n456\n")
        fake_kill = MagicMock()

        with (
            patch.object(amgg_cloudxr.os, "name", "posix"),
            patch.object(amgg_cloudxr.os, "getuid", return_value=1000, create=True),
            patch.object(amgg_cloudxr.os, "getpid", return_value=999),
            patch.object(amgg_cloudxr.os, "getppid", return_value=456),
            patch.object(amgg_cloudxr.subprocess, "run", return_value=fake_run_result),
            patch.object(amgg_cloudxr.os, "kill", fake_kill),
            patch.object(amgg_cloudxr.time, "monotonic", side_effect=[0.0, 4.0]),
        ):
            pids = amgg_cloudxr.cleanup_stale_cloudxr_runtime(["amgg_record_demos.py", "--xr"], {})

        self.assertEqual(pids, [123])
        fake_kill.assert_any_call(123, amgg_cloudxr.signal.SIGTERM)

    def test_skips_runtime_cleanup_when_disabled(self) -> None:
        with patch.object(amgg_cloudxr.subprocess, "run") as fake_run:
            pids = amgg_cloudxr.cleanup_stale_cloudxr_runtime(
                ["amgg_record_demos.py", "--xr"], {"AMGG_CLOUDXR_KILL_STALE": "0"}
            )

        self.assertEqual(pids, [])
        fake_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
