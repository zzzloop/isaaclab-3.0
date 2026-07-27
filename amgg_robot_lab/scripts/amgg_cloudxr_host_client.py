# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch CloudXR with the bundled Isaac Teleop Web Client.

The upstream ``python -m isaacteleop.cloudxr --host-client`` launcher
downloads the WebXR static client into ``~/.cloudxr/static-client`` on
first use.  Some AMGG lab machines cannot reach GitHub Pages reliably, so
this wrapper pre-stages the matching client files from the repository before
delegating to the upstream launcher.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


_CLIENT_VERSION = "release-1.3.x"
_REQUIRED_CLIENT_FILES = ("index.html", "bundle.js")
_OPTIONAL_CLIENT_FILES = ("bundle.emulator.js", "favicon.ico")


def _repo_client_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "third_party" / "isaacteleop_client" / _CLIENT_VERSION


def _target_client_dir() -> Path:
    configured = os.environ.get("TELEOP_WEB_CLIENT_STATIC_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cloudxr" / "static-client"


def _copy_if_missing_or_empty(source: Path, target: Path) -> bool:
    if target.is_file() and target.stat().st_size > 0:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True


def prepare_static_client() -> Path:
    """Copy the bundled CloudXR.js client into the Isaac Teleop cache.

    Returns:
        Path to the static client directory used by Isaac Teleop.

    Raises:
        FileNotFoundError: If the bundled client is incomplete.
    """
    source_dir = _repo_client_dir()
    missing = [name for name in _REQUIRED_CLIENT_FILES if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Bundled Isaac Teleop client is incomplete: missing {missing} in {source_dir}")

    target_dir = _target_client_dir()
    copied = []
    for name in (*_REQUIRED_CLIENT_FILES, *_OPTIONAL_CLIENT_FILES):
        source = source_dir / name
        if source.is_file() and _copy_if_missing_or_empty(source, target_dir / name):
            copied.append(name)

    os.environ["TELEOP_WEB_CLIENT_STATIC_DIR"] = str(target_dir)
    if copied:
        print(f"[AMGG] Installed CloudXR Web Client files to {target_dir}: {', '.join(copied)}", flush=True)
    else:
        print(f"[AMGG] CloudXR Web Client cache already present: {target_dir}", flush=True)
    return target_dir


def _ensure_default_launcher_args() -> None:
    if "--accept-eula" not in sys.argv:
        sys.argv.append("--accept-eula")
    if "--host-client" not in sys.argv:
        sys.argv.append("--host-client")


def main() -> None:
    prepare_static_client()
    _ensure_default_launcher_args()

    from isaacteleop.cloudxr.__main__ import main as cloudxr_main

    cloudxr_main()


if __name__ == "__main__":
    main()
