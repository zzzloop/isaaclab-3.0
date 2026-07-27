# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small AMGG helpers for CloudXR/OpenXR startup hygiene."""

from __future__ import annotations

import os
import stat
import sys
from collections.abc import MutableMapping, Sequence
from pathlib import Path


def _argument_value(arguments: Sequence[str], name: str) -> str | None:
    for index, argument in enumerate(arguments):
        if argument == name:
            if index + 1 >= len(arguments):
                return None
            return arguments[index + 1]
        if argument.startswith(f"{name}="):
            return argument.split("=", maxsplit=1)[1]
    return None


def _has_argument(arguments: Sequence[str], name: str) -> bool:
    return any(argument == name or argument.startswith(f"{name}=") for argument in arguments)


def cleanup_stale_cloudxr_ipc(
    arguments: Sequence[str] | None = None,
    environment: MutableMapping[str, str] | None = None,
    install_dir: Path | None = None,
) -> bool:
    """Remove a stale CloudXR IPC socket before auto-launching a new runtime.

    Args:
        arguments: Process argument vector. Defaults to :attr:`sys.argv`.
        environment: Process environment. Defaults to :attr:`os.environ`.
        install_dir: CloudXR installation/cache directory. Defaults to
            ``~/.cloudxr``.

    Returns:
        Whether a stale IPC socket was removed.
    """
    arguments = sys.argv if arguments is None else arguments
    environment = os.environ if environment is None else environment

    if environment.get("AMGG_CLOUDXR_CLEANUP", "1").strip() == "0":
        return False
    if not _has_argument(arguments[1:], "--xr"):
        return False
    if _has_argument(arguments[1:], "--no-auto_launch_cloudxr"):
        return False
    cloudxr_env = _argument_value(arguments[1:], "--cloudxr_env")
    if cloudxr_env is not None and cloudxr_env.strip().lower() in {"", "none", "off", "false", "0"}:
        return False

    ipc_socket = (Path.home() / ".cloudxr" if install_dir is None else install_dir) / "run" / "ipc_cloudxr"
    try:
        mode = ipc_socket.lstat().st_mode
    except FileNotFoundError:
        return False
    except OSError as error:
        print(f"[AMGG] Warning: failed to inspect CloudXR IPC socket {ipc_socket}: {error}", flush=True)
        return False

    if not (stat.S_ISSOCK(mode) or ipc_socket.is_symlink()):
        print(f"[AMGG] Warning: CloudXR IPC path exists but is not a socket; leaving it in place: {ipc_socket}", flush=True)
        return False

    try:
        ipc_socket.unlink()
    except FileNotFoundError:
        return False
    except OSError as error:
        print(f"[AMGG] Warning: failed to remove stale CloudXR IPC socket {ipc_socket}: {error}", flush=True)
        return False

    print(f"[AMGG] Removed stale CloudXR IPC socket before launch: {ipc_socket}", flush=True)
    return True
