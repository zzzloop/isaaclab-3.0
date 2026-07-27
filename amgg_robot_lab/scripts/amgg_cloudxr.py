# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small AMGG helpers for CloudXR/OpenXR startup hygiene."""

from __future__ import annotations

import os
import signal
import stat
import subprocess
import sys
import time
from collections.abc import MutableMapping, Sequence
from pathlib import Path

_STALE_CLOUDXR_PROCESS_PATTERN = (
    "cloudxr-service|cloudxr-runtime|isaacteleop.cloudxr|CloudXRLauncher|CloudXR Runtime"
)


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


def _should_cleanup_cloudxr(arguments: Sequence[str], environment: MutableMapping[str, str]) -> bool:
    if environment.get("AMGG_CLOUDXR_CLEANUP", "1").strip() == "0":
        return False
    if not _has_argument(arguments[1:], "--xr"):
        return False
    if _has_argument(arguments[1:], "--no-auto_launch_cloudxr"):
        return False
    cloudxr_env = _argument_value(arguments[1:], "--cloudxr_env")
    return cloudxr_env is None or cloudxr_env.strip().lower() not in {"", "none", "off", "false", "0"}


def cleanup_stale_cloudxr_runtime(
    arguments: Sequence[str] | None = None,
    environment: MutableMapping[str, str] | None = None,
) -> list[int]:
    """Terminate stale user-owned CloudXR runtime processes before launch.

    Args:
        arguments: Process argument vector. Defaults to :attr:`sys.argv`.
        environment: Process environment. Defaults to :attr:`os.environ`.

    Returns:
        PIDs that were signaled.
    """
    arguments = sys.argv if arguments is None else arguments
    environment = os.environ if environment is None else environment

    if not _should_cleanup_cloudxr(arguments, environment):
        return []
    if environment.get("AMGG_CLOUDXR_KILL_STALE", "1").strip() == "0":
        return []
    if os.name != "posix" or not hasattr(os, "getuid"):
        return []

    try:
        result = subprocess.run(
            ["pgrep", "-u", str(os.getuid()), "-f", _STALE_CLOUDXR_PROCESS_PATTERN],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as error:
        print(f"[AMGG] Warning: failed to inspect stale CloudXR processes: {error}", flush=True)
        return []

    own_pid = os.getpid()
    parent_pid = os.getppid()
    pids = []
    for line in result.stdout.splitlines():
        try:
            pid = int(line.strip())
        except ValueError:
            continue
        if pid not in {own_pid, parent_pid}:
            pids.append(pid)

    if not pids:
        return []

    signaled_pids = []
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            signaled_pids.append(pid)
        except ProcessLookupError:
            continue
        except OSError as error:
            print(f"[AMGG] Warning: failed to terminate stale CloudXR process {pid}: {error}", flush=True)

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and signaled_pids:
        alive_pids = []
        for pid in signaled_pids:
            try:
                os.kill(pid, 0)
                alive_pids.append(pid)
            except ProcessLookupError:
                continue
            except OSError:
                continue
        if not alive_pids:
            break
        signaled_pids = alive_pids
        time.sleep(0.1)

    for pid in signaled_pids:
        try:
            os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
        except ProcessLookupError:
            continue
        except OSError as error:
            print(f"[AMGG] Warning: failed to kill stale CloudXR process {pid}: {error}", flush=True)

    print(f"[AMGG] Terminated stale CloudXR runtime process(es): {pids}", flush=True)
    return pids


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

    if not _should_cleanup_cloudxr(arguments, environment):
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
