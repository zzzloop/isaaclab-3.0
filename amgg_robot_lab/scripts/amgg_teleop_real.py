# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Guarded AMGG Unitree G1 real-robot upper-body teleoperation endpoint."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
from pathlib import Path
import sys
import time

from amgg_robot_lab.real import (
    AMGG_G1_HARDWARE_COMMAND_NAMES,
    UnitreeG1BackendCfg,
    UnitreeG1DryRunBackend,
    UnitreeG1UpperBodyBackend,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("dry-run", "unitree-g1"),
        default="dry-run",
        help="Real-robot transport. Defaults to dry-run and never moves hardware.",
    )
    parser.add_argument("--network_interface", default=None, help="DDS NIC for Unitree SDK2, for example eth0.")
    parser.add_argument("--rate_hz", type=float, default=30.0, help="Command loop rate [Hz].")
    parser.add_argument(
        "--hold_seconds",
        type=float,
        default=3.0,
        help="Hold measured pose when no command JSONL is given.",
    )
    parser.add_argument(
        "--command_jsonl",
        default=None,
        help=(
            "Optional JSONL file containing 26-D commands. Use '-' for stdin. "
            "Each line may be a 26-element list or {'positions': [...]} / {'positions': {'joint': value}}."
        ),
    )
    parser.add_argument("--disable_arms", action="store_true", help="Do not publish arm SDK commands.")
    parser.add_argument("--disable_hands", action="store_true", help="Do not publish Inspire hand commands.")
    parser.add_argument("--enable_motion", action="store_true", help="Open the software command gate after checks.")
    parser.add_argument(
        "--physical_estop_ready",
        action="store_true",
        help="Confirm a hardware emergency stop is reachable.",
    )
    parser.add_argument(
        "--robot_supported",
        action="store_true",
        help="Confirm the G1 is safely supported/fixture-held.",
    )
    parser.add_argument("--operator_clear", action="store_true", help="Confirm people and loose objects are clear.")
    parser.add_argument(
        "--print_command_contract",
        action="store_true",
        help="Print the required 26-D command order and exit.",
    )
    return parser.parse_args()


def _make_backend(args: argparse.Namespace):
    if args.backend == "dry-run":
        return UnitreeG1DryRunBackend()
    cfg = UnitreeG1BackendCfg(
        network_interface=args.network_interface,
        enable_arms=not args.disable_arms,
        enable_hands=not args.disable_hands,
    )
    return UnitreeG1UpperBodyBackend(cfg)


def _require_motion_checks(args: argparse.Namespace) -> None:
    if not args.enable_motion:
        raise SystemExit("Refusing to move hardware without --enable_motion. Dry-read with this flag omitted first.")
    missing = []
    if not args.physical_estop_ready:
        missing.append("--physical_estop_ready")
    if not args.robot_supported:
        missing.append("--robot_supported")
    if not args.operator_clear:
        missing.append("--operator_clear")
    if missing:
        raise SystemExit(f"Refusing to move hardware. Missing physical safety confirmations: {', '.join(missing)}")


def _command_from_json(value) -> tuple[float, ...]:
    if isinstance(value, dict):
        value = value.get("positions", value.get("joint_positions", value))
    if isinstance(value, dict):
        missing = [name for name in AMGG_G1_HARDWARE_COMMAND_NAMES if name not in value]
        if missing:
            raise ValueError(f"Command dictionary is missing joints: {missing}")
        command = tuple(float(value[name]) for name in AMGG_G1_HARDWARE_COMMAND_NAMES)
    elif isinstance(value, list | tuple):
        command = tuple(float(item) for item in value)
    else:
        raise ValueError("Command JSON must be a 26-element list or a joint-name dictionary.")
    if len(command) != len(AMGG_G1_HARDWARE_COMMAND_NAMES):
        raise ValueError(f"Expected {len(AMGG_G1_HARDWARE_COMMAND_NAMES)} command values, received {len(command)}.")
    return command


def _iter_jsonl_commands(path: str) -> Iterable[tuple[float, ...]]:
    handle = sys.stdin if path == "-" else Path(path).open("r", encoding="utf-8")
    try:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                yield _command_from_json(json.loads(stripped))
            except Exception as error:
                raise ValueError(f"Invalid command JSONL line {line_number}: {error}") from error
    finally:
        if handle is not sys.stdin:
            handle.close()


def _current_command_from_state(state) -> tuple[float, ...]:
    body = state.joint_positions_rad[:29]
    hand = state.joint_positions_rad[29:]
    arm_indices = tuple(range(15, 29))
    return tuple(body[index] for index in arm_indices) + tuple(hand)


def _sleep_until_next_tick(start_s: float, period_s: float) -> None:
    elapsed_s = time.monotonic() - start_s
    if elapsed_s < period_s:
        time.sleep(period_s - elapsed_s)


def main() -> None:
    """Start the guarded AMGG real-robot command endpoint."""
    args = _parse_args()
    if args.print_command_contract:
        for index, name in enumerate(AMGG_G1_HARDWARE_COMMAND_NAMES):
            print(f"{index:02d}: {name}")
        return

    if args.rate_hz <= 0.0:
        raise SystemExit("--rate_hz must be positive.")
    if args.backend == "unitree-g1" and args.enable_motion:
        _require_motion_checks(args)

    backend = _make_backend(args)
    period_s = 1.0 / args.rate_hz
    backend.connect()
    try:
        state = backend.read_state()
        print(
            f"[AMGG] Connected backend={args.backend}; state_dim={len(state.joint_positions_rad)}; "
            f"command_dim={len(AMGG_G1_HARDWARE_COMMAND_NAMES)}",
            flush=True,
        )
        if not args.enable_motion:
            print("[AMGG] Motion disabled; exiting after successful state read.", flush=True)
            return

        backend.enable()
        if args.command_jsonl is None:
            command = _current_command_from_state(state)
            deadline_s = time.monotonic() + args.hold_seconds
            print(f"[AMGG] Holding measured upper-body pose for {args.hold_seconds:.2f} s.", flush=True)
            while time.monotonic() < deadline_s:
                tick_s = time.monotonic()
                backend.send_joint_position_targets(command, tick_s)
                _sleep_until_next_tick(tick_s, period_s)
        else:
            print(f"[AMGG] Streaming commands from {args.command_jsonl}.", flush=True)
            for command in _iter_jsonl_commands(args.command_jsonl):
                tick_s = time.monotonic()
                backend.send_joint_position_targets(command, tick_s)
                _sleep_until_next_tick(tick_s, period_s)
    except KeyboardInterrupt:
        print("\n[AMGG] Interrupted; closing command gate.", flush=True)
    finally:
        backend.stop()
        backend.disconnect()


if __name__ == "__main__":
    main()
