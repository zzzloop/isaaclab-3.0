# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay a run_sim Pico JSONL recording through exactly the same UDP input."""

import argparse
import json
import math
import socket
import time
import uuid
from pathlib import Path


def main():
    """Replay original sample timing, poses and button events."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=15051)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error("speed must be finite and positive")
    session, started, first, previous = uuid.uuid4().hex, time.monotonic(), None, -math.inf
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock, args.recording.open(encoding="utf-8") as stream:
        for seq, line in enumerate(stream):
            row = json.loads(line)
            timestamp = float(row["elapsed_s"])
            if not math.isfinite(timestamp) or timestamp < previous:
                raise ValueError("Recording timestamps must be finite and nondecreasing")
            previous = timestamp
            if first is None:
                first = timestamp
            time.sleep(max(0, (timestamp - first) / args.speed - (time.monotonic() - started)))
            packet = row["packet"]
            packet.update(session=session, seq=seq)
            sock.sendto(json.dumps(packet, allow_nan=False).encode(), (args.host, args.port))
    print("Replay complete; simulation watchdog will pause.")


if __name__ == "__main__":
    main()
