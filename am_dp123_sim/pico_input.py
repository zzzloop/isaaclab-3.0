# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validated Pico packets and clutch calibration, independent of ROS and Isaac Sim."""

from __future__ import annotations

import json
import socket
import time

import numpy as np
from model import ARMS, TIPS, RobotModel, pose_matrix, rotation

# Same LH Pico (right/up/forward) -> RH robot (forward/left/up) basis as
# ROS_v2/new_teleop_pico_ws/.../core/preprocessor.py. It is a reflection,
# so orientations MUST use the similarity transform P R P^-1.
PICO_TO_ROBOT = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def validate_packet(packet: dict) -> dict:
    """Validate a versioned raw frame (poses [m], quaternion xyzw, triggers [0, 1])."""
    if not isinstance(packet, dict) or packet.get("version") != 1:
        raise ValueError("Unsupported Pico packet version")
    if type(packet.get("seq")) is not int or packet["seq"] < 0:
        raise ValueError("Invalid sequence")
    if not isinstance(packet.get("session"), str) or not 1 <= len(packet["session"]) <= 64:
        raise ValueError("Invalid session")
    for name in ("head", "left", "right"):
        pose_matrix(packet[name])
    for side in ARMS:
        joy = packet[f"{side}_joy"]
        trigger = joy["trigger"]
        if not isinstance(trigger, (float, int)) or not np.isfinite(trigger) or not 0 <= trigger <= 1:
            raise ValueError("Invalid trigger")
        if any(type(joy.get(key)) is not bool for key in ("a", "b")):
            raise ValueError("Invalid buttons")
    return packet


class PicoReceiver:
    """Receive newest raw frames without blocking the simulation loop."""

    def __init__(self, host: str, port: int):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((host, port))
        self.socket.setblocking(False)
        self.received_at = -np.inf
        self.session = None
        self.seq = -1
        self.invalid = 0

    def poll(self) -> dict | None:
        """Return the latest new valid packet; reject malformed and reordered packets."""
        newest = None
        for _ in range(128):
            try:
                data, _ = self.socket.recvfrom(16384)
            except BlockingIOError:
                break
            try:
                packet = validate_packet(json.loads(data))
            except (ValueError, TypeError, KeyError, UnicodeError, IndexError):
                self.invalid += 1
                continue
            if self.session != packet["session"]:
                # A sender restart is accepted only after the prior stream has timed out.
                if time.monotonic() - self.received_at < 0.5:
                    continue
                self.session, self.seq = packet["session"], -1
            if packet["seq"] <= self.seq:
                continue
            self.seq = packet["seq"]
            self.received_at = time.monotonic()
            newest = packet
        return newest

    def close(self) -> None:
        """Release the UDP socket."""
        self.socket.close()


class PicoRetargeter:
    """Clutched, head-relative translation with calibrated absolute wrist rotation."""

    def __init__(self, model: RobotModel, scale: float = 1.0):
        self.model, self.scale = model, scale
        self.enabled = False
        self.last_a = False
        self.reference = {}
        self.anchors = {}
        self.yaw = np.eye(3)

    def pause(self) -> None:
        """Freeze targets until a fresh release-and-press of both A buttons."""
        self.enabled = False
        self.last_a = True

    def _poses(self, packet: dict) -> dict[str, np.ndarray]:
        poses = {}
        for name in ("head", "left", "right"):
            pose = pose_matrix(packet[name])
            pose[:3, :3] = PICO_TO_ROBOT @ pose[:3, :3] @ PICO_TO_ROBOT.T
            pose[:3, 3] = PICO_TO_ROBOT @ pose[:3, 3]
            poses[name] = pose
        for side in ARMS:
            poses[side][:3, 3] -= poses["head"][:3, 3]
        return poses

    def update(self, packet: dict, actual: dict[str, float]) -> dict[str, np.ndarray] | None:
        """Return calibrated base-frame arm targets ([m], rotation), or freeze."""
        poses = self._poses(packet)
        a = all(packet[f"{side}_joy"]["a"] for side in ARMS)
        b = all(packet[f"{side}_joy"]["b"] for side in ARMS)
        if b:
            self.enabled = False
        elif a and not self.last_a and not self.enabled:
            head = poses["head"][:3, :3]
            self.yaw = rotation([0, 0, 1], -np.arctan2(head[1, 0], head[0, 0]))
            self.reference = {side: poses[side].copy() for side in ARMS}
            self.anchors = {side: self.model.forward(TIPS[side], actual)[0] for side in ARMS}
            self.enabled = True
        self.last_a = a
        if not self.enabled:
            return None
        result = {}
        for side in ARMS:
            current, ref, anchor = poses[side], self.reference[side], self.anchors[side]
            target = anchor.copy()
            target[:3, 3] += self.scale * self.yaw @ (current[:3, 3] - ref[:3, 3])
            target[:3, :3] = self.yaw @ current[:3, :3] @ ref[:3, :3].T @ self.yaw.T @ anchor[:3, :3]
            result[side] = target
        return result
