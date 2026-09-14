# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical AM-DP123 link and tool-frame names."""

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AmDp123FrameContract:
    """Required robot frames shared by simulation, IK, and hardware backends.

    The TCP offsets are the wrist-to-hand-base translation expressed in the wrist
    link frame [m]. They are recorded for downstream tooling only: the Pink IK
    targets the wrist links, so no offset is applied when solving.
    """

    base_link: str = "base_link"
    torso_link: str = "waist_link3"
    head_link: str = "head_link2"
    head_camera_link: str = "head_camera_link"
    left_wrist_link: str = "left_arm_link7"
    right_wrist_link: str = "right_arm_link7"
    left_hand_base_link: str = "left_arm_hand_link"
    right_hand_base_link: str = "right_arm_hand_link"
    left_wrist_camera_link: str = "left_arm_camera_link"
    right_wrist_camera_link: str = "right_arm_camera_link"
    left_tcp_offset_m: tuple[float, float, float] = (-0.0055, -0.16988, -0.04737)
    right_tcp_offset_m: tuple[float, float, float] = (-0.01485, 0.1748, -0.01884)

    def validate(self) -> None:
        """Validate that all required frame names are present and unique."""
        names = (
            self.base_link,
            self.torso_link,
            self.head_link,
            self.head_camera_link,
            self.left_wrist_link,
            self.right_wrist_link,
            self.left_hand_base_link,
            self.right_hand_base_link,
            self.left_wrist_camera_link,
            self.right_wrist_camera_link,
        )
        if any(not name for name in names):
            raise ValueError("AM-DP123 frame contract is incomplete.")
        if len(set(names)) != len(names):
            raise ValueError("AM-DP123 frame names must be unique.")
        offsets = (self.left_tcp_offset_m, self.right_tcp_offset_m)
        if any(len(offset) != 3 or not all(math.isfinite(value) for value in offset) for offset in offsets):
            raise ValueError("AM-DP123 TCP offsets must contain three finite values.")


AM_DP123_FRAMES = AmDp123FrameContract()
