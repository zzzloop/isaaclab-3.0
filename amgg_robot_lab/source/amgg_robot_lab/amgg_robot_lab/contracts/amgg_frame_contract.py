# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical AMGG link and tool-frame names."""

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AmggFrameContract:
    """Required robot frames and TCP offsets shared by all AMGG backends.

    TCP offsets are expressed in their parent-link frames [m].
    """

    base_link: str = "base_link"
    torso_link: str = "Body0422_Link"
    left_wrist_link: str = "ArmL07Output_Link"
    right_wrist_link: str = "ArmR07Output_Link"
    left_tcp_parent_link: str = "left_gripper_base_link"
    right_tcp_parent_link: str = "right_gripper_base_link"
    left_tcp_link: str = "left_tcp_link"
    right_tcp_link: str = "right_tcp_link"
    left_tcp_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.12)
    right_tcp_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.12)

    def validate(self) -> None:
        """Validate that all required frame names are present and unique."""
        names = (
            self.base_link,
            self.torso_link,
            self.left_wrist_link,
            self.right_wrist_link,
            self.left_tcp_parent_link,
            self.right_tcp_parent_link,
            self.left_tcp_link,
            self.right_tcp_link,
        )
        if any(not name for name in names):
            raise ValueError("AMGG frame contract is incomplete.")
        if len(set(names)) != len(names):
            raise ValueError("AMGG frame names must be unique.")
        offsets = (self.left_tcp_offset_m, self.right_tcp_offset_m)
        if any(len(offset) != 3 or not all(math.isfinite(value) for value in offset) for offset in offsets):
            raise ValueError("AMGG TCP offsets must contain three finite values.")


AMGG_FRAMES = AmggFrameContract()
