# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical AMGG link and tool-frame names."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AmggFrameContract:
    """Required robot frames shared by FK, IK, simulation, and hardware."""

    base_link: str = "base_link"
    torso_link: str = "Body0422_Link"
    left_wrist_link: str = "ArmL07Output_Link"
    right_wrist_link: str = "ArmR07Output_Link"
    left_tcp_link: str = "left_tcp_link"
    right_tcp_link: str = "right_tcp_link"

    def validate(self) -> None:
        """Validate that all required frame names are present and unique."""
        names = (
            self.base_link,
            self.torso_link,
            self.left_wrist_link,
            self.right_wrist_link,
            self.left_tcp_link,
            self.right_tcp_link,
        )
        if any(not name for name in names):
            raise ValueError("AMGG frame contract is incomplete.")
        if len(set(names)) != len(names):
            raise ValueError("AMGG frame names must be unique.")


AMGG_FRAMES = AmggFrameContract()
