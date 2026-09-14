# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared AM-DP123 kinematics data types."""

from dataclasses import dataclass


class AmDp123KinematicsError(RuntimeError):
    """Raised when the AM-DP123 kinematics model or a solve request is invalid."""


@dataclass(frozen=True, slots=True)
class AmDp123Pose:
    """Rigid pose represented in a named reference frame.

    Attributes:
        position_m: Cartesian position [m], ordered ``(x, y, z)``.
        quaternion_xyzw: Unit quaternion, ordered ``(x, y, z, w)``.
        reference_frame: Frame in which the pose is expressed.
    """

    position_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    reference_frame: str
