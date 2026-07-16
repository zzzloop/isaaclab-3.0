# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared AMGG kinematics data types."""

from dataclasses import dataclass


class AmggKinematicsError(RuntimeError):
    """Raised when the AMGG kinematics model or a solve request is invalid."""


@dataclass(frozen=True, slots=True)
class AmggPose:
    """Rigid pose represented in a named reference frame.

    Attributes:
        position_m: Cartesian position [m], ordered ``(x, y, z)``.
        quaternion_xyzw: Unit quaternion, ordered ``(x, y, z, w)``.
        reference_frame: Frame in which the pose is expressed.
    """

    position_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    reference_frame: str
