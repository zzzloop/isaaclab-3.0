# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical AMGG camera names, attachment frames, and capture settings."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class AmggCameraSpec:
    """One camera shared by simulation, real recording, and dataset conversion.

    Attributes:
        name: Stable suffix used by ``observation.images.<name>``.
        parent_link: Robot link to which the camera is attached.
        width_px: Image width in pixels.
        height_px: Image height in pixels.
        fps: Capture rate [Hz].
        translation_m: Parent-to-camera translation [m], ordered ``(x, y, z)``.
        quaternion_xyzw: Parent-to-camera unit quaternion, ordered ``(x, y, z, w)``.
    """

    name: str
    parent_link: str
    width_px: int
    height_px: int
    fps: int
    translation_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]

    def validate(self) -> None:
        """Validate camera identity, dimensions, rate, and finite extrinsics."""
        if not self.name or not self.parent_link:
            raise ValueError("AMGG camera name and parent link must be non-empty.")
        if self.width_px <= 0 or self.height_px <= 0 or self.fps <= 0:
            raise ValueError(f"Camera '{self.name}' dimensions and FPS must be positive.")
        if not all(isfinite(value) for value in self.translation_m + self.quaternion_xyzw):
            raise ValueError(f"Camera '{self.name}' contains non-finite extrinsics.")


# Populate only after the exact real/sim camera set and extrinsics are confirmed.
AMGG_CAMERAS: tuple[AmggCameraSpec, ...] = ()

