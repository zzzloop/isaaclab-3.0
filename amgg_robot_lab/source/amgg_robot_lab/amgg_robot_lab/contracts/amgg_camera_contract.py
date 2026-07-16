# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical AMGG camera names, attachment frames, and capture settings."""

from dataclasses import dataclass
from math import isclose, isfinite, sqrt


@dataclass(frozen=True, slots=True)
class AmggCameraSpec:
    """One camera shared by simulation, real recording, and conversion.

    Attributes:
        name: Stable suffix used by ``observation.images.<name>``.
        parent_link: Robot link to which the camera is attached.
        width_px: Image width in pixels.
        height_px: Image height in pixels.
        fps: Capture rate [Hz].
        translation_m: Parent-to-camera translation [m], ordered ``(x, y, z)``.
        quaternion_xyzw: Parent-to-camera quaternion, ordered ``(x, y, z, w)``.
    """

    name: str
    parent_link: str
    width_px: int
    height_px: int
    fps: int
    translation_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]

    def validate(self) -> None:
        """Validate camera identity, dimensions, rate, and extrinsics."""
        if not self.name or not self.parent_link:
            raise ValueError("AMGG camera name and parent link must be non-empty.")
        if self.width_px <= 0 or self.height_px <= 0 or self.fps <= 0:
            raise ValueError(f"Camera '{self.name}' dimensions and FPS must be positive.")
        if not all(isfinite(value) for value in self.translation_m + self.quaternion_xyzw):
            raise ValueError(f"Camera '{self.name}' contains non-finite extrinsics.")
        norm = sqrt(sum(value * value for value in self.quaternion_xyzw))
        if not isclose(norm, 1.0, abs_tol=1e-5):
            raise ValueError(f"Camera '{self.name}' quaternion is not normalized.")


# These are simulation defaults.  Real-camera extrinsics must be replaced by
# hand-eye calibration values without changing the four stable dataset keys.
AMGG_CAMERAS: tuple[AmggCameraSpec, ...] = (
    AmggCameraSpec("head", "Head03_Link", 640, 480, 30, (0.04, 0.0, 0.02), (0.0, 0.7071068, 0.0, 0.7071068)),
    AmggCameraSpec(
        "left_wrist",
        "left_gripper_base_link",
        640,
        480,
        30,
        (0.025, 0.0, 0.055),
        (0.0, 0.7071068, 0.0, 0.7071068),
    ),
    AmggCameraSpec(
        "right_wrist",
        "right_gripper_base_link",
        640,
        480,
        30,
        (0.025, 0.0, 0.055),
        (0.0, 0.7071068, 0.0, 0.7071068),
    ),
    AmggCameraSpec("overview", "base_link", 640, 480, 30, (1.65, 0.0, 1.35), (0.0, 0.8924104, 0.0, -0.4512247)),
)
AMGG_CAMERA_BY_NAME = {camera.name: camera for camera in AMGG_CAMERAS}


def require_amgg_camera_contract() -> None:
    """Validate stable camera names and simulation defaults."""
    names = [camera.name for camera in AMGG_CAMERAS]
    if len(names) != 4 or len(set(names)) != 4:
        raise ValueError("The AMGG dataset contract requires exactly four unique cameras.")
    for camera in AMGG_CAMERAS:
        camera.validate()
