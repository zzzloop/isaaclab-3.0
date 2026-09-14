# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical AM-DP123 camera names, attachment frames, and capture settings."""

from dataclasses import dataclass
from math import atan, degrees, isclose, isfinite, sqrt
from statistics import fmean


@dataclass(frozen=True, slots=True)
class AmDp123CameraSpec:
    """One camera shared by simulation, XR image panels, and recording.

    The intrinsics default to the simulation values the environment ships with. The
    AM-DP123 URDF has no camera sensors, so none of these numbers describe the real
    robot: replace them with measured values after calibrating the hardware.

    Attributes:
        name: Stable name used by the scene and by ``observation.images.<name>``.
        parent_link: Robot link to which the camera is attached.
        width_px: Image width in pixels.
        height_px: Image height in pixels.
        fps: Capture rate [Hz].
        translation_m: Parent-to-camera translation [m], ordered ``(x, y, z)``.
        quaternion_xyzw: Parent-to-camera quaternion, ordered ``(x, y, z, w)``, expressed
            in the Isaac Lab ``"ros"`` camera convention.
        focal_length_mm: Pinhole focal length [mm]. Simulation default.
        horizontal_aperture_mm: Pinhole sensor width [mm]. Simulation default.
        focus_distance_m: Pinhole focus distance [m]. Simulation default.
        clipping_range_m: Near and far clipping planes [m]. Simulation default.
    """

    name: str
    parent_link: str
    width_px: int
    height_px: int
    fps: int
    translation_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    focal_length_mm: float = 18.0
    horizontal_aperture_mm: float = 20.955
    focus_distance_m: float = 1.0
    clipping_range_m: tuple[float, float] = (0.05, 10.0)

    def validate(self) -> None:
        """Validate camera identity, capture settings, intrinsics, and extrinsics."""
        if not self.name or not self.parent_link:
            raise ValueError("AM-DP123 camera name and parent link must be non-empty.")
        if self.width_px <= 0 or self.height_px <= 0 or self.fps <= 0:
            raise ValueError(f"Camera '{self.name}' dimensions and FPS must be positive.")
        if not all(isfinite(value) for value in self.translation_m + self.quaternion_xyzw):
            raise ValueError(f"Camera '{self.name}' contains non-finite extrinsics.")
        if self.focal_length_mm <= 0.0 or self.horizontal_aperture_mm <= 0.0 or self.focus_distance_m <= 0.0:
            raise ValueError(f"Camera '{self.name}' needs a positive focal length, aperture, and focus distance.")
        near_plane, far_plane = self.clipping_range_m
        if not 0.0 < near_plane < far_plane:
            raise ValueError(f"Camera '{self.name}' clipping range must satisfy 0 < near < far.")
        norm = sqrt(sum(value * value for value in self.quaternion_xyzw))
        if not isclose(norm, 1.0, abs_tol=1e-5):
            raise ValueError(f"Camera '{self.name}' quaternion is not normalized.")


# The offsets place each camera on the lens plane of its mounting bracket and
# align its optical axis with the bracket's local forward axis:
#   * head bracket lens plane      : local +z at 0.0288 m, 0.10 m wide housing
#   * left-arm bracket lens plane  : local +x at 0.0340 m
#   * right-arm bracket lens plane : local -x at 0.0323 m
# The head pair sits 0.03 m either side of the bracket centre for a 0.06 m
# baseline. Real-camera extrinsics must be replaced by hand-eye calibration
# values without changing the four stable names.
AM_DP123_CAMERAS: tuple[AmDp123CameraSpec, ...] = (
    AmDp123CameraSpec(
        "head_left",
        "head_camera_link",
        640,
        480,
        30,
        (0.0, 0.03, 0.0288),
        (0.0, 0.0, -0.70710678, 0.70710678),
    ),
    AmDp123CameraSpec(
        "head_right",
        "head_camera_link",
        640,
        480,
        30,
        (0.0, -0.03, 0.0288),
        (0.0, 0.0, -0.70710678, 0.70710678),
    ),
    AmDp123CameraSpec(
        "left_wrist",
        "left_arm_camera_link",
        640,
        480,
        30,
        (0.034, 0.0, 0.0),
        (0.5, 0.5, 0.5, 0.5),
    ),
    AmDp123CameraSpec(
        "right_wrist",
        "right_arm_camera_link",
        640,
        480,
        30,
        (-0.0323, 0.0, 0.0),
        (0.5, -0.5, -0.5, 0.5),
    ),
)
AM_DP123_CAMERA_BY_NAME = {camera.name: camera for camera in AM_DP123_CAMERAS}

AM_DP123_STEREO_CAMERA_NAMES: tuple[str, str] = ("head_left", "head_right")
"""Left and right head cameras that form the XR stereo pair."""


def am_dp123_stereo_baseline_m() -> float:
    """Return the head stereo baseline [m] measured between both camera positions."""
    left, right = (AM_DP123_CAMERA_BY_NAME[name] for name in AM_DP123_STEREO_CAMERA_NAMES)
    return float(sqrt(sum((left.translation_m[i] - right.translation_m[i]) ** 2 for i in range(3))))


def am_dp123_camera_fov_deg(camera: AmDp123CameraSpec) -> tuple[float, float]:
    """Return the ``(horizontal, vertical)`` field of view [deg] implied by a camera's intrinsics.

    Args:
        camera: Camera contract entry whose pinhole intrinsics define the field of view.

    Returns:
        Field of view [deg] along the image width and height.
    """
    horizontal = 2.0 * atan(camera.horizontal_aperture_mm / (2.0 * camera.focal_length_mm))
    vertical_ratio = camera.height_px / camera.width_px
    vertical = 2.0 * atan(vertical_ratio * camera.horizontal_aperture_mm / (2.0 * camera.focal_length_mm))
    return float(degrees(horizontal)), float(degrees(vertical))


def require_am_dp123_camera_contract() -> None:
    """Validate stable camera names and the head stereo pair geometry."""
    names = [camera.name for camera in AM_DP123_CAMERAS]
    if len(names) != 4 or len(set(names)) != 4:
        raise ValueError("The AM-DP123 camera contract requires exactly four unique cameras.")
    for camera in AM_DP123_CAMERAS:
        camera.validate()
    left, right = (AM_DP123_CAMERA_BY_NAME[name] for name in AM_DP123_STEREO_CAMERA_NAMES)
    if left.parent_link != right.parent_link:
        raise ValueError("The AM-DP123 stereo pair must share one parent link.")
    if left.quaternion_xyzw != right.quaternion_xyzw:
        raise ValueError("The AM-DP123 stereo pair must share one orientation.")
    if left.width_px != right.width_px or left.height_px != right.height_px or left.fps != right.fps:
        raise ValueError("The AM-DP123 stereo pair must share resolution and rate.")
    if not isclose(left.translation_m[1], -right.translation_m[1], abs_tol=1e-9):
        raise ValueError("The AM-DP123 stereo pair must be mirrored about the mount centreline.")
    if abs(fmean((left.translation_m[0], right.translation_m[0]))) > 1e-9:
        raise ValueError("The AM-DP123 stereo pair must share one forward offset.")
    baseline = am_dp123_stereo_baseline_m()
    if not 0.03 <= baseline <= 0.12:
        raise ValueError(f"AM-DP123 stereo baseline {baseline:.4f} m is outside the usable range.")
