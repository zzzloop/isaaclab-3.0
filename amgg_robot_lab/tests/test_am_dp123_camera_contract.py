# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Public AM-DP123 camera contract and its physical mounting derivation."""

from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from math import isclose, radians, tan

import numpy as np
import pytest

from amgg_robot_lab.assets import AM_DP123_ASSET_DATA_DIR, AM_DP123_URDF_PATH
from amgg_robot_lab.contracts import (
    AM_DP123_CAMERA_BY_NAME,
    AM_DP123_CAMERAS,
    AM_DP123_STEREO_CAMERA_NAMES,
    AmDp123CameraSpec,
    am_dp123_camera_fov_deg,
    am_dp123_stereo_baseline_m,
    require_am_dp123_camera_contract,
)
from amgg_robot_lab.kinematics import get_am_dp123_kinematics
from amgg_robot_lab.kinematics.am_dp123_urdf_kinematics import quaternion_xyzw_to_matrix

URDF_ROOT = ET.parse(AM_DP123_URDF_PATH).getroot()
LINK_NAMES = {link.get("name") for link in URDF_ROOT.findall("link")}


def _mesh_vertices(filename: str) -> np.ndarray:
    """Return the vertices of a packaged STL mesh."""
    data = (AM_DP123_ASSET_DATA_DIR / filename.removeprefix("package://AM-DP123/")).read_bytes()
    if data[:5] != b"solid":  # binary STL
        count = struct.unpack_from("<I", data, 80)[0]
        records = np.frombuffer(data, dtype=np.uint8, count=count * 50, offset=84).reshape(count, 50)
        return records[:, 12:48].copy().view("<f4").reshape(-1, 3).astype(np.float64)
    lines = [line.split()[1:4] for line in data.decode("ascii", "ignore").splitlines() if "vertex" in line]
    return np.array([[float(value) for value in line] for line in lines])


def _link_visual_vertices(link_name: str) -> np.ndarray:
    """Return every visual mesh vertex of a link, expressed in the link frame."""
    link = next(element for element in URDF_ROOT.findall("link") if element.get("name") == link_name)
    collected = []
    for visual in link.findall("visual"):
        mesh = visual.find("geometry/mesh")
        if mesh is None:
            continue
        origin = visual.find("origin")
        offset = np.array([float(value) for value in (origin.get("xyz") if origin is not None else "0 0 0").split()])
        collected.append(_mesh_vertices(mesh.get("filename")) + offset)
    return np.concatenate(collected)


def _camera_axes_in_link(quaternion_xyzw: tuple[float, float, float, float]) -> tuple[np.ndarray, ...]:
    """Return the camera axes ``(right, up, forward)`` in the parent link frame.

    Isaac Lab applies the offset pose in the ``"ros"`` convention, where the camera
    looks along ``+Z`` and image-up is ``-Y``.
    """
    rotation = quaternion_xyzw_to_matrix(quaternion_xyzw)
    return (
        rotation @ np.array((1.0, 0.0, 0.0)),
        rotation @ np.array((0.0, -1.0, 0.0)),
        rotation @ np.array((0.0, 0.0, 1.0)),
    )


def test_camera_contract_validates():
    """The contract exposes four unique cameras with a mirrored stereo pair."""
    require_am_dp123_camera_contract()
    assert len(AM_DP123_CAMERAS) == 4
    assert AM_DP123_STEREO_CAMERA_NAMES == ("head_left", "head_right")
    assert set(AM_DP123_CAMERA_BY_NAME) == {"head_left", "head_right", "left_wrist", "right_wrist"}
    assert isclose(am_dp123_stereo_baseline_m(), 0.06, abs_tol=1e-9)


def test_cameras_mount_on_existing_links():
    """Every camera parent is a real URDF link and shared by the stereo pair."""
    for camera in AM_DP123_CAMERAS:
        assert camera.parent_link in LINK_NAMES
    left, right = (AM_DP123_CAMERA_BY_NAME[name] for name in AM_DP123_STEREO_CAMERA_NAMES)
    assert left.parent_link == right.parent_link == "head_camera_link"
    assert left.translation_m[1] == -right.translation_m[1]
    assert left.quaternion_xyzw == right.quaternion_xyzw


def test_camera_extrinsics_are_normalized():
    """Camera quaternions must be unit length for a valid rotation."""
    for camera in AM_DP123_CAMERAS:
        assert isclose(float(np.linalg.norm(camera.quaternion_xyzw)), 1.0, abs_tol=1e-6)
        assert camera.width_px == 640
        assert camera.height_px == 480
        assert camera.fps == 30


def test_camera_intrinsics_are_explicit_and_consistent():
    """The contract ships the intrinsics, and their field of view matches the aspect ratio."""
    for camera in AM_DP123_CAMERAS:
        assert camera.focal_length_mm > 0.0
        assert camera.horizontal_aperture_mm > 0.0
        assert camera.focus_distance_m > 0.0
        assert 0.0 < camera.clipping_range_m[0] < camera.clipping_range_m[1]
        horizontal_deg, vertical_deg = am_dp123_camera_fov_deg(camera)
        # A pinhole sensor keeps tan(hfov/2)/tan(vfov/2) equal to the image aspect ratio.
        assert isclose(
            tan(radians(horizontal_deg) / 2.0) / tan(radians(vertical_deg) / 2.0),
            camera.width_px / camera.height_px,
            rel_tol=1e-9,
        )
        assert 40.0 < horizontal_deg < 90.0
        assert 30.0 < vertical_deg < 70.0


def test_camera_contract_rejects_inverted_clipping_range():
    """A near plane behind the far plane must be rejected."""
    with pytest.raises(ValueError):
        AmDp123CameraSpec(
            "head_left",
            "head_camera_link",
            640,
            480,
            30,
            (0.0, 0.03, 0.0288),
            (0.0, 0.0, -0.70710678, 0.70710678),
            clipping_range_m=(10.0, 0.05),
        ).validate()


def test_cameras_sit_on_the_mount_lens_plane():
    """Each camera offset reaches the lens plane of its own bracket mesh."""
    for camera in AM_DP123_CAMERAS:
        _, _, forward = _camera_axes_in_link(camera.quaternion_xyzw)
        vertices = _link_visual_vertices(camera.parent_link)
        offset_along_forward = float(np.dot(camera.translation_m, forward))
        mesh_front = float(np.max(vertices @ forward))
        assert isclose(offset_along_forward, mesh_front, abs_tol=0.003), camera.name
        assert float(np.linalg.norm(np.cross(camera.translation_m, forward))) < 0.06, camera.name


def test_stereo_baseline_fits_inside_the_head_bracket():
    """The stereo pair spans less than the head bracket width."""
    left, _ = (AM_DP123_CAMERA_BY_NAME[name] for name in AM_DP123_STEREO_CAMERA_NAMES)
    right_axis, _, _ = _camera_axes_in_link(left.quaternion_xyzw)
    vertices = _link_visual_vertices(left.parent_link)
    extent = float(np.max(vertices @ right_axis) - np.min(vertices @ right_axis))
    assert am_dp123_stereo_baseline_m() <= extent


def test_cameras_look_forward_at_the_zero_pose():
    """With every joint at zero, all four optical axes point along the robot x-axis."""
    model = get_am_dp123_kinematics()
    zero_pose: dict[str, float] = {}
    for camera in AM_DP123_CAMERAS:
        link_rotation = model.forward(camera.parent_link, zero_pose)[:3, :3]
        right, up, forward = _camera_axes_in_link(camera.quaternion_xyzw)
        np.testing.assert_allclose(link_rotation @ forward, (1.0, 0.0, 0.0), atol=0.05, err_msg=camera.name)
        np.testing.assert_allclose(link_rotation @ up, (0.0, 0.0, 1.0), atol=0.05, err_msg=camera.name)
        np.testing.assert_allclose(link_rotation @ right, (0.0, -1.0, 0.0), atol=0.05, err_msg=camera.name)


def test_camera_contract_rejects_mismatched_stereo_pair(monkeypatch):
    """A stereo pair with different rates must be rejected."""
    import amgg_robot_lab.contracts.am_dp123_camera_contract as camera_contract

    left = AM_DP123_CAMERA_BY_NAME["head_left"]
    broken = AmDp123CameraSpec(
        "head_right",
        left.parent_link,
        left.width_px,
        left.height_px,
        left.fps + 1,
        (0.0, -0.03, 0.0288),
        left.quaternion_xyzw,
    )
    monkeypatch.setattr(
        camera_contract,
        "AM_DP123_CAMERAS",
        (left, broken, AM_DP123_CAMERA_BY_NAME["left_wrist"], AM_DP123_CAMERA_BY_NAME["right_wrist"]),
    )
    monkeypatch.setattr(
        camera_contract,
        "AM_DP123_CAMERA_BY_NAME",
        {"head_left": left, "head_right": broken, "left_wrist": AM_DP123_CAMERA_BY_NAME["left_wrist"]},
    )
    with pytest.raises(ValueError):
        camera_contract.require_am_dp123_camera_contract()
