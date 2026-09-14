# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The packaged AM-DP123 URDF must match the public contracts."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from math import isclose

from amgg_robot_lab.assets import AM_DP123_ASSET_DATA_DIR, AM_DP123_URDF_PATH
from amgg_robot_lab.contracts import (
    AM_DP123_FRAMES,
    AM_DP123_HAND_JOINT_NAMES,
    AM_DP123_JOINT_SPECS,
    AM_DP123_LOCOMOTION_JOINT_NAMES,
)

URDF_ROOT = ET.parse(AM_DP123_URDF_PATH).getroot()
LINK_NAMES = {link.get("name") for link in URDF_ROOT.findall("link")}
JOINTS = {joint.get("name"): joint for joint in URDF_ROOT.findall("joint")}


def test_urdf_topology_is_stable():
    """Link, joint, and joint-type counts are the ABI of the shipped asset."""
    assert len(LINK_NAMES) == 61
    assert len(JOINTS) == 60
    movable = {name: joint.get("type") for name, joint in JOINTS.items() if joint.get("type") != "fixed"}
    assert Counter(movable.values()) == {"revolute": 23, "continuous": 8}


def test_contract_joints_match_urdf_limits():
    """Every contract joint keeps the upstream position, effort, and velocity limits."""
    for spec in AM_DP123_JOINT_SPECS:
        element = JOINTS[spec.name]
        assert element.get("type") == "revolute"
        limit = element.find("limit")
        assert isclose(float(limit.get("lower")), spec.lower_limit_rad, abs_tol=1e-12)
        assert isclose(float(limit.get("upper")), spec.upper_limit_rad, abs_tol=1e-12)
        assert isclose(float(limit.get("effort")), spec.max_effort_nm, abs_tol=1e-9)
        assert isclose(float(limit.get("velocity")), spec.max_velocity_rad_s, abs_tol=1e-9)


def test_locomotion_joints_are_continuous_and_uncounted():
    """The eight mobile-base joints stay outside the state and command ABIs."""
    assert len(AM_DP123_LOCOMOTION_JOINT_NAMES) == 8
    for name in AM_DP123_LOCOMOTION_JOINT_NAMES:
        assert JOINTS[name].get("type") == "continuous"
    contract_names = {spec.name for spec in AM_DP123_JOINT_SPECS}
    assert contract_names.isdisjoint(AM_DP123_LOCOMOTION_JOINT_NAMES)


def test_contract_frames_exist_as_links():
    """Every frame the extension targets is a real URDF link."""
    contract = AM_DP123_FRAMES
    for name in (
        contract.base_link,
        contract.torso_link,
        contract.head_link,
        contract.head_camera_link,
        contract.left_wrist_link,
        contract.right_wrist_link,
        contract.left_hand_base_link,
        contract.right_hand_base_link,
        contract.left_wrist_camera_link,
        contract.right_wrist_camera_link,
    ):
        assert name in LINK_NAMES
    child_links = {joint.find("child").get("link") for joint in JOINTS.values()}
    assert contract.base_link not in child_links


def test_mesh_references_resolve_inside_the_package():
    """All visual and collision meshes use the packaged ROS package and exist on disk."""
    references = [mesh.get("filename") for mesh in URDF_ROOT.iter("mesh")]
    assert len(references) == 122
    assert len(set(references)) == 61
    for filename in references:
        assert filename.startswith("package://AM-DP123/meshes/")
        assert (AM_DP123_ASSET_DATA_DIR / filename.removeprefix("package://AM-DP123/")).is_file()
    for link in URDF_ROOT.findall("link"):
        assert link.find("visual/geometry/mesh") is not None
        assert link.find("collision/geometry/mesh") is not None


def test_hand_jaws_keep_the_upstream_mimic_coupling():
    """The four hand joints are independent DOF whose targets reproduce the mimic law.

    URDF importer 3.0 ignores ``convert_mimic_joints_to_normal_joints``, so the
    simulator drives both jaws of each hand explicitly. The upstream ``<mimic>``
    entries declare ``jaw2 = -jaw1``; the contract's one-sided limits preserve
    that relation at every closure.
    """
    for side in ("left", "right"):
        jaw_a, jaw_b = (f"{side}_arm_hand_joint{index}_0" for index in (1, 2))
        assert jaw_a in AM_DP123_HAND_JOINT_NAMES and jaw_b in AM_DP123_HAND_JOINT_NAMES
        mimic = JOINTS[jaw_b].find("mimic")
        assert mimic is not None
        assert mimic.get("joint") == jaw_a
        assert isclose(float(mimic.get("multiplier")), -1.0, abs_tol=1e-12)
        assert isclose(float(mimic.get("offset")), 0.0, abs_tol=1e-12)
