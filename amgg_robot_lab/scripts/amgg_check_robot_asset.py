# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate the AMGG URDF, meshes, limits, contracts, and kinematics."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from amgg_robot_lab.assets import AMGG_URDF_PATH
from amgg_robot_lab.contracts import (
    AMGG_FRAMES,
    AMGG_HOME_POSITIONS,
    AMGG_OBSERVED_JOINT_NAMES,
    require_amgg_camera_contract,
    require_amgg_joint_contract,
)
from amgg_robot_lab.kinematics import get_amgg_kinematics


def validate_robot_asset(urdf_path: Path = AMGG_URDF_PATH) -> dict[str, int]:
    """Validate the normalized model and return structural counts.

    Args:
        urdf_path: Normalized URDF to validate.

    Returns:
        Counts for links, joints, actuated joints, and meshes.
    """
    if not urdf_path.is_file():
        raise FileNotFoundError(f"Missing AMGG URDF: {urdf_path}")
    root = ET.parse(urdf_path).getroot()
    links = [element.get("name", "") for element in root.findall("link")]
    joints = root.findall("joint")
    joint_names = [element.get("name", "") for element in joints]
    if len(links) != len(set(links)) or len(joint_names) != len(set(joint_names)):
        raise ValueError("AMGG URDF contains duplicate link or joint names.")
    missing_meshes = []
    for mesh in root.findall(".//mesh"):
        path = (urdf_path.parent / mesh.get("filename", "")).resolve()
        if not path.is_file():
            missing_meshes.append(str(path))
    if missing_meshes:
        raise FileNotFoundError(f"Missing AMGG meshes: {missing_meshes}")
    for joint in joints:
        if joint.get("type") == "fixed":
            continue
        limit = joint.find("limit")
        if limit is None or float(limit.get("velocity", "0")) <= 0.0 or float(limit.get("effort", "0")) <= 0.0:
            raise ValueError(f"Joint '{joint.get('name')}' has unusable effort or velocity limits.")
    require_amgg_joint_contract()
    require_amgg_camera_contract()
    if not set(AMGG_OBSERVED_JOINT_NAMES).issubset(joint_names):
        raise ValueError("The normalized URDF does not implement the observed-joint contract.")
    required_frames = {
        AMGG_FRAMES.base_link,
        AMGG_FRAMES.torso_link,
        AMGG_FRAMES.left_wrist_link,
        AMGG_FRAMES.right_wrist_link,
        AMGG_FRAMES.left_tcp_link,
        AMGG_FRAMES.right_tcp_link,
    }
    if not required_frames.issubset(links):
        raise ValueError("The normalized URDF does not implement the frame contract.")

    model = get_amgg_kinematics()
    for tip in (AMGG_FRAMES.left_tcp_link, AMGG_FRAMES.right_tcp_link):
        transform = model.forward(tip, AMGG_HOME_POSITIONS)
        if not np.isfinite(transform).all() or not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0)):
            raise ValueError(f"FK produced an invalid transform for {tip}.")
    return {
        "links": len(links),
        "joints": len(joints),
        "actuated_joints": len(model.actuated_joint_names),
        "meshes": len(root.findall(".//mesh")),
    }


def main() -> None:
    """Run validation and print a compact acceptance report."""
    counts = validate_robot_asset()
    print("AMGG robot asset validation passed")
    print(", ".join(f"{key}={value}" for key, value in counts.items()))


if __name__ == "__main__":
    main()
