# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Normalize the supplied AMGG SolidWorks URDF for Isaac Lab.

The source URDF is archived as ``amgg_robot_raw.urdf`` with only line endings
and trailing whitespace normalized. This script creates a deterministic
simulation URDF with unique joint names, usable drive limits,
package-independent mesh paths, TCP frames, and explicitly identified research
grippers. It intentionally does not overwrite the source package.
"""

from __future__ import annotations

import argparse
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

JOINT_RENAMES = {
    # The source contains two ArmR01_Joint entries.  The revolute one is R02.
    ("ArmR01_Joint", "ArmR02_Link"): "ArmR02_Joint",
    ("ArmR05_Link", "ArmR05_Link"): "ArmR05_Joint",
    ("ArmR06_Link", "ArmR06_Link"): "ArmR06_Joint",
    ("ArmR07_Link", "ArmR07_Link"): "ArmR07_Joint",
    ("ArmR07Output_Link", "ArmR07Output_Link"): "ArmR07Output_Joint",
}


def _usd_safe_name(name: str) -> str:
    """Replace characters that USD importers otherwise sanitize implicitly."""
    safe = re.sub(r"[^A-Za-z0-9_]", "_", name)
    return f"_{safe}" if safe[:1].isdigit() else safe


def _scalar_text(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def _add_origin(parent: ET.Element, xyz: tuple[float, float, float], rpy=(0.0, 0.0, 0.0)) -> None:
    ET.SubElement(parent, "origin", {"xyz": _scalar_text(xyz), "rpy": _scalar_text(rpy)})


def _add_box_link(
    robot: ET.Element,
    name: str,
    size: tuple[float, float, float],
    mass: float,
    color: tuple[float, float, float, float],
) -> None:
    link = ET.SubElement(robot, "link", {"name": name})
    inertial = ET.SubElement(link, "inertial")
    _add_origin(inertial, (0.0, 0.0, 0.0))
    ET.SubElement(inertial, "mass", {"value": f"{mass:.6g}"})
    x, y, z = size
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": f"{mass * (y * y + z * z) / 12.0:.9g}",
            "ixy": "0",
            "ixz": "0",
            "iyy": f"{mass * (x * x + z * z) / 12.0:.9g}",
            "iyz": "0",
            "izz": f"{mass * (x * x + y * y) / 12.0:.9g}",
        },
    )
    for tag in ("visual", "collision"):
        element = ET.SubElement(link, tag)
        _add_origin(element, (0.0, 0.0, 0.0))
        geometry = ET.SubElement(element, "geometry")
        ET.SubElement(geometry, "box", {"size": _scalar_text(size)})
        if tag == "visual":
            material = ET.SubElement(element, "material", {"name": f"{name}_material"})
            ET.SubElement(material, "color", {"rgba": _scalar_text(color)})


def _add_fixed_joint(
    robot: ET.Element,
    name: str,
    parent: str,
    child: str,
    xyz: tuple[float, float, float],
) -> None:
    joint = ET.SubElement(robot, "joint", {"name": name, "type": "fixed"})
    _add_origin(joint, xyz)
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})


def _add_prismatic_joint(
    robot: ET.Element,
    name: str,
    parent: str,
    child: str,
    xyz: tuple[float, float, float],
    axis: tuple[float, float, float],
) -> None:
    joint = ET.SubElement(robot, "joint", {"name": name, "type": "prismatic"})
    _add_origin(joint, xyz)
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})
    ET.SubElement(joint, "axis", {"xyz": _scalar_text(axis)})
    ET.SubElement(joint, "limit", {"lower": "0", "upper": "0.025", "effort": "30", "velocity": "0.2"})
    ET.SubElement(joint, "dynamics", {"damping": "2.0", "friction": "0.05"})


def _add_research_gripper(robot: ET.Element, side: str, wrist_link: str) -> None:
    prefix = f"{side}_gripper"
    base_link = f"{prefix}_base_link"
    negative_link = f"{prefix}_negative_finger_link"
    positive_link = f"{prefix}_positive_finger_link"
    tcp_link = f"{side}_tcp_link"

    _add_box_link(robot, base_link, (0.08, 0.08, 0.04), 0.20, (0.12, 0.16, 0.22, 1.0))
    _add_box_link(robot, negative_link, (0.015, 0.020, 0.10), 0.04, (0.85, 0.25, 0.12, 1.0))
    _add_box_link(robot, positive_link, (0.015, 0.020, 0.10), 0.04, (0.85, 0.25, 0.12, 1.0))
    ET.SubElement(robot, "link", {"name": tcp_link})

    _add_fixed_joint(robot, f"{prefix}_mount_joint", wrist_link, base_link, (0.0, 0.0, 0.025))
    _add_prismatic_joint(
        robot,
        f"{prefix}_negative_finger_joint",
        base_link,
        negative_link,
        (0.0, -0.035, 0.07),
        (0.0, 1.0, 0.0),
    )
    _add_prismatic_joint(
        robot,
        f"{prefix}_positive_finger_joint",
        base_link,
        positive_link,
        (0.0, 0.035, 0.07),
        (0.0, -1.0, 0.0),
    )
    _add_fixed_joint(robot, f"{side}_tcp_joint", base_link, tcp_link, (0.0, 0.0, 0.12))


def _velocity_for_joint(joint_name: str, joint_type: str) -> float:
    if "Drive_wheel" in joint_name or "Driven" in joint_name:
        return 12.0
    if joint_name.startswith("Turn"):
        return 2.0
    if joint_name.startswith("Waist") or joint_name.startswith("Body"):
        return 1.5
    if joint_name.startswith("Head"):
        return 1.5
    if joint_name.startswith("Arm") or joint_name.startswith("AM_D02"):
        return 2.0
    return 8.0 if joint_type == "continuous" else 2.0


def normalize_urdf(source_urdf: Path, output_urdf: Path) -> None:
    """Generate the normalized AMGG URDF.

    Args:
        source_urdf: Original SolidWorks-exported URDF.
        output_urdf: Destination for the normalized URDF.
    """
    tree = ET.parse(source_urdf)
    robot = tree.getroot()
    robot.set("name", "amgg_robot")

    link_name_map = {link.get("name", ""): _usd_safe_name(link.get("name", "")) for link in robot.findall("link")}
    for link in robot.findall("link"):
        link.set("name", link_name_map[link.get("name", "")])

    for mesh in robot.findall(".//mesh"):
        mesh_name = Path(mesh.get("filename", "")).name
        mesh.set("filename", f"../meshes/{mesh_name}")

    for joint in robot.findall("joint"):
        child = joint.find("child")
        child_link = "" if child is None else child.get("link", "")
        rename = JOINT_RENAMES.get((joint.get("name", ""), child_link))
        if rename:
            joint.set("name", rename)
        joint.set("name", _usd_safe_name(joint.get("name", "")))
        parent = joint.find("parent")
        if parent is not None:
            parent.set("link", link_name_map[parent.get("link", "")])
        if child is not None:
            child.set("link", link_name_map[child_link])
        if joint.get("type") == "fixed":
            continue
        limit = joint.find("limit")
        if limit is None:
            limit = ET.SubElement(joint, "limit")
        limit.set("velocity", f"{_velocity_for_joint(joint.get('name', ''), joint.get('type', '')):.6g}")
        if "effort" not in limit.attrib:
            limit.set("effort", "10")
        dynamics = joint.find("dynamics")
        if dynamics is None:
            dynamics = ET.SubElement(joint, "dynamics")
        dynamics.set("damping", "0.2")
        dynamics.set("friction", "0.02")

    _add_research_gripper(robot, "left", "ArmL07Output_Link")
    _add_research_gripper(robot, "right", "ArmR07Output_Link")

    ET.indent(tree, space="  ")
    output_urdf.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_urdf, encoding="utf-8", xml_declaration=True)
    with output_urdf.open("a", encoding="utf-8") as output_file:
        output_file.write("\n")


def prepare_asset(source_dir: Path, destination_dir: Path) -> None:
    """Copy source assets and build the normalized model.

    Args:
        source_dir: Root of the provided ROS description package.
        destination_dir: AMGG package ``assets/data`` directory.
    """
    urdf_candidates = sorted((source_dir / "urdf").glob("*.urdf"))
    if len(urdf_candidates) != 1:
        raise ValueError(f"Expected exactly one source URDF, found {len(urdf_candidates)} in {source_dir / 'urdf'}")
    mesh_candidates = sorted((source_dir / "meshes").glob("*.STL"))
    if not mesh_candidates:
        raise ValueError(f"No STL meshes found in {source_dir / 'meshes'}")

    urdf_dir = destination_dir / "urdf"
    mesh_dir = destination_dir / "meshes"
    metadata_dir = destination_dir / "source_metadata"
    urdf_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    raw_urdf = urdf_dir / "amgg_robot_raw.urdf"
    source_text = urdf_candidates[0].read_text(encoding="utf-8-sig")
    normalized_source_text = "\n".join(line.rstrip() for line in source_text.splitlines()) + "\n"
    raw_urdf.write_text(normalized_source_text, encoding="utf-8")
    for mesh in mesh_candidates:
        shutil.copy2(mesh, mesh_dir / mesh.name)
    for pattern in ("*.csv", "*.yaml", "*.log"):
        for path in source_dir.rglob(pattern):
            shutil.copy2(path, metadata_dir / path.name)

    normalize_urdf(raw_urdf, urdf_dir / "amgg_robot.urdf")
    print(f"Prepared {len(mesh_candidates)} meshes and normalized URDF under {destination_dir}")


def main() -> None:
    """Parse command-line arguments and prepare the AMGG model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path, help="Provided ROS description package")
    parser.add_argument(
        "--destination_dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "source"
        / "amgg_robot_lab"
        / "amgg_robot_lab"
        / "assets"
        / "data",
    )
    args = parser.parse_args()
    prepare_asset(args.source_dir.resolve(), args.destination_dir.resolve())


if __name__ == "__main__":
    main()
