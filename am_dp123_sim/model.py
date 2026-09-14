# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent AM-DP123 URDF model and bounded arm IK; only NumPy is needed."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
URDF = ROOT / "assets" / "AM-DP123" / "urdf" / "AM-DP123.urdf"
ARMS = {side: [f"openarm_{side}_joint{i}" for i in range(1, 8)] for side in ("left", "right")}
TIPS = {side: f"{side}_arm_link7" for side in ARMS}


def rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return a rotation about an axis by an angle [rad]."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def rotation_error(target: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return the world-frame shortest rotation vector [rad], including near pi."""
    delta = target @ actual.T
    angle = np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1))
    vector = np.array([delta[2, 1] - delta[1, 2], delta[0, 2] - delta[2, 0], delta[1, 0] - delta[0, 1]])
    if angle < 1e-7:
        return vector / 2
    if np.pi - angle < 1e-5:
        values, vectors = np.linalg.eigh((delta + delta.T) / 2)
        axis = vectors[:, np.argmax(values)]
        if axis @ vector < 0:
            axis = -axis
        return angle * axis
    return angle / (2 * np.sin(angle)) * vector


def pose_matrix(pose: list[float]) -> np.ndarray:
    """Convert [x, y, z, qx, qy, qz, qw] ([m], unit quaternion) to a transform."""
    values = np.asarray(pose, dtype=float)
    if values.shape != (7,) or not np.all(np.isfinite(values)):
        raise ValueError("Pose must contain seven finite numbers")
    norm = np.linalg.norm(values[3:])
    if not 0.5 < norm < 1.5:
        raise ValueError("Invalid pose quaternion")
    x, y, z, w = values[3:] / norm
    result = np.eye(4)
    result[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    result[:3, 3] = values[:3]
    return result


class RobotModel:
    """Read the original URDF; retain its axes, origins, limits and fixed joints."""

    def __init__(self, path: Path = URDF):
        self.path = Path(path)
        self.xml = ET.parse(path)
        self.joints = {}
        self.parents = {}
        self.limits = {}
        self.efforts = {}
        self.velocities = {}
        for node in self.xml.getroot().findall("joint"):
            name = node.attrib["name"]
            origin = node.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            rpy = np.fromstring(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            transform = np.eye(4)
            transform[:3, 3] = xyz
            transform[:3, :3] = rotation([0, 0, 1], rpy[2]) @ rotation([0, 1, 0], rpy[1]) @ rotation([1, 0, 0], rpy[0])
            kind = node.attrib["type"]
            axis_node = node.find("axis")
            axis = np.fromstring(axis_node.get("xyz", "1 0 0") if axis_node is not None else "1 0 0", sep=" ")
            if kind != "fixed":
                if kind not in ("revolute", "continuous") or np.linalg.norm(axis) < 1e-10:
                    raise ValueError(f"Unsupported joint: {name}")
                axis /= np.linalg.norm(axis)
                limit = node.find("limit")
                self.limits[name] = (
                    (-np.inf, np.inf)
                    if kind == "continuous"
                    else (float(limit.get("lower")), float(limit.get("upper")))
                )
                self.efforts[name] = float(limit.get("effort", "10")) if limit is not None else 10.0
                self.velocities[name] = float(limit.get("velocity", "1")) if limit is not None else 1.0
            parent, child = node.find("parent").get("link"), node.find("child").get("link")
            self.joints[name] = (parent, child, kind, transform, axis)
            self.parents[child] = name
        self.home = {name: float(np.clip(0, *limits)) for name, limits in self.limits.items()}
        # Slight elbow bend avoids a completely straight initial arm.
        self.home.update({"openarm_left_joint4": 0.3, "openarm_right_joint4": 0.3})
        for side, names in ARMS.items():
            if not set(names).issubset(self.limits) or TIPS[side] not in self.parents:
                raise ValueError(f"URDF does not match the AM-DP123 {side} arm contract")

    def forward(
        self, tip: str, positions: dict[str, float], names: tuple[str, ...] | list[str] = ()
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return base-to-tip pose ([m], rotation) and geometric Jacobian ([m/rad], angular)."""
        chain = []
        link = tip
        while link in self.parents:
            name = self.parents[link]
            chain.append(name)
            link = self.joints[name][0]
        transform = np.eye(4)
        frames = {}
        for name in reversed(chain):
            _, _, kind, origin, axis = self.joints[name]
            transform = transform @ origin
            frames[name] = (transform[:3, 3].copy(), transform[:3, :3] @ axis)
            if kind != "fixed":
                motion = np.eye(4)
                motion[:3, :3] = rotation(axis, positions[name])
                transform = transform @ motion
        jacobian = np.zeros((6, len(names)))
        for i, name in enumerate(names):
            point, axis = frames[name]
            jacobian[:3, i] = np.cross(axis, transform[:3, 3] - point)
            jacobian[3:, i] = axis
        return transform, jacobian

    def solve(self, side: str, target: np.ndarray, seed: dict[str, float]) -> tuple[dict[str, float], float, float]:
        """Solve one arm; return joints [rad], position error [m] and rotation error [rad]."""
        names = ARMS[side]
        q = dict(seed)
        weights = np.array([1, 1, 1, 0.35, 0.35, 0.35])
        for _ in range(60):
            actual, jacobian = self.forward(TIPS[side], q, names)
            error = np.r_[target[:3, 3] - actual[:3, 3], rotation_error(target[:3, :3], actual[:3, :3])]
            if np.linalg.norm(error[:3]) < 0.002 and np.linalg.norm(error[3:]) < 0.02:
                break
            weighted = weights[:, None] * jacobian
            step = weighted.T @ np.linalg.solve(weighted @ weighted.T + 0.02**2 * np.eye(6), weights * error)
            step *= min(1.0, 0.10 / max(np.max(np.abs(step)), 1e-12))
            for name, delta in zip(names, step, strict=True):
                q[name] = float(np.clip(q[name] + delta, *self.limits[name]))
        actual, _ = self.forward(TIPS[side], q)
        return (
            q,
            float(np.linalg.norm(target[:3, 3] - actual[:3, 3])),
            float(np.linalg.norm(rotation_error(target[:3, :3], actual[:3, :3]))),
        )

    def prepare_urdf(self) -> Path:
        """Create an import copy with absolute mesh paths; never change source assets."""
        tree = ET.parse(self.path)
        for mesh in tree.getroot().iter("mesh"):
            filename = mesh.attrib["filename"]
            prefix = "package://AM-DP123/"
            if not filename.startswith(prefix):
                raise ValueError(f"Unexpected mesh URI: {filename}")
            path = self.path.parent.parent / filename[len(prefix) :]
            if not path.is_file():
                raise FileNotFoundError(f"Missing mesh: {path}")
            with path.open("rb") as stream:
                header = stream.read(80)
            if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise FileNotFoundError(f"Missing mesh or unexpanded Git LFS pointer: {path}; run git lfs pull")
            mesh.set("filename", path.resolve().as_posix())
        output = ROOT / ".cache" / "AM-DP123.urdf"
        output.parent.mkdir(parents=True, exist_ok=True)
        tree.write(output, encoding="utf-8", xml_declaration=True)
        return output
