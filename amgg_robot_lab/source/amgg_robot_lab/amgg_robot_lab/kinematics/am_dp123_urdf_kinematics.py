# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dependency-light AM-DP123 URDF FK, Jacobian, and damped least-squares IK."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import acos, cos, sin
from pathlib import Path

import numpy as np

from .am_dp123_kinematics_model import AmDp123KinematicsError


@dataclass(frozen=True, slots=True)
class UrdfJoint:
    """Kinematic information for one URDF joint.

    Attributes:
        name: Joint name from the URDF.
        joint_type: URDF joint type.
        parent: Parent link name.
        child: Child link name.
        origin: Parent-to-joint homogeneous transform; translation [m].
        axis: Unit motion axis in the joint frame.
        lower: Lower limit [m or rad, depending on joint type].
        upper: Upper limit [m or rad, depending on joint type].
    """

    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class IkTarget:
    """One Cartesian target used by the multi-end-effector solver.

    Attributes:
        link_name: Name of the target link.
        transform: Root-to-target homogeneous transform; translation [m].
        position_weight: Weight applied to translational error.
        orientation_weight: Weight applied to rotational error.
    """

    link_name: str
    transform: np.ndarray
    position_weight: float = 1.0
    orientation_weight: float = 0.35


@dataclass(frozen=True, slots=True)
class IkResult:
    """Result and diagnostics from an iterative IK solve.

    Attributes:
        joint_positions: Solved positions [m or rad, depending on joint type].
        converged: Whether the requested tolerances were reached.
        iterations: Number of solver iterations performed.
        position_error_m: Maximum target position error [m].
        orientation_error_rad: Maximum target orientation error [rad].
    """

    joint_positions: dict[str, float]
    converged: bool
    iterations: int
    position_error_m: float
    orientation_error_rad: float


def _vector(text: str | None, length: int) -> np.ndarray:
    values = np.zeros(length) if not text else np.asarray([float(value) for value in text.split()])
    if values.shape != (length,):
        raise AmDp123KinematicsError(f"Expected {length} values, received {text!r}.")
    return values.astype(np.float64)


def _rotation_from_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr, cp, sp, cy, sy = cos(roll), sin(roll), cos(pitch), sin(pitch), cos(yaw), sin(yaw)
    rotation_x = np.array(((1, 0, 0), (0, cr, -sr), (0, sr, cr)), dtype=np.float64)
    rotation_y = np.array(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)), dtype=np.float64)
    rotation_z = np.array(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)), dtype=np.float64)
    return rotation_z @ rotation_y @ rotation_x


def _origin_transform(element: ET.Element | None) -> np.ndarray:
    transform = np.eye(4)
    if element is not None:
        transform[:3, :3] = _rotation_from_rpy(_vector(element.get("rpy"), 3))
        transform[:3, 3] = _vector(element.get("xyz"), 3)
    return transform


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        raise AmDp123KinematicsError("A revolute joint has a zero-length axis.")
    x, y, z = axis / axis_norm
    c, s, cross = cos(angle), sin(angle), 1.0 - cos(angle)
    transform = np.eye(4)
    transform[:3, :3] = (
        (c + x * x * cross, x * y * cross - z * s, x * z * cross + y * s),
        (y * x * cross + z * s, c + y * y * cross, y * z * cross - x * s),
        (z * x * cross - y * s, z * y * cross + x * s, c + z * z * cross),
    )
    return transform


def _motion_transform(joint: UrdfJoint, position: float) -> np.ndarray:
    if joint.joint_type in {"revolute", "continuous"}:
        return _axis_angle(joint.axis, position)
    transform = np.eye(4)
    if joint.joint_type == "prismatic":
        transform[:3, 3] = joint.axis * position
    return transform


def _rotation_log(rotation: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = acos(cosine)
    skew = np.array((rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]))
    if angle < 1e-8:
        return 0.5 * skew
    if abs(angle - np.pi) < 1e-5:
        axis = np.sqrt(np.maximum((np.diag(rotation) + 1.0) * 0.5, 0.0))
        axis[0] = np.copysign(axis[0], skew[0])
        axis[1] = np.copysign(axis[1], skew[1])
        axis[2] = np.copysign(axis[2], skew[2])
        return angle * axis / max(np.linalg.norm(axis), 1e-12)
    return angle * skew / (2.0 * sin(angle))


class AmDp123UrdfKinematics:
    """URDF tree model supporting offline validation and hardware-side IK."""

    def __init__(self, urdf_path: str | Path) -> None:
        """Parse a URDF model.

        Args:
            urdf_path: File-system path to a URDF.
        """
        self.urdf_path = Path(urdf_path).resolve()
        root = ET.parse(self.urdf_path).getroot()
        self.link_names = tuple(link.get("name", "") for link in root.findall("link"))
        if any(not name for name in self.link_names) or len(set(self.link_names)) != len(self.link_names):
            raise AmDp123KinematicsError("URDF link names must be non-empty and unique.")
        joints: list[UrdfJoint] = []
        for element in root.findall("joint"):
            parent, child = element.find("parent"), element.find("child")
            if parent is None or child is None:
                raise AmDp123KinematicsError(f"Joint {element.get('name')} is missing a parent or child.")
            joint_type = element.get("type", "fixed")
            axis_element = element.find("axis")
            axis = _vector(axis_element.get("xyz") if axis_element is not None else None, 3)
            if joint_type in {"revolute", "continuous", "prismatic"}:
                norm = np.linalg.norm(axis)
                if norm < 1e-12:
                    raise AmDp123KinematicsError(f"Joint {element.get('name')} has an invalid axis.")
                axis /= norm
            limit = element.find("limit")
            if joint_type == "continuous":
                lower, upper = -np.inf, np.inf
            elif limit is not None:
                lower, upper = float(limit.get("lower", "-inf")), float(limit.get("upper", "inf"))
            else:
                lower, upper = -np.inf, np.inf
            joints.append(
                UrdfJoint(
                    element.get("name", ""),
                    joint_type,
                    parent.get("link", ""),
                    child.get("link", ""),
                    _origin_transform(element.find("origin")),
                    axis,
                    lower,
                    upper,
                )
            )
        names = [joint.name for joint in joints]
        if any(not name for name in names) or len(set(names)) != len(names):
            raise AmDp123KinematicsError("URDF joint names must be non-empty and unique.")
        roots = set(self.link_names) - {joint.child for joint in joints}
        if len(roots) != 1:
            raise AmDp123KinematicsError(f"URDF must contain one root link, found {sorted(roots)}.")
        self.root_link = roots.pop()
        self.joints = tuple(joints)
        self.joint_by_name = {joint.name: joint for joint in joints}
        self.joint_by_child = {joint.child: joint for joint in joints}
        self.actuated_joint_names = tuple(
            joint.name for joint in joints if joint.joint_type in {"revolute", "continuous", "prismatic"}
        )

    def chain(self, tip_link: str) -> tuple[UrdfJoint, ...]:
        """Return the root-to-tip joint chain of a link.

        Args:
            tip_link: Name of the terminal link.

        Returns:
            Ordered joints from the URDF root to :paramref:`tip_link`.

        Raises:
            AmDp123KinematicsError: If :paramref:`tip_link` is unknown.
        """
        if tip_link not in self.link_names:
            raise AmDp123KinematicsError(f"Unknown URDF link: {tip_link}")
        chain: list[UrdfJoint] = []
        link = tip_link
        while link in self.joint_by_child:
            joint = self.joint_by_child[link]
            chain.append(joint)
            link = joint.parent
        chain.reverse()
        return tuple(chain)

    def _evaluate_chain(
        self, tip_link: str, joint_positions: Mapping[str, float]
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        transform = np.eye(4)
        joint_frames: dict[str, np.ndarray] = {}
        for joint in self.chain(tip_link):
            joint_frame = transform @ joint.origin
            joint_frames[joint.name] = joint_frame
            transform = joint_frame @ _motion_transform(joint, float(joint_positions.get(joint.name, 0.0)))
        return transform, joint_frames

    def forward(self, tip_link: str, joint_positions: Mapping[str, float]) -> np.ndarray:
        """Compute a root-to-link homogeneous transform.

        Args:
            tip_link: Name of the terminal link.
            joint_positions: Joint positions [m or rad, depending on joint type].

        Returns:
            Root-to-link homogeneous transform; translation [m].
        """
        return self._evaluate_chain(tip_link, joint_positions)[0]

    def geometric_jacobian(
        self, tip_link: str, joint_names: Sequence[str], joint_positions: Mapping[str, float]
    ) -> np.ndarray:
        """Compute a root-frame spatial geometric Jacobian.

        Args:
            tip_link: Name of the terminal link.
            joint_names: Ordered joints defining the Jacobian columns.
            joint_positions: Joint positions [m or rad, depending on joint type].

        Returns:
            Spatial Jacobian whose translational rows are [m/m or m/rad] and
            rotational rows are [rad/m or rad/rad], depending on joint type.
        """
        tip_transform, joint_frames = self._evaluate_chain(tip_link, joint_positions)
        tip_position = tip_transform[:3, 3]
        jacobian = np.zeros((6, len(joint_names)))
        chain_names = {joint.name for joint in self.chain(tip_link)}
        for index, name in enumerate(joint_names):
            if name not in chain_names:
                continue
            joint = self.joint_by_name.get(name)
            if joint is None or joint.joint_type == "fixed":
                continue
            frame = joint_frames[name]
            axis_world = frame[:3, :3] @ joint.axis
            if joint.joint_type in {"revolute", "continuous"}:
                jacobian[:3, index] = np.cross(axis_world, tip_position - frame[:3, 3])
                jacobian[3:, index] = axis_world
            elif joint.joint_type == "prismatic":
                jacobian[:3, index] = axis_world
        return jacobian

    def solve(
        self,
        targets: Sequence[IkTarget],
        joint_names: Sequence[str],
        seed: Mapping[str, float],
        *,
        max_iterations: int = 250,
        damping: float = 0.04,
        max_step_rad: float = 0.12,
        position_tolerance_m: float = 1e-4,
        orientation_tolerance_rad: float = 2e-3,
    ) -> IkResult:
        """Solve simultaneous Cartesian targets with joint-limit projection.

        Args:
            targets: Cartesian link targets; translations [m].
            joint_names: Ordered joints to solve.
            seed: Initial positions [m or rad, depending on joint type].
            max_iterations: Maximum solver iterations.
            damping: Damped least-squares regularization factor.
            max_step_rad: Per-iteration revolute-joint step limit [rad].
            position_tolerance_m: Required position accuracy [m].
            orientation_tolerance_rad: Required orientation accuracy [rad].

        Returns:
            Joint solution and convergence diagnostics.

        Raises:
            AmDp123KinematicsError: If the request has no target or contains
                duplicate or unknown joints or a malformed transform.
        """
        if not targets:
            raise AmDp123KinematicsError("IK requires at least one target.")
        names = tuple(joint_names)
        unknown = set(names) - set(self.actuated_joint_names)
        if len(set(names)) != len(names) or unknown:
            raise AmDp123KinematicsError(f"Invalid IK joints: duplicate names or unknown {sorted(unknown)}")
        for target in targets:
            if np.asarray(target.transform).shape != (4, 4):
                raise AmDp123KinematicsError(f"Target for {target.link_name} is not a 4x4 transform.")
        q = np.asarray([float(seed.get(name, 0.0)) for name in names])
        lower = np.asarray([self.joint_by_name[name].lower for name in names])
        upper = np.asarray([self.joint_by_name[name].upper for name in names])
        q = np.clip(q, lower, upper)
        final_position_error = final_orientation_error = np.inf
        for iteration in range(max_iterations + 1):
            positions = dict(zip(names, q, strict=True))
            errors, jacobians = [], []
            position_errors, orientation_errors = [], []
            for target in targets:
                current = self.forward(target.link_name, positions)
                position_error = np.asarray(target.transform)[:3, 3] - current[:3, 3]
                orientation_error = _rotation_log(np.asarray(target.transform)[:3, :3] @ current[:3, :3].T)
                errors.append(
                    np.concatenate(
                        (target.position_weight * position_error, target.orientation_weight * orientation_error)
                    )
                )
                jacobian = self.geometric_jacobian(target.link_name, names, positions)
                jacobian[:3] *= target.position_weight
                jacobian[3:] *= target.orientation_weight
                jacobians.append(jacobian)
                position_errors.append(float(np.linalg.norm(position_error)))
                orientation_errors.append(float(np.linalg.norm(orientation_error)))
            final_position_error, final_orientation_error = max(position_errors), max(orientation_errors)
            if final_position_error <= position_tolerance_m and final_orientation_error <= orientation_tolerance_rad:
                return IkResult(
                    dict(zip(names, q, strict=True)), True, iteration, final_position_error, final_orientation_error
                )
            if iteration == max_iterations:
                break
            error, jacobian = np.concatenate(errors), np.vstack(jacobians)
            regularizer = damping * damping * np.eye(jacobian.shape[0])
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + regularizer, error)
            q = np.clip(q + np.clip(delta, -max_step_rad, max_step_rad), lower, upper)
        return IkResult(
            dict(zip(names, q, strict=True)), False, max_iterations, final_position_error, final_orientation_error
        )


def quaternion_xyzw_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """Convert an XYZW quaternion into a rotation matrix.

    Args:
        quaternion: Quaternion ordered ``(x, y, z, w)``.

    Returns:
        Normalized 3-by-3 rotation matrix.

    Raises:
        AmDp123KinematicsError: If the quaternion norm is zero.
    """
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm((x, y, z, w))
    if norm < 1e-12:
        raise AmDp123KinematicsError("Quaternion norm is zero.")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )


def matrix_to_quaternion_xyzw(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a rotation matrix into a normalized XYZW quaternion.

    Args:
        rotation: Three-dimensional rotation matrix.

    Returns:
        Unit quaternion ordered ``(x, y, z, w)``.
    """
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = np.trace(matrix)
    if trace > 0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array(
            (
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            )
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        indices = ((0, 1, 2), (1, 2, 0), (2, 0, 1))[index]
        i, j, k = indices
        scale = 2.0 * np.sqrt(max(1.0 + matrix[i, i] - matrix[j, j] - matrix[k, k], 1e-12))
        quaternion = np.zeros(4)
        quaternion[i] = 0.25 * scale
        quaternion[j] = (matrix[i, j] + matrix[j, i]) / scale
        quaternion[k] = (matrix[i, k] + matrix[k, i]) / scale
        quaternion[3] = (matrix[k, j] - matrix[j, k]) / scale
    quaternion /= np.linalg.norm(quaternion)
    return tuple(float(value) for value in quaternion)
