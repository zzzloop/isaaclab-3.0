# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task observations and automatic evaluation terms for the AMGG G1 suite."""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch

from ..amgg_g1_workspace import AMGG_G1_TASK_LAYOUTS

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


_GOAL_TOLERANCES_M = {"clutter_transfer": 0.075, "bimanual_reorient": 0.10, "precision_insert": 0.04}
_GOALS = {
    task_slug: (*layout["goal"], _GOAL_TOLERANCES_M[task_slug]) for task_slug, layout in AMGG_G1_TASK_LAYOUTS.items()
}


def _object(env: ManagerBasedRLEnv, name: str = "object") -> RigidObject:
    return env.scene[name]


def _position_env(env: ManagerBasedRLEnv, name: str = "object") -> torch.Tensor:
    return _object(env, name).data.root_pos_w.torch - env.scene.env_origins


def _settled(env: ManagerBasedRLEnv, name: str = "object", max_speed: float = 0.10) -> torch.Tensor:
    velocity = _object(env, name).data.root_vel_w.torch
    linear_ok = torch.linalg.vector_norm(velocity[:, :3], dim=1) < max_speed
    angular_ok = torch.linalg.vector_norm(velocity[:, 3:], dim=1) < 1.0
    return linear_ok & angular_ok


def g1_object_state(env: ManagerBasedRLEnv, object_name: str = "object") -> torch.Tensor:
    """Return object pose and spatial velocity in the environment frame."""
    obj = _object(env, object_name)
    position = obj.data.root_pos_w.torch - env.scene.env_origins
    return torch.cat((position, obj.data.root_quat_w.torch, obj.data.root_vel_w.torch), dim=1)


def g1_rh56dfx_contact_forces(env: ManagerBasedRLEnv, sensor_name: str = "finger_contact") -> torch.Tensor:
    """Return 12 RH56DFX-shaped contact-force magnitudes [N].

    PhysX supplies link contact forces, while the real RH56DFX supplies six
    actuator force channels per hand. This observation aggregates simulated
    link forces into the real motor order. It is a proxy for algorithm
    development, not a calibrated model of the hardware sensor transfer
    function.
    """
    sensor = env.scene[sensor_name]
    force_norm = torch.linalg.vector_norm(sensor.data.net_forces_w.torch, dim=-1)
    body_names = sensor.body_names or []
    channels = []
    for side in ("R", "L"):
        for finger in ("pinky", "ring", "middle", "index", "thumb", "thumb"):
            indices = [index for index, name in enumerate(body_names) if f"{side}_{finger}" in name]
            if indices:
                channels.append(force_norm[:, indices].amax(dim=1))
            else:
                channels.append(torch.zeros(env.num_envs, device=env.device))
    return torch.stack(channels, dim=1)


def g1_task_goal(env: ManagerBasedRLEnv, task_slug: str) -> torch.Tensor:
    """Return the task goal position [m] and primary tolerance [m]."""
    return torch.tensor(_GOALS[task_slug], device=env.device).repeat(env.num_envs, 1)


def g1_task_progress(env: ManagerBasedRLEnv, task_slug: str) -> torch.Tensor:
    """Return a bounded geometric task-progress diagnostic."""
    position = _position_env(env)
    goal = g1_task_goal(env, task_slug)
    distance = torch.linalg.vector_norm(position - goal[:, :3], dim=1)
    scale = {"clutter_transfer": 0.75, "bimanual_reorient": 0.55, "precision_insert": 0.75}[task_slug]
    return (1.0 - distance / scale).clamp(0.0, 1.0).unsqueeze(1)


def clutter_transfer_success(
    env: ManagerBasedRLEnv,
    xy_tolerance: float = 0.09,
    z_tolerance: float = 0.055,
    max_speed: float = 0.15,
) -> torch.Tensor:
    """Check stable placement of the selected block in the target region."""
    position = _position_env(env)
    goal = position.new_tensor(_GOALS["clutter_transfer"][:3])
    xy_ok = torch.linalg.vector_norm(position[:, :2] - goal[:2], dim=1) < xy_tolerance
    z_ok = torch.abs(position[:, 2] - goal[2]) < z_tolerance
    return xy_ok & z_ok & _settled(env, max_speed=max_speed)


def bimanual_reorient_success(
    env: ManagerBasedRLEnv,
    xy_tolerance: float = 0.10,
    z_tolerance: float = 0.075,
    alignment_cosine: float = 0.85,
    max_long_axis_z_component: float = 0.25,
    max_speed: float = 0.20,
) -> torch.Tensor:
    """Check stable bar placement on the supports without requiring a top face."""
    obj = _object(env)
    position = _position_env(env)
    goal = position.new_tensor(_GOALS["bimanual_reorient"][:3])
    xy_ok = torch.linalg.vector_norm(position[:, :2] - goal[:2], dim=1) < xy_tolerance
    z_ok = torch.abs(position[:, 2] - goal[2]) < z_tolerance
    local_x = torch.zeros((env.num_envs, 3), device=env.device)
    local_x[:, 0] = 1.0
    long_axis_w = math_utils.quat_apply(obj.data.root_quat_w.torch, local_x)
    aligned = torch.abs(long_axis_w[:, 0]) > alignment_cosine
    horizontal = torch.abs(long_axis_w[:, 2]) < max_long_axis_z_component
    return xy_ok & z_ok & aligned & horizontal & _settled(env, max_speed=max_speed)


def precision_insert_success(
    env: ManagerBasedRLEnv,
    xy_tolerance: float = 0.04,
    z_tolerance: float = 0.07,
    vertical_axis_cosine: float = 0.88,
    max_speed: float = 0.15,
) -> torch.Tensor:
    """Check stable insertion into the guide socket without requiring a top face."""
    obj = _object(env)
    position = _position_env(env)
    goal = position.new_tensor(_GOALS["precision_insert"][:3])
    xy_ok = torch.linalg.vector_norm(position[:, :2] - goal[:2], dim=1) < xy_tolerance
    z_ok = torch.abs(position[:, 2] - goal[2]) < z_tolerance
    local_z = torch.zeros((env.num_envs, 3), device=env.device)
    local_z[:, 2] = 1.0
    long_axis_w = math_utils.quat_apply(obj.data.root_quat_w.torch, local_z)
    vertical = torch.abs(long_axis_w[:, 2]) > vertical_axis_cosine
    return xy_ok & z_ok & vertical & _settled(env, max_speed=max_speed)


def g1_object_dropped(env: ManagerBasedRLEnv, minimum_height: float = 0.72) -> torch.Tensor:
    """Fail when the task object falls below the recoverable workspace [m]."""
    return _position_env(env)[:, 2] < minimum_height


def g1_object_escaped(
    env: ManagerBasedRLEnv,
    x_bounds: tuple[float, float] = (-0.65, 0.65),
    y_bounds: tuple[float, float] = (0.05, 1.05),
    maximum_height: float = 1.65,
) -> torch.Tensor:
    """Fail when the task object exits the reproducible evaluation workspace."""
    position = _position_env(env)
    return (
        (position[:, 0] < x_bounds[0])
        | (position[:, 0] > x_bounds[1])
        | (position[:, 1] < y_bounds[0])
        | (position[:, 1] > y_bounds[1])
        | (position[:, 2] > maximum_height)
    )


def g1_unsafe_robot_state(env: ManagerBasedRLEnv, max_joint_speed: float = 12.0) -> torch.Tensor:
    """Fail on non-finite robot state or implausibly high joint speed [rad/s]."""
    robot: Articulation = env.scene["robot"]
    joint_pos = robot.data.joint_pos.torch
    joint_vel = robot.data.joint_vel.torch
    finite = torch.isfinite(joint_pos).all(dim=1) & torch.isfinite(joint_vel).all(dim=1)
    return (~finite) | (torch.abs(joint_vel).max(dim=1).values > max_joint_speed)
