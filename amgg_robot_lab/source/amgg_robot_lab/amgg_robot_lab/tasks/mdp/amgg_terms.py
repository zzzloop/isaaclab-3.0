# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observations, metrics, success terms, and failure terms for AMGG tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.managers import SceneEntityCfg

from amgg_robot_lab.contracts import AMGG_FRAMES

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def _object(env: ManagerBasedRLEnv, name: str) -> RigidObject:
    return env.scene[name]


def _position_env(env: ManagerBasedRLEnv, name: str) -> torch.Tensor:
    return _object(env, name).data.root_pos_w.torch - env.scene.env_origins


def _settled(env: ManagerBasedRLEnv, name: str, max_speed: float) -> torch.Tensor:
    velocity = _object(env, name).data.root_vel_w.torch
    return torch.linalg.vector_norm(velocity[:, :3], dim=1) < max_speed


def body_pose_env(
    env: ManagerBasedRLEnv, link_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Return one rigid-body or virtual TCP pose in the environment frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    if link_name in robot.data.body_names:
        body_id = robot.data.body_names.index(link_name)
        pose = robot.data.body_link_pose_w.torch[:, body_id].clone()
    else:
        tcp_frames = {
            AMGG_FRAMES.left_tcp_link: (AMGG_FRAMES.left_tcp_parent_link, AMGG_FRAMES.left_tcp_offset_m),
            AMGG_FRAMES.right_tcp_link: (AMGG_FRAMES.right_tcp_parent_link, AMGG_FRAMES.right_tcp_offset_m),
        }
        if link_name not in tcp_frames:
            raise ValueError(f"Body '{link_name}' is unavailable. Available bodies: {robot.data.body_names}")
        parent_link, offset_m = tcp_frames[link_name]
        if parent_link not in robot.data.body_names:
            raise ValueError(
                f"TCP parent body '{parent_link}' is unavailable for '{link_name}'. "
                f"Available bodies: {robot.data.body_names}"
            )
        parent_id = robot.data.body_names.index(parent_link)
        parent_pose = robot.data.body_link_pose_w.torch[:, parent_id]
        offset = parent_pose.new_tensor(offset_m).expand(env.num_envs, -1)
        position, quaternion = math_utils.combine_frame_transforms(parent_pose[:, :3], parent_pose[:, 3:7], offset)
        pose = torch.cat((position, quaternion), dim=-1)
    pose[:, :3] -= env.scene.env_origins
    return pose


def object_states(env: ManagerBasedRLEnv, object_names: tuple[str, ...]) -> torch.Tensor:
    """Concatenate object position, quaternion, and spatial velocity."""
    values = []
    for name in object_names:
        obj = _object(env, name)
        position = obj.data.root_pos_w.torch - env.scene.env_origins
        values.append(torch.cat((position, obj.data.root_quat_w.torch, obj.data.root_vel_w.torch), dim=1))
    return torch.cat(values, dim=1)


def task_goal(env: ManagerBasedRLEnv, task_slug: str) -> torch.Tensor:
    """Return numeric goal descriptors for policy and dataset consumers."""
    goals = {
        "pick_place": (0.75, -0.22, 0.66, 0.10),
        "bimanual_lift": (0.60, 0.00, 0.90, 0.15),
        "handover": (0.70, -0.28, 0.66, 0.10),
        "sort": (0.76, 0.25, 0.66, 0.76, -0.25, 0.66),
    }
    return torch.tensor(goals[task_slug], device=env.device).repeat(env.num_envs, 1)


def task_progress(env: ManagerBasedRLEnv, task_slug: str) -> torch.Tensor:
    """Return a bounded, monotonic-style task progress metric."""
    if task_slug == "pick_place":
        error = torch.linalg.vector_norm(_position_env(env, "object") - task_goal(env, task_slug)[:, :3], dim=1)
        progress = 1.0 - error / 0.7
    elif task_slug == "bimanual_lift":
        progress = (_position_env(env, "bar")[:, 2] - 0.66) / 0.24
    elif task_slug == "handover":
        position = _position_env(env, "handover_object")
        progress = (0.30 - position[:, 1]) / 0.58
    elif task_slug == "sort":
        goals = task_goal(env, task_slug)
        red_error = torch.linalg.vector_norm(_position_env(env, "red_object") - goals[:, :3], dim=1)
        blue_error = torch.linalg.vector_norm(_position_env(env, "blue_object") - goals[:, 3:], dim=1)
        progress = 1.0 - (red_error + blue_error) / 1.4
    else:
        raise ValueError(f"Unknown AMGG task slug: {task_slug}")
    return progress.clamp(0.0, 1.0).unsqueeze(1)


def pick_place_success(
    env: ManagerBasedRLEnv,
    object_name: str = "object",
    target=(0.75, -0.22, 0.66),
    xy_tolerance: float = 0.09,
    z_tolerance: float = 0.07,
    max_speed: float = 0.12,
) -> torch.Tensor:
    """Check that the pick object is stably inside its target zone."""
    position = _position_env(env, object_name)
    target_tensor = torch.tensor(target, device=env.device)
    xy_ok = torch.linalg.vector_norm(position[:, :2] - target_tensor[:2], dim=1) < xy_tolerance
    z_ok = torch.abs(position[:, 2] - target_tensor[2]) < z_tolerance
    return xy_ok & z_ok & _settled(env, object_name, max_speed)


def bimanual_lift_success(
    env: ManagerBasedRLEnv,
    minimum_height: float = 0.86,
    max_speed: float = 0.16,
    max_tcp_distance: float = 0.16,
) -> torch.Tensor:
    """Check level, stable, two-ended support of the lifted bar."""
    bar = _object(env, "bar")
    position = _position_env(env, "bar")
    quat = bar.data.root_quat_w.torch
    local_z = torch.zeros((env.num_envs, 3), device=env.device)
    local_z[:, 2] = 1.0
    level = math_utils.quat_apply(quat, local_z)[:, 2] > 0.94
    local_left = torch.zeros((env.num_envs, 3), device=env.device)
    local_left[:, 1] = 0.22
    left_end = position + math_utils.quat_apply(quat, local_left)
    right_end = position + math_utils.quat_apply(quat, -local_left)
    left_tcp = body_pose_env(env, "left_tcp_link")[:, :3]
    right_tcp = body_pose_env(env, "right_tcp_link")[:, :3]
    left_ok = torch.linalg.vector_norm(left_tcp - left_end, dim=1) < max_tcp_distance
    right_ok = torch.linalg.vector_norm(right_tcp - right_end, dim=1) < max_tcp_distance
    return (position[:, 2] > minimum_height) & level & left_ok & right_ok & _settled(env, "bar", max_speed)


def handover_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Check stable delivery from the left start area to the right goal."""
    return pick_place_success(
        env,
        object_name="handover_object",
        target=(0.70, -0.28, 0.66),
        xy_tolerance=0.10,
        z_tolerance=0.07,
        max_speed=0.12,
    )


def sort_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Check stable color-matched placement of both sorting objects."""
    red = pick_place_success(env, "red_object", (0.76, 0.25, 0.66), 0.10, 0.07, 0.12)
    blue = pick_place_success(env, "blue_object", (0.76, -0.25, 0.66), 0.10, 0.07, 0.12)
    return red & blue


def any_object_below_height(
    env: ManagerBasedRLEnv, object_names: tuple[str, ...], minimum_height: float = 0.38
) -> torch.Tensor:
    """Fail when any task object falls below the recoverable workspace."""
    failed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in object_names:
        failed |= _position_env(env, name)[:, 2] < minimum_height
    return failed


def any_object_outside_workspace(
    env: ManagerBasedRLEnv,
    object_names: tuple[str, ...],
    x_bounds=(0.15, 1.05),
    y_bounds=(-0.70, 0.70),
    maximum_height: float = 1.35,
) -> torch.Tensor:
    """Fail when any object exits the paper evaluation workspace."""
    failed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in object_names:
        position = _position_env(env, name)
        failed |= (
            (position[:, 0] < x_bounds[0])
            | (position[:, 0] > x_bounds[1])
            | (position[:, 1] < y_bounds[0])
            | (position[:, 1] > y_bounds[1])
            | (position[:, 2] > maximum_height)
        )
    return failed


def unsafe_robot_state(env: ManagerBasedRLEnv, max_joint_speed: float = 8.0) -> torch.Tensor:
    """Fail on non-finite state or implausibly high joint speed."""
    robot: Articulation = env.scene["robot"]
    joint_pos = robot.data.joint_pos.torch
    joint_vel = robot.data.joint_vel.torch
    finite = torch.isfinite(joint_pos).all(dim=1) & torch.isfinite(joint_vel).all(dim=1)
    return (~finite) | (torch.abs(joint_vel).max(dim=1).values > max_joint_speed)
