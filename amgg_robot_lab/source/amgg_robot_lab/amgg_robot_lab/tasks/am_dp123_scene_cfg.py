# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared AM-DP123 scene, observation, reset, and termination configuration.

This module intentionally contains no XR, PICO, Pink IK, or IsaacTeleop imports. Policy
evaluation and teleoperation tasks can therefore share the simulated robot and sensors
without coupling their controller dependencies.
"""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from amgg_robot_lab.assets import get_am_dp123_robot_cfg
from amgg_robot_lab.contracts import AM_DP123_CAMERA_BY_NAME, AM_DP123_FRAMES, AM_DP123_STATE_JOINT_NAMES

AM_DP123_TABLE_TOP_HEIGHT_M = 0.78
"""Workbench top height [m]."""

AM_DP123_TABLE_THICKNESS_M = 0.08
"""Workbench top thickness [m]."""


def _rigid_material(color: tuple[float, float, float]) -> dict:
    return {
        "visual_material": sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.65),
        "physics_material": sim_utils.RigidBodyMaterialCfg(static_friction=1.1, dynamic_friction=0.9),
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
            max_depenetration_velocity=2.0,
        ),
        "collision_props": sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.05),
    }


def _target_marker(size: tuple[float, float, float], color: tuple[float, float, float]) -> sim_utils.CuboidCfg:
    return sim_utils.CuboidCfg(
        size=size,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, opacity=0.42, roughness=0.8),
    )


def _camera(name: str, prim_name: str) -> CameraCfg:
    """Return the camera configuration described by the AM-DP123 camera contract."""
    camera = AM_DP123_CAMERA_BY_NAME[name]
    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{camera.parent_link}/{prim_name}",
        update_period=1.0 / camera.fps,
        height=camera.height_px,
        width=camera.width_px,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=camera.focal_length_mm / 10.0,
            focus_distance=camera.focus_distance_m,
            horizontal_aperture=camera.horizontal_aperture_mm / 10.0,
            clipping_range=camera.clipping_range_m,
        ),
        offset=CameraCfg.OffsetCfg(pos=camera.translation_m, rot=camera.quaternion_xyzw, convention="ros"),
    )


@configclass
class AmDp123SceneCfg(InteractiveSceneCfg):
    """AM-DP123 robot, workbench, lighting, and four-camera rig."""

    robot = get_am_dp123_robot_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.84, 0.0, AM_DP123_TABLE_TOP_HEIGHT_M - AM_DP123_TABLE_THICKNESS_M / 2.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(1.00, 1.20, AM_DP123_TABLE_THICKNESS_M),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.30, 0.32, 0.36), roughness=0.85),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.1, dynamic_friction=0.9),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        ),
    )
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.46, 0.10, AM_DP123_TABLE_TOP_HEIGHT_M + 0.025), rot=(0.0, 0.0, 0.0, 1.0)
        ),
        spawn=sim_utils.CuboidCfg(size=(0.05, 0.05, 0.05), **_rigid_material((0.95, 0.45, 0.05))),
    )
    place_target = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PlaceTarget",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.46, -0.10, AM_DP123_TABLE_TOP_HEIGHT_M + 0.004)),
        spawn=_target_marker((0.09, 0.09, 0.008), (0.10, 0.85, 0.22)),
    )
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        collision_group=-1,
        spawn=sim_utils.GroundPlaneCfg(color=(0.12, 0.12, 0.14), size=(8.0, 8.0)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.82, 0.84, 0.90), intensity=2400.0),
    )
    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.2, -1.2, 2.3)),
        spawn=sim_utils.DistantLightCfg(color=(1.0, 0.92, 0.82), intensity=900.0, angle=35.0),
    )
    head_left = _camera("head_left", "HeadLeftCamera")
    head_right = _camera("head_right", "HeadRightCamera")
    left_wrist = _camera("left_wrist", "LeftWristCamera")
    right_wrist = _camera("right_wrist", "RightWristCamera")


@configclass
class PolicyCfg(ObsGroup):
    """State, wrist, object, and camera observations."""

    actions = ObsTerm(func=base_mdp.last_action)
    robot_joint_pos = ObsTerm(
        func=base_mdp.joint_pos,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(AM_DP123_STATE_JOINT_NAMES), preserve_order=True)
        },
    )
    robot_joint_vel = ObsTerm(
        func=base_mdp.joint_vel,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(AM_DP123_STATE_JOINT_NAMES), preserve_order=True)
        },
    )
    left_wrist_pose = ObsTerm(
        func=base_mdp.body_pose_w,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[AM_DP123_FRAMES.left_wrist_link])},
    )
    right_wrist_pose = ObsTerm(
        func=base_mdp.body_pose_w,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[AM_DP123_FRAMES.right_wrist_link])},
    )
    object_position = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("cube")})
    image_head_left = ObsTerm(
        func=base_mdp.image,
        params={"sensor_cfg": SceneEntityCfg("head_left"), "data_type": "rgb", "normalize": False, "clone": False},
    )
    image_head_right = ObsTerm(
        func=base_mdp.image,
        params={"sensor_cfg": SceneEntityCfg("head_right"), "data_type": "rgb", "normalize": False, "clone": False},
    )
    image_left_wrist = ObsTerm(
        func=base_mdp.image,
        params={"sensor_cfg": SceneEntityCfg("left_wrist"), "data_type": "rgb", "normalize": False, "clone": False},
    )
    image_right_wrist = ObsTerm(
        func=base_mdp.image,
        params={"sensor_cfg": SceneEntityCfg("right_wrist"), "data_type": "rgb", "normalize": False, "clone": False},
    )

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset the robot to its home pose and the cube back onto the table."""

    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
    reset_object = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.01, 0.01], "y": [-0.01, 0.01]},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    cube_dropped = DoneTerm(
        func=base_mdp.root_height_below_minimum,
        params={"minimum_height": 0.45, "asset_cfg": SceneEntityCfg("cube")},
    )


__all__ = [
    "AM_DP123_TABLE_THICKNESS_M",
    "AM_DP123_TABLE_TOP_HEIGHT_M",
    "AmDp123SceneCfg",
    "EventCfg",
    "ObservationsCfg",
    "PolicyCfg",
    "TerminationsCfg",
]
