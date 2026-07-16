# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Four procedural AMGG manipulation environments with automatic evaluation."""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.pink_ik import DampingTaskCfg, FrameTaskCfg, NullSpacePostureTaskCfg, PinkIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab_teleop import IsaacTeleopCfg, XrCfg

from amgg_robot_lab.assets import AMGG_ASSET_DATA_DIR, AMGG_URDF_PATH, get_amgg_robot_cfg
from amgg_robot_lab.contracts import (
    AMGG_CAMERA_BY_NAME,
    AMGG_GRIPPER_JOINT_NAMES,
    AMGG_IK_JOINT_NAMES,
    AMGG_OBSERVED_JOINT_NAMES,
)
from amgg_robot_lab.teleop import build_amgg_pico_pipeline

from . import mdp


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
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.18),
    }


def _target_marker(size: tuple[float, float, float], color: tuple[float, float, float]) -> sim_utils.CuboidCfg:
    return sim_utils.CuboidCfg(
        size=size,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, opacity=0.42, roughness=0.8),
    )


def _camera(name: str, prim_name: str) -> CameraCfg:
    camera = AMGG_CAMERA_BY_NAME[name]
    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{camera.parent_link}/{prim_name}",
        update_period=1.0 / camera.fps,
        height=camera.height_px,
        width=camera.width_px,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=1.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=camera.translation_m, rot=camera.quaternion_xyzw, convention="ros"),
    )


@configclass
class AmggBaseSceneCfg(InteractiveSceneCfg):
    """Robot, workbench, lighting, and synchronized four-camera rig."""

    robot = get_amgg_robot_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.64, 0.0, 0.57), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=sim_utils.CuboidCfg(
            size=(1.00, 1.20, 0.08),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.28, 0.30, 0.34), roughness=0.9),
        ),
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
    head_camera = _camera("head", "HeadCamera")
    left_wrist_camera = _camera("left_wrist", "LeftWristCamera")
    right_wrist_camera = _camera("right_wrist", "RightWristCamera")
    overview_camera = _camera("overview", "OverviewCamera")


@configclass
class AmggPickPlaceSceneCfg(AmggBaseSceneCfg):
    """Single-object placement scene."""

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.48, 0.23, 0.66), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=sim_utils.CuboidCfg(size=(0.065, 0.065, 0.065), **_rigid_material((0.95, 0.34, 0.06))),
    )
    target = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PickTarget",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.75, -0.22, 0.614)),
        spawn=_target_marker((0.20, 0.20, 0.008), (0.10, 0.85, 0.22)),
    )


@configclass
class AmggBimanualLiftSceneCfg(AmggBaseSceneCfg):
    """Two-handed bar-lifting scene."""

    bar = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bar",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.60, 0.0, 0.66), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=sim_utils.CuboidCfg(size=(0.075, 0.50, 0.06), **_rigid_material((0.08, 0.34, 0.95))),
    )
    height_goal = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/LiftHeightGoal",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.60, 0.0, 0.86)),
        spawn=_target_marker((0.16, 0.58, 0.008), (0.12, 0.85, 0.88)),
    )


@configclass
class AmggHandoverSceneCfg(AmggBaseSceneCfg):
    """Cross-workspace handover scene."""

    handover_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HandoverObject",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.50, 0.28, 0.69), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=sim_utils.CylinderCfg(radius=0.035, height=0.14, **_rigid_material((0.95, 0.75, 0.06))),
    )
    handover_target = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/HandoverTarget",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.70, -0.28, 0.614)),
        spawn=_target_marker((0.20, 0.20, 0.008), (0.76, 0.20, 0.90)),
    )


@configclass
class AmggSortSceneCfg(AmggBaseSceneCfg):
    """Two-object color sorting scene."""

    red_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RedObject",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.12, 0.66), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=sim_utils.CuboidCfg(size=(0.06, 0.06, 0.06), **_rigid_material((0.90, 0.08, 0.06))),
    )
    blue_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BlueObject",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, -0.12, 0.66), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=sim_utils.CuboidCfg(size=(0.06, 0.06, 0.06), **_rigid_material((0.06, 0.22, 0.92))),
    )
    red_target = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RedTarget",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.76, 0.25, 0.614)),
        spawn=_target_marker((0.19, 0.19, 0.008), (0.92, 0.10, 0.08)),
    )
    blue_target = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BlueTarget",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.76, -0.25, 0.614)),
        spawn=_target_marker((0.19, 0.19, 0.008), (0.08, 0.20, 0.92)),
    )


@configclass
class ActionsCfg:
    """18-D absolute dual-TCP action plus four trigger-driven fingers."""

    upper_body_ik = mdp.AmggPinkInverseKinematicsActionCfg(
        pink_controlled_joint_names=list(AMGG_IK_JOINT_NAMES),
        hand_joint_names=list(AMGG_GRIPPER_JOINT_NAMES),
        target_eef_link_names={"left_tcp": "left_tcp_link", "right_tcp": "right_tcp_link"},
        asset_name="robot",
        enable_gravity_compensation=True,
        controller=PinkIKControllerCfg(
            urdf_path=str(AMGG_URDF_PATH),
            mesh_path=str(AMGG_ASSET_DATA_DIR / "meshes"),
            articulation_name="robot",
            base_link_name="base_link",
            num_hand_joints=4,
            show_ik_warnings=False,
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                FrameTaskCfg(
                    frame="left_tcp_link", position_cost=8.0, orientation_cost=1.0, lm_damping=10.0, gain=0.45
                ),
                FrameTaskCfg(
                    frame="right_tcp_link", position_cost=8.0, orientation_cost=1.0, lm_damping=10.0, gain=0.45
                ),
                DampingTaskCfg(cost=0.4),
                NullSpacePostureTaskCfg(
                    cost=0.35,
                    lm_damping=1.0,
                    controlled_frames=["left_tcp_link", "right_tcp_link"],
                    controlled_joints=list(AMGG_IK_JOINT_NAMES),
                ),
            ],
            fixed_input_tasks=[],
        ),
    )


@configclass
class AmggBasePolicyCfg(ObsGroup):
    """Shared state, action, TCP, and vision observations."""

    actions = ObsTerm(func=base_mdp.last_action)
    robot_joint_pos = ObsTerm(
        func=base_mdp.joint_pos,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(AMGG_OBSERVED_JOINT_NAMES), preserve_order=True)},
    )
    robot_joint_vel = ObsTerm(
        func=base_mdp.joint_vel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(AMGG_OBSERVED_JOINT_NAMES), preserve_order=True)},
    )
    left_tcp_pose = ObsTerm(func=mdp.body_pose_env, params={"link_name": "left_tcp_link"})
    right_tcp_pose = ObsTerm(func=mdp.body_pose_env, params={"link_name": "right_tcp_link"})
    image_head = ObsTerm(
        func=base_mdp.image,
        params={"sensor_cfg": SceneEntityCfg("head_camera"), "data_type": "rgb", "normalize": False},
    )
    image_left_wrist = ObsTerm(
        func=base_mdp.image,
        params={"sensor_cfg": SceneEntityCfg("left_wrist_camera"), "data_type": "rgb", "normalize": False},
    )
    image_right_wrist = ObsTerm(
        func=base_mdp.image,
        params={"sensor_cfg": SceneEntityCfg("right_wrist_camera"), "data_type": "rgb", "normalize": False},
    )
    image_overview = ObsTerm(
        func=base_mdp.image,
        params={"sensor_cfg": SceneEntityCfg("overview_camera"), "data_type": "rgb", "normalize": False},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class PickPolicyCfg(AmggBasePolicyCfg):
    object_state = ObsTerm(func=mdp.object_states, params={"object_names": ("object",)})
    goal = ObsTerm(func=mdp.task_goal, params={"task_slug": "pick_place"})
    progress = ObsTerm(func=mdp.task_progress, params={"task_slug": "pick_place"})


@configclass
class LiftPolicyCfg(AmggBasePolicyCfg):
    object_state = ObsTerm(func=mdp.object_states, params={"object_names": ("bar",)})
    goal = ObsTerm(func=mdp.task_goal, params={"task_slug": "bimanual_lift"})
    progress = ObsTerm(func=mdp.task_progress, params={"task_slug": "bimanual_lift"})


@configclass
class HandoverPolicyCfg(AmggBasePolicyCfg):
    object_state = ObsTerm(func=mdp.object_states, params={"object_names": ("handover_object",)})
    goal = ObsTerm(func=mdp.task_goal, params={"task_slug": "handover"})
    progress = ObsTerm(func=mdp.task_progress, params={"task_slug": "handover"})


@configclass
class SortPolicyCfg(AmggBasePolicyCfg):
    object_state = ObsTerm(func=mdp.object_states, params={"object_names": ("red_object", "blue_object")})
    goal = ObsTerm(func=mdp.task_goal, params={"task_slug": "sort"})
    progress = ObsTerm(func=mdp.task_progress, params={"task_slug": "sort"})


@configclass
class PickObservationsCfg:
    policy: PickPolicyCfg = PickPolicyCfg()


@configclass
class LiftObservationsCfg:
    policy: LiftPolicyCfg = LiftPolicyCfg()


@configclass
class HandoverObservationsCfg:
    policy: HandoverPolicyCfg = HandoverPolicyCfg()


@configclass
class SortObservationsCfg:
    policy: SortPolicyCfg = SortPolicyCfg()


@configclass
class BaseEventsCfg:
    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})


def _object_reset(name: str, x_range, y_range) -> EventTerm:
    return EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": x_range, "y": y_range, "yaw": (-0.35, 0.35)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg(name),
        },
    )


@configclass
class PickEventsCfg(BaseEventsCfg):
    reset_object = _object_reset("object", (-0.035, 0.035), (-0.045, 0.045))


@configclass
class LiftEventsCfg(BaseEventsCfg):
    reset_bar = _object_reset("bar", (-0.025, 0.025), (-0.025, 0.025))


@configclass
class HandoverEventsCfg(BaseEventsCfg):
    reset_object = _object_reset("handover_object", (-0.03, 0.03), (-0.035, 0.035))


@configclass
class SortEventsCfg(BaseEventsCfg):
    reset_red = _object_reset("red_object", (-0.025, 0.025), (-0.025, 0.025))
    reset_blue = _object_reset("blue_object", (-0.025, 0.025), (-0.025, 0.025))


@configclass
class PickTerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    dropped = DoneTerm(func=mdp.any_object_below_height, params={"object_names": ("object",)})
    escaped = DoneTerm(func=mdp.any_object_outside_workspace, params={"object_names": ("object",)})
    unsafe_robot = DoneTerm(func=mdp.unsafe_robot_state)
    success = DoneTerm(func=mdp.pick_place_success)


@configclass
class LiftTerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    dropped = DoneTerm(func=mdp.any_object_below_height, params={"object_names": ("bar",)})
    escaped = DoneTerm(func=mdp.any_object_outside_workspace, params={"object_names": ("bar",)})
    unsafe_robot = DoneTerm(func=mdp.unsafe_robot_state)
    success = DoneTerm(func=mdp.bimanual_lift_success)


@configclass
class HandoverTerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    dropped = DoneTerm(func=mdp.any_object_below_height, params={"object_names": ("handover_object",)})
    escaped = DoneTerm(func=mdp.any_object_outside_workspace, params={"object_names": ("handover_object",)})
    unsafe_robot = DoneTerm(func=mdp.unsafe_robot_state)
    success = DoneTerm(func=mdp.handover_success)


@configclass
class SortTerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    dropped = DoneTerm(func=mdp.any_object_below_height, params={"object_names": ("red_object", "blue_object")})
    escaped = DoneTerm(func=mdp.any_object_outside_workspace, params={"object_names": ("red_object", "blue_object")})
    unsafe_robot = DoneTerm(func=mdp.unsafe_robot_state)
    success = DoneTerm(func=mdp.sort_success)


@configclass
class AmggBaseEnvCfg(ManagerBasedRLEnvCfg):
    """Shared runtime, PICO, and recorder behavior."""

    actions: ActionsCfg = ActionsCfg()
    commands = None
    rewards = None
    curriculum = None
    idle_action = [
        0.4509800,
        0.2206270,
        0.7691035,
        0.7899079,
        -0.1334403,
        0.0637877,
        0.5951221,
        0.4489726,
        -0.2203487,
        0.7707756,
        0.5868265,
        -0.0542144,
        0.1198586,
        0.7989551,
        1.0,
        1.0,
        1.0,
        1.0,
    ]

    def __post_init__(self):
        self.decimation = 4
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = 4
        self.sim.device = "cuda:0"
        self.viewer.eye = (2.0, -1.6, 1.55)
        self.viewer.lookat = (0.58, 0.0, 0.72)
        self.xr = XrCfg(anchor_pos=(0.0, 0.0, 0.0), anchor_rot=(0.0, 0.0, 0.0, 1.0))
        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=lambda: build_amgg_pico_pipeline()[0],
            sim_device=self.sim.device,
            xr_cfg=self.xr,
        )


@configclass
class AmggPickPlaceEnvCfg(AmggBaseEnvCfg):
    scene: AmggPickPlaceSceneCfg = AmggPickPlaceSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    observations: PickObservationsCfg = PickObservationsCfg()
    events: PickEventsCfg = PickEventsCfg()
    terminations: PickTerminationsCfg = PickTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 35.0


@configclass
class AmggBimanualLiftEnvCfg(AmggBaseEnvCfg):
    scene: AmggBimanualLiftSceneCfg = AmggBimanualLiftSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    observations: LiftObservationsCfg = LiftObservationsCfg()
    events: LiftEventsCfg = LiftEventsCfg()
    terminations: LiftTerminationsCfg = LiftTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 40.0


@configclass
class AmggHandoverEnvCfg(AmggBaseEnvCfg):
    scene: AmggHandoverSceneCfg = AmggHandoverSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    observations: HandoverObservationsCfg = HandoverObservationsCfg()
    events: HandoverEventsCfg = HandoverEventsCfg()
    terminations: HandoverTerminationsCfg = HandoverTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 40.0


@configclass
class AmggSortEnvCfg(AmggBaseEnvCfg):
    scene: AmggSortSceneCfg = AmggSortSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    observations: SortObservationsCfg = SortObservationsCfg()
    events: SortEventsCfg = SortEventsCfg()
    terminations: SortTerminationsCfg = SortTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 55.0
