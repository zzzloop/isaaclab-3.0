# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AM-DP123 PICO XR teleoperation environment."""

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
from isaaclab.visualizers import VisualizerCfg
from isaaclab_teleop import IsaacTeleopCfg, XrCameraFeedCfg, XrCameraFeedLayoutCfg, XrCfg

from amgg_robot_lab.assets import AM_DP123_ASSET_DATA_DIR, AM_DP123_URDF_PATH, get_am_dp123_robot_cfg
from amgg_robot_lab.contracts import (
    AM_DP123_CAMERA_BY_NAME,
    AM_DP123_FRAMES,
    AM_DP123_HAND_JOINT_NAMES,
    AM_DP123_IK_JOINT_NAMES,
    AM_DP123_STATE_JOINT_NAMES,
    AM_DP123_STEREO_CAMERA_NAMES,
)
from amgg_robot_lab.teleop import AM_DP123_IDLE_ACTION, build_am_dp123_pico_pipeline

from . import mdp

AM_DP123_TABLE_TOP_HEIGHT_M = 0.78
AM_DP123_TABLE_THICKNESS_M = 0.08


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
    """AM-DP123 robot, workbench, lighting, and the four-camera rig."""

    robot = get_am_dp123_robot_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
    # The tabletop is close to a standard standing workbench height and leaves
    # the home wrists roughly 0.17 m above the manipulation surface.
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
class ActionsCfg:
    """18-D absolute dual-wrist action plus four trigger-driven finger joints."""

    upper_body_ik = mdp.AmDp123PinkInverseKinematicsActionCfg(
        pink_controlled_joint_names=list(AM_DP123_IK_JOINT_NAMES),
        hand_joint_names=list(AM_DP123_HAND_JOINT_NAMES),
        target_eef_link_names={
            "left_wrist": AM_DP123_FRAMES.left_wrist_link,
            "right_wrist": AM_DP123_FRAMES.right_wrist_link,
        },
        asset_name="robot",
        enable_gravity_compensation=True,
        controller=PinkIKControllerCfg(
            urdf_path=str(AM_DP123_URDF_PATH),
            # Pinocchio resolves ``package://AM-DP123/meshes/...`` by looking for a
            # directory named after the package inside each search path, so the
            # search path is the data directory that contains ``AM-DP123/``.
            mesh_path=str(AM_DP123_ASSET_DATA_DIR.parent),
            articulation_name="robot",
            base_link_name=AM_DP123_FRAMES.base_link,
            num_hand_joints=len(AM_DP123_HAND_JOINT_NAMES),
            show_ik_warnings=False,
            # The AM-DP123 waist limits are narrow, so the solver keeps its last
            # iterate instead of aborting. Re-calibrate on the server before
            # switching this to ``True``.
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                FrameTaskCfg(
                    frame=AM_DP123_FRAMES.left_wrist_link,
                    position_cost=8.0,
                    orientation_cost=1.0,
                    lm_damping=10.0,
                    gain=0.45,
                ),
                FrameTaskCfg(
                    frame=AM_DP123_FRAMES.right_wrist_link,
                    position_cost=8.0,
                    orientation_cost=1.0,
                    lm_damping=10.0,
                    gain=0.45,
                ),
                DampingTaskCfg(cost=0.4),
                NullSpacePostureTaskCfg(
                    cost=0.35,
                    lm_damping=1.0,
                    controlled_frames=[AM_DP123_FRAMES.left_wrist_link, AM_DP123_FRAMES.right_wrist_link],
                    controlled_joints=list(AM_DP123_IK_JOINT_NAMES),
                ),
            ],
            fixed_input_tasks=[],
        ),
    )


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


@configclass
class AmDp123PicoXrEnvCfg(ManagerBasedRLEnvCfg):
    """PICO XR teleoperation environment for the AM-DP123 mobile manipulator."""

    scene: AmDp123SceneCfg = AmDp123SceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    idle_action = list(AM_DP123_IDLE_ACTION)

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 60.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = 2
        self.sim.device = "cuda:0"
        self.num_rerenders_on_reset = 3
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(2.0, -1.6, 1.8), lookat=(0.55, 0.0, 0.9))
        self.xr = XrCfg(anchor_pos=(0.0, 0.0, 0.0), anchor_rot=(0.0, 0.0, 0.0, 1.0))
        pipeline, retargeters = build_am_dp123_pico_pipeline()
        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=lambda: pipeline,
            retargeters_to_tune=lambda: retargeters,
            sim_device=self.sim.device,
            xr_cfg=self.xr,
            xr_camera_feeds=[
                XrCameraFeedCfg(camera_name=name, label=name.replace("_", " ").title())
                for name in AM_DP123_STEREO_CAMERA_NAMES
            ],
            xr_camera_feed_layout=XrCameraFeedLayoutCfg(mode="horizontal", placement="head_locked"),
        )
        self.image_obs_list = list(AM_DP123_STEREO_CAMERA_NAMES)
