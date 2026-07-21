# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Three research tasks using Isaac Lab's official Unitree G1 Inspire asset."""

from __future__ import annotations

from pathlib import Path

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.manipulation.pick_place import mdp as official_mdp
from isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_unitree_g1_inspire_hand_env_cfg import (
    ObjectTableSceneCfg,
    PickPlaceG1InspireFTPEnvCfg,
)

from amgg_robot_lab.contracts import (
    AMGG_G1_HAND_MOTOR_SIM_JOINT_NAMES,
    AMGG_G1_SIM_OBSERVATION_JOINT_NAMES,
)

from . import mdp
from .amgg_g1_workspace import (
    AMGG_G1_OBJECT_RESET_X_HALF_RANGE_M,
    AMGG_G1_OBJECT_RESET_Y_HALF_RANGE_M,
    AMGG_G1_TASK_LAYOUTS,
    AMGG_G1_TASK_OBJECT_RESET_RANGES,
    AMGG_G1_TASK_OBJECT_ROTATIONS,
)

_CLUTTER_LAYOUT = AMGG_G1_TASK_LAYOUTS["clutter_transfer"]
_RANDOM_CLUTTER_LAYOUT = AMGG_G1_TASK_LAYOUTS["random_clutter_transfer"]
_BIMANUAL_LAYOUT = AMGG_G1_TASK_LAYOUTS["bimanual_reorient"]
_PRECISION_LAYOUT = AMGG_G1_TASK_LAYOUTS["precision_insert"]
_RANDOM_PRECISION_LAYOUT = AMGG_G1_TASK_LAYOUTS["random_precision_insert"]
_BUCKET_LAYOUT = AMGG_G1_TASK_LAYOUTS["random_cube_bucket"]
_BUCKET_URDF_PATH = (
    Path(__file__).resolve().parents[1] / "assets" / "data" / "objects" / "bucket" / "bucket.urdf"
)


def _dynamic_material(color: tuple[float, float, float], mass: float = 0.25) -> dict:
    return {
        "visual_material": sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.65),
        "physics_material": sim_utils.RigidBodyMaterialCfg(
            static_friction=1.15,
            dynamic_friction=0.9,
            restitution=0.0,
        ),
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.08,
            angular_damping=0.15,
            max_linear_velocity=3.0,
            max_angular_velocity=720.0,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=8,
            max_depenetration_velocity=0.25,
            max_contact_impulse=0.5,
            enable_gyroscopic_forces=True,
        ),
        "collision_props": sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        "mass_props": sim_utils.MassPropertiesCfg(mass=mass),
    }


def _static_box(
    prim_path: str,
    position: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[float, float, float],
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim_path,
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.8),
        ),
    )


def _goal_marker(position: tuple[float, float, float], size: tuple[float, float, float], color) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/GoalMarker",
        init_state=AssetBaseCfg.InitialStateCfg(pos=position),
        spawn=sim_utils.CuboidCfg(
            size=size,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, opacity=0.42, roughness=0.8),
        ),
    )


def _collision_box(
    prim_path: str,
    position: tuple[float, float, float],
    size: tuple[float, float, float],
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim_path,
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.72, 0.22), opacity=0.0),
        ),
    )


def _camera(prim_path: str, position, rotation) -> CameraCfg:
    return CameraCfg(
        prim_path=prim_path,
        update_period=1.0 / 30.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.15,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 4.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=position, rot=rotation, convention="ros"),
    )


@configclass
class AmggG1BaseSceneCfg(ObjectTableSceneCfg):
    """Official fixed-base G1 with a clean table, RGB cameras, and finger contacts."""

    # Replace the inherited warehouse packing-table USD.  Its trays and crates
    # occlude task objects and make controlled visual-domain studies difficult.
    packing_table = _static_box(
        "{ENV_REGEX_NS}/WorkTable",
        (0.0, 0.55, 0.955),
        (1.10, 0.82, 0.09),
        (0.34, 0.37, 0.42),
    )
    table_leg_front_left = _static_box(
        "{ENV_REGEX_NS}/TableLegFrontLeft",
        (-0.48, 0.22, 0.455),
        (0.06, 0.06, 0.91),
        (0.20, 0.22, 0.25),
    )
    table_leg_front_right = _static_box(
        "{ENV_REGEX_NS}/TableLegFrontRight",
        (0.48, 0.22, 0.455),
        (0.06, 0.06, 0.91),
        (0.20, 0.22, 0.25),
    )
    table_leg_back_left = _static_box(
        "{ENV_REGEX_NS}/TableLegBackLeft",
        (-0.48, 0.88, 0.455),
        (0.06, 0.06, 0.91),
        (0.20, 0.22, 0.25),
    )
    table_leg_back_right = _static_box(
        "{ENV_REGEX_NS}/TableLegBackRight",
        (0.48, 0.88, 0.455),
        (0.06, 0.06, 0.91),
        (0.20, 0.22, 0.25),
    )

    front_camera = _camera(
        "{ENV_REGEX_NS}/FrontCamera",
        (0.0, 0.12, 1.67675),
        (0.9801, 0.0, 0.0, -0.19848),
    )
    overview_camera = _camera(
        "{ENV_REGEX_NS}/OverviewCamera",
        (1.05, 1.45, 1.55),
        (-0.33545082, -0.74522487, 0.52550691, 0.23654836),
    )
    finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/[LR]_.*",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
    )


@configclass
class AmggG1ClutterTransferSceneCfg(AmggG1BaseSceneCfg):
    """Randomized target block among movable visual distractors."""

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_CLUTTER_LAYOUT["object"], rot=AMGG_G1_TASK_OBJECT_ROTATIONS["clutter_transfer"]
        ),
        spawn=sim_utils.CuboidCfg(size=(0.070, 0.070, 0.070), **_dynamic_material((0.95, 0.28, 0.04))),
    )
    distractor_a = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/DistractorA",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_CLUTTER_LAYOUT["distractor_a"]),
        spawn=sim_utils.CuboidCfg(size=(0.065, 0.065, 0.070), **_dynamic_material((0.08, 0.35, 0.92))),
    )
    distractor_b = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/DistractorB",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_CLUTTER_LAYOUT["distractor_b"]),
        spawn=sim_utils.CylinderCfg(radius=0.035, height=0.080, **_dynamic_material((0.88, 0.78, 0.08))),
    )
    goal = _goal_marker(_CLUTTER_LAYOUT["goal_marker"], (0.17, 0.17, 0.008), (0.08, 0.85, 0.22))


@configclass
class AmggG1RandomClutterTransferSceneCfg(AmggG1BaseSceneCfg):
    """Randomized orange-target transfer among multiple visual distractors."""

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_RANDOM_CLUTTER_LAYOUT["object"], rot=AMGG_G1_TASK_OBJECT_ROTATIONS["random_clutter_transfer"]
        ),
        spawn=sim_utils.CuboidCfg(size=(0.070, 0.070, 0.070), **_dynamic_material((0.95, 0.28, 0.04))),
    )
    distractor_a = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/DistractorA",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_RANDOM_CLUTTER_LAYOUT["distractor_a"]),
        spawn=sim_utils.CuboidCfg(size=(0.060, 0.060, 0.065), **_dynamic_material((0.08, 0.35, 0.92))),
    )
    distractor_b = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/DistractorB",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_RANDOM_CLUTTER_LAYOUT["distractor_b"]),
        spawn=sim_utils.CylinderCfg(radius=0.032, height=0.075, **_dynamic_material((0.88, 0.78, 0.08))),
    )
    distractor_c = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/DistractorC",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_RANDOM_CLUTTER_LAYOUT["distractor_c"]),
        spawn=sim_utils.CuboidCfg(size=(0.055, 0.070, 0.060), **_dynamic_material((0.58, 0.16, 0.86))),
    )
    distractor_d = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/DistractorD",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_RANDOM_CLUTTER_LAYOUT["distractor_d"]),
        spawn=sim_utils.CuboidCfg(size=(0.050, 0.050, 0.080), **_dynamic_material((0.10, 0.72, 0.72))),
    )
    goal = _goal_marker(_RANDOM_CLUTTER_LAYOUT["goal_marker"], (0.17, 0.17, 0.008), (0.08, 0.85, 0.22))


@configclass
class AmggG1RandomCubeBucketSceneCfg(AmggG1BaseSceneCfg):
    """Randomized cube placement into a reachable bucket mesh with distractors."""

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_BUCKET_LAYOUT["object"], rot=AMGG_G1_TASK_OBJECT_ROTATIONS["random_cube_bucket"]
        ),
        spawn=sim_utils.CuboidCfg(size=(0.045, 0.045, 0.045), **_dynamic_material((0.95, 0.28, 0.04))),
    )
    distractor_a = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BucketDistractorA",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_BUCKET_LAYOUT["distractor_a"]),
        spawn=sim_utils.CuboidCfg(size=(0.045, 0.045, 0.045), **_dynamic_material((0.08, 0.35, 0.92))),
    )
    distractor_b = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BucketDistractorB",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_BUCKET_LAYOUT["distractor_b"]),
        spawn=sim_utils.CuboidCfg(size=(0.045, 0.045, 0.045), **_dynamic_material((0.88, 0.78, 0.08))),
    )
    distractor_c = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BucketDistractorC",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_BUCKET_LAYOUT["distractor_c"]),
        spawn=sim_utils.CuboidCfg(size=(0.045, 0.045, 0.045), **_dynamic_material((0.58, 0.16, 0.86))),
    )
    bucket = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Bucket",
        init_state=AssetBaseCfg.InitialStateCfg(pos=_BUCKET_LAYOUT["bucket"]),
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(_BUCKET_URDF_PATH),
            fix_base=True,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )
    bucket_collision_left = _collision_box(
        "{ENV_REGEX_NS}/BucketCollisionLeft", _BUCKET_LAYOUT["bucket_collision_left"], (0.018, 0.120, 0.10)
    )
    bucket_collision_right = _collision_box(
        "{ENV_REGEX_NS}/BucketCollisionRight", _BUCKET_LAYOUT["bucket_collision_right"], (0.018, 0.120, 0.10)
    )
    bucket_collision_near = _collision_box(
        "{ENV_REGEX_NS}/BucketCollisionNear", _BUCKET_LAYOUT["bucket_collision_near"], (0.160, 0.018, 0.10)
    )
    bucket_collision_far = _collision_box(
        "{ENV_REGEX_NS}/BucketCollisionFar", _BUCKET_LAYOUT["bucket_collision_far"], (0.160, 0.018, 0.10)
    )
    goal = _goal_marker(_BUCKET_LAYOUT["goal_marker"], (0.105, 0.075, 0.006), (0.10, 0.85, 0.18))


@configclass
class AmggG1BimanualReorientSceneCfg(AmggG1BaseSceneCfg):
    """Long-object reorientation onto a two-support fixture."""

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_BIMANUAL_LAYOUT["object"], rot=AMGG_G1_TASK_OBJECT_ROTATIONS["bimanual_reorient"]
        ),
        spawn=sim_utils.CuboidCfg(size=(0.30, 0.055, 0.055), **_dynamic_material((0.06, 0.32, 0.95), 0.32)),
    )
    left_support = _static_box(
        "{ENV_REGEX_NS}/LeftSupport", _BIMANUAL_LAYOUT["left_support"], (0.070, 0.080, 0.10), (0.16, 0.70, 0.78)
    )
    right_support = _static_box(
        "{ENV_REGEX_NS}/RightSupport",
        _BIMANUAL_LAYOUT["right_support"],
        (0.070, 0.080, 0.10),
        (0.16, 0.70, 0.78),
    )
    goal = _goal_marker(_BIMANUAL_LAYOUT["goal_marker"], (0.34, 0.075, 0.008), (0.10, 0.85, 0.88))


@configclass
class AmggG1PrecisionInsertSceneCfg(AmggG1BaseSceneCfg):
    """Upright keyed block insertion through a tight four-wall guide."""

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_PRECISION_LAYOUT["object"], rot=AMGG_G1_TASK_OBJECT_ROTATIONS["precision_insert"]
        ),
        spawn=sim_utils.CuboidCfg(size=(0.045, 0.045, 0.14), **_dynamic_material((0.96, 0.72, 0.05), 0.20)),
    )
    guide_left = _static_box(
        "{ENV_REGEX_NS}/GuideLeft", _PRECISION_LAYOUT["guide_left"], (0.025, 0.090, 0.11), (0.55, 0.12, 0.75)
    )
    guide_right = _static_box(
        "{ENV_REGEX_NS}/GuideRight",
        _PRECISION_LAYOUT["guide_right"],
        (0.025, 0.090, 0.11),
        (0.55, 0.12, 0.75),
    )
    guide_near = _static_box(
        "{ENV_REGEX_NS}/GuideNear", _PRECISION_LAYOUT["guide_near"], (0.060, 0.018, 0.11), (0.55, 0.12, 0.75)
    )
    guide_far = _static_box(
        "{ENV_REGEX_NS}/GuideFar", _PRECISION_LAYOUT["guide_far"], (0.060, 0.018, 0.11), (0.55, 0.12, 0.75)
    )
    goal = _goal_marker(_PRECISION_LAYOUT["goal_marker"], (0.055, 0.055, 0.008), (0.72, 0.18, 0.90))


@configclass
class AmggG1RandomPrecisionInsertSceneCfg(AmggG1BaseSceneCfg):
    """Precision insertion with a wider randomized key start pose."""

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_RANDOM_PRECISION_LAYOUT["object"], rot=AMGG_G1_TASK_OBJECT_ROTATIONS["random_precision_insert"]
        ),
        spawn=sim_utils.CuboidCfg(size=(0.045, 0.045, 0.14), **_dynamic_material((0.96, 0.72, 0.05), 0.20)),
    )
    guide_left = _static_box(
        "{ENV_REGEX_NS}/GuideLeft",
        _RANDOM_PRECISION_LAYOUT["guide_left"],
        (0.025, 0.090, 0.11),
        (0.55, 0.12, 0.75),
    )
    guide_right = _static_box(
        "{ENV_REGEX_NS}/GuideRight",
        _RANDOM_PRECISION_LAYOUT["guide_right"],
        (0.025, 0.090, 0.11),
        (0.55, 0.12, 0.75),
    )
    guide_near = _static_box(
        "{ENV_REGEX_NS}/GuideNear",
        _RANDOM_PRECISION_LAYOUT["guide_near"],
        (0.060, 0.018, 0.11),
        (0.55, 0.12, 0.75),
    )
    guide_far = _static_box(
        "{ENV_REGEX_NS}/GuideFar",
        _RANDOM_PRECISION_LAYOUT["guide_far"],
        (0.060, 0.018, 0.11),
        (0.55, 0.12, 0.75),
    )
    goal = _goal_marker(_RANDOM_PRECISION_LAYOUT["goal_marker"], (0.055, 0.055, 0.008), (0.72, 0.18, 0.90))


@configclass
class AmggG1PolicyCfg(ObsGroup):
    """Shared state, task diagnostics, force proxy, and RGB observations."""

    actions = ObsTerm(func=base_mdp.last_action)
    robot_joint_pos = ObsTerm(
        func=base_mdp.joint_pos,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=list(AMGG_G1_SIM_OBSERVATION_JOINT_NAMES), preserve_order=True
            )
        },
    )
    robot_joint_vel = ObsTerm(
        func=base_mdp.joint_vel,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=list(AMGG_G1_SIM_OBSERVATION_JOINT_NAMES), preserve_order=True
            )
        },
    )
    rh56dfx_motor_proxy = ObsTerm(
        func=base_mdp.joint_pos,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=list(AMGG_G1_HAND_MOTOR_SIM_JOINT_NAMES), preserve_order=True
            )
        },
    )
    tactile = ObsTerm(func=mdp.g1_rh56dfx_contact_forces)
    left_eef_pos = ObsTerm(func=official_mdp.get_eef_pos, params={"link_name": "left_wrist_yaw_link"})
    left_eef_quat = ObsTerm(func=official_mdp.get_eef_quat, params={"link_name": "left_wrist_yaw_link"})
    right_eef_pos = ObsTerm(func=official_mdp.get_eef_pos, params={"link_name": "right_wrist_yaw_link"})
    right_eef_quat = ObsTerm(func=official_mdp.get_eef_quat, params={"link_name": "right_wrist_yaw_link"})
    object_state = ObsTerm(func=mdp.g1_object_state)
    image_front = ObsTerm(
        func=base_mdp.image,
        params={"sensor_cfg": SceneEntityCfg("front_camera"), "data_type": "rgb", "normalize": False},
    )
    image_overview = ObsTerm(
        func=base_mdp.image,
        params={"sensor_cfg": SceneEntityCfg("overview_camera"), "data_type": "rgb", "normalize": False},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class ClutterPolicyCfg(AmggG1PolicyCfg):
    goal = ObsTerm(func=mdp.g1_task_goal, params={"task_slug": "clutter_transfer"})
    progress = ObsTerm(func=mdp.g1_task_progress, params={"task_slug": "clutter_transfer"})


@configclass
class RandomClutterPolicyCfg(AmggG1PolicyCfg):
    goal = ObsTerm(func=mdp.g1_task_goal, params={"task_slug": "random_clutter_transfer"})
    progress = ObsTerm(func=mdp.g1_task_progress, params={"task_slug": "random_clutter_transfer"})


@configclass
class RandomCubeBucketPolicyCfg(AmggG1PolicyCfg):
    goal = ObsTerm(func=mdp.g1_task_goal, params={"task_slug": "random_cube_bucket"})
    progress = ObsTerm(func=mdp.g1_task_progress, params={"task_slug": "random_cube_bucket"})


@configclass
class BimanualPolicyCfg(AmggG1PolicyCfg):
    goal = ObsTerm(func=mdp.g1_task_goal, params={"task_slug": "bimanual_reorient"})
    progress = ObsTerm(func=mdp.g1_task_progress, params={"task_slug": "bimanual_reorient"})


@configclass
class PrecisionPolicyCfg(AmggG1PolicyCfg):
    goal = ObsTerm(func=mdp.g1_task_goal, params={"task_slug": "precision_insert"})
    progress = ObsTerm(func=mdp.g1_task_progress, params={"task_slug": "precision_insert"})


@configclass
class RandomPrecisionPolicyCfg(AmggG1PolicyCfg):
    goal = ObsTerm(func=mdp.g1_task_goal, params={"task_slug": "random_precision_insert"})
    progress = ObsTerm(func=mdp.g1_task_progress, params={"task_slug": "random_precision_insert"})


@configclass
class ClutterObservationsCfg:
    policy: ClutterPolicyCfg = ClutterPolicyCfg()


@configclass
class RandomClutterObservationsCfg:
    policy: RandomClutterPolicyCfg = RandomClutterPolicyCfg()


@configclass
class RandomCubeBucketObservationsCfg:
    policy: RandomCubeBucketPolicyCfg = RandomCubeBucketPolicyCfg()


@configclass
class BimanualObservationsCfg:
    policy: BimanualPolicyCfg = BimanualPolicyCfg()


@configclass
class PrecisionObservationsCfg:
    policy: PrecisionPolicyCfg = PrecisionPolicyCfg()


@configclass
class RandomPrecisionObservationsCfg:
    policy: RandomPrecisionPolicyCfg = RandomPrecisionPolicyCfg()


@configclass
class AmggG1EventsCfg:
    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
    reset_object = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-AMGG_G1_OBJECT_RESET_X_HALF_RANGE_M, AMGG_G1_OBJECT_RESET_X_HALF_RANGE_M),
                "y": (-AMGG_G1_OBJECT_RESET_Y_HALF_RANGE_M, AMGG_G1_OBJECT_RESET_Y_HALF_RANGE_M),
                "yaw": (-0.25, 0.25),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class RandomClutterEventsCfg(AmggG1EventsCfg):
    """Randomize target and distractor poses for clutter generalization."""

    reset_object = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": AMGG_G1_TASK_OBJECT_RESET_RANGES["random_clutter_transfer"],
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    reset_distractor_a = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.030, 0.030), "y": (-0.025, 0.025), "yaw": (-0.80, 0.80)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("distractor_a"),
        },
    )
    reset_distractor_b = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.025, 0.025), "y": (-0.025, 0.025), "yaw": (-0.80, 0.80)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("distractor_b"),
        },
    )
    reset_distractor_c = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.030, 0.030), "y": (-0.020, 0.020), "yaw": (-0.80, 0.80)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("distractor_c"),
        },
    )
    reset_distractor_d = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.025, 0.025), "y": (-0.020, 0.020), "yaw": (-0.80, 0.80)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("distractor_d"),
        },
    )


@configclass
class RandomCubeBucketEventsCfg(AmggG1EventsCfg):
    """Randomize the cube start pose over a wider reachable tabletop patch."""

    reset_object = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": AMGG_G1_TASK_OBJECT_RESET_RANGES["random_cube_bucket"],
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    reset_distractor_a = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.025, 0.025), "y": (-0.025, 0.025), "yaw": (-0.60, 0.60)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("distractor_a"),
        },
    )
    reset_distractor_b = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.025, 0.025), "y": (-0.020, 0.020), "yaw": (-0.60, 0.60)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("distractor_b"),
        },
    )
    reset_distractor_c = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.020, 0.020), "y": (-0.020, 0.020), "yaw": (-0.60, 0.60)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("distractor_c"),
        },
    )


@configclass
class BimanualEventsCfg(AmggG1EventsCfg):
    """Reset the longitudinal bar without sweeping it into either hand."""

    reset_object = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": AMGG_G1_TASK_OBJECT_RESET_RANGES["bimanual_reorient"],
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class PrecisionEventsCfg(AmggG1EventsCfg):
    """Keep the key outside the hands and the guide during reset."""

    reset_object = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": AMGG_G1_TASK_OBJECT_RESET_RANGES["precision_insert"],
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class RandomPrecisionEventsCfg(AmggG1EventsCfg):
    """Randomize the key start pose for precision-insertion generalization."""

    reset_object = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": AMGG_G1_TASK_OBJECT_RESET_RANGES["random_precision_insert"],
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class AmggG1FailureTermsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    dropped = DoneTerm(func=mdp.g1_object_dropped)
    escaped = DoneTerm(func=mdp.g1_object_escaped)
    unsafe_robot = DoneTerm(func=mdp.g1_unsafe_robot_state)


@configclass
class ClutterTerminationsCfg(AmggG1FailureTermsCfg):
    success = DoneTerm(func=mdp.clutter_transfer_success)


@configclass
class RandomClutterTerminationsCfg(AmggG1FailureTermsCfg):
    success = DoneTerm(func=mdp.random_clutter_transfer_success)


@configclass
class RandomCubeBucketTerminationsCfg(AmggG1FailureTermsCfg):
    success = DoneTerm(func=mdp.random_cube_bucket_success)


@configclass
class BimanualTerminationsCfg(AmggG1FailureTermsCfg):
    success = DoneTerm(func=mdp.bimanual_reorient_success)


@configclass
class PrecisionTerminationsCfg(AmggG1FailureTermsCfg):
    success = DoneTerm(func=mdp.precision_insert_success)


@configclass
class RandomPrecisionTerminationsCfg(AmggG1FailureTermsCfg):
    success = DoneTerm(func=mdp.random_precision_insert_success)


@configclass
class AmggG1BaseEnvCfg(PickPlaceG1InspireFTPEnvCfg):
    """Shared official G1/Pink/PICO configuration for the research suite."""

    events: AmggG1EventsCfg = AmggG1EventsCfg()

    def __post_init__(self):
        super().__post_init__()
        # The task and the robot hands are both visible from this front
        # three-quarter view.  Sensor cameras remain independent of the viewer.
        self.viewer.eye = (1.45, 1.65, 1.55)
        self.viewer.lookat = (0.0, 0.55, 1.05)
        # The data contract is 30 Hz: 120 Hz physics / 4 control substeps.
        # Rendering once per environment step also removes redundant camera
        # renders inherited from the official 20 Hz demonstration config.
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.seed = 42
        self.num_rerenders_on_reset = 2


@configclass
class AmggG1ClutterTransferEnvCfg(AmggG1BaseEnvCfg):
    scene: AmggG1ClutterTransferSceneCfg = AmggG1ClutterTransferSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )
    observations: ClutterObservationsCfg = ClutterObservationsCfg()
    terminations: ClutterTerminationsCfg = ClutterTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 45.0


@configclass
class AmggG1RandomClutterTransferEnvCfg(AmggG1BaseEnvCfg):
    scene: AmggG1RandomClutterTransferSceneCfg = AmggG1RandomClutterTransferSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )
    observations: RandomClutterObservationsCfg = RandomClutterObservationsCfg()
    events: RandomClutterEventsCfg = RandomClutterEventsCfg()
    terminations: RandomClutterTerminationsCfg = RandomClutterTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 50.0


@configclass
class AmggG1RandomCubeBucketEnvCfg(AmggG1BaseEnvCfg):
    scene: AmggG1RandomCubeBucketSceneCfg = AmggG1RandomCubeBucketSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )
    observations: RandomCubeBucketObservationsCfg = RandomCubeBucketObservationsCfg()
    events: RandomCubeBucketEventsCfg = RandomCubeBucketEventsCfg()
    terminations: RandomCubeBucketTerminationsCfg = RandomCubeBucketTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 45.0


@configclass
class AmggG1BimanualReorientEnvCfg(AmggG1BaseEnvCfg):
    scene: AmggG1BimanualReorientSceneCfg = AmggG1BimanualReorientSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )
    observations: BimanualObservationsCfg = BimanualObservationsCfg()
    events: BimanualEventsCfg = BimanualEventsCfg()
    terminations: BimanualTerminationsCfg = BimanualTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # The official absolute targets place the wrists at x=+/-0.1487 m,
        # which overlaps this task's support fixture. Apply a deterministic
        # task-local outward offset in the action term so PICO and replay agree.
        self.actions.pink_ik_cfg.class_type = mdp.AmggG1BimanualPinkInverseKinematicsAction
        self.episode_length_s = 55.0


@configclass
class AmggG1PrecisionInsertEnvCfg(AmggG1BaseEnvCfg):
    scene: AmggG1PrecisionInsertSceneCfg = AmggG1PrecisionInsertSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )
    observations: PrecisionObservationsCfg = PrecisionObservationsCfg()
    events: PrecisionEventsCfg = PrecisionEventsCfg()
    terminations: PrecisionTerminationsCfg = PrecisionTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 50.0


@configclass
class AmggG1RandomPrecisionInsertEnvCfg(AmggG1BaseEnvCfg):
    scene: AmggG1RandomPrecisionInsertSceneCfg = AmggG1RandomPrecisionInsertSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )
    observations: RandomPrecisionObservationsCfg = RandomPrecisionObservationsCfg()
    events: RandomPrecisionEventsCfg = RandomPrecisionEventsCfg()
    terminations: RandomPrecisionTerminationsCfg = RandomPrecisionTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 55.0


class _AmggG1XrTimingMixin:
    """Use a 60 Hz control/render loop for lower-latency XR teleoperation."""

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 2
        self.sim.render_interval = 2


@configclass
class AmggG1ClutterTransferXrEnvCfg(_AmggG1XrTimingMixin, AmggG1ClutterTransferEnvCfg):
    """Low-latency XR variant of the clutter-transfer task."""


@configclass
class AmggG1RandomClutterTransferXrEnvCfg(_AmggG1XrTimingMixin, AmggG1RandomClutterTransferEnvCfg):
    """Low-latency XR variant of the randomized clutter-transfer task."""


@configclass
class AmggG1RandomCubeBucketXrEnvCfg(_AmggG1XrTimingMixin, AmggG1RandomCubeBucketEnvCfg):
    """Low-latency XR variant of the randomized cube-to-bucket task."""


@configclass
class AmggG1BimanualReorientXrEnvCfg(_AmggG1XrTimingMixin, AmggG1BimanualReorientEnvCfg):
    """Low-latency XR variant of the bimanual-reorientation task."""


@configclass
class AmggG1PrecisionInsertXrEnvCfg(_AmggG1XrTimingMixin, AmggG1PrecisionInsertEnvCfg):
    """Low-latency XR variant of the precision-insertion task."""


@configclass
class AmggG1RandomPrecisionInsertXrEnvCfg(_AmggG1XrTimingMixin, AmggG1RandomPrecisionInsertEnvCfg):
    """Low-latency XR variant of the randomized precision-insertion task."""
