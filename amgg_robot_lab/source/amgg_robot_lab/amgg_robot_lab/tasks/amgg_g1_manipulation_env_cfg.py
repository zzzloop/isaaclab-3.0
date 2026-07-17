# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Three research tasks using Isaac Lab's official Unitree G1 Inspire asset."""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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
            solver_position_iteration_count=24,
            solver_velocity_iteration_count=8,
            max_depenetration_velocity=0.5,
            max_contact_impulse=2.0,
            enable_gyroscopic_forces=True,
        ),
        "collision_props": sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        "mass_props": sim_utils.MassPropertiesCfg(mass=mass),
    }


def _contact_stable_robot() -> ArticulationCfg:
    """Return the official G1 with bounded drives and stronger contact solving."""
    robot = ObjectTableSceneCfg.robot.copy()
    robot.spawn.rigid_props.max_linear_velocity = 3.0
    robot.spawn.rigid_props.max_angular_velocity = 360.0
    robot.spawn.rigid_props.max_depenetration_velocity = 0.5
    robot.spawn.collision_props = sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0)
    robot.spawn.articulation_props.solver_position_iteration_count = 16
    robot.spawn.articulation_props.solver_velocity_iteration_count = 8

    # ``velocity_limit`` is intentionally ignored for implicit actuators. Use
    # the PhysX solver limit so XR target jumps cannot tunnel through props.
    arms = robot.actuators["arms"]
    arms.effort_limit = None
    arms.effort_limit_sim = 80.0
    arms.velocity_limit = None
    arms.velocity_limit_sim = 4.0
    hands = robot.actuators["hands"]
    hands.velocity_limit = None
    hands.velocity_limit_sim = 5.0
    return robot


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

    robot: ArticulationCfg = _contact_stable_robot()

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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.24, 0.43, 1.035)),
        spawn=sim_utils.CuboidCfg(size=(0.070, 0.070, 0.070), **_dynamic_material((0.95, 0.28, 0.04))),
    )
    distractor_a = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/DistractorA",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.05, 0.46, 1.035)),
        spawn=sim_utils.CuboidCfg(size=(0.065, 0.065, 0.070), **_dynamic_material((0.08, 0.35, 0.92))),
    )
    distractor_b = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/DistractorB",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.10, 0.40, 1.04)),
        spawn=sim_utils.CylinderCfg(radius=0.035, height=0.080, **_dynamic_material((0.88, 0.78, 0.08))),
    )
    goal = _goal_marker((0.24, 0.62, 1.003), (0.17, 0.17, 0.008), (0.08, 0.85, 0.22))


@configclass
class AmggG1BimanualReorientSceneCfg(AmggG1BaseSceneCfg):
    """Long-object reorientation onto a two-support fixture."""

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.42, 1.035)),
        spawn=sim_utils.CuboidCfg(size=(0.48, 0.055, 0.055), **_dynamic_material((0.06, 0.32, 0.95), 0.40)),
    )
    left_support = _static_box(
        "{ENV_REGEX_NS}/LeftSupport", (-0.19, 0.68, 1.055), (0.085, 0.13, 0.11), (0.16, 0.70, 0.78)
    )
    right_support = _static_box(
        "{ENV_REGEX_NS}/RightSupport", (0.19, 0.68, 1.055), (0.085, 0.13, 0.11), (0.16, 0.70, 0.78)
    )
    goal = _goal_marker((0.0, 0.68, 1.115), (0.52, 0.09, 0.008), (0.10, 0.85, 0.88))


@configclass
class AmggG1PrecisionInsertSceneCfg(AmggG1BaseSceneCfg):
    """Upright keyed block insertion through a tight four-wall guide."""

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.24, 0.43, 1.07)),
        spawn=sim_utils.CuboidCfg(size=(0.045, 0.045, 0.14), **_dynamic_material((0.96, 0.72, 0.05), 0.20)),
    )
    guide_left = _static_box("{ENV_REGEX_NS}/GuideLeft", (0.178, 0.62, 1.055), (0.025, 0.105, 0.11), (0.55, 0.12, 0.75))
    guide_right = _static_box(
        "{ENV_REGEX_NS}/GuideRight", (0.262, 0.62, 1.055), (0.025, 0.105, 0.11), (0.55, 0.12, 0.75)
    )
    guide_near = _static_box("{ENV_REGEX_NS}/GuideNear", (0.22, 0.578, 1.055), (0.060, 0.025, 0.11), (0.55, 0.12, 0.75))
    guide_far = _static_box("{ENV_REGEX_NS}/GuideFar", (0.22, 0.662, 1.055), (0.060, 0.025, 0.11), (0.55, 0.12, 0.75))
    goal = _goal_marker((0.22, 0.62, 1.003), (0.055, 0.055, 0.008), (0.72, 0.18, 0.90))


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
class BimanualPolicyCfg(AmggG1PolicyCfg):
    goal = ObsTerm(func=mdp.g1_task_goal, params={"task_slug": "bimanual_reorient"})
    progress = ObsTerm(func=mdp.g1_task_progress, params={"task_slug": "bimanual_reorient"})


@configclass
class PrecisionPolicyCfg(AmggG1PolicyCfg):
    goal = ObsTerm(func=mdp.g1_task_goal, params={"task_slug": "precision_insert"})
    progress = ObsTerm(func=mdp.g1_task_progress, params={"task_slug": "precision_insert"})


@configclass
class ClutterObservationsCfg:
    policy: ClutterPolicyCfg = ClutterPolicyCfg()


@configclass
class BimanualObservationsCfg:
    policy: BimanualPolicyCfg = BimanualPolicyCfg()


@configclass
class PrecisionObservationsCfg:
    policy: PrecisionPolicyCfg = PrecisionPolicyCfg()


@configclass
class AmggG1EventsCfg:
    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
    reset_object = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.035, 0.035), "y": (-0.035, 0.035), "yaw": (-0.25, 0.25)},
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
class BimanualTerminationsCfg(AmggG1FailureTermsCfg):
    success = DoneTerm(func=mdp.bimanual_reorient_success)


@configclass
class PrecisionTerminationsCfg(AmggG1FailureTermsCfg):
    success = DoneTerm(func=mdp.precision_insert_success)


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
class AmggG1BimanualReorientEnvCfg(AmggG1BaseEnvCfg):
    scene: AmggG1BimanualReorientSceneCfg = AmggG1BimanualReorientSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )
    observations: BimanualObservationsCfg = BimanualObservationsCfg()
    terminations: BimanualTerminationsCfg = BimanualTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 55.0


@configclass
class AmggG1PrecisionInsertEnvCfg(AmggG1BaseEnvCfg):
    scene: AmggG1PrecisionInsertSceneCfg = AmggG1PrecisionInsertSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )
    observations: PrecisionObservationsCfg = PrecisionObservationsCfg()
    terminations: PrecisionTerminationsCfg = PrecisionTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 50.0


class _AmggG1XrTimingMixin:
    """Use a 60 Hz control/render loop for lower-latency XR teleoperation."""

    def __post_init__(self):
        super().__post_init__()
        # Keep the 60 Hz control/render rate while doubling contact substeps.
        # GPU PhysX does not support scene-level sweep CCD, so 240 Hz physics
        # plus bounded articulation drives is the robust tunneling safeguard.
        self.sim.dt = 1.0 / 240.0
        self.decimation = 4
        self.sim.render_interval = 4


@configclass
class AmggG1ClutterTransferXrEnvCfg(_AmggG1XrTimingMixin, AmggG1ClutterTransferEnvCfg):
    """Low-latency XR variant of the clutter-transfer task."""


@configclass
class AmggG1BimanualReorientXrEnvCfg(_AmggG1XrTimingMixin, AmggG1BimanualReorientEnvCfg):
    """Low-latency XR variant of the bimanual-reorientation task."""


@configclass
class AmggG1PrecisionInsertXrEnvCfg(_AmggG1XrTimingMixin, AmggG1PrecisionInsertEnvCfg):
    """Low-latency XR variant of the precision-insertion task."""
