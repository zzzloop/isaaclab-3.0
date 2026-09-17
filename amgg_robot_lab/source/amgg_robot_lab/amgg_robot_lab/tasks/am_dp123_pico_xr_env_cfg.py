# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AM-DP123 PICO XR teleoperation environment."""

from __future__ import annotations

from isaaclab.controllers.pink_ik import DampingTaskCfg, FrameTaskCfg, NullSpacePostureTaskCfg, PinkIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg
from isaaclab_teleop import IsaacTeleopCfg, XrCameraFeedCfg, XrCameraFeedLayoutCfg, XrCfg

from amgg_robot_lab.assets import AM_DP123_ASSET_DATA_DIR, AM_DP123_URDF_PATH
from amgg_robot_lab.contracts import (
    AM_DP123_FRAMES,
    AM_DP123_HAND_JOINT_NAMES,
    AM_DP123_IK_JOINT_NAMES,
    AM_DP123_STEREO_CAMERA_NAMES,
)
from amgg_robot_lab.teleop import AM_DP123_IDLE_ACTION, build_am_dp123_pico_pipeline

from . import mdp
from .am_dp123_scene_cfg import (
    AM_DP123_TABLE_THICKNESS_M,
    AM_DP123_TABLE_TOP_HEIGHT_M,
    AmDp123SceneCfg,
    EventCfg,
    ObservationsCfg,
    PolicyCfg,
    TerminationsCfg,
)


@configclass
class ActionsCfg:
    """18-D absolute dual-hand action plus four trigger-driven finger joints."""

    upper_body_ik = mdp.AmDp123PinkInverseKinematicsActionCfg(
        pink_controlled_joint_names=list(AM_DP123_IK_JOINT_NAMES),
        hand_joint_names=list(AM_DP123_HAND_JOINT_NAMES),
        target_eef_link_names={
            "left_hand": AM_DP123_FRAMES.left_hand_base_link,
            "right_hand": AM_DP123_FRAMES.right_hand_base_link,
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
                    frame=AM_DP123_FRAMES.left_hand_base_link,
                    # Match the proven first-generation controller's task
                    # balance: hand position dominates wrist orientation.
                    position_cost=50.0,
                    orientation_cost=1.0,
                    lm_damping=0.1,
                    gain=1.0,
                ),
                FrameTaskCfg(
                    frame=AM_DP123_FRAMES.right_hand_base_link,
                    position_cost=50.0,
                    orientation_cost=1.0,
                    lm_damping=0.1,
                    gain=1.0,
                ),
                DampingTaskCfg(cost=0.1),
                NullSpacePostureTaskCfg(
                    # A small posture prior resolves the seventh arm DOF
                    # without pinning shoulders and elbows near their home pose.
                    cost=0.02,
                    lm_damping=0.0,
                    controlled_frames=[AM_DP123_FRAMES.left_hand_base_link, AM_DP123_FRAMES.right_hand_base_link],
                    controlled_joints=list(AM_DP123_IK_JOINT_NAMES),
                ),
            ],
            fixed_input_tasks=[],
        ),
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
        pipeline = build_am_dp123_pico_pipeline()
        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=lambda: pipeline,
            sim_device=self.sim.device,
            xr_cfg=self.xr,
            xr_camera_feeds=[
                XrCameraFeedCfg(camera_name=name, label=name.replace("_", " ").title())
                for name in AM_DP123_STEREO_CAMERA_NAMES
            ],
            xr_camera_feed_layout=XrCameraFeedLayoutCfg(mode="horizontal", placement="head_locked"),
        )
        self.image_obs_list = list(AM_DP123_STEREO_CAMERA_NAMES)


__all__ = [
    "AM_DP123_TABLE_THICKNESS_M",
    "AM_DP123_TABLE_TOP_HEIGHT_M",
    "ActionsCfg",
    "AmDp123PicoXrEnvCfg",
    "AmDp123SceneCfg",
    "EventCfg",
    "ObservationsCfg",
    "PolicyCfg",
    "TerminationsCfg",
]
