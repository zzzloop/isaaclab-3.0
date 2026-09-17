# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AM-DP123 PI0.5 policy evaluation environment.

This task is decoupled from the PICO teleoperation task: it drives the 18 commanded
joints directly with :class:`~isaaclab.envs.mdp.JointPositionActionCfg` and carries no
IsaacTeleop pipeline, XR camera feed, controller, or Pink IK. Only the scene, camera
observations, reset events, and terminations are shared with the teleoperation task.
"""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg

from amgg_robot_lab.contracts import AM_DP123_CONTROLLED_JOINT_NAMES, AM_DP123_JOINT_SPECS

from .am_dp123_scene_cfg import AmDp123SceneCfg, EventCfg, ObservationsCfg, TerminationsCfg

AM_DP123_PI05_SIM_DT = 1.0 / 120.0
"""Physics time step [s]."""

AM_DP123_PI05_DECIMATION = 4
"""Physics steps per policy step."""

AM_DP123_PI05_CONTROL_DT = AM_DP123_PI05_SIM_DT * AM_DP123_PI05_DECIMATION
"""Policy control period [s], i.e. the 30 Hz command rate."""

AM_DP123_PI05_CLIP = {
    spec.name: (spec.lower_limit_rad, spec.upper_limit_rad) for spec in AM_DP123_JOINT_SPECS if spec.command_enabled
}
"""Exact per-joint position clip taken from the joint contract."""


@configclass
class Pi05ActionsCfg:
    """Single 18-D absolute joint-position action term for the PI0.5 adapter."""

    joint_positions = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(AM_DP123_CONTROLLED_JOINT_NAMES),
        preserve_order=True,
        scale=1.0,
        offset=0.0,
        use_default_offset=False,
        clip=AM_DP123_PI05_CLIP,
    )


@configclass
class AmDp123Pi05EvalEnvCfg(ManagerBasedRLEnvCfg):
    """Direct joint-position policy evaluation environment for the AM-DP123."""

    scene: AmDp123SceneCfg = AmDp123SceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: Pi05ActionsCfg = Pi05ActionsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self) -> None:
        self.decimation = AM_DP123_PI05_DECIMATION
        self.episode_length_s = 60.0
        self.compute_final_obs = True
        self.sim.dt = AM_DP123_PI05_SIM_DT
        self.sim.render_interval = 2
        self.sim.device = "cuda:0"
        self.num_rerenders_on_reset = 3
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(2.0, -1.6, 1.8), lookat=(0.55, 0.0, 0.9))
