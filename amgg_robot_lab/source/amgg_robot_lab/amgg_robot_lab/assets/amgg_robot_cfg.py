# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab asset configuration for the normalized AMGG robot."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from amgg_robot_lab.contracts import AMGG_HOME_POSITIONS

if TYPE_CHECKING:
    from isaaclab.assets import ArticulationCfg

AMGG_ASSET_DATA_DIR = Path(__file__).resolve().parent / "data"
AMGG_URDF_PATH = AMGG_ASSET_DATA_DIR / "urdf" / "amgg_robot.urdf"
AMGG_RAW_URDF_PATH = AMGG_ASSET_DATA_DIR / "urdf" / "amgg_robot_raw.urdf"


def get_amgg_robot_cfg() -> ArticulationCfg:
    """Build the AMGG fixed-base articulation configuration."""
    if not AMGG_URDF_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {AMGG_URDF_PATH}. Run scripts/amgg_prepare_robot_asset.py before launching Isaac Lab."
        )
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    return ArticulationCfg(
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(AMGG_URDF_PATH),
            fix_base=True,
            merge_fixed_joints=False,
            self_collision=False,
            collision_type="Convex Decomposition",
            robot_type="Mobile Manipulators",
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=3.0,
                retain_accelerations=False,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.25),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos=dict(AMGG_HOME_POSITIONS),
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.92,
        actuators={
            "waist": ImplicitActuatorCfg(
                joint_names_expr=["Waist01_Joint", "Waist02_Joint", "Body0422_Joint"],
                effort_limit_sim={"Waist01_Joint": 376.0, "Waist02_Joint": 367.0, "Body0422_Joint": 367.0},
                velocity_limit_sim=1.5,
                stiffness=900.0,
                damping=65.0,
                armature=0.02,
            ),
            "shoulders_elbows": ImplicitActuatorCfg(
                joint_names_expr=["Arm[LR]02_Joint", "AM_D02.*_Joint", "Arm[LR]0[45]_Joint"],
                effort_limit_sim=40.0,
                velocity_limit_sim=2.0,
                stiffness=280.0,
                damping=28.0,
                armature=0.015,
            ),
            "wrists": ImplicitActuatorCfg(
                joint_names_expr=["Arm[LR]06_Joint", "Arm[LR]07_Joint", "Arm[LR]07Output_Joint"],
                effort_limit_sim=9.0,
                velocity_limit_sim=2.0,
                stiffness=80.0,
                damping=10.0,
                armature=0.01,
            ),
            "grippers": ImplicitActuatorCfg(
                joint_names_expr=[".*_gripper_.*_finger_joint"],
                effort_limit_sim=30.0,
                velocity_limit_sim=0.2,
                stiffness=1500.0,
                damping=60.0,
                armature=0.001,
            ),
            "head": ImplicitActuatorCfg(
                joint_names_expr=["Head0[23]_Joint"],
                effort_limit_sim=2.9,
                velocity_limit_sim=1.5,
                stiffness=35.0,
                damping=5.0,
            ),
            "passive_base": ImplicitActuatorCfg(
                joint_names_expr=["Turn[LR]_Joint", ".*wheel[LR]_Joint", "Driven[LR]0[12]_Joint"],
                effort_limit_sim=10.0,
                velocity_limit_sim=12.0,
                stiffness=0.0,
                damping=0.3,
            ),
        },
    )
