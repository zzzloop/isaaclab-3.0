# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab asset configuration for the AM-DP123 mobile manipulator."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from amgg_robot_lab.contracts import AM_DP123_HOME_POSITIONS, AM_DP123_LOCOMOTION_JOINT_NAMES

if TYPE_CHECKING:
    from isaaclab.assets import ArticulationCfg

AM_DP123_ASSET_DATA_DIR = Path(__file__).resolve().parent / "data" / "AM-DP123"
"""Directory that holds the unmodified AM-DP123 URDF and its 61 STL meshes."""

AM_DP123_URDF_PATH = AM_DP123_ASSET_DATA_DIR / "urdf" / "AM-DP123.urdf"
AM_DP123_MESH_DIR = AM_DP123_ASSET_DATA_DIR / "meshes"

AM_DP123_ROS_PACKAGE_NAME = "AM-DP123"
"""ROS package name used by the ``package://`` mesh references inside the URDF."""

# The lowest mesh vertex of the robot in the ``base_link`` frame at the zero pose
# is z = -0.07289 m (rear-right steer link), so a welded base spawned at this
# height rests the wheels and steer links exactly on the ground plane.
AM_DP123_BASE_SPAWN_HEIGHT_M = 0.0729
AM_DP123_BASE_SPAWN_ORIENTATION_XYZW = (0.0, 0.0, 0.0, 1.0)
"""Identity base orientation in Isaac Lab's ``xyzw`` quaternion order."""


def get_am_dp123_robot_cfg() -> ArticulationCfg:
    """Build the AM-DP123 fixed-base articulation configuration."""
    if not AM_DP123_URDF_PATH.is_file():
        raise FileNotFoundError(f"Missing {AM_DP123_URDF_PATH}")
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    return ArticulationCfg(
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(AM_DP123_URDF_PATH),
            fix_base=True,
            merge_fixed_joints=False,
            self_collision=False,
            collision_type="Convex Decomposition",
            robot_type="Mobile Manipulators",
            # ``package://AM-DP123/meshes/*.STL`` resolves against the packaged
            # asset directory, which keeps the URDF byte-identical to upstream.
            ros_package_paths=[{"name": AM_DP123_ROS_PACKAGE_NAME, "path": str(AM_DP123_ASSET_DATA_DIR)}],
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
            pos=(0.0, 0.0, AM_DP123_BASE_SPAWN_HEIGHT_M),
            rot=AM_DP123_BASE_SPAWN_ORIENTATION_XYZW,
            joint_pos={**{".*": 0.0}, **AM_DP123_HOME_POSITIONS},
            joint_vel={".*": 0.0},
        ),
        # Keep the exact URDF limits: the hand fingers close to their one-sided
        # limits, so a soft-limit factor would silently shorten the grasp travel.
        soft_joint_pos_limit_factor=1.0,
        actuators={
            "waist": ImplicitActuatorCfg(
                joint_names_expr=["waist_joint[123]"],
                effort_limit_sim={"waist_joint1": 376.0, "waist_joint2": 367.0, "waist_joint3": 367.0},
                velocity_limit_sim=2.62,
                stiffness=900.0,
                damping=65.0,
                armature=0.02,
            ),
            "shoulders_elbows": ImplicitActuatorCfg(
                joint_names_expr=["openarm_left_joint[1-4]", "openarm_right_joint[1-4]"],
                effort_limit_sim=40.0,
                velocity_limit_sim=3.14,
                stiffness=280.0,
                damping=28.0,
                armature=0.015,
            ),
            "wrists": ImplicitActuatorCfg(
                joint_names_expr=["openarm_left_joint[5-7]", "openarm_right_joint[5-7]"],
                effort_limit_sim=7.0,
                velocity_limit_sim=3.14,
                stiffness=80.0,
                damping=10.0,
                armature=0.01,
            ),
            "hands": ImplicitActuatorCfg(
                joint_names_expr=["left_arm_hand_joint[12]_0", "right_arm_hand_joint[12]_0"],
                effort_limit_sim=1.0,
                velocity_limit_sim=1.0,
                stiffness=100.0,
                damping=5.0,
                armature=0.001,
            ),
            "head": ImplicitActuatorCfg(
                joint_names_expr=["head_joint[12]"],
                effort_limit_sim=2.9,
                velocity_limit_sim=3.14,
                stiffness=35.0,
                damping=5.0,
            ),
            "passive_base": ImplicitActuatorCfg(
                joint_names_expr=list(AM_DP123_LOCOMOTION_JOINT_NAMES),
                effort_limit_sim=10.0,
                velocity_limit_sim=12.0,
                stiffness=0.0,
                damping=0.3,
            ),
        },
    )
