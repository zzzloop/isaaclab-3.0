# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stable Unitree G1 with RH56DFX simulation and hardware contracts."""

AMGG_G1_EMBODIMENT = "unitree_g1_29dof_rh56dfx"
AMGG_G1_SCHEMA_VERSION = "1.0.0"

AMGG_G1_BODY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# This is the exact order consumed by the official G1 Inspire Pink action.
AMGG_G1_SIM_HAND_JOINT_NAMES = (
    "L_index_proximal_joint",
    "L_middle_proximal_joint",
    "L_pinky_proximal_joint",
    "L_ring_proximal_joint",
    "L_thumb_proximal_yaw_joint",
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_pinky_proximal_joint",
    "R_ring_proximal_joint",
    "R_thumb_proximal_yaw_joint",
    "L_index_intermediate_joint",
    "L_middle_intermediate_joint",
    "L_pinky_intermediate_joint",
    "L_ring_intermediate_joint",
    "L_thumb_proximal_pitch_joint",
    "R_index_intermediate_joint",
    "R_middle_intermediate_joint",
    "R_pinky_intermediate_joint",
    "R_ring_intermediate_joint",
    "R_thumb_proximal_pitch_joint",
    "L_thumb_intermediate_joint",
    "R_thumb_intermediate_joint",
    "L_thumb_distal_joint",
    "R_thumb_distal_joint",
)

AMGG_G1_HAND_MOTOR_NAMES = (
    "right_pinky",
    "right_ring",
    "right_middle",
    "right_index",
    "right_thumb_bend",
    "right_thumb_rotation",
    "left_pinky",
    "left_ring",
    "left_middle",
    "left_index",
    "left_thumb_bend",
    "left_thumb_rotation",
)

# Representative USD joints used to build a hardware-shaped state. The
# intermediate/distal phalanges remain available in the simulation-only state.
AMGG_G1_HAND_MOTOR_SIM_JOINT_NAMES = (
    "R_pinky_proximal_joint",
    "R_ring_proximal_joint",
    "R_middle_proximal_joint",
    "R_index_proximal_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_proximal_yaw_joint",
    "L_pinky_proximal_joint",
    "L_ring_proximal_joint",
    "L_middle_proximal_joint",
    "L_index_proximal_joint",
    "L_thumb_proximal_pitch_joint",
    "L_thumb_proximal_yaw_joint",
)

AMGG_G1_OBSERVATION_JOINT_NAMES = AMGG_G1_BODY_JOINT_NAMES + AMGG_G1_HAND_MOTOR_NAMES
AMGG_G1_SIM_OBSERVATION_JOINT_NAMES = AMGG_G1_BODY_JOINT_NAMES + AMGG_G1_SIM_HAND_JOINT_NAMES
AMGG_G1_CONTROLLED_ARM_JOINT_NAMES = AMGG_G1_BODY_JOINT_NAMES[15:]
AMGG_G1_SIM_PROCESSED_ACTION_NAMES = AMGG_G1_CONTROLLED_ARM_JOINT_NAMES + AMGG_G1_SIM_HAND_JOINT_NAMES

AMGG_G1_WRIST_ACTION_NAMES = (
    "left_wrist.x",
    "left_wrist.y",
    "left_wrist.z",
    "left_wrist.qx",
    "left_wrist.qy",
    "left_wrist.qz",
    "left_wrist.qw",
    "right_wrist.x",
    "right_wrist.y",
    "right_wrist.z",
    "right_wrist.qx",
    "right_wrist.qy",
    "right_wrist.qz",
    "right_wrist.qw",
)
AMGG_G1_SIM_RAW_ACTION_NAMES = AMGG_G1_WRIST_ACTION_NAMES + tuple(
    f"sim_hand.{name}" for name in AMGG_G1_SIM_HAND_JOINT_NAMES
)
AMGG_G1_HARDWARE_ACTION_NAMES = AMGG_G1_WRIST_ACTION_NAMES + tuple(
    f"rh56dfx.{name}" for name in AMGG_G1_HAND_MOTOR_NAMES
)

# Select RH56DFX motor proxies from the 24-D hand suffix of the official
# Isaac Lab action: right [pinky, ring, middle, index, thumb bend, thumb rot], then left.
AMGG_G1_SIM_HAND_TO_MOTOR_INDICES = (7, 8, 6, 5, 19, 9, 2, 3, 1, 0, 14, 4)
AMGG_G1_TACTILE_NAMES = tuple(f"force.{name}" for name in AMGG_G1_HAND_MOTOR_NAMES)

AMGG_G1_CAMERA_NAMES = ("front", "overview")


def validate_amgg_g1_contract() -> None:
    """Validate dimensions and uniqueness of the G1 data ABI."""
    if len(AMGG_G1_BODY_JOINT_NAMES) != 29:
        raise ValueError("The Unitree G1 body contract must contain 29 joints.")
    if len(AMGG_G1_SIM_HAND_JOINT_NAMES) != 24:
        raise ValueError("The Inspire hand contract must contain 24 joints.")
    if len(AMGG_G1_OBSERVATION_JOINT_NAMES) != 41 or len(AMGG_G1_SIM_OBSERVATION_JOINT_NAMES) != 53:
        raise ValueError("The RH56DFX hardware/simulation observation contracts must be 41-D/53-D.")
    if len(AMGG_G1_SIM_RAW_ACTION_NAMES) != 38 or len(AMGG_G1_SIM_PROCESSED_ACTION_NAMES) != 38:
        raise ValueError("The official G1 Inspire simulation actions must be 38-D.")
    if len(AMGG_G1_HARDWARE_ACTION_NAMES) != 26:
        raise ValueError("The G1 with RH56DFX hardware action must be 26-D.")
    if len(AMGG_G1_TACTILE_NAMES) != 12:
        raise ValueError("The RH56DFX force observation must be 12-D.")
    if len(set(AMGG_G1_OBSERVATION_JOINT_NAMES)) != len(AMGG_G1_OBSERVATION_JOINT_NAMES):
        raise ValueError("The G1 observation joint names must be unique.")


validate_amgg_g1_contract()
