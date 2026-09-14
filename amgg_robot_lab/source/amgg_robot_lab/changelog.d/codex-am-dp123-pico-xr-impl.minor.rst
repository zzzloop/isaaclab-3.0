# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

Added
^^^^^

* Added the AM-DP123 external extension with the unmodified AM-DP123 URDF and 61
  STL meshes, joint/frame/camera ABI contracts, offline URDF FK/Jacobian/DLS-IK,
  the 18-D Pink inverse-kinematics action, and the PICO XR teleoperation pipeline
  registered as ``Isaac-AM-DP123-Pico-XR-v0``.

  Migration: the extension carries AM-DP123 joint, frame, and camera names
  instead of the previous AMGG names; downstream code must import the new
  ``amgg_robot_lab.contracts.am_dp123_*`` contracts.
