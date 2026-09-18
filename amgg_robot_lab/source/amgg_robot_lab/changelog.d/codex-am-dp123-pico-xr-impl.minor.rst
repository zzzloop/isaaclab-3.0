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

* Added the ``Isaac-AM-DP123-Pi05-Eval-v0`` PI0.5 simulation validation task. It
  drives the 18-D joint command ABI directly with the official
  ``JointPositionActionCfg`` and is decoupled from the PICO/Pink teleoperation
  pipeline.

* Added ``amgg_robot_lab.policy``: the OpenPI observation payload builder and the
  ``Pi05ActionAdapter`` safety adapter with the confirmed 18-D WebSocket policy
  contract and server-side padding to the 32-D PI0.5 network width.

* Added ``scripts/am_dp123_pi05_eval.py`` with ``mock_hold``, ``mock_sine``, and a
  lazily imported ``remote`` OpenPI WebSocket client, plus bounded episode NPZ/JSON
  recording of states, actions, targets, terminations, and optional images.

Fixed
^^^^^

* Fixed PI0.5 control to include both head joints and expand each one-dimensional
  gripper command to its URDF driving and mimic finger targets without exposing
  mimic joints to the model.

* Fixed the PI0.5 task dependency boundary so importing it no longer loads the
  PICO, Pink IK, or IsaacTeleop stack.

* Fixed PI0.5 episode recording to preserve terminal observations and distinguish
  held safety targets from valid model output.

* Fixed the missing episode acceptance path by adding offline validation and
  simulator replay for recorded PI0.5 joint targets.

* Fixed the BPX remote policy protocol to send the trained left-eye and dual-wrist
  image layout, center-crop images to 224×224, and decode the server's raw 23-D
  actions into the effective 18-D robot control order.

* Fixed the Isaac Lab 3.0 base quaternion so the AM-DP123 starts upright.
* Fixed right-hand trigger routing.
* Fixed PICO arm control by rebasing controller motion at Play, targeting the
  URDF hand-base frames, and removing the XR tuning overlay from normal use.
* Fixed stiff AM-DP123 arm motion by matching the proven first-generation IK
  task weighting and reducing the home-posture bias on shoulder and elbow joints.
* Preserved scheduler GPU selection by delegating device choice to the official launcher.
