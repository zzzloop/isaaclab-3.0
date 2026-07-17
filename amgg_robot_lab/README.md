# AMGG Robot Lab

The active Unitree G1 with RH56DFX research track is documented in
[README_G1_CN.md](README_G1_CN.md). The original custom-URDF track is retained
for auditability but is frozen until a corrected robot description is available.

See [README_CN.md](README_CN.md) for the complete setup, task definitions, PICO workflow, recording commands, LeRobot conversion, server acceptance checklist, and real-robot integration contract.

This isolated Isaac Lab extension contains a normalized AMGG URDF, 32 source meshes, FK/Jacobian/dual-arm IK, four automatically evaluated manipulation tasks, controller-based PICO teleoperation, official HDF5 demo recording wrappers, four-camera observations, LeRobot Dataset v3 conversion, and guarded dry-run/ROS 2 hardware boundaries.

Quick server setup:

```bash
cd ~/zzk_data/IsaacLab
./isaaclab.sh -p -m pip install -e amgg_robot_lab/source/amgg_robot_lab
./isaaclab.sh -p amgg_robot_lab/scripts/amgg_check_robot_asset.py
./isaaclab.sh -p -m pytest amgg_robot_lab/tests -q
```

First simulation launch:

```bash
./isaaclab.sh -p amgg_robot_lab/scripts/amgg_smoke_test.py \
  --task Isaac-AMGG-PickPlace-v0 \
  --num_steps 240 \
  --enable_cameras \
  --visualizer kit
```

The included parallel grippers and camera extrinsics are research simulation defaults. They must be replaced or calibrated before claiming real-hardware equivalence.
