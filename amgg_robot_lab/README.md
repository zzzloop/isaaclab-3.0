# AMGG Robot Lab

`amgg_robot_lab` is the isolated project scaffold for the AMGG robot, custom Isaac Lab tasks,
PICO teleoperation, demonstration recording, LeRobot conversion, and a future real-robot backend.

The scaffold intentionally contains no fabricated robot parameters. Add the robot URDF and meshes
under `source/amgg_robot_lab/amgg_robot_lab/assets/data/`, then populate the joint and frame contracts
before implementing simulation or hardware control.

## Boundaries

- `assets/`: simulation model paths and `ArticulationCfg` construction.
- `contracts/`: canonical joint order and frame names shared by simulation, recording, and hardware.
- `kinematics/`: URDF-backed FK and IK.
- `tasks/`: manager-based Isaac Lab scene and MDP configuration.
- `teleop/`: PICO input, retargeting, and safety checks.
- `recording/`: HDF5 recording terms and dataset schema.
- `real/`: hardware-neutral interface and the future ROS 2 or vendor-SDK backend.
- `scripts/`: asset checks, real-robot teleoperation/recording, and LeRobot conversion entry points.

## Server placement

This copy is staged inside the local Isaac Lab checkout because that is the available workspace. On
the server, copy `amgg_robot_lab/` next to, rather than inside, the Isaac Lab checkout. For example:

```text
~/zzk_data/
├── IsaacLab/
└── amgg_robot_lab/
```

Install it in the Isaac Lab Python environment with:

```bash
cd ~/zzk_data/amgg_robot_lab
../IsaacLab/isaaclab.sh -p -m pip install -e source/amgg_robot_lab
```

## Required robot inputs

1. URDF and all referenced meshes.
2. Canonical controlled-joint order and observed-joint order.
3. Joint limits, home positions, velocity limits, and effort limits.
4. Base, torso, wrist, TCP, gripper, and camera-parent link names.
5. Fixed, wheeled, or floating-base semantics.
6. Real-robot SDK or ROS 2 command/state interfaces.

