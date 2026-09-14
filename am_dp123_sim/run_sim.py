# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AM-DP123 fixed-base Pico validation scene, using this checkout's Isaac Lab API."""

import argparse
import csv
import json
import time
from pathlib import Path

from model import ARMS, ROOT, TIPS, RobotModel, pose_matrix, rotation_error
from pico_input import PicoReceiver, PicoRetargeter

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bind", default="127.0.0.1")
parser.add_argument("--port", type=int, default=15051)
parser.add_argument("--mode", choices=("kinematic", "servo"), default="kinematic")
parser.add_argument("--config", type=Path, default=ROOT / "teleop_config.json")
parser.add_argument("--timeout", type=float, default=0.5, help="Pico receipt timeout [s]")
parser.add_argument("--base_height", type=float, default=0.35, help="Fixed base height above ground [m]")
parser.add_argument("--steps", type=int, default=0, help="Stop after this many steps; zero means unlimited")
parser.add_argument("--output_dir", type=Path, default=ROOT / "outputs")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_from_matrix


def load_configuration():
    """Load bounded joint settings [rad] and positive tracking thresholds [m, rad]."""
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for key in ("position_scale", "max_joint_speed", "position_tolerance", "orientation_tolerance"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not np.isfinite(args.timeout) or args.timeout <= 0 or not np.isfinite(args.base_height):
        raise ValueError("Invalid timeout or base_height")
    model = RobotModel()
    for name, value in config["home"].items():
        if (
            name not in model.limits
            or not np.isfinite(value)
            or not model.limits[name][0] <= value <= model.limits[name][1]
        ):
            raise ValueError(f"Invalid home position: {name}={value}")
        model.home[name] = value
    for key in ("gripper_open", "gripper_closed"):
        values = config[key]
        if len(values) != 2:
            raise ValueError(f"{key} requires two angles [rad]")
        for side in ARMS:
            for i, value in enumerate(values, 1):
                lower, upper = model.limits[f"{side}_arm_hand_joint{i}_0"]
                if not np.isfinite(value) or not lower <= value <= upper:
                    raise ValueError(f"Invalid {key} angle: {value}")
    return config, model


def main():
    """Spawn the robot and track calibrated Pico targets with diagnostics."""
    config, model = load_configuration()
    asset_path = model.prepare_urdf()
    dt = 1 / 60
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args.device))
    sim.set_camera_view([2.5, 2.5, 2.2], [0, 0, 1.0])
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/Ground", ground)
    light = sim_utils.DomeLightCfg(intensity=2500)
    light.func("/World/Light", light)
    spawn = sim_utils.UrdfFileCfg(
        asset_path=str(asset_path),
        fix_base=True,
        merge_fixed_joints=False,
        usd_dir=str(ROOT / ".cache" / "usd"),
        force_usd_conversion=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
    )
    # Servo gains are visual-validation defaults, not identified hardware gains.
    cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=spawn,
        init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, args.base_height), joint_pos=model.home),
        soft_joint_pos_limit_factor=1.0,
        actuators={
            "joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=model.efforts,
                velocity_limit_sim=model.velocities,
                stiffness=150.0,
                damping=20.0,
            )
        },
    )
    robot = Articulation(cfg)
    # Locally generated RGB axes avoid a dependency on remote marker USD assets.
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Targets",
        markers={
            name: sim_utils.CuboidCfg(size=size, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))
            for name, size, color in (
                ("x", (0.12, 0.008, 0.008), (1.0, 0.1, 0.1)),
                ("y", (0.008, 0.12, 0.008), (0.1, 1.0, 0.1)),
                ("z", (0.008, 0.008, 0.12), (0.1, 0.1, 1.0)),
            )
        },
    )
    markers = VisualizationMarkers(marker_cfg)
    sim.reset()
    names = robot.joint_names
    if set(names) != set(model.limits):
        raise RuntimeError(
            f"Imported joint mismatch: missing={set(model.limits) - set(names)}, extra={set(names) - set(model.limits)}"
        )
    body_ids = {side: robot.body_names.index(tip) for side, tip in TIPS.items()}
    command = dict(model.home)
    target_tensor = torch.tensor([[command[name] for name in names]], device=sim.device, dtype=torch.float32)
    robot.write_joint_position_to_sim_index(position=target_tensor)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(target_tensor))
    robot.reset()
    robot.update(dt)
    receiver = PicoReceiver(args.bind, args.port)
    retargeter = PicoRetargeter(model, config["position_scale"])
    targets = {side: model.forward(TIPS[side], command)[0] for side in ARMS}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1000000000:09d}"
    status, last_status, last_report, tick = "WAITING", None, -np.inf, 0
    last_command_at = time.monotonic()
    print(f"AM-DP123 ready: {args.mode}, UDP {args.bind}:{args.port}; A+A calibrates/enables, B+B pauses.")
    try:
        with (
            (args.output_dir / f"{stamp}_metrics.csv").open("w", newline="", encoding="utf-8") as metrics_file,
            (args.output_dir / f"{stamp}_pico.jsonl").open("w", encoding="utf-8") as raw_file,
        ):
            metrics = csv.writer(metrics_file)
            metrics.writerow(
                [
                    "elapsed_s",
                    "status",
                    "side",
                    "target_position_error_m",
                    "target_orientation_error_rad",
                    "joint_tracking_max_rad",
                    "fk_import_position_error_m",
                    "fk_import_orientation_error_rad",
                ]
            )
            started = time.monotonic()
            session = None
            while simulation_app.is_running() and (not args.steps or tick < args.steps):
                begin = time.monotonic()
                if not sim.is_playing():
                    retargeter.pause()
                    receiver.poll()
                    sim.render()
                    time.sleep(dt)
                    continue
                actual_q = dict(zip(names, robot.data.joint_pos.torch[0].cpu().numpy().tolist(), strict=True))
                # Pause before accepting a new packet so reconnecting cannot bypass the watchdog.
                if begin - receiver.received_at > args.timeout:
                    retargeter.pause()
                    status = "STALE / PAUSED"
                packet = receiver.poll()
                if packet is not None:
                    if packet["session"] != session:
                        retargeter.pause()
                        session = packet["session"]
                    raw_file.write(json.dumps({"elapsed_s": begin - started, "packet": packet}, allow_nan=False) + "\n")
                    updated = retargeter.update(packet, actual_q)
                    if updated is None:
                        status = "PAUSED"
                    else:
                        targets = updated
                        candidate, errors = dict(command), []
                        for side in ARMS:
                            candidate, pos_error, rot_error = model.solve(side, targets[side], candidate)
                            errors.append((pos_error, rot_error))
                        if all(
                            p <= config["position_tolerance"] and r <= config["orientation_tolerance"]
                            for p, r in errors
                        ):
                            for side in ARMS:
                                trigger = packet[f"{side}_joy"]["trigger"]
                                gripper_range = zip(config["gripper_open"], config["gripper_closed"], strict=True)
                                for i, (opened, closed) in enumerate(gripper_range, 1):
                                    candidate[f"{side}_arm_hand_joint{i}_0"] = opened + trigger * (closed - opened)
                            # Bound each control update by elapsed wall time, including low-rate streams.
                            control_dt = min(begin - last_command_at, 1 / 30)
                            for name in names:
                                step = min(config["max_joint_speed"], model.velocities[name]) * control_dt
                                command[name] += float(np.clip(candidate[name] - command[name], -step, step))
                            status = "ACTIVE"
                        else:
                            status = "IK REJECTED / HOLD"
                        last_command_at = begin
                target_tensor = torch.tensor(
                    [[command[name] for name in names]], device=sim.device, dtype=torch.float32
                )
                if args.mode == "kinematic":
                    robot.write_joint_position_to_sim_index(position=target_tensor)
                    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(target_tensor))
                robot.set_joint_position_target_index(target=target_tensor)
                robot.write_data_to_sim()
                sim.step()
                if args.mode == "kinematic":
                    robot.write_joint_position_to_sim_index(position=target_tensor)
                    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(target_tensor))
                robot.update(dt)
                tick += 1
                translations, matrices = [], []
                for side in ARMS:
                    for axis in range(3):
                        point = targets[side][:3, 3] + targets[side][:3, axis] * 0.06
                        translations.append(point + [0, 0, args.base_height])
                        matrices.append(targets[side][:3, :3])
                markers.visualize(
                    translations=np.asarray(translations),
                    orientations=quat_from_matrix(torch.tensor(np.asarray(matrices), dtype=torch.float32)),
                    marker_indices=[0, 1, 2, 0, 1, 2],
                )
                if begin - last_report >= 0.5:
                    actual_q = dict(zip(names, robot.data.joint_pos.torch[0].cpu().numpy().tolist(), strict=True))
                    tracking = max(abs(command[name] - actual_q[name]) for name in names)
                    body_poses = robot.data.body_link_pose_w.torch[0].cpu().numpy()
                    messages = []
                    for side in ARMS:
                        actual = pose_matrix(body_poses[body_ids[side]].tolist())
                        actual[2, 3] -= args.base_height
                        fk, _ = model.forward(TIPS[side], actual_q)
                        ep = float(np.linalg.norm(targets[side][:3, 3] - actual[:3, 3]))
                        er = float(np.linalg.norm(rotation_error(targets[side][:3, :3], actual[:3, :3])))
                        fp = float(np.linalg.norm(fk[:3, 3] - actual[:3, 3]))
                        fr = float(np.linalg.norm(rotation_error(fk[:3, :3], actual[:3, :3])))
                        metrics.writerow([begin - started, status, side, ep, er, tracking, fp, fr])
                        messages.append(f"{side}: {ep * 1000:.1f} mm / {np.degrees(er):.1f} deg; FK {fp * 1000:.2f} mm")
                    print(status + " | " + " | ".join(messages))
                    metrics_file.flush()
                    raw_file.flush()
                    last_report = begin
                if status != last_status:
                    print(f"State: {status}")
                    last_status = status
                time.sleep(max(0, dt - (time.monotonic() - begin)))
    finally:
        receiver.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
