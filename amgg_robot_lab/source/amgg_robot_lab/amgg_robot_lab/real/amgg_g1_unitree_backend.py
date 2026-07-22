# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Guarded Unitree G1 DDS backend for AMGG real-robot teleoperation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import isfinite
import time

from amgg_robot_lab.contracts import (
    AMGG_G1_BODY_JOINT_NAMES,
    AMGG_G1_CONTROLLED_ARM_JOINT_NAMES,
    AMGG_G1_HAND_MOTOR_NAMES,
    AMGG_G1_OBSERVATION_JOINT_NAMES,
)

from .amgg_robot_backend import AmggRobotState

AMGG_G1_HARDWARE_COMMAND_NAMES = AMGG_G1_CONTROLLED_ARM_JOINT_NAMES + AMGG_G1_HAND_MOTOR_NAMES

_G1_BODY_INDEX_BY_NAME = {name: index for index, name in enumerate(AMGG_G1_BODY_JOINT_NAMES)}
_G1_ARM_INDICES = tuple(_G1_BODY_INDEX_BY_NAME[name] for name in AMGG_G1_CONTROLLED_ARM_JOINT_NAMES)
_G1_ARM_SDK_ENABLE_INDEX = 29

# Conservative software envelopes for bring-up.  These are intentionally tighter
# than the mechanical extremes; loosen them only after measuring the real setup.
_G1_ARM_LIMITS_RAD: dict[str, tuple[float, float]] = {
    "left_shoulder_pitch_joint": (-2.5, 2.5),
    "left_shoulder_roll_joint": (-0.4, 2.6),
    "left_shoulder_yaw_joint": (-2.5, 2.5),
    "left_elbow_joint": (-1.2, 2.3),
    "left_wrist_roll_joint": (-1.6, 1.6),
    "left_wrist_pitch_joint": (-1.2, 1.2),
    "left_wrist_yaw_joint": (-1.6, 1.6),
    "right_shoulder_pitch_joint": (-2.5, 2.5),
    "right_shoulder_roll_joint": (-2.6, 0.4),
    "right_shoulder_yaw_joint": (-2.5, 2.5),
    "right_elbow_joint": (-2.3, 1.2),
    "right_wrist_roll_joint": (-1.6, 1.6),
    "right_wrist_pitch_joint": (-1.2, 1.2),
    "right_wrist_yaw_joint": (-1.6, 1.6),
}
_G1_HAND_LIMITS_RAD = {name: (0.0, 1.7) for name in AMGG_G1_HAND_MOTOR_NAMES}
_G1_COMMAND_LIMITS = tuple(
    _G1_ARM_LIMITS_RAD[name] if name in _G1_ARM_LIMITS_RAD else _G1_HAND_LIMITS_RAD[name]
    for name in AMGG_G1_HARDWARE_COMMAND_NAMES
)
_G1_MAX_VELOCITIES = tuple(
    0.8 if name in AMGG_G1_CONTROLLED_ARM_JOINT_NAMES else 1.5 for name in AMGG_G1_HARDWARE_COMMAND_NAMES
)


@dataclass(frozen=True, slots=True)
class UnitreeG1BackendCfg:
    """DDS topics and conservative gains for Unitree G1 upper-body teleoperation.

    Args:
        network_interface: DDS network interface name, such as ``eth0`` or
          ``enp3s0``.  Use ``None`` only for SDK defaults.
        low_state_topic: G1 low-state DDS topic.
        arm_command_topic: G1 arm SDK command topic.  This is preferred over
          whole-body low-level ``rt/lowcmd`` for manipulation bring-up.
        hand_command_topic: Inspire RH56DFX command topic from Unitree's
          ``DFX_inspire_service``.
        hand_state_topic: Inspire RH56DFX state topic from Unitree's
          ``DFX_inspire_service``.
        state_timeout_s: Maximum time to wait for fresh DDS state [s].
        command_timeout_s: Maximum gap between command updates [s].
        arm_kp: Position gain for arm joints.
        arm_kd: Damping gain for arm joints.
        hand_kp: Position gain for hand motors.
        hand_kd: Damping gain for hand motors.
        enable_arms: Whether to publish arm SDK commands.
        enable_hands: Whether to publish Inspire hand commands.
    """

    network_interface: str | None = None
    low_state_topic: str = "rt/lowstate"
    arm_command_topic: str = "rt/arm_sdk"
    hand_command_topic: str = "rt/inspire/cmd"
    hand_state_topic: str = "rt/inspire/state"
    state_timeout_s: float = 0.25
    command_timeout_s: float = 0.20
    arm_kp: float = 30.0
    arm_kd: float = 1.5
    hand_kp: float = 1.0
    hand_kd: float = 0.05
    enable_arms: bool = True
    enable_hands: bool = True


@dataclass(slots=True)
class G1HardwareCommandLimiter:
    """Rate-limit 26-D G1 arm + Inspire hand position commands."""

    tracking_error_limit_rad: float = 0.35
    timeout_s: float = 0.20
    _last_command: tuple[float, ...] | None = field(default=None, init=False)
    _last_timestamp_s: float | None = field(default=None, init=False)

    def reset(self, measured_positions: Sequence[float], timestamp_s: float) -> None:
        """Initialize the limiter from measured 26-D command joints [rad]."""
        self._last_command = _validate_g1_command(measured_positions)
        self._last_timestamp_s = float(timestamp_s)

    def filter(
        self,
        requested_positions: Sequence[float],
        measured_positions: Sequence[float],
        timestamp_s: float,
    ) -> tuple[float, ...]:
        """Validate, clamp, and rate-limit one 26-D G1 command sample."""
        requested = _validate_g1_command(requested_positions)
        measured = _validate_g1_command(measured_positions)
        timestamp_s = float(timestamp_s)
        if self._last_command is None or self._last_timestamp_s is None:
            self.reset(measured, timestamp_s)
        assert self._last_command is not None and self._last_timestamp_s is not None
        dt = timestamp_s - self._last_timestamp_s
        if dt <= 0.0 or dt > self.timeout_s:
            raise ValueError(f"G1 command timing violation: dt={dt:.6f} s.")

        safe_command: list[float] = []
        for index, (target, state, previous, limits, velocity) in enumerate(
            zip(requested, measured, self._last_command, _G1_COMMAND_LIMITS, _G1_MAX_VELOCITIES, strict=True)
        ):
            if abs(previous - state) > self.tracking_error_limit_rad:
                joint_name = AMGG_G1_HARDWARE_COMMAND_NAMES[index]
                raise ValueError(f"G1 joint '{joint_name}' tracking error exceeds the safety limit.")
            bounded = min(max(target, limits[0]), limits[1])
            max_delta = velocity * dt
            bounded = min(max(bounded, previous - max_delta), previous + max_delta)
            if not isfinite(bounded):
                raise ValueError(f"G1 command {index} became non-finite.")
            safe_command.append(bounded)
        self._last_command = tuple(safe_command)
        self._last_timestamp_s = timestamp_s
        return self._last_command


class UnitreeG1UpperBodyBackend:
    """Unitree SDK2 DDS backend for fixed-base G1 arm and Inspire-hand commands.

    This backend deliberately avoids locomotion and whole-body low-level torque
    control.  It only opens a software command gate after fresh state has been
    observed and :meth:`enable` is called by a CLI that performs physical
    safety checks.
    """

    def __init__(self, cfg: UnitreeG1BackendCfg = UnitreeG1BackendCfg()) -> None:
        self.cfg = cfg
        self._channel_factory_initialize = None
        self._arm_publisher = None
        self._low_state_subscriber = None
        self._hand_publisher = None
        self._hand_state_subscriber = None
        self._low_cmd_factory = None
        self._hand_cmd_factory = None
        self._crc = None
        self._enabled = False
        self._limiter = G1HardwareCommandLimiter(timeout_s=cfg.command_timeout_s)

    def connect(self) -> None:
        """Initialize DDS readers and writers without enabling motion."""
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
            from unitree_sdk2py.utils.crc import CRC
        except ImportError as error:
            raise RuntimeError(
                "Install Unitree SDK2 Python on the robot-control PC: "
                "pip install unitree_sdk2py, or install from unitreerobotics/unitree_sdk2_python."
            ) from error

        if self.cfg.network_interface:
            ChannelFactoryInitialize(0, self.cfg.network_interface)
        else:
            ChannelFactoryInitialize(0)
        self._channel_factory_initialize = ChannelFactoryInitialize
        self._low_cmd_factory = unitree_hg_msg_dds__LowCmd_
        self._low_state_subscriber = ChannelSubscriber(self.cfg.low_state_topic, LowState_)
        self._low_state_subscriber.Init()
        if self.cfg.enable_arms:
            self._arm_publisher = ChannelPublisher(self.cfg.arm_command_topic, LowCmd_)
            self._arm_publisher.Init()
        self._crc = CRC()

        if self.cfg.enable_hands:
            self._connect_inspire_hand(ChannelPublisher, ChannelSubscriber)

    def _connect_inspire_hand(self, publisher_type, subscriber_type) -> None:
        """Create optional Inspire-hand DDS endpoints if the service is installed."""
        try:
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmds_
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
        except ImportError as error:
            raise RuntimeError(
                "Install Unitree's DFX_inspire_service / matching SDK2 Python IDL before enabling Inspire hands."
            ) from error

        self._hand_cmd_factory = unitree_go_msg_dds__MotorCmds_
        self._hand_publisher = publisher_type(self.cfg.hand_command_topic, MotorCmds_)
        self._hand_publisher.Init()
        self._hand_state_subscriber = subscriber_type(self.cfg.hand_state_topic, MotorStates_)
        self._hand_state_subscriber.Init()

    def enable(self) -> None:
        """Open the software command gate from the current measured pose."""
        state = self.read_state()
        self._limiter.reset(_command_positions_from_state(state), state.timestamp_s)
        self._enabled = True

    def read_state(self) -> AmggRobotState:
        """Read one complete 41-D G1 body + Inspire hand state."""
        if self._low_state_subscriber is None:
            raise RuntimeError("Connect the Unitree G1 backend before reading state.")
        low_state = self._read_dds(self._low_state_subscriber, self.cfg.state_timeout_s, "G1 low state")
        body_positions, body_velocities = _extract_body_state(low_state)
        hand_positions, hand_velocities = self._read_hand_state()
        return AmggRobotState(
            timestamp_s=time.monotonic(),
            joint_positions_rad=body_positions + hand_positions,
            joint_velocities_rad_s=body_velocities + hand_velocities,
        )

    def _read_hand_state(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        if not self.cfg.enable_hands:
            return (0.0,) * len(AMGG_G1_HAND_MOTOR_NAMES), (0.0,) * len(AMGG_G1_HAND_MOTOR_NAMES)
        if self._hand_state_subscriber is None:
            raise RuntimeError("The Inspire hand state subscriber is not connected.")
        hand_state = self._read_dds(self._hand_state_subscriber, self.cfg.state_timeout_s, "Inspire hand state")
        motors = _sequence_attr(hand_state, ("states", "motor_state", "state"))
        if len(motors) < len(AMGG_G1_HAND_MOTOR_NAMES):
            raise RuntimeError("The Inspire hand state message does not contain 12 motors.")
        positions = tuple(float(getattr(motors[index], "q", 0.0)) for index in range(len(AMGG_G1_HAND_MOTOR_NAMES)))
        velocities = tuple(float(getattr(motors[index], "dq", 0.0)) for index in range(len(AMGG_G1_HAND_MOTOR_NAMES)))
        return positions, velocities

    def send_joint_position_targets(self, command_rad: tuple[float, ...], timestamp_s: float) -> None:
        """Send one 26-D arm + hand joint-position target after safety filtering."""
        if not self._enabled:
            raise RuntimeError("The Unitree G1 software command gate is closed.")
        state = self.read_state()
        measured = _command_positions_from_state(state)
        command = self._limiter.filter(command_rad, measured, timestamp_s)
        if self.cfg.enable_arms:
            self._publish_arm_command(command[: len(AMGG_G1_CONTROLLED_ARM_JOINT_NAMES)])
        if self.cfg.enable_hands:
            self._publish_hand_command(command[len(AMGG_G1_CONTROLLED_ARM_JOINT_NAMES) :])

    def _publish_arm_command(self, arm_command: Sequence[float]) -> None:
        if self._arm_publisher is None or self._low_cmd_factory is None:
            raise RuntimeError("The Unitree G1 arm publisher is not connected.")
        low_cmd = self._low_cmd_factory()
        low_cmd.mode_pr = 0
        low_cmd.mode_machine = 0
        for motor_index, target in zip(_G1_ARM_INDICES, arm_command, strict=True):
            motor_cmd = low_cmd.motor_cmd[motor_index]
            motor_cmd.mode = 1
            motor_cmd.q = float(target)
            motor_cmd.dq = 0.0
            motor_cmd.tau = 0.0
            motor_cmd.kp = self.cfg.arm_kp
            motor_cmd.kd = self.cfg.arm_kd
        if len(low_cmd.motor_cmd) > _G1_ARM_SDK_ENABLE_INDEX:
            low_cmd.motor_cmd[_G1_ARM_SDK_ENABLE_INDEX].q = 1.0
        if self._crc is not None and hasattr(self._crc, "Crc"):
            low_cmd.crc = self._crc.Crc(low_cmd)
        self._arm_publisher.Write(low_cmd)

    def _publish_hand_command(self, hand_command: Sequence[float]) -> None:
        if self._hand_publisher is None or self._hand_cmd_factory is None:
            raise RuntimeError("The Inspire hand publisher is not connected.")
        message = self._hand_cmd_factory()
        motors = _sequence_attr(message, ("cmds", "motor_cmd", "cmd"))
        if len(motors) < len(AMGG_G1_HAND_MOTOR_NAMES):
            raise RuntimeError("The Inspire hand command message does not contain 12 motors.")
        for index, target in enumerate(hand_command):
            motor_cmd = motors[index]
            _set_if_present(motor_cmd, "q", float(target))
            _set_if_present(motor_cmd, "dq", 0.0)
            _set_if_present(motor_cmd, "tau", 0.0)
            _set_if_present(motor_cmd, "kp", self.cfg.hand_kp)
            _set_if_present(motor_cmd, "kd", self.cfg.hand_kd)
        self._hand_publisher.Write(message)

    def stop(self) -> None:
        """Close the software command gate."""
        self._enabled = False

    def disconnect(self) -> None:
        """Close the software command gate and drop DDS handles."""
        self.stop()
        self._arm_publisher = None
        self._low_state_subscriber = None
        self._hand_publisher = None
        self._hand_state_subscriber = None

    @staticmethod
    def _read_dds(subscriber, timeout_s: float, label: str):
        sample = subscriber.Read(timeout_s)
        if sample is None:
            raise TimeoutError(f"No fresh {label} was received within {timeout_s:.3f} s.")
        return sample


class UnitreeG1DryRunBackend:
    """No-motion G1 backend that follows the 41-D state and 26-D command ABI."""

    def __init__(self) -> None:
        self._connected = False
        self._enabled = False
        self._timestamp_s = 0.0
        self._body_positions = (0.0,) * len(AMGG_G1_BODY_JOINT_NAMES)
        self._hand_positions = (0.0,) * len(AMGG_G1_HAND_MOTOR_NAMES)

    def connect(self) -> None:
        """Mark the deterministic dry-run backend as connected."""
        self._connected = True

    def enable(self) -> None:
        """Open the dry-run software command gate."""
        if not self._connected:
            raise RuntimeError("Connect the G1 dry-run backend before enabling it.")
        self._enabled = True

    def read_state(self) -> AmggRobotState:
        """Return deterministic 41-D G1 state."""
        if not self._connected:
            raise RuntimeError("The G1 dry-run backend is disconnected.")
        return AmggRobotState(
            timestamp_s=self._timestamp_s,
            joint_positions_rad=self._body_positions + self._hand_positions,
            joint_velocities_rad_s=(0.0,) * len(AMGG_G1_OBSERVATION_JOINT_NAMES),
        )

    def send_joint_position_targets(self, command_rad: tuple[float, ...], timestamp_s: float) -> None:
        """Apply a 26-D command to the deterministic dry-run state."""
        if not self._enabled:
            raise RuntimeError("G1 dry-run motion is not enabled.")
        command = _validate_g1_command(command_rad)
        arm_command = command[: len(AMGG_G1_CONTROLLED_ARM_JOINT_NAMES)]
        hand_command = command[len(AMGG_G1_CONTROLLED_ARM_JOINT_NAMES) :]
        body = list(self._body_positions)
        for motor_index, target in zip(_G1_ARM_INDICES, arm_command, strict=True):
            body[motor_index] = target
        self._body_positions = tuple(body)
        self._hand_positions = tuple(hand_command)
        self._timestamp_s = float(timestamp_s)

    def stop(self) -> None:
        """Close the dry-run software command gate."""
        self._enabled = False

    def disconnect(self) -> None:
        """Disconnect the dry-run backend."""
        self.stop()
        self._connected = False


def _validate_g1_command(command_rad: Sequence[float]) -> tuple[float, ...]:
    frozen = tuple(float(value) for value in command_rad)
    if len(frozen) != len(AMGG_G1_HARDWARE_COMMAND_NAMES):
        raise ValueError(
            f"Expected {len(AMGG_G1_HARDWARE_COMMAND_NAMES)} G1 hardware commands, received {len(frozen)}."
        )
    if not all(isfinite(value) for value in frozen):
        raise ValueError("G1 hardware command contains a non-finite value.")
    return frozen


def _extract_body_state(low_state) -> tuple[tuple[float, ...], tuple[float, ...]]:
    motors = _sequence_attr(low_state, ("motor_state", "motor_states"))
    if len(motors) < len(AMGG_G1_BODY_JOINT_NAMES):
        raise RuntimeError("The G1 low-state message does not contain 29 body motors.")
    positions = tuple(float(getattr(motors[index], "q")) for index in range(len(AMGG_G1_BODY_JOINT_NAMES)))
    velocities = tuple(float(getattr(motors[index], "dq", 0.0)) for index in range(len(AMGG_G1_BODY_JOINT_NAMES)))
    return positions, velocities


def _command_positions_from_state(state: AmggRobotState) -> tuple[float, ...]:
    if len(state.joint_positions_rad) != len(AMGG_G1_OBSERVATION_JOINT_NAMES):
        raise ValueError("G1 state does not follow the 41-D observation contract.")
    body = state.joint_positions_rad[: len(AMGG_G1_BODY_JOINT_NAMES)]
    hand = state.joint_positions_rad[len(AMGG_G1_BODY_JOINT_NAMES) :]
    return tuple(body[index] for index in _G1_ARM_INDICES) + hand


def _sequence_attr(message, names: tuple[str, ...]):
    for name in names:
        if hasattr(message, name):
            return getattr(message, name)
    raise AttributeError(f"Message {type(message).__name__} does not expose any of {names}.")


def _set_if_present(message, name: str, value: float) -> None:
    if hasattr(message, name):
        setattr(message, name, value)
