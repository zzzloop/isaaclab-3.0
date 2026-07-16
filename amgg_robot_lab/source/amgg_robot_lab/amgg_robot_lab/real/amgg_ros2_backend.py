# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurable ROS 2 JointState/JointTrajectory backend for AMGG."""

from __future__ import annotations

from dataclasses import dataclass

from amgg_robot_lab.contracts import AMGG_CONTROLLED_JOINT_NAMES, AMGG_OBSERVED_JOINT_NAMES
from amgg_robot_lab.teleop.amgg_safety import AmggCommandLimiter

from .amgg_robot_backend import AmggRobotState


@dataclass(frozen=True, slots=True)
class AmggRos2BackendCfg:
    """ROS 2 transport settings that must be matched to the real controller."""

    node_name: str = "amgg_teleop_bridge"
    state_topic: str = "/joint_states"
    command_topic: str = "/joint_trajectory_controller/joint_trajectory"
    state_timeout_s: float = 0.25
    command_horizon_s: float = 0.10


class AmggRos2Backend:
    """Publish ordered position targets and consume ordered measured state.

    This adapter does not replace a hardware emergency stop, drive-enable
    service, collision monitor, or vendor safety controller. ``enable`` only
    opens the software command gate after a complete fresh state is received.
    """

    def __init__(self, cfg: AmggRos2BackendCfg = AmggRos2BackendCfg()) -> None:
        self.cfg = cfg
        self._node = None
        self._publisher = None
        self._executor = None
        self._state: AmggRobotState | None = None
        self._enabled = False
        self._owns_rclpy = False
        self._limiter = AmggCommandLimiter(timeout_s=cfg.state_timeout_s)

    def connect(self) -> None:
        """Create ROS interfaces without enabling command output."""
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from sensor_msgs.msg import JointState
            from trajectory_msgs.msg import JointTrajectory
        except ImportError as error:
            raise RuntimeError("Install and source ROS 2 with sensor_msgs and trajectory_msgs.") from error
        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True
        self._node = rclpy.create_node(self.cfg.node_name)
        self._publisher = self._node.create_publisher(JointTrajectory, self.cfg.command_topic, 10)
        self._node.create_subscription(JointState, self.cfg.state_topic, self._state_callback, 20)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)

    def _state_callback(self, message) -> None:
        names = list(message.name)
        missing = set(AMGG_OBSERVED_JOINT_NAMES) - set(names)
        if missing:
            return
        indices = [names.index(name) for name in AMGG_OBSERVED_JOINT_NAMES]
        positions = tuple(float(message.position[index]) for index in indices)
        if len(message.velocity) == len(names):
            velocities = tuple(float(message.velocity[index]) for index in indices)
        else:
            velocities = (0.0,) * len(indices)
        if self._node is None:
            return
        timestamp_s = self._node.get_clock().now().nanoseconds * 1e-9
        self._state = AmggRobotState(timestamp_s, positions, velocities)

    def read_state(self) -> AmggRobotState:
        """Spin once and return a complete, fresh canonical state."""
        if self._executor is None or self._node is None:
            raise RuntimeError("Connect the AMGG ROS 2 backend first.")
        self._executor.spin_once(timeout_sec=self.cfg.state_timeout_s)
        if self._state is None:
            raise TimeoutError("No complete canonical AMGG JointState was received.")
        now_s = self._node.get_clock().now().nanoseconds * 1e-9
        if now_s - self._state.timestamp_s > self.cfg.state_timeout_s:
            raise TimeoutError("The AMGG JointState stream is stale.")
        return self._state

    def enable(self) -> None:
        """Open the software command gate after checking fresh measured state."""
        state = self.read_state()
        self._limiter.reset(state.joint_positions_rad[: len(AMGG_CONTROLLED_JOINT_NAMES)], state.timestamp_s)
        self._enabled = True

    def send_joint_position_targets(self, command_rad: tuple[float, ...], timestamp_s: float) -> None:
        """Rate-limit and publish one canonical JointTrajectory target."""
        if not self._enabled or self._publisher is None or self._node is None:
            raise RuntimeError("AMGG ROS 2 command gate is closed.")
        if not isinstance(timestamp_s, (float, int)):
            raise ValueError("AMGG source timestamp must be numeric.")
        state = self.read_state()
        measured = state.joint_positions_rad[: len(AMGG_CONTROLLED_JOINT_NAMES)]
        now_s = self._node.get_clock().now().nanoseconds * 1e-9
        command = self._limiter.filter(command_rad, measured, now_s)
        from builtin_interfaces.msg import Duration
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        message = JointTrajectory()
        message.joint_names = list(AMGG_CONTROLLED_JOINT_NAMES)
        point = JointTrajectoryPoint()
        point.positions = list(command)
        horizon_ns = int(self.cfg.command_horizon_s * 1e9)
        point.time_from_start = Duration(sec=horizon_ns // 1_000_000_000, nanosec=horizon_ns % 1_000_000_000)
        message.points = [point]
        self._publisher.publish(message)

    def stop(self) -> None:
        """Close the software gate; the hardware controller must enforce hold."""
        self._enabled = False

    def disconnect(self) -> None:
        """Destroy ROS resources and close the command gate."""
        self.stop()
        if self._executor is not None and self._node is not None:
            self._executor.remove_node(self._node)
        if self._node is not None:
            self._node.destroy_node()
        if self._owns_rclpy:
            import rclpy

            rclpy.shutdown()
        self._executor = self._publisher = self._node = None
