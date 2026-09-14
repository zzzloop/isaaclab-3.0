# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline math, asset and UDP tests; does not start Isaac Sim, ROS or hardware."""

import copy
import json
import socket
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from model import ARMS, TIPS, URDF, RobotModel, pose_matrix, rotation, rotation_error
from pico_input import PICO_TO_ROBOT, PicoReceiver, PicoRetargeter, validate_packet


def sample():
    """Build a synthetic Pico frame with released buttons."""
    return {
        "version": 1,
        "session": "test",
        "seq": 0,
        "head": [0, 1.6, 0, 0, 0, 0, 1],
        "left": [-0.3, 1.2, 0.3, 0, 0, 0, 1],
        "right": [0.3, 1.2, 0.3, 0, 0, 0, 1],
        "left_joy": {"trigger": 0.0, "a": False, "b": False},
        "right_joy": {"trigger": 0.0, "a": False, "b": False},
    }


class OfflineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = RobotModel()

    def test_asset_integrity(self):
        root = ET.parse(URDF).getroot()
        self.assertEqual(len(self.model.limits), 31)
        mesh_paths = set()
        for mesh in root.iter("mesh"):
            path = URDF.parent.parent / mesh.get("filename").removeprefix("package://AM-DP123/")
            self.assertTrue(path.is_file(), str(path))
            self.assertGreater(path.stat().st_size, 84)
            mesh_paths.add(path)
        self.assertEqual(len(mesh_paths), 61)
        for link in root.findall("link"):
            inertia = link.find("inertial/inertia")
            if inertia is not None:
                values = {key: float(value) for key, value in inertia.attrib.items()}
                tensor = np.array(
                    [
                        [values["ixx"], values["ixy"], values["ixz"]],
                        [values["ixy"], values["iyy"], values["iyz"]],
                        [values["ixz"], values["iyz"], values["izz"]],
                    ]
                )
                self.assertGreater(np.linalg.eigvalsh(tensor).min(), 0, link.get("name"))

    def test_urdf_jacobian_against_finite_difference(self):
        for side, names in ARMS.items():
            q = dict(self.model.home)
            for name in names:
                lo, hi = self.model.limits[name]
                q[name] = (lo + hi) / 2
            actual, jacobian = self.model.forward(TIPS[side], q, names)
            for column, name in enumerate(names):
                perturbed = dict(q)
                perturbed[name] += 1e-6
                moved, _ = self.model.forward(TIPS[side], perturbed)
                numeric = np.r_[
                    (moved[:3, 3] - actual[:3, 3]) / 1e-6, rotation_error(moved[:3, :3], actual[:3, :3]) / 1e-6
                ]
                np.testing.assert_allclose(jacobian[:, column], numeric, atol=2e-6)

    def test_ik_reaches_known_poses_without_moving_other_joints(self):
        rng = np.random.default_rng(123)
        for side, names in ARMS.items():
            for _ in range(6):
                expected = dict(self.model.home)
                for name in names:
                    expected[name] = float(np.clip(expected[name] + rng.uniform(-0.08, 0.08), *self.model.limits[name]))
                target, _ = self.model.forward(TIPS[side], expected)
                solved, ep, er = self.model.solve(side, target, self.model.home)
                self.assertLess(ep, 0.003)
                self.assertLess(er, 0.025)
                for name in self.model.limits:
                    self.assertTrue(self.model.limits[name][0] <= solved[name] <= self.model.limits[name][1])
                    if name not in names:
                        self.assertEqual(solved[name], self.model.home[name])

    def test_unreachable_target_has_large_residual(self):
        target, _ = self.model.forward(TIPS["left"], self.model.home)
        target[0, 3] += 10
        _, ep, _ = self.model.solve("left", target, self.model.home)
        self.assertGreater(ep, 5)

    def test_pi_rotation_and_xyzw_convention(self):
        axis = np.array([1, -2, 3]) / np.sqrt(14)
        matrix = rotation(axis, np.pi)
        vector = rotation_error(matrix, np.eye(3))
        np.testing.assert_allclose(rotation(vector, np.linalg.norm(vector)), matrix, atol=1e-7)
        matrix = pose_matrix([0, 0, 0, 0, 0, np.sqrt(0.5), np.sqrt(0.5)])
        np.testing.assert_allclose(matrix[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-7)

    def test_calibration_direction_pause_rearm(self):
        retargeter = PicoRetargeter(self.model)
        packet = sample()
        self.assertIsNone(retargeter.update(packet, self.model.home))
        for side in ARMS:
            packet[f"{side}_joy"]["a"] = True
        targets = retargeter.update(packet, self.model.home)
        for side in ARMS:
            np.testing.assert_allclose(targets[side], self.model.forward(TIPS[side], self.model.home)[0], atol=1e-10)
        packet["left"][2] += 0.05  # Pico forward -> robot +X, not -X.
        moved = retargeter.update(packet, self.model.home)
        np.testing.assert_allclose(moved["left"][:3, 3] - targets["left"][:3, 3], [0.05, 0, 0], atol=1e-10)
        np.testing.assert_allclose(moved["right"], targets["right"])
        retargeter.pause()
        self.assertIsNone(retargeter.update(packet, self.model.home))
        for side in ARMS:
            packet[f"{side}_joy"]["a"] = False
        self.assertIsNone(retargeter.update(packet, self.model.home))
        for side in ARMS:
            packet[f"{side}_joy"]["a"] = True
        rearmed = retargeter.update(packet, self.model.home)
        np.testing.assert_allclose(rearmed["left"], targets["left"], atol=1e-10)

    def test_basis_rotations_stay_proper(self):
        reflected = PICO_TO_ROBOT @ rotation([1, 0, 0], 0.4) @ PICO_TO_ROBOT.T
        self.assertAlmostEqual(np.linalg.det(reflected), 1)
        np.testing.assert_allclose(reflected @ reflected.T, np.eye(3), atol=1e-10)

    def test_invalid_packets(self):
        for key, value in (("left", [0] * 7), ("right", [float("nan")] * 7), ("seq", -1)):
            packet = sample()
            packet[key] = value
            with self.assertRaises(ValueError):
                validate_packet(packet)

    def test_udp_duplicate_reordered_and_invalid_packets(self):
        receiver = PicoReceiver("127.0.0.1", 0)
        try:
            address = receiver.socket.getsockname()
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                packet = sample()
                sender.sendto(json.dumps(packet).encode(), address)
                self.assertEqual(receiver.poll()["seq"], 0)
                accepted_at = receiver.received_at
                sender.sendto(json.dumps(packet).encode(), address)
                sender.sendto(b"invalid JSON", address)
                self.assertIsNone(receiver.poll())
                self.assertEqual(receiver.received_at, accepted_at)
                self.assertEqual(receiver.invalid, 1)
                packet = copy.deepcopy(packet)
                packet["seq"] = 2
                sender.sendto(json.dumps(packet).encode(), address)
                packet["seq"] = 1
                sender.sendto(json.dumps(packet).encode(), address)
                self.assertEqual(receiver.poll()["seq"], 2)
        finally:
            receiver.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
