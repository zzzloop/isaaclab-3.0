# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression contract for contact-stable G1 XR configuration."""

import ast
import unittest
from pathlib import Path


def validate_contact_stability_source(source: str) -> None:
    """Validate the source-level settings available without an Isaac Sim runtime."""
    ast.parse(source)
    required_settings = {
        "240 Hz XR physics": "self.sim.dt = 1.0 / 240.0",
        "arm effort limit": "arms.effort_limit_sim = 80.0",
        "arm velocity limit": "arms.velocity_limit_sim = 4.0",
        "hand velocity limit": "hands.velocity_limit_sim = 5.0",
        "object contact margin": "contact_offset=0.005",
        "object contact impulse": "max_contact_impulse=2.0",
        "robot contact override": "robot: ArticulationCfg = _contact_stable_robot()",
        "official scene instance": "ObjectTableSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)",
        "instance robot copy": "official_scene.robot.copy()",
    }
    missing = [name for name, setting in required_settings.items() if setting not in source]
    if missing:
        raise AssertionError(f"Missing contact-stability settings: {missing}")
    if "ObjectTableSceneCfg.robot" in source:
        raise AssertionError("configclass fields must be read from an instance, not the scene class")


class TestAmggG1PhysicsConfig(unittest.TestCase):
    """Prevent regression to unbounded high-speed XR contact dynamics."""

    def test_contact_stability_settings(self) -> None:
        source_path = (
            Path(__file__).resolve().parents[1]
            / "source"
            / "amgg_robot_lab"
            / "amgg_robot_lab"
            / "tasks"
            / "amgg_g1_manipulation_env_cfg.py"
        )
        validate_contact_stability_source(source_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
