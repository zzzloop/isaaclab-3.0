# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression test for imports from the Isaac Lab repository root."""

import subprocess
import sys
import unittest
from pathlib import Path


class TestAmggDevelopmentImport(unittest.TestCase):
    """Ensure the outer project directory does not shadow the source package."""

    def test_assets_import_from_repository_root(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import amgg_robot_lab.assets as assets; print(assets.AMGG_URDF_PATH.name)",
            ],
            cwd=repository_root,
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "amgg_robot.urdf")


if __name__ == "__main__":
    unittest.main()
