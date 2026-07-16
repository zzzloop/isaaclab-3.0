# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate AMGG URDF, mesh references, joints, and required frames."""

from amgg_robot_lab.assets import AMGG_URDF_PATH


def main() -> None:
    """Validate the supplied AMGG model before launching Isaac Sim."""
    if not AMGG_URDF_PATH.is_file():
        raise SystemExit(f"Missing AMGG URDF: {AMGG_URDF_PATH}")
    raise SystemExit("AMGG asset parser will be implemented after the model package is supplied.")


if __name__ == "__main__":
    main()
