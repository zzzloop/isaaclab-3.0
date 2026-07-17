# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMGG robot simulation, teleoperation, recording, and deployment package."""

__version__ = "0.1.0"

# Keep URDF/FK/dataset tooling usable in lightweight Python environments. The
# Isaac Lab runtime always provides Gymnasium and therefore registers tasks.
try:
    import gymnasium  # noqa: F401
except ModuleNotFoundError:
    pass
else:
    from . import tasks  # noqa: F401, E402
