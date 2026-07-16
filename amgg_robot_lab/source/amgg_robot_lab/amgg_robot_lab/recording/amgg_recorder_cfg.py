# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab recorder configuration used by the official demo recorder."""


def build_amgg_recorder_cfg():
    """Build the recorder used by ``scripts/tools/record_demos.py``."""
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg

    return ActionStateRecorderManagerCfg()
