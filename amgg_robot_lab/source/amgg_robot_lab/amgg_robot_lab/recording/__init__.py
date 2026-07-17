# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMGG demonstration recording and dataset contracts."""

from .amgg_dataset_schema import (
    AMGG_ACTION_KEY,
    AMGG_ENVIRONMENT_STATE_KEY,
    AMGG_IMAGE_PREFIX,
    AMGG_OBSERVATION_STATE_KEY,
    AMGG_SCHEMA_VERSION,
    AMGG_TCP_STATE_KEY,
    AmggDatasetSpec,
    make_amgg_dataset_spec,
)
from .amgg_g1_dataset_schema import AmggG1DatasetSpec, make_amgg_g1_dataset_spec

__all__ = [
    "AMGG_ACTION_KEY",
    "AMGG_ENVIRONMENT_STATE_KEY",
    "AMGG_IMAGE_PREFIX",
    "AMGG_OBSERVATION_STATE_KEY",
    "AMGG_SCHEMA_VERSION",
    "AMGG_TCP_STATE_KEY",
    "AmggG1DatasetSpec",
    "AmggDatasetSpec",
    "make_amgg_g1_dataset_spec",
    "make_amgg_dataset_spec",
]
