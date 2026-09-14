# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the ``amgg_robot_lab`` extension."""

from pathlib import Path

import toml
from setuptools import find_packages, setup

EXTENSION_ROOT = Path(__file__).resolve().parent
EXTENSION_METADATA = toml.load(EXTENSION_ROOT / "config" / "extension.toml")

setup(
    name="amgg_robot_lab",
    version=EXTENSION_METADATA["package"]["version"],
    description=EXTENSION_METADATA["package"]["description"],
    author=EXTENSION_METADATA["package"]["author"],
    maintainer=EXTENSION_METADATA["package"]["maintainer"],
    url=EXTENSION_METADATA["package"]["repository"],
    keywords=EXTENSION_METADATA["package"]["keywords"],
    packages=find_packages(),
    include_package_data=True,
    package_data={"amgg_robot_lab": ["assets/data/**/*"]},
    python_requires=">=3.12",
    license="BSD-3-Clause",
    zip_safe=False,
)
