# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Development-tree import bridge for the nested AMGG extension package.

The external project directory and its Python package intentionally share the
``amgg_robot_lab`` name. PEP 660 editable finders are ordered after Python's
standard path finder, so running a script from the Isaac Lab repository root
can otherwise resolve this directory as an empty namespace package. Extending
the package path keeps repository-root scripts and tests consistent with a
regular wheel installation.
"""

from __future__ import annotations

import importlib
from pathlib import Path

_SOURCE_PACKAGE_DIR = Path(__file__).resolve().parent / "source" / "amgg_robot_lab" / "amgg_robot_lab"
if not (_SOURCE_PACKAGE_DIR / "__init__.py").is_file():
    raise ImportError(f"AMGG source package was not found at {_SOURCE_PACKAGE_DIR}")

__path__.append(str(_SOURCE_PACKAGE_DIR))
__version__ = "0.1.0"

try:
    import gymnasium  # noqa: F401
except ModuleNotFoundError:
    pass
else:
    importlib.import_module(f"{__name__}.tasks")
