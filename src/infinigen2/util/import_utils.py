# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

"""Utilities for dynamic imports and finding generator functions."""

from pathlib import Path

try:
    from rapidfuzz import process

    rapidfuzz = process
except ImportError:
    rapidfuzz = None


def module_path():
    return Path(__file__).parent.parent
