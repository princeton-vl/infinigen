# Copyright (C) 2023, Princeton University.

# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

import logging
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    __version__ = version("infinigen")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

# Optional optimized packages, off unless asked for; see _perf_selection.
from infinigen import _perf_selection as _perf_selection  # noqa: E402
from infinigen._perf_selection import (  # noqa: E402
    optimized_packages as optimized_packages,
)
from infinigen._perf_selection import use_optimized as use_optimized


def module_parent_path():
    return Path(__file__).parent.parent
