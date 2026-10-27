# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

"""Optional optimized implementations, reached through gates on the default ones.

`infinigen.core.init` always imports this package so its gin settings register, but
nothing optimized runs unless a package is selected.
"""

import gin

from infinigen.perf import options  # noqa: F401  (registers the settings with gin)
from infinigen.perf.kernels import available, optional_import

__all__ = [
    "apply_gin_selection",
    "available",
    "optional_import",
    "status",
]


def apply_gin_selection():
    """Apply any parsed `optimized.*` gin selection and return the names selected."""
    if gin.get_bindings("optimized"):
        configure_optimized()
    return _selected_names()


def _selected_names():
    from infinigen import _perf_selection

    return sorted(n for n, on in _perf_selection.optimized_packages().items() if on)


def status():
    """One-line summary of the current selection, for logs and bug reports."""
    names = _selected_names()
    return "infinigen.perf: " + (", ".join(names) if names else "none (default path)")


@gin.configurable("optimized")
def configure_optimized(packages=True, exclude=()):
    """Select optimized packages from gin: `packages` is True for all or a list of names
    such as `["trees", "leaves"]`, minus `exclude`. Applied by `apply_gin_selection`."""
    from infinigen import _perf_selection

    return _perf_selection.use_optimized(packages, exclude=exclude)
