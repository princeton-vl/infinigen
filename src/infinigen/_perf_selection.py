# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

"""The selection of optimized implementations that every gate consults.

Nothing is selected by default. Set it with `optimized.packages` in gin (recorded in
each stage's operative gin config) or `infinigen.use_optimized()` in Python.
"""

import logging

logger = logging.getLogger(__name__)

# Every optimized implementation a caller may ask for.
SELECTABLE = (
    "TERRAIN",
    "PLACEMENT",
    "TREES",
    "LEAVES",
    "MONOCOT",
    "SMALL_PLANTS",
    "CORALS",
    "CREATURES",
    "MUSHROOM",
    # shared utilities the optimized packages rely on (attribute reads and
    # writes, garbage collection, collections)
    "CORE",
    # process-level settings: input prefetch, compressed stage outputs, OptiX
    # kernel cache
    "RUNTIME",
)

# Names as a caller writes them.
NAMES = {suffix.lower(): suffix for suffix in SELECTABLE}

# What use_optimized() was told, or None when nothing has been selected.
_selection: dict[str, bool] | None = None


def selected(suffix: str) -> bool:
    """Whether optimized package `suffix` is selected."""
    if _selection is None:
        return False
    return _selection.get(suffix, False)


def _resolve(name: str) -> str:
    try:
        return NAMES[name.lower().replace("-", "_")]
    except KeyError:
        known = ", ".join(sorted(NAMES))
        raise ValueError(
            f"unknown optimized package {name!r}; known: {known}"
        ) from None


def use_optimized(packages=True, *, exclude=()) -> dict[str, bool]:
    """Select every optimized package (`True`), none (`False`) or a list of names.

    `exclude` drops names from it. Takes effect at the next call of each gated
    definition; objects already built keep their implementation. Returns the
    resolved selection, name to on/off.
    """
    excluded = {_resolve(n) for n in exclude}

    if packages is True:
        chosen = {suffix: suffix not in excluded for suffix in SELECTABLE}
    elif packages is False:
        chosen = {suffix: False for suffix in SELECTABLE}
    else:
        if isinstance(packages, str):
            packages = [packages]
        wanted = {_resolve(n) for n in packages}
        overlap = wanted & excluded
        if overlap:
            names = ", ".join(sorted(n.lower() for n in overlap))
            raise ValueError(f"{names} is both selected and excluded")
        chosen = {suffix: suffix in wanted for suffix in SELECTABLE}

    global _selection
    _selection = chosen

    from infinigen import _optimized

    _optimized.reset()
    logger.info(
        "infinigen.perf: %s",
        ", ".join(sorted(n for n, suffix in NAMES.items() if chosen[suffix])) or "none",
    )
    return {name: chosen[suffix] for name, suffix in NAMES.items()}


def optimized_packages() -> dict[str, bool]:
    """The current selection, however it was made."""
    return {name: selected(suffix) for name, suffix in NAMES.items()}
