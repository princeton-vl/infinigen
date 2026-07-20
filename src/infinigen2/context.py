# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import copy
import logging
import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from typing import Any, Literal

import procfunc as pf

__all__ = [
    "ERROR_MODES",
    "ErrorMode",
    "InfinigenContext",
    "globals",
    "override_globals",
    "raise_or_warn",
    "set_strict",
    "strict_names",
]

ErrorMode = Literal["ignore", "warn", "error"]

ERROR_MODES = ("ignore", "warn", "error")

_ERROR_MODE_PREFIX = "error_mode_"
_PF_WARN_MODE_PREFIX = "warn_mode_"


@dataclass
class InfinigenContext:
    """Global context controlling the severity of infinigen's runtime checks."""

    error_mode_black_frame: ErrorMode
    """A rendered RGB frame whose mean pixel value is ~zero, i.e. an all-black render."""

    error_mode_unconverged_samples: ErrorMode
    """Adaptive sampling hit the max-samples cap with pixels still above the noise threshold."""

    error_mode_missing_attribute: ErrorMode
    """A shader samples a named attribute/color absent on the mesh (Cycles reads zeros)."""

    error_mode_hidden_render_object: ErrorMode
    """An object passed to the render is hidden from camera, so it silently never renders."""


def _env_error_mode(name: str, default: ErrorMode) -> ErrorMode:
    mode = os.environ.get(name, default)
    assert mode in ERROR_MODES, f"{name}={mode!r} must be one of {ERROR_MODES}"
    return mode  # type: ignore[return-value]


globals = InfinigenContext(
    error_mode_black_frame=_env_error_mode("INFINIGEN_ERROR_MODE_BLACK_FRAME", "warn"),
    error_mode_unconverged_samples=_env_error_mode(
        "INFINIGEN_ERROR_MODE_UNCONVERGED_SAMPLES", "warn"
    ),
    error_mode_missing_attribute=_env_error_mode(
        "INFINIGEN_ERROR_MODE_MISSING_ATTRIBUTE", "warn"
    ),
    error_mode_hidden_render_object=_env_error_mode(
        "INFINIGEN_ERROR_MODE_HIDDEN_RENDER_OBJECT", "error"
    ),
)


def _prefixed_names(context: Any, prefix: str) -> tuple[str, ...]:
    return tuple(
        f.name[len(prefix) :] for f in fields(context) if f.name.startswith(prefix)
    )


def strict_names() -> tuple[str, ...]:
    """Every check accepted by set_strict and --strict, across both the infinigen
    context (error_mode_<name> fields) and the procfunc context (warn_mode_<name>
    fields), with the prefix stripped (e.g. "unconverged_samples", "empty_geonodes")."""
    ifg = _prefixed_names(InfinigenContext, _ERROR_MODE_PREFIX)
    procfunc = _prefixed_names(pf.context.globals, _PF_WARN_MODE_PREFIX)
    return ifg + procfunc


def set_strict(names: Iterable[str]) -> None:
    """Upgrade the named runtime checks to their fatal severity so a violation raises
    instead of warning: infinigen checks to "error", procfunc checks to "throw".
    Upgrade-only; never relaxes a check. Dispatches each name to whichever context
    owns it (see strict_names)."""
    ifg = _prefixed_names(InfinigenContext, _ERROR_MODE_PREFIX)
    procfunc = _prefixed_names(pf.context.globals, _PF_WARN_MODE_PREFIX)
    for name in names:
        if name not in ifg and name not in procfunc:
            raise ValueError(f"unknown strict check {name!r}; valid: {strict_names()}")
        if name in ifg:
            setattr(globals, _ERROR_MODE_PREFIX + name, "error")
        else:
            setattr(pf.context.globals, _PF_WARN_MODE_PREFIX + name, "throw")


def raise_or_warn(mode: ErrorMode, error: Exception, logger: logging.Logger) -> None:
    if mode == "error":
        raise error
    if mode == "warn":
        logger.warning(str(error))


@contextmanager
def override_globals(
    new_context: InfinigenContext | None = None,
    **overrides: Any,
) -> Iterator[None]:
    orig = copy.deepcopy(globals)

    if new_context is not None:
        for key, value in asdict(new_context).items():
            setattr(globals, key, value)

    for key, value in overrides.items():
        setattr(globals, key, value)

    try:
        yield
    finally:
        for key, value in asdict(orig).items():
            setattr(globals, key, value)
