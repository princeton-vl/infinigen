# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import copy
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from typing import Any, Literal

import procfunc as pf

__all__ = [
    "ERROR_MODES",
    "ErrorMode",
    "InfinigenContext",
    "check_names",
    "globals",
    "override_globals",
    "parse_error_modes",
    "raise_or_warn",
    "set_error_modes",
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

    error_mode_displacement_coords: ErrorMode
    """A named-attribute node drives Displacement, which Cycles does not evaluate."""

    error_mode_shader_complexity: ErrorMode
    """A material exceeds SHADER_NODE_COUNT_FAIL flattened shader nodes."""

    error_mode_uv_coords: ErrorMode
    """A material samples a UV layer the mesh lacks, or whose UVs are degenerate."""

    error_mode_material_normal_input: ErrorMode
    """A ShaderNodeNormalMap, or a linked 'Normal'/'Coat Normal' input, encodes bump
    that the displacement pass discards."""

    error_mode_material_texture_vector: ErrorMode
    """A ShaderNodeTex* has an unlinked Vector input, so Cycles samples Generated
    coordinates instead of the intended sample vector."""

    error_mode_material_floating_interface: ErrorMode
    """A node group contains a floating output/input node instead of routing through
    the group interface."""

    error_mode_finite_geometry: ErrorMode
    """A mesh has non-finite (NaN/Inf) vertex coordinates."""

    error_mode_singular_transform: ErrorMode
    """An object has a singular (zero-scale) world transform, so it renders flat."""

    error_mode_cycles_shader: ErrorMode
    """Cycles reported a shader/SVM error on stderr during the render."""


def _mode(name: str, default: ErrorMode) -> ErrorMode:
    var = "INFINIGEN_ERROR_MODE_" + name.upper()
    mode = os.environ.get(var, default)
    assert mode in ERROR_MODES, f"{var}={mode!r} must be one of {ERROR_MODES}"
    return mode  # type: ignore[return-value]


globals = InfinigenContext(
    error_mode_black_frame=_mode("black_frame", "warn"),
    error_mode_unconverged_samples=_mode("unconverged_samples", "warn"),
    error_mode_missing_attribute=_mode("missing_attribute", "warn"),
    error_mode_hidden_render_object=_mode("hidden_render_object", "error"),
    error_mode_displacement_coords=_mode("displacement_coords", "error"),
    error_mode_shader_complexity=_mode("shader_complexity", "error"),
    error_mode_uv_coords=_mode("uv_coords", "error"),
    error_mode_material_normal_input=_mode("material_normal_input", "error"),
    error_mode_material_texture_vector=_mode("material_texture_vector", "error"),
    error_mode_material_floating_interface=_mode(
        "material_floating_interface", "error"
    ),
    error_mode_finite_geometry=_mode("finite_geometry", "error"),
    error_mode_singular_transform=_mode("singular_transform", "error"),
    error_mode_cycles_shader=_mode("cycles_shader", "error"),
)


def _prefixed_names(context: Any, prefix: str) -> tuple[str, ...]:
    return tuple(
        f.name[len(prefix) :] for f in fields(context) if f.name.startswith(prefix)
    )


def check_names() -> tuple[str, ...]:
    """Every check whose severity is settable, across both the infinigen context
    (error_mode_<name> fields) and the procfunc context (warn_mode_<name> fields),
    with the prefix stripped (e.g. "unconverged_samples", "empty_geonodes")."""
    ifg = _prefixed_names(InfinigenContext, _ERROR_MODE_PREFIX)
    procfunc = _prefixed_names(pf.context.globals, _PF_WARN_MODE_PREFIX)
    return ifg + procfunc


def set_error_modes(modes: dict[str, ErrorMode]) -> None:
    """Set the named runtime checks to exactly the given severities, relaxing as well
    as tightening. Dispatches each name to whichever context owns it (see check_names),
    mapping "error" to procfunc's "throw"."""
    ifg = _prefixed_names(InfinigenContext, _ERROR_MODE_PREFIX)
    procfunc = _prefixed_names(pf.context.globals, _PF_WARN_MODE_PREFIX)
    for name, mode in modes.items():
        if name in ifg:
            setattr(globals, _ERROR_MODE_PREFIX + name, mode)
        elif name in procfunc:
            pf_mode = "throw" if mode == "error" else mode
            setattr(pf.context.globals, _PF_WARN_MODE_PREFIX + name, pf_mode)
        else:
            raise ValueError(f"unknown check {name!r}; valid: {check_names()}")


def parse_error_modes(pairs: list[str]) -> dict[str, ErrorMode]:
    """Parse CHECK=MODE strings, e.g. "uv_coords=warn", validating both halves."""
    modes: dict[str, ErrorMode] = {}
    for pair in pairs:
        name, sep, mode = pair.partition("=")
        if not sep:
            raise ValueError(f"expected CHECK=MODE, got {pair!r}")
        if name not in check_names():
            raise ValueError(f"unknown check {name!r}; valid: {check_names()}")
        if mode not in ERROR_MODES:
            raise ValueError(
                f"invalid mode {mode!r} for {name!r}; valid: {ERROR_MODES}"
            )
        modes[name] = mode  # type: ignore[assignment]
    return modes


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
