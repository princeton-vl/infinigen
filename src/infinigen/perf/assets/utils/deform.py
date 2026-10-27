# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

import numpy as np

AXES = {"X": 0, "Y": 1, "Z": 2}
BEND_EPS = 1e-4  # smaller bend factors leave the coordinates unchanged


def simple_deform(co, mode, angle, deform_axis="X", limits=(0.0, 1.0)):
    """Blender's SIMPLE_DEFORM (BEND or TWIST; no origin, vertex group or lock axis)
    on a copy of the (N, 3) `co`. The angle spans the `limits` fraction of the extent
    along the limit axis (the deform axis for twist; Z for a bend about X or Y, X for
    a bend about Z); points outside are clamped and their overshoot added back."""
    co = np.asarray(co, dtype=np.float64).copy()
    axis = AXES[deform_axis] if isinstance(deform_axis, str) else int(deform_axis)
    if mode == "BEND":
        limit_axis = 2 if axis in (0, 1) else 0
    else:
        limit_axis = axis
    lower = co[:, limit_axis].min()
    upper = co[:, limit_axis].max()
    lo, hi = min(limits), max(limits)
    smd_limit = (lower + (upper - lower) * lo, lower + (upper - lower) * hi)
    factor = angle / max(np.finfo(np.float32).eps, smd_limit[1] - smd_limit[0])
    if mode == "BEND" and abs(factor) < BEND_EPS:
        return co
    dcut = np.zeros_like(co)
    c = co[:, limit_axis]
    above = c > smd_limit[1]
    below = c < smd_limit[0]
    dcut[above, limit_axis] = c[above] - smd_limit[1]
    dcut[below, limit_axis] = c[below] - smd_limit[0]
    c = np.clip(c, smd_limit[0], smd_limit[1])
    co[:, limit_axis] = c
    x, y, z = co[:, 0], co[:, 1], co[:, 2]
    out = np.empty_like(co)
    if mode == "TWIST":
        theta = co[:, axis] * factor
        sint, cost = np.sin(theta), np.cos(theta)
        if axis == 0:
            out[:, 0] = x
            out[:, 1] = y * cost - z * sint
            out[:, 2] = y * sint + z * cost
        elif axis == 1:
            out[:, 0] = x * cost + z * sint
            out[:, 1] = y
            out[:, 2] = -x * sint + z * cost
        else:
            out[:, 0] = x * cost - y * sint
            out[:, 1] = x * sint + y * cost
            out[:, 2] = z
        out += dcut
        return out
    if mode != "BEND":
        raise ValueError(f"unsupported simple deform mode {mode}")
    theta = (z if axis in (0, 1) else x) * factor
    sint, cost = np.sin(theta), np.cos(theta)
    inv = 1.0 / factor
    if axis == 0:
        out[:, 0] = x + dcut[:, 0]
        out[:, 1] = y * cost + (1.0 - cost) * inv + sint * dcut[:, 2]
        out[:, 2] = -(y - inv) * sint + cost * dcut[:, 2]
    elif axis == 1:
        out[:, 0] = x * cost + (1.0 - cost) * inv + sint * dcut[:, 2]
        out[:, 1] = y + dcut[:, 1]
        out[:, 2] = -(x - inv) * sint + cost * dcut[:, 2]
    else:
        out[:, 0] = -(y - inv) * sint + cost * dcut[:, 0]
        out[:, 1] = (y - inv) * cost + inv + sint * dcut[:, 0]
        out[:, 2] = z + dcut[:, 2]
    return out


def rotation_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def apply_rotation(co, matrix):
    """Coordinates after `transform_apply(rotation=True)` with the 3x3 `matrix`."""
    return np.asarray(co, dtype=np.float64) @ np.asarray(matrix, dtype=np.float64).T
