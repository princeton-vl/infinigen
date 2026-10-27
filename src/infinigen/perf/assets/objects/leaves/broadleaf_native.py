# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

"""Broadleaf fields evaluated natively from curve tables sampled once in Blender."""

from dataclasses import dataclass

import numpy as np

FIELD_NAMES = (
    "mask",
    "z",
    "coordinate_x",
    "coordinate_y",
    "offset",
    "subvein_offset",
    "vein_value",
    "x_modulated",
)
# Scale and bias each field is encoded with; the raster values are rounded
# through this encoding in float32.
ENCODING = {
    "mask": (1.0, 0.0),
    "z": (8.0, 0.25),
    "coordinate_x": (0.25, 0.5),
    "coordinate_y": (0.25, 0.5),
    "offset": (1.0, 0.0),
    "subvein_offset": (0.25, 0.5),
    "vein_value": (1.0, 0.0),
    "x_modulated": (0.25, 0.5),
}
VARIANTS = (
    (
        [
            (0, 0),
            (0.0182, 0.05),
            (0.3364, 0.2386),
            (0.6045, 0.4812),
            (0.7, 0.725),
            (0.8273, 0.8437),
            (1, 1),
        ],
        ["AUTO", "AUTO", "AUTO", "VECTOR", "AUTO", "AUTO", "AUTO"],
    ),
    (
        [(0, 0), (0.0182, 0.05), (0.3364, 0.2386), (0.8091, 0.7312), (1, 0.9937)],
        ["AUTO"] * 5,
    ),
    (
        [
            (0, 0),
            (0.0182, 0.05),
            (0.2909, 0.2199),
            (0.4182, 0.3063),
            (0.7045, 0.3),
            (1, 0.8562),
        ],
        ["AUTO", "AUTO", "AUTO", "VECTOR", "AUTO", "AUTO"],
    ),
)


@dataclass(frozen=True)
class PreparedBroadleaf:
    parameters: np.ndarray
    tables: np.ndarray
    ranges: np.ndarray


def prepare_broadleaf(params):
    import bpy

    specifications = [
        (params["midrib_curve_control_points"], None),
        (params["shape_curve_control_points"], None),
        *VARIANTS,
        ([(0, 0), (0.0182, 0.05), (0.3364, 0.2386), (0.7227, 0.75), (1, 1)], None),
        ([(0, 0), (0.5182, 1), (1, 1)], None),
        ([(0.0045, 0.0063), (0.0409, 0.0375), (0.4182, 0.05), (1, 0)], None),
    ]
    material = bpy.data.materials.new("native_field_curve_preparation")
    material.use_nodes = True
    tables = []
    ranges = []
    try:
        for points, handles in specifications:
            node = material.node_tree.nodes.new("ShaderNodeFloatCurve")
            mapping = node.mapping
            c = mapping.curves[0]
            for i, point in enumerate(points):
                if i < 2:
                    c.points[i].location = point
                else:
                    c.points.new(*point)
                if handles is not None:
                    c.points[i].handle_type = handles[i]
            mapping.update()
            lo = min(p.location.x for p in c.points)
            hi = max(p.location.x for p in c.points)
            ranges.append([lo, hi])
            tables.append(
                [
                    mapping.evaluate(
                        c,
                        float(
                            np.float32(lo) + np.float32(i / 256) * np.float32(hi - lo)
                        ),
                    )
                    for i in range(257)
                ]
            )
    finally:
        bpy.data.materials.remove(material)
    # these products are formed in Python before the float32 conversion, as for
    # the default's shader constants
    p = [
        params[k]
        for k in (
            "midrib_length",
            "stem_length",
            "vein_asymmetry",
            "vein_angle",
            "vein_density",
            "vein_mask_random_seed",
            "jigsaw_scale",
            "jigsaw_depth",
        )
    ]
    p += [
        params["vein_density"] * params["vein_mask_random_seed"],
        params["vein_density"] * (params["vein_mask_random_seed"] * 2.7),
        params["jigsaw_depth"] * 0.05,
    ]
    return PreparedBroadleaf(
        np.asarray(p, np.float32),
        np.asarray(tables, np.float32),
        np.asarray(ranges, np.float32),
    )


def evaluate_broadleaf(prepared, xy, *, coverage_mask=None):
    from infinigen.perf.assets.utils import _native_fields

    xy = np.asarray(xy, dtype=np.float32)
    if xy.ndim < 2 or xy.shape[-1] != 2:
        raise ValueError("expected coordinates with final dimension 2")
    shape = xy.shape[:-1]
    if coverage_mask is not None and np.shape(coverage_mask) != shape:
        raise ValueError("mask must match coordinate shape")
    output = _native_fields.evaluate_broadleaf(
        xy.reshape(-1, 2),
        prepared.parameters,
        prepared.tables,
        prepared.ranges,
        None if coverage_mask is None else np.asarray(coverage_mask).reshape(-1),
    )
    output = output.reshape(*shape, 8)
    return {name: output[..., i] for i, name in enumerate(FIELD_NAMES)}


def broadleaf_coverage_mask(size, params, prepared=None):
    """Conservative mask of the blade and the full noisy midrib, bounded by the curve
    lookup tables. The midrib test uses abs(y), so it reaches past both ends of the
    blade's [-0.6, 0.6]."""
    if prepared is None:
        prepared = prepare_broadleaf(params)
    mid = float(params["midrib_length"])
    stem = float(params["stem_length"])
    # outside the parameter ranges these bounds assume, evaluate everywhere
    if (
        not 0 <= mid <= 0.8
        or not 0 <= stem <= 1
        or params["jigsaw_depth"] < 0
        or not np.array_equal(
            prepared.ranges[:2], np.array([[0, 1], [0, 1]], np.float32)
        )
    ):
        return np.ones((size, size), bool)
    coordinates = np.arange(size, dtype=np.float32) * np.float32(
        2 / (size - 1)
    ) - np.float32(1)
    coverage = np.zeros((size, size), bool)
    for row, y in enumerate(coordinates):
        t = float(np.clip((float(y) + 0.6) / 1.2, 0, 1))
        i = int(t * 256)
        # Native interpolation is a convex combination of adjacent LUT entries.
        # Extra neighbors cover float32 indexing roundoff, including endpoints.
        samples = np.clip(prepared.tables[:2, max(0, i - 2) : min(257, i + 4)], 0, 1)
        low = 2 * float(samples[0].min()) - 1
        high = 2 * float(samples[0].max()) - 1
        radius = float(samples[1].max()) + 0.00201
        dy = max(abs(float(y)) - 0.6, 0)
        if dy <= radius:
            coverage[row] |= (coordinates >= low - radius) & (
                coordinates <= high + radius
            )
        if abs(float(y)) <= stem + 0.00001:
            width = mid * float(np.clip((mid - float(y)) / (70 + mid), 0, 1)) + 0.01001
            coverage[row] |= (coordinates >= low - width) & (
                coordinates <= high + width
            )
    for _ in range(2):
        padded = np.pad(coverage, 1)
        coverage = np.logical_or.reduce(
            [
                padded[dy : dy + size, dx : dx + size]
                for dy in range(3)
                for dx in range(3)
            ]
        )
    return coverage


def rasterize_broadleaf_native_fields(size, params):
    """Every broadleaf field on the coverage box, rounded through `ENCODING`."""
    from infinigen.perf.assets.utils.heightfield.coverage import (
        canvas_from_coverage_mask,
    )

    prepared = prepare_broadleaf(params)
    coverage = broadleaf_coverage_mask(size, params, prepared)
    canvas = canvas_from_coverage_mask(size, coverage)
    region = np.s_[
        canvas.y_start : canvas.y_stop + 1, canvas.x_start : canvas.x_stop + 1
    ]
    coverage = coverage[region]
    step = np.float32(2.0 / (size - 1))
    x = np.arange(
        canvas.x_start, canvas.x_stop + 1, dtype=np.float32
    ) * step - np.float32(1)
    y = np.arange(
        canvas.y_start, canvas.y_stop + 1, dtype=np.float32
    ) * step - np.float32(1)
    gx, gy = np.meshgrid(x, y)
    coordinates = np.stack([gx, gy], axis=-1)
    fields = evaluate_broadleaf(prepared, coordinates, coverage_mask=coverage)
    result = {}
    for name, values in fields.items():
        scale, bias = map(np.float32, ENCODING[name])
        encoded = values * scale + bias
        encoded[~coverage] = 0
        result[name] = np.ascontiguousarray((encoded - bias) / scale)
    return result
