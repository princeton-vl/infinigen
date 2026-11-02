# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

"""Native v2, ginko and maple leaf fields, from curve tables sampled in Blender."""

import numpy as np

MAPLE_SHAPES = [
    (
        [
            (0, 0),
            (0.1156, 0.075),
            (0.2109, 0.2719),
            (0.2602, 0.2344),
            (0.3633, 0.2625),
            (0.4171, 0.5545),
            (0.4336, 0.5344),
            (0.4568, 0.7094),
            (0.4749, 0.6012),
            (0.4882, 0.6636),
            (0.5352, 0.4594),
            (0.5484, 0.4375),
            (0.5648, 0.4469),
            (0.6366, 0.7331),
            (0.6719, 0.6562),
            (0.7149, 0.8225),
            (0.768, 0.6344),
            (0.7928, 0.6853),
            (0.8156, 0.5125),
            (0.8297, 0.4906),
            (0.85, 0.5125),
            (0.8988, 0.747),
            (0.9297, 0.6937),
            (0.9648, 0.8937),
            (0.9797, 0.8656),
            (0.9883, 0.8938),
            (1, 1),
        ],
        [2, 5, 7, 9, 13, 15, 17, 21, 23, 25],
    ),
    (
        [
            (0, 0),
            (0.1156, 0.075),
            (0.2109, 0.2719),
            (0.2602, 0.2344),
            (0.3633, 0.2625),
            (0.4336, 0.5344),
            (0.4568, 0.7094),
            (0.4749, 0.6012),
            (0.5352, 0.4594),
            (0.5484, 0.4375),
            (0.5648, 0.4469),
            (0.6719, 0.6562),
            (0.7149, 0.8225),
            (0.768, 0.6344),
            (0.8156, 0.5125),
            (0.8297, 0.4906),
            (0.85, 0.5125),
            (0.9297, 0.6937),
            (0.9883, 0.8938),
            (1, 1),
        ],
        [2, 6, 12, 18],
    ),
]


def curve_tables(specifications):
    import bpy

    material = bpy.data.materials.new("native_planar_curves")
    material.use_nodes = True
    tables = []
    ranges = []
    try:
        for points, vectors in specifications:
            node = material.node_tree.nodes.new("ShaderNodeFloatCurve")
            mapping = node.mapping
            c = mapping.curves[0]
            for i, point in enumerate(points):
                if i < 2:
                    c.points[i].location = point
                else:
                    c.points.new(*point)
            for i in vectors:
                c.points[i].handle_type = "VECTOR"
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
    return np.asarray(tables, np.float32), np.asarray(ranges, np.float32)


def prepare(kind, p):
    if kind == "v2":
        params = [
            p[k]
            for k in (
                "midrib_length",
                "midrib_width",
                "stem_length",
                "vein_asymmetry",
                "vein_angle",
                "vein_density",
                "jigsaw_scale",
                "jigsaw_depth",
            )
        ]
        specs = [
            (p[k], [])
            for k in (
                "midrib_shape_control_points",
                "vein_shape_control_points",
                "leaf_shape_control_points",
            )
        ]
    elif kind == "ginko":
        params = [
            p["vein_length"],
            np.deg2rad(p["angle"]),
            p["multiplier"],
            p["scale_vein"],
            p["scale_wave"],
            p["scale_margin"],
        ]
        specs = [
            (p[k], [])
            for k in ("stem_curve_control_points", "shape_curve_control_points")
        ]
    elif kind == "maple":
        params = [p["angle"] / 180 * np.pi, p["multiplier"], *p["native_vein_phases"]]
        specs = [
            (p["stem_curve_control_points"], []),
            *MAPLE_SHAPES,
            (
                [
                    (0, 0.5),
                    (0.1432, 0.5406),
                    (0.2591, 0.5062),
                    (0.3705, 0.5406),
                    (0.4591, 0.425),
                    (0.5932, 0.4562),
                    (0.7432, 0.3562),
                    (0.8727, 0.5062),
                    (1, 0.5),
                ],
                [],
            ),
            ([(0, 0), (0.5932, 0.1969), (1, 1)], []),
        ]
    else:
        raise ValueError(f"unsupported native leaf {kind}")
    specs += [(p["y_wave_control_points"], []), (p["x_wave_control_points"], [2])]
    tables, ranges = curve_tables(specs)
    return np.asarray(params, np.float32), tables, ranges


def coarse_coverage_mask(kind, shape, prepared, stride):
    """Coarse coverage mask whose nearest-neighbour upsample is conservative."""
    from infinigen.perf.assets.utils import _native_fields

    h, w = shape
    ch = max(2, int(np.ceil(h / stride)))
    cw = max(2, int(np.ceil(w / stride)))
    # Match the bins used by heightfield.coverage.upsample_mask.
    rows = np.minimum(((np.arange(ch) + 0.5) * h / ch).astype(np.intp), h - 1)
    columns = np.minimum(((np.arange(cw) + 0.5) * w / cw).astype(np.intp), w - 1)
    x = columns.astype(np.float32) / np.float32(w - 1) * 2 - 1
    y = rows.astype(np.float32) / np.float32(h - 1) * 2 - 1
    kind_id = ("v2", "ginko", "maple").index(kind)
    # the per-asset reach covers contour motion and thin parts that fall between
    # coarse samples, since upsample_mask adds no padding
    reach_factor = {"v2": 12, "ginko": 4, "maple": 4}[kind]
    reach = np.float32(reach_factor * np.hypot(2 / ch, 2 / cw))
    return (
        _native_fields.planar_coverage(x, y, kind_id, *prepared, reach=float(reach))
        != 0
    )


def rasterize(kind, p, shape, *, defer_bend=False, use_coverage=True):
    from infinigen.perf.assets.utils import _native_fields
    from infinigen.perf.assets.utils.heightfield.coverage import (
        coverage_slices,
        upsample_mask,
    )

    h, w = shape
    if h < 2 or w < 2:
        raise ValueError("raster axes require at least two samples")
    x = np.arange(w, dtype=np.float32) / np.float32(w - 1) * 2 - 1
    y = np.arange(h, dtype=np.float32) / np.float32(h - 1) * 2 - 1
    prepared = prepare(kind, p)
    kind_id = ("v2", "ginko", "maple").index(kind)
    # the ginko silhouette spans nearly the full raster, so a coarse prepass
    # would only add work
    coarse_enabled = use_coverage and kind in ("v2", "maple")
    if coarse_enabled:
        stride = {"v2": 8, "maple": 4}[kind]
        coarse = coarse_coverage_mask(kind, (h, w), prepared, stride)
        supplied_mask = upsample_mask(coarse, (h, w))
        region = coverage_slices(supplied_mask, padding=0)
        coverage = _native_fields.planar_coverage(
            x[region[1]], y[region[0]], kind_id, *prepared
        )
    else:
        coverage = (
            _native_fields.planar_coverage(x, y, kind_id, *prepared)
            if use_coverage
            else None
        )
        region = (
            coverage_slices(coverage)
            if coverage is not None
            else (slice(0, h), slice(0, w))
        )
    gx, gy = np.meshgrid(x[region[1]], y[region[0]])
    xy = np.stack([gx, gy], axis=-1)
    crop_mask = (
        None if coverage is None else (coverage if coarse_enabled else coverage[region])
    )
    result = _native_fields.evaluate_planar_leaf(
        xy.reshape(-1, 2),
        kind_id,
        *prepared,
        defer_bend=defer_bend,
        coverage_mask=crop_mask,
    )
    if defer_bend:
        result, bounds = result
    values = result.reshape(*gx.shape, 6)
    mask = np.zeros((h, w), np.uint8)
    mask[region] = values[..., 0].astype(np.uint8)
    if not mask.any():
        raise ValueError(f"{kind} native field produced an empty silhouette")
    # expand back to the full raster, so the meshing sees the same domain whether
    # or not the evaluation was cropped
    position = np.empty((h, w, 3), np.float32)
    position[..., 0] = x[None, :]
    y_shift = 0 if defer_bend else y[np.flatnonzero(mask.any(axis=1))[0]]
    position[..., 1] = (y - y_shift)[:, None]
    position[..., 2] = 0
    position[region][..., 2] = values[..., 1]
    # samples outside the silhouette are zero rather than extrapolated
    position[..., 2][mask == 0] = 0
    if kind == "v2":
        cropped_attrs = {
            "offset": values[..., 2],
            "coordinate": np.stack([gx, gy, np.zeros_like(gx)], axis=-1),
        }
    elif kind == "ginko":
        cropped_attrs = {"vein": values[..., 2], "shape": values[..., 3]}
    else:
        cropped_attrs = {"vein": values[..., 2]}
    attrs = {}
    for name, value in cropped_attrs.items():
        attrs[name] = np.zeros((h, w) + value.shape[2:], np.float32)
        attrs[name][region] = value
    for value in attrs.values():
        value[mask == 0] = 0
    if defer_bend:
        return mask, position, attrs, (prepared, bounds)
    return mask, position, attrs


def create_mesh(obj, kind, params):
    from infinigen.perf.assets.utils import _native_fields
    from infinigen.perf.assets.utils.heightfield import mesh_route

    raster = rasterize(kind, params, mesh_route.raster_shape(kind), defer_bend=True)
    mask, position, attrs = raster[:3]
    obj = mesh_route._convert_geometry_simplified(
        obj, mask, position, attrs, list(obj.data.materials), kind
    )
    prepared, bounds = raster[3]
    xyz = np.empty((len(obj.data.vertices), 3), np.float32)
    obj.data.vertices.foreach_get("co", xyz.ravel())
    bent = _native_fields.bend_planar_leaf(
        xyz, ("v2", "ginko", "maple").index(kind), *prepared, bounds
    )
    obj.data.vertices.foreach_set("co", bent.ravel())
    obj.data.update()
    return obj
