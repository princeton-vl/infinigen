# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lingjie Mei: original Infinigen drawing utilities (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/assets/utils/draw.py)
# - David Yan, Jia Deng: optimized implementation

"""`curve2mesh` and `remesh_fill` (CORE) skip an operator call and a geometry-nodes
pass; `leaf` (MONOCOT) builds its sheet from arrays for all callers, monocot or not."""

import bpy
import numpy as np

from infinigen.assets.utils import draw as _default
from infinigen.assets.utils.decorate import remove_vertices
from infinigen.core.placement.detail import sharp_remesh_with_attrs
from infinigen.core.surface import read_attr_data
from infinigen.core.util import blender as butil
from infinigen.perf import options


def monocot_legacy():
    """True when `direct_monocot_leaves.enabled = False` asks for the default,
    operator-based monocot leaf sheets, decoration and agave join."""
    return not options.direct_monocot_leaves()


def curve2mesh(obj):
    points = obj.data.splines[0].bezier_points
    cos = np.array([p.co for p in points])
    length = np.linalg.norm(cos[:-1] - cos[1:], axis=-1)
    min_length = 5e-3
    with butil.ViewportMode(obj, "EDIT"):
        for i in range(len(points)):
            if points[i].handle_left_type == "FREE":
                points[i].handle_left_type = "ALIGNED"
            if points[i].handle_right_type == "FREE":
                points[i].handle_right_type = "ALIGNED"
        for i in reversed(range(len(points) - 1)):
            points = list(obj.data.splines[0].bezier_points)
            number_cuts = min(int(length[i] / min_length) - 1, 64)
            if number_cuts < 0:
                continue
            # curve.select_all(action="DESELECT") for a spline with no hidden
            # points, without the operator call
            for p in points:
                p.select_control_point = False
                p.select_left_handle = False
                p.select_right_handle = False
            points[i].select_control_point = True
            points[i + 1].select_control_point = True
            bpy.ops.curve.subdivide(number_cuts=number_cuts)
    obj.data.splines[0].resolution_u = 1
    with butil.SelectObjects(obj):
        bpy.ops.object.convert(target="MESH")
    obj = bpy.context.active_object
    butil.modify_mesh(obj, "WELD", merge_threshold=1e-3)
    return obj


def remesh_fill(obj, resolution=0.005):
    n = len(obj.data.vertices)
    butil.modify_mesh(obj, "SOLIDIFY", thickness=0.1)
    # "top" marks the vertices the solidify added, as the default's geometry-nodes
    # attribute does
    top = obj.data.attributes.new("top", "FLOAT", "POINT")
    top.data.foreach_set(
        "value", (np.arange(len(obj.data.vertices)) >= n).astype(np.float32)
    )
    sharp_remesh_with_attrs(obj, resolution)
    is_top = read_attr_data(obj, "top") > 1e-3
    remove_vertices(obj, lambda x, y, z: is_top)
    obj.data.attributes.remove(obj.data.attributes["top"])
    return obj


def leaf(x_anchors, y_anchors, vector_locations=(), subdivision=64, face_size=None):
    """Planar leaf sheet in the Bezier outline through the anchors (mirrored in y),
    on `remesh_fill`'s lattice, with a "boundary" vertex group. Built from arrays
    unless `direct_monocot_leaves` is off, keeping whole outline cells the operators
    trim, so it has somewhat more faces than the default."""
    if not monocot_legacy():
        from infinigen.perf.assets.objects.monocot.leaf_direct import leaf_direct

        obj = leaf_direct(x_anchors, y_anchors, vector_locations, face_size)
        if obj is not None:
            return obj
    return _default.leaf.default(
        x_anchors, y_anchors, vector_locations, subdivision, face_size
    )
