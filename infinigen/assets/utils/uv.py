# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import logging

# Authors: Lingjie Mei
from collections.abc import Iterable

import bpy
import numpy as np
from sklearn.linear_model import LinearRegression

from infinigen.assets.materials import common
from infinigen.assets.utils.decorate import (
    read_co,
    read_edges,
    read_loop_edges,
    read_loop_starts,
    read_loop_totals,
    read_loop_vertices,
    read_normal,
    read_uv,
    select_faces,
    write_uv,
)
from infinigen.core.util import blender as butil

logger = logging.getLogger(__name__)


def face_corner2faces(obj):
    loop_starts = read_loop_starts(obj)
    faces = np.zeros(len(obj.data.loops), dtype=int)
    faces[loop_starts] = 1
    faces = np.cumsum(faces) - 1
    return faces


def unwrap_faces(obj, selection=None):
    if isinstance(obj, Iterable):
        for o in obj:
            unwrap_faces(o, selection)
        return
    butil.select_none()
    selection = common.get_selection(obj, selection)
    if len(obj.data.uv_layers) == 0:
        smart = True
    else:
        uv = read_uv(obj)[selection.astype(bool)[face_corner2faces(obj)]]
        smart = (np.isnan(uv) | (np.abs(uv) < 0.1)).sum() / uv.size > 0.5
    butil.select_none()
    with butil.ViewportMode(obj, "EDIT"):
        bpy.ops.mesh.select_mode(type="FACE")
        select_faces(obj, selection)
        if smart:
            bpy.ops.uv.smart_project()
        else:
            bpy.ops.uv.unwrap()


def str2vec(axis):
    if not isinstance(axis, str):
        return axis
    match axis[-1].lower():
        case "x":
            vec = 1, 0, 0
        case "y":
            vec = 0, 1, 0
        case "z":
            vec = 0, 0, 1
        case "u":
            vec = -1, 0, 0
        case "v":
            vec = 0, -1, 0
        case "w":
            vec = 0, 0, -1
        case _:
            raise NotImplementedError
    vec = np.array(vec)
    if axis[0] == "-":
        vec = -vec
    return vec


def compute_uv_direction(obj, x="x", y="y", selection=None):
    ensure_uv(obj, selection)
    x, y = str2vec(x), str2vec(y)
    co = read_co(obj)
    edges = read_edges(obj)
    loop_vertices = read_loop_vertices(obj)
    loop_edges = read_loop_edges(obj)
    uv = read_uv(obj)
    if selection is None:
        selection = np.full(len(uv), True)
    selection = selection.astype(bool)
    loop_starts = read_loop_starts(obj)
    loop_totals = read_loop_totals(obj)
    next_vertices = edges[loop_edges].sum(1) - loop_vertices
    next_loops = np.arange(len(uv)) + 1
    next_loops[loop_starts + loop_totals - 1] -= loop_totals
    uv_diff = uv[next_loops] - uv
    co_diff = co[next_vertices] - co[loop_vertices]
    lr = LinearRegression()
    lr.fit(co_diff[selection], uv_diff[selection])
    lr.coef_[lr.coef_ > 1e3] = 0
    axes = lr.predict(np.stack([x, y]))
    axes = axes / (np.linalg.norm(axes, axis=-1) + 1e-6)
    pred = uv @ axes.T
    pred_sel = pred[selection]
    x_min, x_max = np.min(pred_sel[:, 0]), np.max(pred_sel[:, 0])
    y_min, y_max = np.min(pred_sel[:, 1]), np.max(pred_sel[:, 1])
    if x_max - x_min > y_max - y_min:
        scale = 1 / (x_max - x_min + 1e-4)
        mid = (y_max + y_min) / 2
        pred = np.stack(
            [(pred[:, 0] - x_min) * scale, (pred[:, 1] - mid) * scale + 0.5], -1
        )
        bbox = (
            0,
            1,
            0.5 - 0.5 * (y_max - y_min) * scale,
            0.5 + 0.5 * (y_max - y_min) * scale,
        )
    else:
        scale = 1 / (y_max - y_min + 1e-4)
        mid = (x_max + x_min) / 2
        pred = np.stack(
            [(pred[:, 0] - mid) * scale + 0.5, (pred[:, 1] - y_min) * scale], -1
        )
        bbox = (
            0.5 - 0.5 * (x_max - x_min) * scale,
            0.5 + 0.5 * (x_max - x_min) * scale,
            0,
            1,
        )
    new_uv = np.where(selection[:, np.newaxis], pred, uv)
    write_uv(obj, new_uv)
    return bbox


def max_bbox(bboxes):
    return (
        min(b[0] for b in bboxes),
        max(b[1] for b in bboxes),
        min(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    )


def wrap_sides(obj, surface, axes, xs, ys, groupings=None, selection=None, **kwargs):
    fc2f = face_corner2faces(obj)
    axes = np.array([str2vec(axis) for axis in axes])
    faces = np.argmax(read_normal(obj) @ axes.T, -1)
    selection = common.get_selection(obj, selection)
    faces = np.where(selection, faces, -1)
    bboxes, selections = [], []
    for i in range(len(axes)):
        selected = faces == i
        selections.append(selected)
        unwrap_faces(obj, selected)
        bboxes.append(
            compute_uv_direction(obj, str2vec(xs[i]), str2vec(ys[i]), selected[fc2f])
        )
    if groupings is None:
        groupings = [[i] for i in range(len(axes))]
    for indices in groupings:
        selected = sum(selections[i] for i in indices)
        try:
            surface.apply(
                obj, selected, bbox=max_bbox([bboxes[i] for i in indices]), **kwargs
            )
        except TypeError:
            logger.debug(
                f"apply() for {surface=} with kwarg bbox failed, trying again without"
            )
            surface.apply(obj, selected, **kwargs)


def wrap_front_back(obj, surface, shared=True, **kwargs):
    wrap_sides(obj, surface, "vy", "xu", "zz", [[0, 1]] if shared else None, **kwargs)


def wrap_top_bottom(obj, surface, shared=True, **kwargs):
    wrap_sides(obj, surface, "zw", "xu", "yy", [[0, 1]] if shared else None, **kwargs)


def wrap_front_back_side(obj, surface, shared=True, **kwargs):
    wrap_sides(
        obj, surface, "vuy", "xyu", "zzz", [[0, 2], [1]] if shared else None, **kwargs
    )


def wrap_four_sides(obj, surface, shared=True, **kwargs):
    wrap_sides(
        obj,
        surface,
        "vxyu",
        "xyuv",
        "zzzz",
        [[0, 2], [1, 3]] if shared else None,
        **kwargs,
    )


def wrap_six_sides(obj, surface, shared=True, **kwargs):
    wrap_sides(
        obj,
        surface,
        "vxyuzw",
        "xyuvxx",
        "zzzzyv",
        [[0, 2], [1, 3], [4, 5]] if shared else None,
        **kwargs,
    )


def unwrap_normal(obj, selection=None, axis=None, axis_=None):
    ensure_uv(obj)
    normal = read_normal(obj)
    loop_vertices = read_loop_vertices(obj)
    co = read_co(obj)
    loop_totals = read_loop_totals(obj)
    normal = normal[np.arange(len(obj.data.polygons)).repeat(loop_totals)]
    selection = common.get_selection(obj, selection).repeat(loop_totals)
    if axis is not None:
        axis = str2vec(axis)
        axis_ = np.cross(normal, axis)
        axis = axis[np.newaxis, :]
    elif axis_ is not None:
        axis_ = str2vec(axis_)
        axis = np.cross(normal, axis_)
        axis_ = axis_[np.newaxis, :]
    else:
        axis = np.zeros(3)
        i = np.argmin(np.abs(normal)[selection.astype(bool)].sum(0))
        axis[i] = 1
        axis = axis[np.newaxis, :] - np.inner(axis, normal)[:, np.newaxis] * normal
        axis /= np.maximum(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-4)
        axis_ = np.cross(normal, axis)
    uv = np.stack(
        [(co[loop_vertices] * axis).sum(1), (co[loop_vertices] * axis_).sum(1)], -1
    )
    uv = np.where(selection[:, np.newaxis], uv, read_uv(obj))
    write_uv(obj, uv)


def ensure_uv(obj, selection=None):
    if len(obj.data.uv_layers) == 0:
        unwrap_faces(obj, selection)
