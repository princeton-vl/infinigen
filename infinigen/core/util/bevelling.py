# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

import bmesh
import bpy
import mathutils
import numpy as np

from infinigen.core.nodes.node_wrangler import Nodes

from .blender import ViewportMode


def special_bounds(obj):
    inf = 1e5
    points = []
    for v in obj.data.vertices:
        points.append(v.co)
    points = np.array(points)
    mask = np.sum(points**2, axis=-1) ** 0.5 < 0.5 * inf
    return points[mask].min(axis=0), points[mask].max(axis=0)


def on_bound_edges(points, points_min, points_max):
    flags = [0, 0, 0]
    eps = 1e-4
    for i in range(3):
        if abs(points[i] - points_min[i]) < eps:
            flags[i] = -1
        elif abs(points[i] - points_max[i]) < eps:
            flags[i] = 1
    return flags


def get_bevel_edges(obj):
    inf = 1e5
    points_min, points_max = special_bounds(obj)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    edges = []
    for edge in bm.edges:
        on_bounds_flag = [0, 0, 0]
        flags = []
        mags = []
        for i in range(2):
            pos = np.array([edge.verts[i].co.x, edge.verts[i].co.y, edge.verts[i].co.z])
            flags.append(on_bound_edges(pos, points_min, points_max))
            mags.append(np.sum(pos**2) ** 0.5)
        for j in range(3):
            on_bounds_flag[j] = flags[0][j] != 0 and flags[0][j] == flags[1][j]
        if np.sum(on_bounds_flag) >= 2:
            edges.append(edge.index)
        elif mags[0] > 0.5 * inf and mags[0] < 1.5 * inf:
            edges.append(edge.index)
    return edges


def add_bevel(obj, edges, offset=0.03, segments=8):
    with ViewportMode(obj, mode="EDIT"):
        bpy.ops.mesh.select_mode(type="EDGE")
        bpy.ops.mesh.select_all(action="DESELECT")
        bm = bmesh.from_edit_mesh(obj.data)
        for edge in bm.edges:
            if edge.index in edges:
                edge.select_set(True)
        bpy.ops.mesh.bevel(
            offset=offset, offset_pct=0, segments=segments, release_confirm=True
        )
    return obj


def complete_bevel(nw, geometry, preprocess):
    inf = 1e5
    geometry = nw.new_node(Nodes.RealizeInstances, [geometry])
    if not preprocess:
        return geometry
    return nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": (geometry, 0),
            "Offset": nw.new_node(
                Nodes.Vector, attrs={"vector": mathutils.Vector((inf, 0, 0))}
            ),
        },
    )


def complete_no_bevel(nw, geometry, preprocess):
    inf = 1e5
    geometry = nw.new_node(Nodes.RealizeInstances, [geometry])
    if not preprocess:
        return geometry
    return nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": (geometry, 0),
            "Offset": nw.new_node(
                Nodes.Vector, attrs={"vector": mathutils.Vector((2 * inf, 0, 0))}
            ),
        },
    )
