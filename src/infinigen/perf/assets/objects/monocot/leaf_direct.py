# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

"""Leaf sheet built from arrays instead of by operators.

The optimized `draw.leaf` uses it unless `direct_monocot_leaves` is off. Its lattice
matches the default operator chain's; its faces differ.
"""

import bpy
import numpy as np

from infinigen.assets.utils.object import mesh2obj

from . import leaf_grid


def mesh_from_polygons(verts, loop_verts, loop_totals, name=""):
    """Mesh datablock of polygons given as flat loop vertices and per-polygon counts."""
    loop_verts = np.asarray(loop_verts, dtype=np.int32).reshape(-1)
    loop_totals = np.asarray(loop_totals, dtype=np.int32).reshape(-1)
    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(len(verts))
    mesh.vertices.foreach_set(
        "co", np.ascontiguousarray(verts, dtype=np.float32).ravel()
    )
    if len(loop_totals):
        starts = np.concatenate([[0], np.cumsum(loop_totals[:-1])]).astype(np.int32)
        mesh.loops.add(len(loop_verts))
        mesh.polygons.add(len(loop_totals))
        # loop_start alone defines the polygons; loop_total is derived from the offsets
        mesh.polygons.foreach_set("loop_start", starts)
        mesh.loops.foreach_set("vertex_index", loop_verts)
    mesh.update(calc_edges=True)
    return mesh


def mesh_from_arrays(verts, quads, tris=None, name=""):
    """Mesh datablock of `quads` followed by `tris`."""
    quads = np.asarray(quads, dtype=np.int64).reshape(-1, 4)
    tris = (
        np.zeros((0, 3), dtype=np.int64)
        if tris is None
        else np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    )
    loop_verts = np.concatenate([quads.ravel(), tris.ravel()])
    loop_totals = np.concatenate([np.full(len(quads), 4), np.full(len(tris), 3)])
    return mesh_from_polygons(verts, loop_verts, loop_totals, name)


def leaf_direct(x_anchors, y_anchors, vector_locations, face_size):
    weld = 0.0 if face_size is None else face_size / 2
    verts, (quads, tris), boundary = leaf_grid.leaf_sheet(
        x_anchors, y_anchors, vector_locations, weld=weld
    )
    if len(quads) + len(tris) == 0:
        return None
    verts = np.concatenate([verts, np.zeros((len(verts), 1))], 1)
    obj = mesh2obj(mesh_from_arrays(verts, quads, tris))
    group = obj.vertex_groups.new(name="boundary")
    group.add(np.nonzero(boundary)[0].tolist(), 1.0, "REPLACE")
    return obj
