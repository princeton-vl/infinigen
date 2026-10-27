# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lingjie Mei: original Infinigen mesh decoration utilities (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/assets/utils/decorate.py)
# - David Yan, Jia Deng: optimized implementation

"""Reads and writes in each property's own dtype, and deletions outside edit mode.

`distance2boundary` takes the edges with one face as the boundary, which can differ
from what `region_to_loop` selects, so its selection may differ from the default's.
"""

from collections.abc import Iterable
from itertools import compress

import bmesh
import bpy  # noqa: I001
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from infinigen.assets.utils.decorate import read_edge_center
from infinigen.core.surface import write_attr_data
from infinigen.perf.core.util.isolation import IsolatedScene


def mesh_boundary(obj):
    """Boolean masks of the boundary edges (one adjacent face) and their vertices."""
    mesh = obj.data
    faces_per_edge = np.bincount(read_loop_edges(obj), minlength=len(mesh.edges))
    edge_mask = faces_per_edge == 1
    vert_mask = np.zeros(len(mesh.vertices), dtype=bool)
    vert_mask[read_edges(obj)[edge_mask].ravel()] = True
    return edge_mask, vert_mask


def distance2boundary(obj):
    # select the boundary, as the default does with region_to_loop
    edge_mask, frontier = mesh_boundary(obj)
    mesh = obj.data
    mesh.vertices.foreach_set("select", frontier)
    mesh.edges.foreach_set("select", edge_mask)
    if len(mesh.polygons):
        loop_edge_selected = edge_mask[read_loop_edges(obj)]
        face_selected = np.logical_and.reduceat(
            loop_edge_selected, read_loop_starts(obj)
        )
        mesh.polygons.foreach_set("select", face_selected)
    # edge-hop distance from the boundary, normalised by the largest; unreachable
    # vertices get 0
    n = len(mesh.vertices)
    edges = read_edges(obj)
    sources = np.flatnonzero(frontier)
    if len(sources) == 0 or len(edges) == 0:
        distance = np.zeros(n)
    else:
        graph = coo_matrix(
            (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)
        )
        distance = dijkstra(
            graph, directed=False, indices=sources, min_only=True, unweighted=True
        )
        reached = np.isfinite(distance)
        d = distance[reached].max() if reached.any() else 0.0
        distance[~reached] = 0.0
        distance /= max(d, 1)
    write_attr_data(obj, "distance", distance)
    return distance


def _foreach_get(collection, prop, n, dims, dtype, out_dtype):
    arr = np.empty(n * dims, dtype=dtype)
    collection.foreach_get(prop, arr)
    if out_dtype is not None and arr.dtype != out_dtype:
        arr = arr.astype(out_dtype)
    return arr


def _foreach_set_float(collection, prop, arr):
    arr = np.asarray(arr).reshape(-1)
    if arr.dtype != np.float32:
        # Blender converts other dtypes value by value through a double; round
        # the same way
        arr = np.asarray(arr, dtype=np.float64).astype(np.float32)
    collection.foreach_set(prop, np.ascontiguousarray(arr))


def _foreach_set_int(collection, prop, arr):
    arr = np.asarray(arr).reshape(-1)
    if not (np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.bool_)):
        raise TypeError(f"expected an integer array for {prop}, got {arr.dtype}")
    collection.foreach_set(prop, np.ascontiguousarray(arr, dtype=np.int32))


def _selected_elements(seq, to_delete, n):
    """BMesh elements of `seq` at the nonzero entries of `to_delete`."""
    mask = np.zeros(n, dtype=bool)
    mask[np.nonzero(to_delete)[0]] = True
    return list(compress(seq[:], mask))


class _ObjectModeBMesh:
    """Edit `obj`'s mesh as a BMesh outside edit mode, flushing the selection by the
    scene's select mode as edit mode does. Only for operators needing no context."""

    def __init__(self, obj):
        self.obj = obj

    def __enter__(self):
        self.bm = bmesh.new()
        self.bm.from_mesh(self.obj.data)
        modes = {
            name
            for name, on in zip(
                ("VERT", "EDGE", "FACE"),
                bpy.context.scene.tool_settings.mesh_select_mode,
            )
            if on
        }
        if modes:
            self.bm.select_mode = modes
        self.bm.select_flush_mode()
        return self.bm

    def __exit__(self, *_):
        self.bm.to_mesh(self.obj.data)
        self.bm.free()
        return False


def read_area(obj):
    arr = _foreach_get(
        obj.data.polygons, "area", len(obj.data.polygons), 1, np.float32, np.float64
    )
    return arr.reshape(-1)


def read_base_co(obj):
    IsolatedScene.ensure_evaluable(obj)
    dg = bpy.context.evaluated_depsgraph_get()
    obj = obj.evaluated_get(dg)
    mesh = obj.to_mesh()
    arr = _foreach_get(
        mesh.vertices, "co", len(mesh.vertices), 3, np.float32, np.float64
    )
    return arr.reshape(-1, 3)


def read_center(obj):
    arr = _foreach_get(
        obj.data.polygons, "center", len(obj.data.polygons), 3, np.float32, np.float64
    )
    return arr.reshape(-1, 3)


def read_co(obj):
    arr = _foreach_get(
        obj.data.vertices, "co", len(obj.data.vertices), 3, np.float32, np.float64
    )
    return arr.reshape(-1, 3)


def read_edges(obj):
    arr = _foreach_get(
        obj.data.edges, "vertices", len(obj.data.edges), 2, np.int32, int
    )
    return arr.reshape(-1, 2)


def read_loop_edges(obj):
    arr = _foreach_get(
        obj.data.loops, "edge_index", len(obj.data.loops), 1, np.int32, int
    )
    return arr.reshape(-1)


def read_loop_starts(obj):
    return _foreach_get(
        obj.data.polygons, "loop_start", len(obj.data.polygons), 1, np.int32, int
    )


def read_loop_totals(obj):
    return _foreach_get(
        obj.data.polygons, "loop_total", len(obj.data.polygons), 1, np.int32, int
    )


def read_loop_vertices(obj):
    arr = _foreach_get(
        obj.data.loops, "vertex_index", len(obj.data.loops), 1, np.int32, int
    )
    return arr.reshape(-1)


def read_material_index(obj):
    return _foreach_get(
        obj.data.polygons, "material_index", len(obj.data.polygons), 1, np.int32, int
    )


def read_normal(obj):
    arr = _foreach_get(
        obj.data.polygons, "normal", len(obj.data.polygons), 3, np.float32, np.float64
    )
    return arr.reshape(-1, 3)


def read_selected(obj, domain="VERT"):
    match domain:
        case "VERT":
            arr = _foreach_get(
                obj.data.vertices, "select", len(obj.data.vertices), 1, bool, int
            )
        case "EDGE":
            arr = _foreach_get(
                obj.data.edges, "select", len(obj.data.edges), 1, bool, int
            )
        case _:
            arr = np.zeros(len(obj.data.faces), int)
            obj.data.faces.foreach_get("select", arr)
    return arr.ravel()


def read_uv(obj):
    data = obj.data.uv_layers.active.data
    arr = _foreach_get(data, "uv", len(obj.data.loops), 2, np.float32, np.float64)
    return arr.reshape(-1, 2)


def remove_edges(obj, to_delete):
    if not isinstance(to_delete, Iterable):
        x, y, z = read_edge_center(obj).T
        to_delete = to_delete(x, y, z)
    with _ObjectModeBMesh(obj) as bm:
        geom = _selected_elements(bm.edges, to_delete, len(bm.edges))
        bmesh.ops.delete(bm, geom=geom, context="EDGES_FACES")
    return obj


def remove_vertices(obj, to_delete):
    if not isinstance(to_delete, Iterable):
        x, y, z = read_co(obj).T
        to_delete = to_delete(x, y, z)
    with _ObjectModeBMesh(obj) as bm:
        geom = _selected_elements(bm.verts, to_delete, len(bm.verts))
        bmesh.ops.delete(bm, geom=geom)
    return obj


def write_co(obj, arr):
    try:
        _foreach_set_float(obj.data.vertices, "co", arr)
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to set vertices.co on {obj.name=}. Object has {len(obj.data.vertices)} verts, "
            f"{arr.shape=}"
        ) from e


def write_material_index(obj, arr):
    _foreach_set_int(obj.data.polygons, "material_index", arr)


def write_uv(obj, arr):
    _foreach_set_float(obj.data.uv_layers.active.data, "uv", arr)
