# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging
from typing import NamedTuple

import bpy
import numpy as np
import procfunc as pf
from mathutils.bvhtree import BVHTree

from infinigen2 import context
from infinigen2.exporters.render_error_check.util import (
    context_meshes,
    evaluated_mesh,
)

logger = logging.getLogger(__name__)

MIN_EDGE_LENGTH = 1e-6
MIN_FACE_AREA = MIN_EDGE_LENGTH**2
COINCIDENT_DISTANCE = 1e-4
COINCIDENT_PARALLEL_DOT = 0.999
COINCIDENT_LATERAL = 1e-5


class DegenerateGeometryError(ValueError):
    pass


class MeshDefects(NamedTuple):
    zero_area_faces: int
    zero_length_edges: int
    nonmanifold_edges: int
    flipped_edges: int


def _foreach(
    collection: bpy.types.bpy_prop_collection, attr: str, dtype: type, width: int
) -> np.ndarray:
    arr = np.empty(len(collection) * width, dtype=dtype)
    collection.foreach_get(attr, arr)
    return arr.reshape(-1, width) if width > 1 else arr


def _nearby_hits(tree: BVHTree, centroids: np.ndarray) -> np.ndarray:
    rows = []
    for i, point in enumerate(centroids):
        for location, normal, j, _ in tree.find_nearest_range(
            point, COINCIDENT_DISTANCE
        ):
            rows.append((i, j, *location, *normal))
    return np.array(rows, dtype=np.float64).reshape(-1, 8)


def coincident_triangle_count(mesh: bpy.types.Mesh) -> int:
    mesh.calc_loop_triangles()
    co = _foreach(mesh.vertices, "co", np.float64, 3)
    # collapsed triangles are counted as zero-area faces; left in, they all hit each other
    keep = _foreach(mesh.loop_triangles, "area", np.float64, 1) >= MIN_FACE_AREA
    tris = _foreach(mesh.loop_triangles, "vertices", np.int64, 3)[keep]
    normals = _foreach(mesh.loop_triangles, "normal", np.float64, 3)[keep]
    materials = _foreach(mesh.loop_triangles, "material_index", np.int64, 1)[keep]
    centroids = co[tris].mean(axis=1)
    co_list = co.tolist()

    # only cross-material pairs z-fight; querying per material skips dense same-material neighbours
    fighting = np.zeros(len(tris), dtype=bool)
    for material in np.unique(materials):
        own = np.flatnonzero(materials == material)
        others = np.flatnonzero(materials != material)
        tree = BVHTree.FromPolygons(co_list, tris[own].tolist(), epsilon=0.0)
        hits = _nearby_hits(tree, centroids[others])
        i = others[hits[:, 0].astype(np.int64)]
        j = own[hits[:, 1].astype(np.int64)]
        offset = centroids[i] - hits[:, 2:5]
        perpendicular = np.einsum("ij,ij->i", offset, hits[:, 5:8])
        lateral_sq = np.einsum("ij,ij->i", offset, offset) - perpendicular**2
        shares_vertex = (tris[i][:, :, None] == tris[j][:, None, :]).any(axis=(1, 2))
        parallel = (
            np.einsum("ij,ij->i", normals[i], normals[j]) > COINCIDENT_PARALLEL_DOT
        )
        coincident = ~shares_vertex & parallel & (lateral_sq < COINCIDENT_LATERAL**2)
        fighting[i[coincident]] = True
    return int(np.count_nonzero(fighting))


def mesh_defects(mesh: bpy.types.Mesh) -> MeshDefects:
    co = _foreach(mesh.vertices, "co", np.float64, 3)
    edges = _foreach(mesh.edges, "vertices", np.int64, 2)
    areas = _foreach(mesh.polygons, "area", np.float64, 1)
    loop_verts = _foreach(mesh.loops, "vertex_index", np.int64, 1)
    loop_edges = _foreach(mesh.loops, "edge_index", np.int64, 1)

    edge_lengths = np.linalg.norm(co[edges[:, 0]] - co[edges[:, 1]], axis=1)
    faces_per_edge = np.bincount(loop_edges, minlength=len(edges))
    forward = loop_verts == edges[loop_edges, 0]
    forward_per_edge = np.bincount(loop_edges, weights=forward, minlength=len(edges))
    flipped = (faces_per_edge == 2) & (forward_per_edge != 1)

    return MeshDefects(
        zero_area_faces=int(np.count_nonzero(areas < MIN_FACE_AREA)),
        zero_length_edges=int(np.count_nonzero(edge_lengths < MIN_EDGE_LENGTH)),
        nonmanifold_edges=int(np.count_nonzero(faces_per_edge > 2)),
        flipped_edges=int(np.count_nonzero(flipped)),
    )


def degenerate_geometry_defects(
    objects: list[pf.MeshObject] | None = None,
) -> dict[str, MeshDefects]:
    found = {}
    for obj in context_meshes(objects):
        if obj.type != "MESH":
            continue
        mesh = evaluated_mesh(obj)
        if mesh is None or len(mesh.polygons) == 0:
            continue
        defects = mesh_defects(mesh)
        if any(defects):
            found[obj.name] = defects
    return found


def assert_geometry_nondegenerate(objects: list[pf.MeshObject] | None = None) -> None:
    bad = degenerate_geometry_defects(objects)
    if bad:
        error = DegenerateGeometryError(
            "meshes have degenerate faces/edges or broken topology, which shade "
            f"with undefined normals or crease and tear under subdivision: {bad}"
        )
        context.raise_or_warn(
            context.globals.error_mode_degenerate_geometry, error, logger
        )


def coincident_face_counts(
    objects: list[pf.MeshObject] | None = None,
) -> dict[str, int]:
    found = {}
    for obj in context_meshes(objects):
        if obj.type != "MESH" or len(obj.data.polygons) == 0:
            continue
        count = coincident_triangle_count(obj.data)
        if count:
            found[obj.name] = count
    return found
