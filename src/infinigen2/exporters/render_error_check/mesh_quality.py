# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging
from typing import NamedTuple

import bpy
import numpy as np
import procfunc as pf

from infinigen2 import context
from infinigen2.exporters.render_error_check.util import (
    context_meshes,
    evaluated_mesh,
)

logger = logging.getLogger(__name__)

MIN_EDGE_LENGTH = 1e-6
MIN_FACE_AREA = MIN_EDGE_LENGTH**2


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
