# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Alexander Raistrick: original Infinigen level-of-detail utilities (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/core/placement/detail.py)
# - David Yan, Jia Deng: optimized implementation

import bpy
import numpy as np

from infinigen.core.placement.detail import logger, subdivide_to_face_size
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.perf.core.nodes.nodegroups import transfer_attributes


# Not gated; called by the optimized adapt_mesh_resolution below.
# nearest_point_attributes lets the optimized trees copy their discrete skeleton
# attributes from the nearest vertex; without it these match the defaults.
def remesh_with_attrs(
    obj,
    face_size,
    apply=True,
    min_remesh_size=None,
    attributes=None,
    nearest_point_attributes=None,
):
    logger.debug(
        f"remesh_with_attrs on {obj.name=} with {face_size=:.4f} {attributes=}"
    )

    temp_copy = deep_clone_obj(obj)

    remesh_size = (
        face_size if min_remesh_size is None else max(face_size, min_remesh_size)
    )
    butil.modify_mesh(obj, type="REMESH", apply=True, voxel_size=remesh_size)

    transfer_attributes.transfer_all_with_nearest(
        source=temp_copy,
        target=obj,
        attributes=attributes,
        uvs=True,
        nearest_point_attributes=nearest_point_attributes,
    )
    bpy.data.objects.remove(temp_copy, do_unlink=True)

    if remesh_size > face_size:
        subdivide_to_face_size(obj, remesh_size, face_size, apply=True)

    return obj


def sharp_remesh_with_attrs(
    obj,
    face_size,
    apply=True,
    min_remesh_size=None,
    attributes=None,
    nearest_point_attributes=None,
):
    temp_copy = deep_clone_obj(obj)

    remesh_size = (
        face_size if min_remesh_size is None else max(face_size, min_remesh_size)
    )
    butil.modify_mesh(
        obj,
        "REMESH",
        apply=apply,
        mode="SHARP",
        octree_depth=int(np.ceil(np.log2((max(obj.dimensions) + 0.01) / remesh_size))),
        use_remove_disconnected=False,
    )

    transfer_attributes.transfer_all_with_nearest(
        source=temp_copy,
        target=obj,
        attributes=attributes,
        uvs=True,
        nearest_point_attributes=nearest_point_attributes,
    )
    bpy.data.objects.remove(temp_copy, do_unlink=True)

    return obj


def min_max_edgelen(mesh):
    verts = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", verts)
    verts = verts.reshape((-1, 3))

    edges = np.empty(len(mesh.edges) * 2, dtype=np.int32)
    mesh.edges.foreach_get("vertices", edges)
    edges = edges.reshape((-1, 2))

    lens = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=-1)
    if len(lens) <= 4:
        return lens.min(), lens.max()

    lower = len(lens) // 4
    upper = len(lens) - (len(lens) + 3) // 4
    lens.partition((lower, upper))
    return lens[lower], lens[upper]


def adapt_mesh_resolution(obj, face_size, method, approx=0.2, **kwargs):
    assert obj.type == "MESH"
    assert 0 <= approx and approx <= 0.5

    logger.debug(
        f"adapt_mesh_resolution on {obj.name} with {method=} to {face_size=:.6f}"
    )

    if len(obj.data.polygons) == 0:
        logger.debug(
            f"Ignoring adapt_mesh_resolution on {obj.name=} due to no polygons"
        )
        return

    if method == "subdivide":
        _, lmax = min_max_edgelen(obj.data)
        if lmax > face_size:
            subdivide_to_face_size(
                obj, from_facesize=lmax, to_facesize=face_size, **kwargs
            )
    elif method == "subdiv_by_area":
        areas = np.zeros(len(obj.data.polygons))
        obj.data.polygons.foreach_get("area", areas)
        approx_facesize = np.sqrt(np.percentile(areas, q=1 - approx))
        if approx_facesize > face_size:
            subdivide_to_face_size(
                obj, from_facesize=approx_facesize, to_facesize=face_size, **kwargs
            )
        else:
            logger.debug(
                f"No subdivision necessary on {obj.name=} {approx_facesize} < {face_size}"
            )
    elif method == "merge_down":
        lmin, _ = min_max_edgelen(obj.data)
        if lmin < face_size:
            butil.merge_by_distance(obj, face_size)
    elif method == "remesh":
        remesh_with_attrs(obj, face_size, **kwargs)
    elif method == "sharp_remesh":
        sharp_remesh_with_attrs(obj, face_size, **kwargs)
    else:
        raise ValueError(f"Unrecognized adapt_mesh_resolution(..., {method=})")
