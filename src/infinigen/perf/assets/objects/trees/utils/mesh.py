# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Alejandro Newell: original Infinigen tree mesh utilities (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/assets/objects/trees/utils/mesh.py)
# - David Yan, Jia Deng: optimized implementation

import bpy
import numpy as np
from mathutils import Vector

from infinigen.assets.objects.trees.utils import mesh as _default
from infinigen.assets.objects.trees.utils.mesh import (
    _UNSTABLE_SAMPLING,
    stabilize_for_sampling,
)

try:
    # GPL-2.0-or-later and optional, so this BSD module imports without it;
    # sampling then uses the default, Blender's particle system.
    from infinigen.infinigen_gpl.perf.particle_sampler import _particle_sampler
except ImportError:
    _particle_sampler = None

C = bpy.context
D = bpy.data

_NATIVE_PRIMITIVES = {
    "MESH_OT_primitive_cone_add",
    "MESH_OT_primitive_cube_add",
    "MESH_OT_primitive_uv_sphere_add",
}
_PRIMITIVE_MESH_CACHE = {}
_PRIMITIVE_SAMPLE_CACHE = {}


def init_mesh(name, verts=[], edges=[], faces=[], coll=None):
    mesh = D.meshes.new(name)
    obj = D.objects.new(mesh.name, mesh)

    if coll is False:
        coll = None
    elif coll is None:
        coll = bpy.context.scene.collection
    else:
        coll = D.collections[coll]

    if coll is not None:
        coll.objects.link(obj)

    if len(faces) == 0:
        mesh.vertices.add(len(verts))
        mesh.vertices.foreach_set("co", np.asarray(verts).reshape(-1))
        mesh.edges.add(len(edges))
        mesh.edges.foreach_set("vertices", np.asarray(edges).reshape(-1))
        # from_pydata creates sharp_face even for an edge-only mesh; match it
        mesh.attributes.new("sharp_face", "BOOLEAN", "FACE")
        mesh.update()
    else:
        mesh.from_pydata(verts, edges, faces)

    return obj


def _validated_legacy_face(indices):
    """Match Blender's MFace zero-sentinel corner rotation."""
    if len(indices) == 3:
        a, b, c = indices
        return (b, c, a, -1) if c == 0 else (a, b, c, -1)
    a, b, c, d = indices
    return (c, d, a, b) if c == 0 or d == 0 else (a, b, c, d)


def _mesh_sampler_buffers(blender_mesh):
    vertices = np.empty((len(blender_mesh.vertices), 3), dtype=np.float32)
    normals = np.empty_like(vertices)
    blender_mesh.vertices.foreach_get("co", vertices.ravel())
    blender_mesh.vertices.foreach_get("normal", normals.ravel())

    blender_mesh.calc_loop_triangles()
    triangles_by_polygon = {}
    for triangle in blender_mesh.loop_triangles:
        triangles_by_polygon.setdefault(triangle.polygon_index, []).append(
            tuple(triangle.vertices)
        )

    faces = []
    smooth = []
    for polygon in blender_mesh.polygons:
        indices = tuple(polygon.vertices)
        if len(indices) in (3, 4):
            polygon_faces = [indices]
        else:
            polygon_faces = triangles_by_polygon[polygon.index]
        for face in polygon_faces:
            faces.append(_validated_legacy_face(face))
            smooth.append(polygon.use_smooth)
    return (
        np.ascontiguousarray(vertices),
        np.ascontiguousarray(normals),
        np.ascontiguousarray(faces, dtype=np.int32),
        np.ascontiguousarray(smooth, dtype=np.uint8),
    )


def _sample_mesh_buffers(buffers, matrix, emit_from, n, seed):
    if emit_from not in {"FACE", "VOLUME"}:
        raise ValueError(
            f"Native mesh sampler does not support emit_from={emit_from!r}"
        )
    vertices, normals, faces, smooth = buffers
    result = _particle_sampler.sample_mesh(
        vertices,
        normals,
        faces,
        smooth,
        np.ascontiguousarray(matrix, dtype=np.float32),
        n,
        seed,
        volume=emit_from == "VOLUME",
    )
    # float64, as the default returns, holding the sampler's float32 values
    return result.astype(np.float64)


def _sample_vtxs_native(obj, emit_from="VOLUME", n=1000, seed=1):
    """Sample mesh points with the native sampler instead of a particle system."""
    if C.object is not None and C.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    C.view_layer.objects.active = obj
    depsgraph = C.evaluated_depsgraph_get()
    evaluated_object = obj.evaluated_get(depsgraph)
    evaluated_mesh = evaluated_object.to_mesh(
        preserve_all_data_layers=True, depsgraph=depsgraph
    )
    try:
        buffers = _mesh_sampler_buffers(evaluated_mesh)
        matrix = np.array(evaluated_object.matrix_world, dtype=np.float32)
        return _sample_mesh_buffers(buffers, matrix, emit_from, n, seed)
    finally:
        evaluated_object.to_mesh_clear()


def sample_vtxs(obj, emit_from="VOLUME", n=1000, seed=1):
    """Sample points on or in a mesh, with the native sampler when the optional GPL
    extension is built and Blender's particle system otherwise."""
    if _particle_sampler is None:
        return _default.sample_vtxs.default(obj, emit_from, n, seed)
    return _sample_vtxs_native(obj, emit_from, n, seed)


def _operator_id(shape_fn):
    try:
        return shape_fn.idname()
    except (AttributeError, TypeError):
        return None


def _primitive_sampler_buffers(shape_fn):
    operator_id = _operator_id(shape_fn)
    if operator_id not in _PRIMITIVE_MESH_CACHE:
        shape_fn(location=(0, 0, 0))
        obj = C.active_object
        if operator_id in _UNSTABLE_SAMPLING:
            stabilize_for_sampling(obj.data)
        blender_mesh = obj.data
        _PRIMITIVE_MESH_CACHE[operator_id] = _mesh_sampler_buffers(blender_mesh)
        D.objects.remove(obj, do_unlink=True)
        if blender_mesh.users == 0:
            D.meshes.remove(blender_mesh)
    return _PRIMITIVE_MESH_CACHE[operator_id]


def get_pts_from_shape(
    shape_fn, n=10, emit_from="VOLUME", loc=(0, 0, 0), scaling=1, pt_offset=0
):
    operator_id = _operator_id(shape_fn)
    # only these primitives are cached and sampled natively
    if _particle_sampler is None or operator_id not in _NATIVE_PRIMITIVES:
        return _default.get_pts_from_shape.default(
            shape_fn, n, emit_from, loc, scaling, pt_offset
        )
    if isinstance(pt_offset, list):
        pt_offset = np.array([pt_offset])
    if isinstance(scaling, list):
        scaling = Vector(scaling)
    buffers = _primitive_sampler_buffers(shape_fn)
    seed = np.random.randint(100)
    scale = np.asarray(scaling, dtype=np.float32)
    if scale.ndim == 0:
        scale = np.full(3, scale, dtype=np.float32)
    matrix = np.eye(4, dtype=np.float32)
    matrix[np.arange(3), np.arange(3)] = scale
    matrix[:3, 3] = np.asarray(loc, dtype=np.float32)
    offset = np.asarray(pt_offset)
    cache_key = (
        operator_id,
        emit_from,
        n,
        seed,
        matrix.tobytes(),
        offset.dtype.str,
        offset.shape,
        offset.tobytes(),
    )
    if cache_key not in _PRIMITIVE_SAMPLE_CACHE:
        pts = _sample_mesh_buffers(buffers, matrix, emit_from, n, seed)
        pts += pt_offset
        _PRIMITIVE_SAMPLE_CACHE[cache_key] = pts
    return _PRIMITIVE_SAMPLE_CACHE[cache_key].copy()
