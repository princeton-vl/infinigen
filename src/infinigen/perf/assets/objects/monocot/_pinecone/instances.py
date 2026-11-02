# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

"""The default Geometry Nodes instance transforms, realized with arrays."""

import bpy
import numpy as np

from infinigen.assets.objects.monocot.growth import MonocotGrowthFactory
from infinigen.core.util import blender as butil


def read(mesh, collection, field, width=1, dtype=np.int32):
    attrs = {
        ("vertices", "co"): ("position", "vector"),
        ("edges", "vertices"): (".edge_verts", "value"),
        ("loops", "vertex_index"): (".corner_vert", "value"),
        ("loops", "edge_index"): (".corner_edge", "value"),
    }
    source = getattr(mesh, collection)
    out = np.empty(len(source) * width, dtype)
    if (collection, field) in attrs:
        name, prop = attrs[collection, field]
        mesh.attributes[name].data.foreach_get(prop, out)
    else:
        source.foreach_get(field, out)
    return out.reshape(-1, width) if width > 1 else out


def snapshot(mesh):
    data = dict(
        co=read(mesh, "vertices", "co", 3, np.float32),
        edges=read(mesh, "edges", "vertices", 2),
        loops=read(mesh, "loops", "vertex_index"),
        loop_edges=read(mesh, "loops", "edge_index"),
        counts=read(mesh, "polygons", "loop_total"),
    )
    attr = mesh.attributes.get("MaskTag")
    values = np.zeros(len(mesh.polygons), np.int32)
    if attr:
        attr.data.foreach_get("value", values)
    data["tags"] = values
    return data


def build_instance(self, i, face_size):
    obj = MonocotGrowthFactory.build_instance(self, i, face_size)
    data = snapshot(obj.data)
    self._pinecone_templates.append(data)
    mesh = bpy.data.meshes.new("pinecone_basis")
    mesh.from_pydata(
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
        [],
        [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)],
    )
    mesh.attributes.new("_pinecone_template", "INT", "POINT").data.foreach_set(
        "value", np.full(4, i, np.int32)
    )
    old = obj.data
    obj.data = mesh
    if old.users == 0:
        bpy.data.meshes.remove(old)
    return obj


def make_collection(self, face_size):
    self._pinecone_templates = []
    return MonocotGrowthFactory.make_collection(self, face_size)


def expand(self, obj):
    mesh = obj.data
    small = snapshot(mesh)
    templates = np.empty(len(mesh.vertices), np.int32)
    mesh.attributes["_pinecone_template"].data.foreach_get("value", templates)
    zrot = np.empty(len(mesh.vertices), np.float32)
    mesh.attributes["z_rotation"].data.foreach_get("value", zrot)
    stem_mask = templates == -1
    vertex_ids = np.flatnonzero(stem_mask)
    stem_edges = np.all(stem_mask[small["edges"]], axis=1)
    offsets = np.r_[0, small["counts"].cumsum()]
    stem_faces = np.logical_and.reduceat(stem_mask[small["loops"]], offsets[:-1])
    stem_loops = np.repeat(stem_faces, small["counts"])
    vm = np.full(len(templates), -1, np.int32)
    vm[vertex_ids] = np.arange(len(vertex_ids), dtype=np.int32)
    em = np.full(len(small["edges"]), -1, np.int32)
    em[stem_edges] = np.arange(np.count_nonzero(stem_edges), dtype=np.int32)
    stem = dict(
        co=small["co"][stem_mask],
        edges=vm[small["edges"][stem_edges]],
        loops=vm[small["loops"][stem_loops]],
        loop_edges=em[small["loop_edges"][stem_loops]],
        counts=small["counts"][stem_faces],
        tags=small["tags"][stem_faces],
    )
    parts = [(stem, None, zrot[stem_mask])]
    basis_ids = np.flatnonzero(~stem_mask).reshape(-1, 4)
    for ids in basis_ids:
        idx = int(templates[ids[0]])
        assert np.all(templates[ids] == idx)
        basis = small["co"][ids]
        mat = (basis[1:] - basis[0]).T
        parts.append((self._pinecone_templates[idx], (mat, basis[0]), zrot[ids[0]]))
    from .core import final_tag_lookup

    unique_parts = {id(t): t for t, _, _ in parts}
    ids = np.unique(
        np.concatenate([np.unique(t["tags"]) for t in unique_parts.values()])
    )
    tag_lut = final_tag_lookup(ids)
    final_tags = {key: tag_lut[t["tags"]] for key, t in unique_parts.items()}
    nv = sum(len(t["co"]) for t, _, _ in parts)
    ne = sum(len(t["edges"]) for t, _, _ in parts)
    nl = sum(len(t["loops"]) for t, _, _ in parts)
    nf = sum(len(t["counts"]) for t, _, _ in parts)
    # allocate once and write into Blender's storage directly where
    # writable_attribute allows it, otherwise copy through RNA
    new = bpy.data.meshes.new("pinecone_assembled")
    new.vertices.add(nv)
    new.edges.add(ne)
    new.loops.add(nl)
    new.polygons.add(nf)
    new.attributes.new("MaskTag", "INT", "FACE")
    new.attributes.new("sharp_face", "BOOLEAN", "FACE").data.foreach_set(
        "value", np.ones(nf, bool)
    )
    new.attributes.new("z_rotation", "FLOAT", "POINT")
    new.attributes.new("uv_map", "FLOAT_VECTOR", "POINT")
    new.uv_layers.new(name="UVMap", do_init=False)
    from infinigen.perf.assets.utils.mesh_io import writable_attribute

    direct = {}

    def buffer(name, n, width, dtype):
        value = writable_attribute(new, name, width, dtype)
        direct[name] = value is not None
        return (
            value
            if value is not None
            else np.empty((n, width) if width > 1 else n, dtype)
        )

    co = buffer("position", nv, 3, np.float32)
    edges = buffer(".edge_verts", ne, 2, np.int32)
    loops = buffer(".corner_vert", nl, 1, np.int32)
    le = buffer(".corner_edge", nl, 1, np.int32)
    counts = np.empty(nf, np.int32)
    tags = buffer("MaskTag", nf, 1, np.int32)
    z = buffer("z_rotation", nv, 1, np.float32)
    vo = eo = lo = fo = 0
    for t, transform, rotation in parts:
        v, e, l, f = len(t["co"]), len(t["edges"]), len(t["loops"]), len(t["counts"])
        if transform is None:
            co[vo : vo + v] = t["co"]
        else:
            np.matmul(t["co"], transform[0].T, out=co[vo : vo + v])
            co[vo : vo + v] += transform[1]
        np.add(t["edges"], vo, out=edges[eo : eo + e])
        np.add(t["loops"], vo, out=loops[lo : lo + l])
        np.add(t["loop_edges"], eo, out=le[lo : lo + l])
        counts[fo : fo + f] = t["counts"]
        tags[fo : fo + f] = final_tags[id(t)]
        z[vo : vo + v] = rotation
        vo += v
        eo += e
        lo += l
        fo += f
    for name, field, value in [
        ("position", "vector", co),
        (".edge_verts", "value", edges),
        (".corner_vert", "value", loops),
        (".corner_edge", "value", le),
        ("MaskTag", "value", tags),
        ("z_rotation", "value", z),
    ]:
        if not direct[name]:
            new.attributes[name].data.foreach_set(field, value.ravel())
    from infinigen.perf.assets.utils.mesh_io import write_faces

    write_faces(new, counts)
    new.vertices[0].co = co[0]
    new.update(calc_edges=False)
    obj.data = new
    if mesh.users == 0:
        bpy.data.meshes.remove(mesh)
    self._pinecone_templates = []
    if hasattr(self, "_pinecone_grid"):
        bpy.data.meshes.remove(self._pinecone_grid)
        del self._pinecone_grid
    if hasattr(self, "_pinecone_shape_basis"):
        del self._pinecone_shape_basis
    return obj


def build_instance_native(self, i, face_size):
    import ctypes as ct

    from numpy.random import uniform

    from infinigen.assets.objects.monocot import pinecone
    from infinigen.core.tagging import tag_object, tag_system
    from infinigen.core.util.random import log_uniform

    from . import core as fast

    ptr = ct.c_void_p
    if not hasattr(build_instance_native, "libs"):
        stucci = fast.load_library("stucci")
        topology = fast.load_library("topology")
        stucci.pinecone_normals.argtypes = [ptr, ct.c_int, ptr, ptr, ct.c_int, ptr]
        stucci.pinecone_stucci.argtypes = [
            ptr,
            ptr,
            ct.c_int,
            ct.c_float,
            ct.c_float,
            ct.c_float,
            ct.c_float,
        ]
        topology.pinecone_edges.argtypes = [ptr, ptr, ct.c_int, ptr, ptr]
        build_instance_native.libs = stucci, topology
    stucci, topology = build_instance_native.libs
    if not hasattr(self, "_pinecone_grid"):
        obj = pinecone.new_circle(vertices=128)
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.fill_grid()
        self._pinecone_grid = obj.data.copy()
    else:
        obj = bpy.data.objects.new("pinecone_raw", self._pinecone_grid.copy())
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
    angles = np.array([-1, -0.8, -0.5, 0, 0.5, 0.8, 1]) * self.angle / 2
    scale = uniform(0.9, 0.95)
    displacement = uniform(0.2, 0.3)
    if not hasattr(self, "_pinecone_shape_basis"):
        from scipy.interpolate import interp1d

        points = fast.coordinates(obj.data).astype(np.float64)
        theta = np.arctan2(points[:, 1], points[:, 0])
        base = interp1d(
            angles,
            [0, 0.7, 0, 1, 0, 0.7, 0],
            "quadratic",
            bounds_error=False,
            fill_value=0,
        )(theta)
        varying = interp1d(
            angles,
            [0, 0, 1, 0, 1, 0, 0],
            "quadratic",
            bounds_error=False,
            fill_value=0,
        )(theta)
        depth = interp1d(
            angles,
            [0, 0, 0, -1, 0, 0, 0],
            "quadratic",
            bounds_error=False,
            fill_value=0,
        )(theta)
        self._pinecone_shape_basis = points, base, varying, depth
    points, base, varying, depth = self._pinecone_shape_basis
    co = (
        ((base + scale * varying)[:, None] * points)
        .astype(np.float32)
        .astype(np.float64)
    )
    co[:, 2] += displacement * depth * np.linalg.norm(co, axis=1)
    obj.data.attributes["position"].data.foreach_set(
        "vector", co.astype(np.float32).ravel()
    )
    obj.data.vertices[0].co = co[0]
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    hull = bmesh.ops.convex_hull(
        bm,
        input=list(bm.verts) + list(bm.edges) + list(bm.faces),
        use_existing_faces=True,
    )
    faces = [f for f in hull["geom"] if isinstance(f, bmesh.types.BMFace)]
    if hull["geom_unused"]:
        bmesh.ops.delete(bm, geom=hull["geom_unused"], context="TAGGED_ONLY")
    bmesh.ops.join_triangles(
        bm,
        faces=faces,
        angle_face_threshold=np.float32(np.deg2rad(40)),
        angle_shape_threshold=np.float32(np.deg2rad(40)),
    )
    bm.to_mesh(obj.data)
    bm.free()
    co = fast.coordinates(obj.data).astype(np.float64)
    x = co[:, 0]
    co[:, 2] += np.where((x >= 0) & (x <= 2), x * (x - 1) * 0.25, 0)
    obj.data.attributes["position"].data.foreach_set(
        "vector", co.astype(np.float32).ravel()
    )
    obj.data.vertices[0].co = co[0]
    obj.scale = [0.1] * 3
    obj.rotation_euler[1] -= uniform(np.pi / 18, np.pi / 12)
    from mathutils import Matrix

    matrix = obj.rotation_euler.to_matrix().to_4x4() @ Matrix.Diagonal(
        (0.1, 0.1, 0.1, 1.0)
    )
    obj.data.transform(matrix)
    obj.scale = (1, 1, 1)
    obj.rotation_euler = (0, 0, 0)
    co, q, tr = fast.remesh_arrays(obj.data, face_size)
    noise_scale = np.float32(log_uniform(0.002, 0.005))
    loops = np.concatenate((q.reshape(-1), tr.reshape(-1)))
    counts = np.r_[np.full(len(q), 4, np.int32), np.full(len(tr), 3, np.int32)]
    normals = np.empty_like(co)
    stucci.pinecone_normals(
        co.ctypes.data,
        len(co),
        loops.ctypes.data,
        counts.ctypes.data,
        len(counts),
        normals.ctypes.data,
    )
    stucci.pinecone_stucci(
        co.ctypes.data, normals.ctypes.data, len(co), noise_scale, 5, 0.001, 0
    )
    tag_object(obj, "pinecone")
    co -= co[np.argmin(co[:, 0])].copy()
    co[:, 0] += np.float32(-0.01)
    if fast.remesh_arrays.connectivity is not None:
        edges, loop_edges = fast.remesh_arrays.connectivity
        ne = len(edges)
    else:
        edges = np.empty((len(loops), 2), np.int32)
        loop_edges = np.empty(len(loops), np.int32)
        ne = topology.pinecone_edges(
            loops.ctypes.data,
            counts.ctypes.data,
            len(counts),
            edges.ctypes.data,
            loop_edges.ctypes.data,
        )
    self._pinecone_templates.append(
        dict(
            co=co,
            edges=edges[:ne],
            loops=loops,
            loop_edges=loop_edges,
            counts=counts,
            tags=np.full(len(counts), tag_system.tag_dict["pinecone"], np.int32),
        )
    )
    mesh = bpy.data.meshes.new("pinecone_basis")
    mesh.from_pydata(
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
        [],
        [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)],
    )
    mesh.attributes.new("_pinecone_template", "INT", "POINT").data.foreach_set(
        "value", np.full(4, i, np.int32)
    )
    old = obj.data
    obj.data = mesh
    if old.users == 0:
        bpy.data.meshes.remove(old)
    return obj
