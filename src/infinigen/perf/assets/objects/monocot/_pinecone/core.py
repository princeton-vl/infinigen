# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

"""Single-CPU pinecone surface generation; preserves all random parameter draws."""

import ctypes as ct
import types

import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.objects.monocot import growth
from infinigen.assets.utils.decorate import displace_vertices
from infinigen.assets.utils.misc import assign_material
from infinigen.assets.utils.nodegroup import geo_radius
from infinigen.assets.utils.object import data2mesh, mesh2obj
from infinigen.core import surface
from infinigen.core.placement.detail import min_max_edgelen
from infinigen.core.tagging import tag_object
from infinigen.core.util import blender as butil

from .libraries import load_library

ptr = ct.c_void_p
noise = load_library("decorate_dispatch")
noise.pinecone_extension.argtypes = [ptr, ct.c_int, ptr]
sub = load_library("subdivide")
sub.subdivide.argtypes = [ptr, ct.c_int, ptr, ptr, ct.c_int, ct.c_int, ptr]
sub.subdivide.restype = ptr
sub.subdivide_copy.argtypes = [ptr, ptr, ptr]


def coordinates(mesh):
    a = np.empty((len(mesh.vertices), 3), np.float32)
    mesh.attributes["position"].data.foreach_get("vector", a.reshape(-1))
    return a


def mesh_from_arrays(co, quads, tris=None):
    if tris is None:
        tris = np.empty((0, 3), np.int32)
    nq, nt = len(quads), len(tris)
    loops = np.concatenate((quads.reshape(-1), tris.reshape(-1)))
    mesh = bpy.data.meshes.new("pinecone_native")
    mesh.vertices.add(len(co))
    mesh.attributes["position"].data.foreach_set("vector", co.reshape(-1))
    mesh.loops.add(len(loops))
    mesh.attributes[".corner_vert"].data.foreach_set("value", loops)
    mesh.polygons.add(nq + nt)
    mesh.polygons.foreach_set(
        "loop_start",
        np.r_[
            np.arange(nq, dtype=np.int32) * 4,
            nq * 4 + np.arange(nt, dtype=np.int32) * 3,
        ],
    )
    mesh.polygons.foreach_set(
        "loop_total", np.r_[np.full(nq, 4, np.int32), np.full(nt, 3, np.int32)]
    )
    mesh.update(calc_edges=True)
    mesh.attributes.new("sharp_face", "BOOLEAN", "FACE").data.foreach_set(
        "value", np.ones(nq + nt, bool)
    )
    for name in [".select_vert", ".select_edge", ".select_poly"]:
        attr = mesh.attributes.get(name)
        if attr is not None:
            mesh.attributes.remove(attr)
    return mesh


def replace(obj, mesh):
    old = obj.data
    obj.data = mesh
    if old.users == 0:
        bpy.data.meshes.remove(old)


lib = load_library("surface_grid")
ptr = ct.c_void_p
lib.pinecone_surface.argtypes = [ptr, ct.c_int, ptr, ct.c_int, ct.c_float, ptr]
lib.pinecone_surface.restype = ptr
lib.pinecone_surface_copy.argtypes = [ptr, ptr, ptr]
lib.pinecone_surface_edge_count.argtypes = [ptr]
lib.pinecone_surface_edge_count.restype = ct.c_int
lib.pinecone_surface_connectivity.argtypes = [ptr, ptr, ptr]
lib.pinecone_surface_phases.argtypes = [ptr, ptr]


def remesh_arrays(mesh, face_size):
    remesh_arrays.connectivity = None
    co = coordinates(mesh)
    mesh.calc_loop_triangles()
    tr = np.empty((len(mesh.loop_triangles), 3), np.int32)
    mesh.loop_triangles.foreach_get("vertices", tr.ravel())
    s = np.zeros(4, np.int32)
    h = lib.pinecone_surface(
        co.ctypes.data, len(co), tr.ctypes.data, len(tr), face_size, s.ctypes.data
    )
    ne = lib.pinecone_surface_edge_count(h)
    edges = np.empty((ne, 2), np.int32)
    le = np.empty(s[1] * 3, np.int32)
    lib.pinecone_surface_connectivity(h, edges.ctypes.data, le.ctypes.data)
    remesh_arrays.connectivity = edges, le
    out = np.empty((s[0], 3), np.float32)
    tri = np.empty((s[1], 3), np.int32)
    lib.pinecone_surface_copy(h, out.ctypes.data, tri.ctypes.data)
    assert s[2] == 0, ("inconsistent cells", s.tolist())
    return out, np.empty((0, 4), np.int32), tri


def native_subdivide(obj, levels):
    for level in range(levels):
        mesh = obj.data
        co = coordinates(mesh)
        loops = np.empty(len(mesh.loops), np.int32)
        mesh.loops.foreach_get("vertex_index", loops)
        counts = np.empty(len(mesh.polygons), np.int32)
        mesh.polygons.foreach_get("loop_total", counts)
        sizes = np.zeros(3, np.int32)
        h = sub.subdivide(
            co.ctypes.data,
            len(co),
            loops.ctypes.data,
            counts.ctypes.data,
            len(counts),
            int(level == levels - 1),
            sizes.ctypes.data,
        )
        new = np.empty((sizes[0], 3), np.float32)
        q = np.empty((sizes[1], 4), np.int32)
        sub.subdivide_copy(h, new.ctypes.data, q.ctypes.data)
        replace(obj, mesh_from_arrays(new, q))


def build_stem(self, face_size):
    obj = mesh2obj(data2mesh([[0, 0, 0], [0, 0, self.stem_offset]], [[0, 1]]))
    butil.modify_mesh(obj, "SUBSURF", True, levels=9, render_levels=9)
    surface.add_geomod(obj, geo_radius, apply=True, input_args=[self.radius, 16])
    _, size = min_max_edgelen(obj.data)
    if size > face_size:
        native_subdivide(obj, min(6, int(np.ceil(np.log2(size / face_size)))))
    texture = bpy.data.textures.new(name="grasses", type="STUCCI")
    texture.noise_scale = 0.1
    butil.modify_mesh(obj, "DISPLACE", strength=0.01, texture=texture)
    tag_object(obj, "stem")
    obj.data.attributes.new("_pinecone_template", "INT", "POINT").data.foreach_set(
        "value", np.full(len(obj.data.vertices), -1, np.int32)
    )
    return obj


def decorate(self, obj):
    if self.z_drag:
        displace_vertices(obj, lambda x, y, z: (0, 0, -self.z_drag * (x * x + y * y)))
    from infinigen.perf.assets.utils.mesh_io import writable_attribute

    co = writable_attribute(obj.data, "position", 3, np.float32)
    direct_positions = co is not None
    if co is None:
        co = coordinates(obj.data)
    params = np.array(
        [uniform(0.2, 0.4), uniform(1.4, 2.8), *uniform(-1, 1, 3)], np.float32
    )
    noise.pinecone_extension(co.ctypes.data, len(co), params.ctypes.data)
    if not hasattr(decorate, "lib"):
        lib = load_library("deform")
        lib.pinecone_deform.argtypes = [
            ptr,
            ct.c_int,
            ct.c_float,
            ct.c_int,
            ct.c_int,
        ]
        decorate.lib = lib
    decorate.lib.pinecone_deform(
        co.ctypes.data, len(co), uniform(-self.twist_angle, self.twist_angle), 0, 2
    )
    decorate.lib.pinecone_deform(
        co.ctypes.data, len(co), uniform(0, self.bend_angle), 1, 0
    )
    if not direct_positions:
        obj.data.attributes["position"].data.foreach_set("vector", co.reshape(-1))
    obj.data.vertices[0].co = co[0]
    obj.scale = uniform(0.8, 1.2), uniform(0.8, 1.2), self.z_scale
    obj.rotation_euler[-1] = uniform(0, np.pi * 2)
    butil.apply_transform(obj)
    assign_material(obj, self.material)


def final_tag_lookup(ids):
    from infinigen.core.tagging import tag_system

    mapping = {int(x): int(x) for x in ids}
    inverse = {v: k for k, v in tag_system.tag_dict.items()}
    for name in ["flower", "monocot_growth"]:
        for original, current in sorted(mapping.items(), key=lambda item: item[1]):
            parts = set(inverse[current].split(".")) if current else set()
            parts.add(name)
            combined = ".".join(sorted(parts))
            value = tag_system.tag_dict.get(combined)
            if value is None:
                value = len(tag_system.tag_dict) + 1
                tag_system.tag_dict[combined] = value
                inverse[value] = combined
            mapping[original] = value
    lut = np.arange(int(ids.max()) + 1, dtype=np.int32)
    for key, value in mapping.items():
        lut[key] = value
    return lut


def create_asset(self, **params):
    obj = self.create_raw(**params)
    from . import instances

    instances.expand(self, obj)
    self.decorate_monocot(obj)
    return obj


# The default create_raw with the "flower" tag skipped; final assembly applies it.
# Built from a copy of its globals, so the default is left unchanged.

_raw = growth.MonocotGrowthFactory.create_raw
_raw_globals = dict(_raw.__globals__)
_raw_globals["tag_object"] = (
    lambda obj, name: None if name == "flower" else tag_object(obj, name)
)
create_raw = types.FunctionType(
    _raw.__code__, _raw_globals, _raw.__name__, _raw.__defaults__, _raw.__closure__
)
