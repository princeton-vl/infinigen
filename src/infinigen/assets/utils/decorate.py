# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Lingjie Mei


import logging
from collections.abc import Iterable

import bmesh
import bpy
import numpy as np
from numpy.random import uniform
from trimesh.points import remove_close

from infinigen.core import surface
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.surface import write_attr_data
from infinigen.core.util import blender as butil
from infinigen.core.util.math import normalize


def multi_res(obj):
    multi_res = obj.modifiers.new(name="multires", type="MULTIRES")
    bpy.ops.object.multires_subdivide(modifier=multi_res.name, mode="CATMULL_CLARK")
    butil.apply_modifiers(obj)


def geo_extension(
    nw: NodeWrangler, noise_strength=0.2, noise_scale=2.0, musgrave_dimensions="3D"
):
    noise_strength = uniform(noise_strength / 2, noise_strength)
    noise_scale = uniform(noise_scale * 0.7, noise_scale * 1.4)
    geometry = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
    )
    pos = nw.new_node(Nodes.InputPosition)
    direction = nw.scale(pos, nw.scalar_divide(1, nw.vector_math("LENGTH", pos)))
    direction = nw.add(direction, uniform(-1, 1, 3))
    musgrave = nw.scalar_multiply(
        nw.scalar_add(
            nw.new_node(
                Nodes.MusgraveTexture,
                [direction],
                input_kwargs={"Scale": noise_scale},
                attrs={"musgrave_dimensions": musgrave_dimensions},
            ),
            0.25,
        ),
        noise_strength,
    )
    geometry = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={"Geometry": geometry, "Offset": nw.scale(musgrave, pos)},
    )
    nw.new_node(Nodes.GroupOutput, input_kwargs={"Geometry": geometry})


def subsurface2face_size(obj, face_size):
    arr = np.zeros(len(obj.data.polygons))
    obj.data.polygons.foreach_get("area", arr)
    area = np.mean(arr)
    if area < 1e-6:
        logging.warning(f"subsurface2face_size found {area=}, quitting to avoid NaN")
        return
    try:
        levels = int(np.ceil(np.log2(area / face_size)))
    except ValueError:
        return  # catch nans
    if levels > 0:
        butil.modify_mesh(obj, "SUBSURF", levels=levels, render_levels=levels)


def read_selected(obj, domain="VERT"):
    match domain:
        case "VERT":
            arr = np.zeros(len(obj.data.vertices), int)
            obj.data.vertices.foreach_get("select", arr)
        case "EDGE":
            arr = np.zeros(len(obj.data.edges), int)
            obj.data.edges.foreach_get("select", arr)
        case _:
            arr = np.zeros(len(obj.data.faces), int)
            obj.data.faces.foreach_get("select", arr)
    return arr.ravel()


def read_co(obj):
    arr = np.zeros(len(obj.data.vertices) * 3)
    obj.data.vertices.foreach_get("co", arr)
    return arr.reshape(-1, 3)


def read_edges(obj):
    arr = np.zeros(len(obj.data.edges) * 2, dtype=int)
    obj.data.edges.foreach_get("vertices", arr)
    return arr.reshape(-1, 2)


def read_edge_center(obj):
    return read_co(obj)[read_edges(obj).reshape(-1)].reshape(-1, 2, 3).mean(1)


def read_edge_direction(obj):
    cos = read_co(obj)[read_edges(obj).reshape(-1)].reshape(-1, 2, 3)
    return normalize(cos[:, 1] - cos[:, 0])


def read_edge_length(obj):
    cos = read_co(obj)[read_edges(obj).reshape(-1)].reshape(-1, 2, 3)
    return np.linalg.norm(cos[:, 1] - cos[:, 0], axis=-1)


def read_center(obj):
    arr = np.zeros(len(obj.data.polygons) * 3)
    obj.data.polygons.foreach_get("center", arr)
    return arr.reshape(-1, 3)


def read_normal(obj):
    arr = np.zeros(len(obj.data.polygons) * 3)
    obj.data.polygons.foreach_get("normal", arr)
    return arr.reshape(-1, 3)


def read_area(obj):
    arr = np.zeros(len(obj.data.polygons))
    obj.data.polygons.foreach_get("area", arr)
    return arr.reshape(-1)


def read_loop_vertices(obj):
    arr = np.zeros(len(obj.data.loops), dtype=int)
    obj.data.loops.foreach_get("vertex_index", arr)
    return arr.reshape(-1)


def read_loop_edges(obj):
    arr = np.zeros(len(obj.data.loops), dtype=int)
    obj.data.loops.foreach_get("edge_index", arr)
    return arr.reshape(-1)


def read_uv(obj):
    arr = np.zeros(len(obj.data.loops) * 2)
    obj.data.uv_layers.active.data.foreach_get("uv", arr)
    return arr.reshape(-1, 2)


def write_uv(obj, arr):
    obj.data.uv_layers.active.data.foreach_set("uv", arr.reshape(-1))


def read_base_co(obj):
    dg = bpy.context.evaluated_depsgraph_get()
    obj = obj.evaluated_get(dg)
    mesh = obj.to_mesh()
    arr = np.zeros(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", arr)
    return arr.reshape(-1, 3)


def write_co(obj, arr):
    try:
        obj.data.vertices.foreach_set("co", arr.reshape(-1))
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to set vertices.co on {obj.name=}. Object has {len(obj.data.vertices)} verts, "
            f"{arr.shape=}"
        ) from e


def read_material_index(obj):
    arr = np.zeros(len(obj.data.polygons), dtype=int)
    obj.data.polygons.foreach_get("material_index", arr)
    return arr


def read_loop_starts(obj):
    arr = np.zeros(len(obj.data.polygons), dtype=int)
    obj.data.polygons.foreach_get("loop_start", arr)
    return arr


def read_loop_totals(obj):
    arr = np.zeros(len(obj.data.polygons), dtype=int)
    obj.data.polygons.foreach_get("loop_total", arr)
    return arr


def write_material_index(obj, arr):
    obj.data.polygons.foreach_set("material_index", arr.reshape(-1))


def set_shade_smooth(obj):
    write_attr_data(
        obj, "use_smooth", np.ones(len(obj.data.polygons), dtype=int), "INT", "FACE"
    )


def displace_vertices(obj, fn):
    co = read_co(obj)
    if not isinstance(fn, Iterable):
        x, y, z = co.T
        fn = fn(x, y, z)
        for i in range(3):
            co[:, i] += fn[i]
    else:
        co += fn
    write_co(obj, co)


def remove_vertices(obj, to_delete):
    if not isinstance(to_delete, Iterable):
        x, y, z = read_co(obj).T
        to_delete = to_delete(x, y, z)
    to_delete = np.nonzero(to_delete)[0]
    with butil.ViewportMode(obj, "EDIT"):
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        geom = [bm.verts[_] for _ in to_delete]
        bmesh.ops.delete(bm, geom=geom)
        bmesh.update_edit_mesh(obj.data)
    return obj


def remove_edges(obj, to_delete):
    if not isinstance(to_delete, Iterable):
        x, y, z = read_edge_center(obj).T
        to_delete = to_delete(x, y, z)
    to_delete = np.nonzero(to_delete)[0]
    with butil.ViewportMode(obj, "EDIT"):
        bm = bmesh.from_edit_mesh(obj.data)
        bm.edges.ensure_lookup_table()
        geom = [bm.edges[_] for _ in to_delete]
        bmesh.ops.delete(bm, geom=geom, context="EDGES_FACES")
        bmesh.update_edit_mesh(obj.data)
    return obj


def remove_faces(obj, to_delete, remove_loose=True):
    if not isinstance(to_delete, Iterable):
        x, y, z = read_center(obj).T
        to_delete = to_delete(x, y, z)
    to_delete = np.nonzero(to_delete)[0]
    with butil.ViewportMode(obj, "EDIT"):
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        geom = [bm.faces[_] for _ in to_delete]
        bmesh.ops.delete(bm, geom=geom, context="FACES_ONLY")
        bmesh.update_edit_mesh(obj.data)
        if remove_loose:
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.select_loose()
            bpy.ops.mesh.delete(type="EDGE")
    return obj


def select_vertices(obj, to_select):
    if not isinstance(to_select, Iterable):
        x, y, z = read_co(obj).T
        to_select = to_select(x, y, z)
    to_select = np.nonzero(to_select)[0]
    with butil.ViewportMode(obj, "EDIT"):
        bpy.ops.mesh.select_mode(type="VERT")
        bpy.ops.mesh.select_all(action="DESELECT")
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        for i in to_select:
            bm.verts[i].select_set(True)
        bm.select_flush(False)
        bmesh.update_edit_mesh(obj.data)
    return obj


def select_edges(obj, to_select):
    if not isinstance(to_select, Iterable):
        x, y, z = read_edge_center(obj).T
        to_select = to_select(x, y, z)
    to_select = np.nonzero(to_select)[0]
    with butil.ViewportMode(obj, "EDIT"):
        bpy.ops.mesh.select_mode(type="EDGE")
        bpy.ops.mesh.select_all(action="DESELECT")
        bm = bmesh.from_edit_mesh(obj.data)
        bm.edges.ensure_lookup_table()
        for i in to_select:
            bm.edges[i].select_set(True)
        bm.select_flush(False)
        bmesh.update_edit_mesh(obj.data)
    return obj


def select_faces(obj, to_select):
    if not isinstance(to_select, Iterable):
        x, y, z = read_center(obj).T
        to_select = to_select(x, y, z)
    to_select = np.nonzero(to_select)[0]
    with butil.ViewportMode(obj, "EDIT"):
        bpy.ops.mesh.select_mode(type="FACE")
        bpy.ops.mesh.select_all(action="DESELECT")
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        for i in to_select:
            bm.faces[i].select_set(True)
        bm.select_flush(False)
        bmesh.update_edit_mesh(obj.data)
    return obj


def write_attribute(obj, fn, name, domain="POINT", data_type="FLOAT"):
    def geo_attribute(nw: NodeWrangler):
        geometry = nw.new_node(
            Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
        )
        attr = surface.eval_argument(nw, fn, position=nw.new_node(Nodes.InputPosition))
        geometry = nw.new_node(
            Nodes.StoreNamedAttribute,
            input_kwargs={"Geometry": geometry, "Name": name, "Value": attr},
            attrs={"domain": domain, "data_type": data_type},
        )
        nw.new_node(Nodes.GroupOutput, input_kwargs={"Geometry": geometry})

    surface.add_geomod(obj, geo_attribute, apply=True)


def distance2boundary(obj):
    with butil.ViewportMode(obj, "EDIT"):
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.region_to_loop()
    with butil.ViewportMode(obj, "EDIT"):
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        distance = np.full(len(obj.data.vertices), -100.0)
        queue = set(v.index for v in bm.verts if v.select)
        d = 0
        while True:
            distance[list(queue)] = d
            next_queue = set()
            for i in queue:
                v = bm.verts[i]
                for e in v.link_edges:
                    next_queue.add(e.other_vert(v).index)
            queue = set(i for i in next_queue if distance[i] < 0)
            if not queue:
                break
            d += 1
    distance[distance < 0] = 0
    distance /= max(d, 1)
    write_attr_data(obj, "distance", distance)
    return distance


def mirror(obj, axis=0):
    obj.scale[axis] = -1
    butil.apply_transform(obj)
    with butil.ViewportMode(obj, "EDIT"):
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.flip_normals()
    return obj


def subsurf(obj, levels, simple=False):
    if levels > 0:
        butil.modify_mesh(
            obj,
            "SUBSURF",
            levels=levels,
            render_levels=levels,
            subdivision_type="SIMPLE" if simple else "CATMULL_CLARK",
        )


def subdivide_edge_ring(obj, cuts=64, axis=(0, 0, 1), **kwargs):
    butil.select_none()
    with butil.ViewportMode(obj, "EDIT"):
        bm = bmesh.from_edit_mesh(obj.data)
        bm.edges.ensure_lookup_table()
        selected = (
            np.abs((read_edge_direction(obj) * np.array(axis)[np.newaxis, :]).sum(1))
            > 1 - 1e-3
        )
        edges = [bm.edges[i] for i in np.nonzero(selected)[0]]
        bmesh.ops.subdivide_edgering(bm, edges=edges, cuts=int(cuts), **kwargs)
        bmesh.update_edit_mesh(obj.data)


def solidify(obj, axis, thickness):
    axes = [0, 1, 2]
    axes.remove(axis)
    u = np.zeros(3)
    u[axes[0]] = thickness
    v = np.zeros(3)
    v[axes[1]] = thickness
    butil.select_none()
    with butil.ViewportMode(obj, "EDIT"):
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.extrude_edges_move(TRANSFORM_OT_translate={"value": u})
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={"value": v})
    obj.location = -(u + v) / 2
    butil.apply_transform(obj, True)
    return obj


def decimate(points, n):
    dist = 0.1
    ratio = 1.2
    while True:
        culled = remove_close(points, dist)[0]
        if len(culled) <= n or dist > 10:
            dist /= ratio
            break
        dist *= ratio
    culled = remove_close(points, dist)[0]
    return np.random.permutation(culled)[:n]


def remove_duplicate_edges(obj):
    remove_faces(obj, np.ones_like(len(obj.data.polygons)), remove_loose=False)
    with butil.ViewportMode(obj, "EDIT"):
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        counts = []
        for v in bm.verts:
            counts.append(len(v.link_edges))
    counts = np.array(counts)
    u, v = read_edges(obj).T
    to_delete = (counts[u] > 2) & (counts[v] > 2)
    remove_edges(obj, to_delete)
