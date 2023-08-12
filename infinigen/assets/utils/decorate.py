# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


from statistics import mean
import logging

import bmesh
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core import surface

from infinigen.core.util import blender as butil
from infinigen.core.util.blender import select_none


def multi_res(obj):
    multi_res = obj.modifiers.new(name='multires', type='MULTIRES')
    bpy.ops.object.multires_subdivide(modifier=multi_res.name, mode='CATMULL_CLARK')
    butil.apply_modifiers(obj)


def geo_extension(nw: NodeWrangler, noise_strength=.2, noise_scale=2., musgrave_dimensions='3D'):
    noise_strength = uniform(noise_strength / 2, noise_strength)
    noise_scale = uniform(noise_scale * .7, noise_scale * 1.4)
    geometry = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
    pos = nw.new_node(Nodes.InputPosition)
    direction = nw.scale(pos, nw.scalar_divide(1, nw.vector_math('LENGTH', pos)))
    direction = nw.add(direction, uniform(-1, 1, 3))
    musgrave = nw.scalar_multiply(nw.scalar_add(
        nw.new_node(Nodes.MusgraveTexture, [direction], input_kwargs={'Scale': noise_scale},
                    attrs={'musgrave_dimensions': musgrave_dimensions}), .25), noise_strength)
    geometry = nw.new_node(Nodes.SetPosition,
                           input_kwargs={'Geometry': geometry, 'Offset': nw.scale(musgrave, pos)})
    nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geometry})


def separate_loose(obj):
    select_none()
    objs = butil.split_object(obj)
    i = np.argmax([len(o.data.vertices) for o in objs])
    obj = objs[i]
    objs.remove(obj)
    butil.delete(objs)
    return obj


def toggle_hide(obj, recursive=True):
    if obj.name in bpy.data.collections:
        for o in obj.objects:
            toggle_hide(o, recursive)
    else:
        obj.hide_set(True)
        obj.hide_render = True
        if recursive:
            for c in obj.children:
                toggle_hide(c)


def toggle_show(obj, recursive=True):
    if obj.name in bpy.data.collections:
        for o in obj.objects:
            toggle_show(o, recursive)
    else:
        obj.hide_set(False)
        obj.hide_render = False
        if recursive:
            for c in obj.children:
                toggle_hide(c)


def join_objects(obj):
    if not isinstance(obj, list):
        obj = [obj]
    if len(obj) == 1:
        return obj[0]
    bpy.context.view_layer.objects.active = obj[0]
    butil.select_none()
    butil.select(obj)
    bpy.ops.object.join()
    obj = bpy.context.active_object
    obj.location = 0, 0, 0
    obj.rotation_euler = 0, 0, 0
    obj.scale = 1, 1, 1
    return obj


def assign_material(obj, material):
    if not isinstance(obj, list):
        obj = [obj]
    for o in obj:
        with butil.SelectObjects(o):
            while len(o.data.materials):
                bpy.ops.object.material_slot_remove()
        if not isinstance(material, list):
            material = [material]
        for m in material:
            o.data.materials.append(m)


def subsurface2face_size(obj, face_size):
    arr = np.zeros(len(obj.data.polygons))
    obj.data.polygons.foreach_get('area', arr)
    area = np.mean(arr)
    if area < 1e-6:
        logging.warning(f'subsurface2face_size found {area=}, quitting to avoid NaN')
        return
    try:
        levels = int(np.ceil(np.log2(area / face_size)))
    except ValueError:
        return  # catch nans
    if levels > 0:
        butil.modify_mesh(obj, 'SUBSURF', levels=levels, render_levels=levels)


def read_co(obj):
    arr = np.zeros(len(obj.data.vertices) * 3)
    obj.data.vertices.foreach_get('co', arr)
    return arr.reshape(-1, 3)


def read_base_co(obj):
    dg = bpy.context.evaluated_depsgraph_get()
    obj = obj.evaluated_get(dg)
    mesh = obj.to_mesh()
    arr = np.zeros(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get('co', arr)
    return arr.reshape(-1, 3)


def write_co(obj, arr):
    obj.data.vertices.foreach_set('co', arr.reshape(-1))


def read_material_index(obj):
    arr = np.zeros(len(obj.data.polygons), dtype=int)
    obj.data.polygons.foreach_get('material_index', arr)
    return arr


def write_material_index(obj, arr):
    obj.data.polygons.foreach_set('material_index', arr.reshape(-1))


def displace_vertices(obj, fn):
    co = read_co(obj)
    x, y, z = co.T
    f = fn(x, y, z)
    for i in range(3):
        co[:, i] += f[i]
    write_co(obj, co)


def remove_vertices(obj, fn):
    x, y, z = read_co(obj).T
    to_delete = np.nonzero(fn(x, y, z))[0]
    with butil.ViewportMode(obj, 'EDIT'):
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        geom = [bm.verts[_] for _ in to_delete]
        bmesh.ops.delete(bm, geom=geom)
        bmesh.update_edit_mesh(obj.data)
    return obj


def write_attribute(obj, fn, name, domain="POINT"):
    def geo_attribute(nw: NodeWrangler):
        geometry = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
        attr = surface.eval_argument(nw, fn, position=nw.new_node(Nodes.InputPosition))
        geometry = nw.new_node(
            Nodes.StoreNamedAttribute, 
            input_kwargs={'Geometry': geometry, 'Name': name, 'Value': attr},
            attrs={'domain': domain})
        nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geometry})

    surface.add_geomod(obj, geo_attribute, apply=True)


def treeify(obj):
    if len(obj.data.vertices) == 0:
        return obj

    obj = separate_loose(obj)
    with butil.ViewportMode(obj, 'EDIT'):
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        included = np.zeros(len(bm.verts))
        i = min((v.co[-1], i) for i, v in enumerate(bm.verts))[1]
        queue = [bm.verts[i]]
        included[i] = 1
        to_keep = []
        while queue:
            v = queue.pop()
            for e in v.link_edges:
                o = e.other_vert(v)
                if not included[o.index]:
                    included[o.index] = 1
                    to_keep.append(e)
                    queue.append(o)
        bmesh.ops.delete(bm, geom=list(set(bm.edges).difference(to_keep)), context='EDGES')
        bmesh.update_edit_mesh(obj.data)
    return obj


def fix_tree(obj):
    with butil.ViewportMode(obj, 'EDIT'), butil.Suppress():
        bpy.ops.mesh.remove_doubles()
        bm = bmesh.from_edit_mesh(obj.data)
        vertices_remove = []
        for v in bm.verts:
            if len(v.link_edges) == 1:
                o = v.link_edges[0].other_vert(v)
                if len(o.link_edges) > 2:
                    vertices_remove.append(v)
        bmesh.ops.delete(bm, geom=vertices_remove)
        bmesh.update_edit_mesh(obj.data)
    return obj


def add_distance_to_boundary(obj):
    with butil.ViewportMode(obj, 'EDIT'):
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.region_to_loop()
    vg = obj.vertex_groups.new(name='distance')
    with butil.ViewportMode(obj, 'EDIT'):
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        distance = np.full(len(obj.data.vertices), -100.)
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
    for i, d in enumerate(distance):
        vg.add([i], d, 'REPLACE')
    return distance
