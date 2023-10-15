# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import bpy
import numpy as np
import trimesh
from mathutils import Vector

import infinigen.core.util.blender as butil
from infinigen.assets.utils.decorate import read_co


def center(obj):
    return (Vector(obj.bound_box[0]) + Vector(obj.bound_box[-2])) * obj.scale / 2.


def origin2lowest(obj, vertical=False):
    co = read_co(obj)
    if not len(co):
        return
    i = np.argmin(co[:, -1])
    if vertical:
        obj.location[-1] = -co[i, -1]
    else:
        obj.location = -co[i]
    butil.apply_transform(obj, loc=True)


def origin2highest(obj):
    co = read_co(obj)
    i = np.argmax(co[:, -1])
    obj.location = -co[i]
    butil.apply_transform(obj, loc=True)


def origin2leftmost(obj):
    co = read_co(obj)
    i = np.argmin(co[:, 0])
    obj.location = -co[i]
    butil.apply_transform(obj, loc=True)


def data2mesh(vertices=(), edges=(), faces=(), name=''):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, edges, faces)
    mesh.update()
    return mesh


def mesh2obj(mesh):
    obj = bpy.data.objects.new(mesh.name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    return obj


def trimesh2obj(trimesh):
    obj = butil.object_from_trimesh(trimesh, '')
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    return obj


def obj2trimesh(obj):
    with butil.ViewportMode(obj, 'EDIT'):
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
    vertices = read_co(obj)
    arr = np.zeros(len(obj.data.polygons) * 3)
    obj.data.polygons.foreach_get('vertices', arr)
    faces = arr.reshape(-1, 3)
    return trimesh.Trimesh(vertices, faces)


def new_cube(**kwargs):
    kwargs['location'] = kwargs.get('location', (0, 0, 0))
    bpy.ops.mesh.primitive_cube_add(**kwargs)
    return bpy.context.active_object


def new_icosphere(**kwargs):
    kwargs['location'] = kwargs.get('location', (0, 0, 0))
    bpy.ops.mesh.primitive_ico_sphere_add(**kwargs)
    return bpy.context.active_object


def new_circle(**kwargs):
    kwargs['location'] = kwargs.get('location', (1, 0, 0))
    bpy.ops.mesh.primitive_circle_add(**kwargs)
    obj = bpy.context.active_object
    butil.apply_transform(obj, loc=True)
    return obj


def new_empty(**kwargs):
    kwargs['location'] = kwargs.get('location', (0, 0, 0))
    bpy.ops.object.empty_add(**kwargs)
    obj = bpy.context.active_object
    obj.scale = kwargs.get('scale', (1, 1, 1))
    return obj


def new_line(scale=1., subdivisions=7):
    obj = mesh2obj(data2mesh([[0, 0, 0], [scale, 0, 0]], [[0, 1]]))
    butil.modify_mesh(obj, 'SUBSURF', levels=subdivisions, render_levels=subdivisions)
    obj.location = 0, 0, 0
    return obj
