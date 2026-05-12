# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Lingjie Mei


import bpy
import numpy as np
import trimesh
from mathutils import Vector

import infinigen.core.util.blender as butil
from infinigen.assets.utils.decorate import read_co
from infinigen.core.util.blender import select_none


def center(obj):
    return (Vector(obj.bound_box[0]) + Vector(obj.bound_box[-2])) * obj.scale / 2.0


def origin2lowest(obj, vertical=False, centered=False, approximate=False):
    co = read_co(obj)
    if not len(co):
        return
    i = np.argmin(co[:, -1])
    if approximate:
        indices = np.argsort(co[:, -1])
        obj.location = -np.mean(co[indices[: len(co) // 10]], 0)
        obj.location[-1] = -co[i, -1]
    elif centered:
        obj.location = -center(obj)
        obj.location[-1] = -co[i, -1]
    elif vertical:
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


def data2mesh(vertices=(), edges=(), faces=(), name=""):
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
    obj = butil.object_from_trimesh(trimesh, "")
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    return obj


def obj2trimesh(obj):
    butil.modify_mesh(obj, "TRIANGULATE", min_vertices=3)
    vertices = read_co(obj)
    arr = np.zeros(len(obj.data.polygons) * 3)
    obj.data.polygons.foreach_get("vertices", arr)
    faces = arr.reshape(-1, 3)
    return trimesh.Trimesh(vertices, faces)


def new_cube(**kwargs):
    kwargs["location"] = kwargs.get("location", (0, 0, 0))
    bpy.ops.mesh.primitive_cube_add(**kwargs)
    return bpy.context.active_object


def new_bbox(x, x_, y, y_, z, z_):
    obj = new_cube()
    obj.location = (x + x_) / 2, (y + y_) / 2, (z + z_) / 2
    obj.scale = (x_ - x) / 2, (y_ - y) / 2, (z_ - z) / 2
    butil.apply_transform(obj, True)
    return obj


def new_bbox_2d(x, x_, y, y_, z=0):
    obj = new_plane()
    obj.location = (x + x_) / 2, (y + y_) / 2, z
    obj.scale = (x_ - x) / 2, (y_ - y) / 2, 1
    butil.apply_transform(obj, True)
    return obj


def new_icosphere(**kwargs):
    kwargs["location"] = kwargs.get("location", (0, 0, 0))
    bpy.ops.mesh.primitive_ico_sphere_add(**kwargs)
    return bpy.context.active_object


def new_circle(**kwargs):
    kwargs["location"] = kwargs.get("location", (1, 0, 0))
    bpy.ops.mesh.primitive_circle_add(**kwargs)
    obj = bpy.context.active_object
    butil.apply_transform(obj, loc=True)
    return obj


def new_base_circle(**kwargs):
    kwargs["location"] = kwargs.get("location", (0, 0, 0))
    bpy.ops.mesh.primitive_circle_add(**kwargs)
    obj = bpy.context.active_object
    return obj


def new_empty(**kwargs):
    kwargs["location"] = kwargs.get("location", (0, 0, 0))
    bpy.ops.object.empty_add(**kwargs)
    obj = bpy.context.active_object
    obj.scale = kwargs.get("scale", (1, 1, 1))
    return obj


def new_plane(**kwargs):
    kwargs["location"] = kwargs.get("location", (0, 0, 0))
    bpy.ops.mesh.primitive_plane_add(**kwargs)
    obj = bpy.context.active_object
    butil.apply_transform(obj, loc=True)
    return obj


def new_cylinder(**kwargs):
    kwargs["location"] = kwargs.get("location", (0, 0, 0.5))
    kwargs["depth"] = kwargs.get("depth", 1)
    bpy.ops.mesh.primitive_cylinder_add(**kwargs)
    obj = bpy.context.active_object
    butil.apply_transform(obj, loc=True)
    return obj


def new_base_cylinder(**kwargs):
    bpy.ops.mesh.primitive_cylinder_add(**kwargs)
    obj = bpy.context.active_object
    butil.apply_transform(obj, loc=True)
    return obj


def new_grid(**kwargs):
    kwargs["location"] = kwargs.get("location", (0, 0, 0))
    bpy.ops.mesh.primitive_grid_add(**kwargs)
    obj = bpy.context.active_object
    butil.apply_transform(obj, loc=True)
    return obj


def new_line(subdivisions=1, scale=1.0):
    vertices = np.stack(
        [
            np.linspace(0, scale, subdivisions + 1),
            np.zeros(subdivisions + 1),
            np.zeros(subdivisions + 1),
        ],
        -1,
    )
    edges = np.stack([np.arange(subdivisions), np.arange(1, subdivisions + 1)], -1)
    obj = mesh2obj(data2mesh(vertices, edges))
    return obj


def join_objects(obj):
    butil.select_none()
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
    butil.select_none()
    return obj


def separate_loose(obj):
    select_none()
    objs = butil.split_object(obj)
    i = np.argmax([len(o.data.vertices) for o in objs])
    obj = objs[i]
    objs.remove(obj)
    butil.delete(objs)
    return obj
