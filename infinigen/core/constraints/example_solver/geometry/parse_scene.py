# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan


import bpy
import fcl
import numpy as np
import trimesh
from mathutils import Matrix

from infinigen.core import tagging
from infinigen.core.constraints.constraint_language.util import sync_trimesh
from infinigen.core.util import blender as butil


def to_trimesh(obj: bpy.types.Object):
    bpy.context.view_layer.update()
    verts = np.array([obj.matrix_world @ v.co for v in obj.data.vertices])
    faces = np.array([p.vertices for p in obj.data.polygons])
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.current_transform = trimesh.transformations.identity_matrix()
    return mesh


def preprocess_obj(obj):
    with butil.ViewportMode(obj, mode="EDIT"):
        butil.select(obj)
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")

    bpy.context.view_layer.update()

    butil.apply_transform(obj, loc=False, rot=False, scale=True)


def preprocess_scene(objects):
    for o in objects:
        preprocess_obj(o)


def parse_scene(objects):
    # convert all bpy.objects into a trimesh.Scene

    preprocess_scene(objects)

    scene = trimesh.Scene()
    for obj in objects:
        add_to_scene(scene, obj)

    return scene


def add_to_scene(scene, obj, preprocess=True):
    if preprocess:
        preprocess_obj(obj)
    obj_matrix_world = Matrix(obj.matrix_world)
    obj.matrix_world = Matrix.Identity(4)
    tmesh = to_trimesh(obj)
    tmesh.metadata["tags"] = tagging.union_object_tags(obj)
    scene.add_geometry(
        geometry=tmesh,
        # transform=np.array(obj.matrix_world),
        geom_name=obj.name + "_mesh",
        node_name=obj.name,
    )
    col = trimesh.collision.CollisionManager()
    T = trimesh.transformations.identity_matrix()
    t = fcl.Transform(T[:3, :3], T[:3, 3])
    tmesh.fcl_obj = col._get_fcl_obj(tmesh)
    tmesh.col_obj = fcl.CollisionObject(tmesh.fcl_obj, t)
    obj.matrix_world = obj_matrix_world
    sync_trimesh(scene, obj.name)
    return tmesh
