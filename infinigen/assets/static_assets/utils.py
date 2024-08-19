# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Karhan Kayan
import bpy

from infinigen.core.util import blender as butil


def create_empty_mesh_object(name, parent=None):
    mesh = bpy.data.meshes.new(name=f"{name}_mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    if parent:
        obj.parent = parent
    return obj


def apply_transforms(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def collapse_hierarchy(root_obj):
    mesh_objects = []

    def process_object(obj, parent=None):
        new_obj = obj
        if obj.type != "MESH":
            new_obj = create_empty_mesh_object(obj.name, parent)
            new_obj.matrix_world = obj.matrix_world
        else:
            if parent:
                new_obj.parent = parent

        mesh_objects.append(new_obj)

        for child in obj.children:
            process_object(child, new_obj)

    process_object(root_obj)

    # Apply all transformations
    for obj in mesh_objects:
        apply_transforms(obj)

    # Join all mesh objects
    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objects:
        obj.select_set(True)

    bpy.context.view_layer.objects.active = mesh_objects[0]
    bpy.ops.object.join()

    final_obj = bpy.context.active_object

    butil.delete(list(butil.iter_object_tree(root_obj)))

    return final_obj
