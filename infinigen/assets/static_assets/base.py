# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Karhan Kayan

import os
from typing import Callable, Dict, Optional

import bpy

from infinigen.assets.static_assets.utils import collapse_hierarchy
from infinigen.core.placement.factory import AssetFactory


class StaticAssetFactory(AssetFactory):
    import_map: Dict[str, Callable] = {
        "dae": bpy.ops.wm.collada_import,
        "abc": bpy.ops.wm.alembic_import,
        "usd": bpy.ops.wm.usd_import,
        "obj": bpy.ops.wm.obj_import,
        "ply": bpy.ops.wm.ply_import,
        "stl": bpy.ops.wm.stl_import,
        "fbx": bpy.ops.import_scene.fbx,
        "glb": bpy.ops.import_scene.gltf,
        "gltf": bpy.ops.import_scene.gltf,
        "blend": bpy.ops.wm.append,
    }

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)

    def import_single_object_from_blend(self, file_path):
        with bpy.data.libraries.load(file_path, link=False) as (data_from, data_to):
            # Ensure there is exactly one object
            if len(data_from.objects) == 1:
                object_name = data_from.objects[0]
            elif len(data_from.objects) == 0:
                raise ValueError("No objects found in the Blender file.")
            else:
                raise ValueError("More than one object found in the Blender file.")

        category = "Object"
        blendfile = file_path
        filepath = os.path.join(blendfile, category, object_name)
        filename = object_name
        directory = os.path.join(blendfile, category)

        bpy.ops.wm.append(filepath=filepath, filename=filename, directory=directory)

    def import_file(self, file_path: str) -> Optional[bpy.types.Object]:
        extension = file_path.split(".")[-1].lower()
        if extension in self.import_map:
            func = self.import_map.get(extension)

            initial_objects = set(bpy.context.scene.objects)

            if extension in ["glb", "gltf"]:
                func(filepath=file_path, merge_vertices=True)
            elif extension == "blend":
                self.import_single_object_from_blend(file_path)
            else:
                func(filepath=file_path)

            new_objects = set(bpy.context.scene.objects) - initial_objects
            for obj in new_objects:
                obj.rotation_mode = "XYZ"

            if new_objects:
                # Check if these objects form a tree structure
                def is_tree(objects):
                    roots = [obj for obj in objects if obj.parent is None]
                    if len(roots) != 1:
                        return False, None
                    root = roots[0]
                    visited = set()

                    def dfs(obj):
                        if obj in visited:
                            return False
                        visited.add(obj)
                        for child in obj.children:
                            if not dfs(child):
                                return False
                        return True

                    if dfs(root):
                        return len(visited) == len(objects), root
                    else:
                        return False, None

                is_tree_structure, root_object = is_tree(new_objects)

                if not is_tree_structure:
                    raise ValueError(
                        "The imported objects do not form a tree structure."
                    )

                collapsed_object = collapse_hierarchy(root_object)
                return collapsed_object
            else:
                return None
        else:
            raise ValueError(f"Unsupported file format: {extension}")

    def create_asset(self, **params) -> bpy.types.Object:
        raise NotImplementedError("This method should be implemented by subclasses")
