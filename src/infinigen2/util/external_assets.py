# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Callable

import bpy
import procfunc as pf

from infinigen2.util.import_utils import module_path

__all__ = [
    "distribution_from_asset_glob",
    "pregenerated_asset_rand",
]

# from procfunc.util.log import Suppress

logger = logging.getLogger(__name__)

SUPPORTED_IMPORT_MAP: dict[str, Callable] = {
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


def _create_empty_mesh_object(name: str, parent: bpy.types.Object | None = None):
    mesh = bpy.data.meshes.new(name=f"{name}_mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    if parent is not None:
        obj.parent = parent
    return obj


def _iter_object_tree(root_obj: bpy.types.Object):
    stack = [root_obj]
    while stack:
        curr = stack.pop()
        yield curr
        stack.extend(curr.children)


def _collapse_hierarchy(root_obj: bpy.types.Object) -> pf.MeshObject:
    mesh_objects: list[bpy.types.Object] = []

    def _apply_transforms(obj: bpy.types.Object):
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    def process_object(obj: bpy.types.Object, parent: bpy.types.Object | None = None):
        new_obj = _create_empty_mesh_object(obj.name, parent=parent)
        new_obj.matrix_world = obj.matrix_world
        if obj.type == "MESH":
            new_obj.data = obj.data.copy()
        mesh_objects.append(new_obj)
        for child in obj.children:
            process_object(child, new_obj)

    process_object(root_obj)

    for obj in mesh_objects:
        _apply_transforms(obj)

    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0]
    bpy.ops.object.join()
    final_obj = bpy.context.active_object

    original_tree = list(_iter_object_tree(root_obj))
    for obj in reversed(original_tree):
        bpy.data.objects.remove(obj, do_unlink=True)

    return pf.MeshObject(final_obj)


def _raise_mesh_base_to_surface(obj: pf.MeshObject, margin: float = 0.002) -> None:
    obj_data = obj.item()
    if (
        obj_data.type != "MESH"
        or obj_data.data is None
        or len(obj_data.data.vertices) == 0
    ):
        return
    min_z = min(v.co.z for v in obj_data.data.vertices)
    obj_data.location.z += -min_z + margin
    pf.ops.mesh.transform_apply(obj, location=True, rotation=True, scale=True)


def _import_single_object_from_blend(file_path: Path):
    with bpy.data.libraries.load(str(file_path), link=False) as (data_from, _data_to):
        if len(data_from.objects) == 1:
            object_name = data_from.objects[0]
        elif len(data_from.objects) == 0:
            raise ValueError("No objects found in the Blender file.")
        else:
            raise ValueError("More than one object found in the Blender file.")

    category = "Object"
    blendfile = str(file_path)
    filepath = os.path.join(blendfile, category, object_name)
    directory = os.path.join(blendfile, category)
    bpy.ops.wm.append(filepath=filepath, filename=object_name, directory=directory)


def _is_tree(objects: set[bpy.types.Object]) -> tuple[bool, bpy.types.Object | None]:
    roots = [obj for obj in objects if obj.parent is None]
    if len(roots) != 1:
        return False, None

    root = roots[0]
    visited = set()

    def dfs(obj: bpy.types.Object):
        if obj in visited:
            return False
        visited.add(obj)
        for child in obj.children:
            if child in objects and not dfs(child):
                return False
        return True

    if not dfs(root):
        return False, None
    return len(visited) == len(objects), root


def _import_asset_file(file_path: Path) -> pf.MeshObject:
    extension = file_path.suffix.lower().lstrip(".")
    if extension not in SUPPORTED_IMPORT_MAP:
        raise ValueError(f"Unsupported file format: {extension}")

    initial_objects = set(bpy.context.scene.objects)
    func = SUPPORTED_IMPORT_MAP[extension]

    if extension in ["glb", "gltf"]:
        func(filepath=str(file_path), merge_vertices=True)
    elif extension == "blend":
        _import_single_object_from_blend(file_path)
    else:
        func(filepath=str(file_path))

    new_objects = set(bpy.context.scene.objects) - initial_objects
    for obj in new_objects:
        obj.rotation_mode = "XYZ"

    if not new_objects:
        raise ValueError(f"Failed to import asset: {file_path}")

    is_tree_structure, root_object = _is_tree(new_objects)
    if not is_tree_structure or root_object is None:
        raise ValueError("The imported objects do not form a tree structure.")

    collapsed_object = _collapse_hierarchy(root_object)
    _raise_mesh_base_to_surface(collapsed_object)
    return collapsed_object


def _resolve_asset_paths(folder: Path) -> list[Path]:
    folder_str = str(folder)
    has_glob = glob.has_magic(folder_str)

    if has_glob:
        candidates = [Path(p) for p in sorted(glob.glob(folder_str))]
    elif folder.is_dir():
        candidates = sorted([p for p in folder.iterdir() if p.is_file()])
    elif folder.exists():
        candidates = [folder]
    else:
        candidates = []

    return [
        p for p in candidates if p.suffix.lower().lstrip(".") in SUPPORTED_IMPORT_MAP
    ]


def _sample_asset_path(paths: list[Path], rng: pf.RNG) -> Path:
    return rng.choice(paths)


def _factory_name_from_path(asset_path: Path) -> str:
    return asset_path.parent.parent.name


def distribution_from_asset_glob(folder: Path):
    paths = _resolve_asset_paths(folder)
    if not paths:
        logger.warning(f"No importable assets found for path pattern: {folder}")

        # Return an empty distribution so optional asset pools can be skipped safely.
        def missing_rand(_rng: pf.RNG) -> pf.MeshObject | None:
            return None

        return missing_rand

    def new_rand(rng: pf.RNG) -> pf.MeshObject:
        asset_path = _sample_asset_path(paths, rng)
        # with Suppress(): # unfortunately doesnt work
        obj = _import_asset_file(asset_path)
        obj.item().name = _factory_name_from_path(asset_path)
        return obj

    return new_rand


def pregenerated_asset_rand(
    relative_glob: str,
) -> Callable[[pf.RNG], pf.MeshObject | None]:
    asset_glob = module_path().parent.parent / Path(relative_glob)
    return distribution_from_asset_glob(asset_glob)
