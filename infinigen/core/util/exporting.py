# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lahav Lipson


from pathlib import Path
import gin

import bpy
import mathutils
import re
import json
from uuid import uuid4
import numpy as np
from itertools import chain, product
from tqdm import tqdm
from infinigen.core.util.math import int_hash
from bpy.types import DepsgraphObjectInstance

def get_mesh_data(obj):
    polys = obj.data.polygons
    verts = obj.data.vertices
    loop_totals = np.full((len(polys),), -1, dtype=np.int32)
    polys.foreach_get("loop_total", loop_totals)
    indices = np.full((loop_totals.sum(),), -1, dtype=np.int32)
    polys.foreach_get("vertices", indices)
    vert_lookup = np.full((len(verts)*3,), np.nan, dtype=np.float32)
    verts.foreach_get("co", vert_lookup)
    vert_lookup = vert_lookup.reshape((-1, 3))
    masktag = np.full(len(verts, ), 0, dtype=np.int32)
    if 'MaskTag' in obj.data.attributes:
        obj.data.attributes['MaskTag'].data.foreach_get("value", masktag)
    assert (loop_totals.size == 0) or (loop_totals.min() >= 0)
    assert (indices.size == 0) or (indices.min() >= 0)
    assert not np.any(np.isnan(vert_lookup))
    return vert_lookup, indices, loop_totals, masktag

def get_curve_data(obj):
    curves = obj.data.curves
    points = obj.data.points
    points_length = np.full(len(curves), -1, dtype=np.int32)
    curves.foreach_get('points_length', points_length)
    points_length = np.unique(points_length)
    assert (points_length.size == 0) or (points_length.size == 1 and points_length[0] == 5), np.unique(points_length)
    vertices = np.full((len(points)*3), np.nan, dtype=np.float32)
    points.foreach_get('position', vertices)
    vertices = vertices.reshape(-1, 3)
    radii = np.full(len(points), np.nan, dtype=np.float32)
    points.foreach_get('radius', radii)
    assert not np.any(np.isnan(vertices))
    assert not np.any(np.isnan(radii))
    return vertices, radii

valid_int32 = lambda x: (-2**31 <= x < 2**31)

# See https://projects.blender.org/blender/blender/issues/60881 for logic
def get_id(i: DepsgraphObjectInstance):
    parent_hash = (int_hash(i.parent.name)-2**31) if (i.parent is not None) else 0
    t = list(i.persistent_id)
    if list(t) == [0]*8:
        return (0, 0, parent_hash)
    a, b, *c = t
    assert c == [2**31-1]*6, t
    assert valid_int32(a) and valid_int32(b), t
    return (a, b, parent_hash)

def get_all_instances():
    vertex_info = {}
    pbar = tqdm(bpy.context.evaluated_depsgraph_get().object_instances)
    for deps_instance in pbar:
        obj = deps_instance.object
        pbar.set_description(f"Finding Instances: {obj.name[:20].ljust(20)}")
        if (obj.type == "MESH") and (deps_instance.is_instance) and ("PARTICLE_SYSTEM" not in {m.type for m in obj.modifiers}):
            mat = np.asarray(deps_instance.matrix_world, dtype=np.float32).copy()
            if obj.data not in vertex_info:
                vert_lookup, indices, loop_totals, masktag = get_mesh_data(obj)
                vertex_info[obj.data] = dict(vertex_lookup=vert_lookup, is_instance=True, masktag=masktag,
                indices=indices, loop_totals=loop_totals, matrices=[], instance_ids=[], name=obj.name)
            vertex_info[obj.data]["matrices"].append(mat)
            vertex_info[obj.data]["instance_ids"].append(get_id(deps_instance))
    return chain.from_iterable(((v['vertex_lookup'].shape[0], v['name']), v) for v in vertex_info.values())

def get_all_non_instances():
    pbar = tqdm(bpy.context.evaluated_depsgraph_get().object_instances)
    for deps_instance in pbar:
        obj = deps_instance.object
        pbar.set_description(f"Finding Non-Instances: {obj.name[:20].ljust(20)}")
        mat = np.asarray(deps_instance.matrix_world, dtype=np.float32).copy()[None]
        if obj.type == "MESH":
            if (not deps_instance.is_instance) and ("PARTICLE_SYSTEM" not in {m.type for m in obj.modifiers}):
                yield (len(obj.data.vertices), obj.name)
                vert_lookup, indices, loop_totals, masktag = get_mesh_data(obj)
                yield dict(vertex_lookup=vert_lookup, indices=indices, loop_totals=loop_totals, name=obj.name, matrices=mat, instance_ids=[get_id(deps_instance)], masktag=masktag, is_instance=False)
        elif obj.type == 'CURVES':
            assert not deps_instance.is_instance
            yield (len(obj.data.points)//5, obj.name) # //5 bc hair is inexpensive
            hair_vertices, hair_radii = get_curve_data(obj)
            yield dict(vertex_lookup=hair_vertices, radii=hair_radii, name=obj.name, matrices=mat, instance_ids=[get_id(deps_instance)],  is_instance=False)

def parse_group_from_name(name: str):
    for reg in ["(.*)\.spawn_asset\(.*", "scatter:(.*)", "([A-Za-z_]+)"]:
        match = re.fullmatch(reg, name)
        if match:
            return match.group(1)

def parse_semantic_from_name(name: str):
    group_name = parse_group_from_name(name) or name
    for reg in ["([A-Za-z_]+)[\.\(].*", "([A-Za-z_]+)"]:
        match = re.fullmatch(reg, group_name)
        if match:
            return match.group(1).replace("Factory", "").replace("_fine", "").title()

def calc_aa_bbox(pts):
    xx, yy, zz = zip(pts.min(axis=0), pts.max(axis=0))
    return np.stack(list(product(xx, yy, zz))) # 8 x 3

def calc_instance_bbox(matrices, verts):
    assert verts.shape[1] == 3
    single_bbox = calc_aa_bbox(verts)
    h_bbox = np.concatenate((single_bbox.T, np.ones((1, 8))), axis=0) # 4 x 8
    all_h_bbox = np.einsum("bij, jk -> bki", matrices, h_bbox) # B x 8 x 4
    assert all_h_bbox.shape[1:] == (8, 4)
    all_bbox = (all_h_bbox[...,:3] / all_h_bbox[..., 3:]) # B x 8 x 3
    combined_bbox = calc_aa_bbox(all_bbox.reshape((-1, 3)))
    return combined_bbox, single_bbox

def get_mesh_id_if_cached(name, num_verts, current_ids, previous_frame_mapping):
    assert isinstance(current_ids, frozenset)
    if releveant_entries := previous_frame_mapping.get(name):
        for (nv, prev_ids), mesh_id in releveant_entries.items():
            assert isinstance(prev_ids, frozenset)
            if (num_verts == nv):
                for idd in current_ids:
                    if idd in prev_ids:
                        return mesh_id
    return None


@gin.configurable
def save_obj_and_instances(output_folder, previous_frame_mesh_id_mapping, current_frame_mesh_id_mapping):
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)
    for atm_name in ['atmosphere', 'atmosphere_fine', 'KoleClouds']:
        if atm_name in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[atm_name])
    if "scatters" in bpy.data.collections:
        for obj in bpy.data.collections["scatters"].objects:
            if "instance_scatter" in obj.modifiers.keys():
                obj.hide_viewport = False
        bpy.data.collections["scatters"].hide_viewport = False

    json_data = []
    instance_mesh_data = get_all_instances()
    singleton_mesh_data = get_all_non_instances()
    npz_number = 1
    filename = output_folder / f"saved_mesh_{npz_number:04d}.npz"
    MAX_NUM_VERTS = int(5e6) # lower if OOM
    running_total_verts = 0
    current_obj_num_verts = None
    npz_data = {}
    object_names_mapping = {}
    for item in chain(instance_mesh_data, singleton_mesh_data):
        if isinstance(item, tuple):
            current_obj_num_verts, object_name = item # Sometimes current_obj_num_verts will be 0. This is fine.
            if object_name not in object_names_mapping:
                object_names_mapping[object_name] = len(object_names_mapping) + 1

            # Flush the .npz to avoid OOM
            if (len(npz_data) > 0) and ((running_total_verts + current_obj_num_verts) >= MAX_NUM_VERTS):
                np.savez(filename, **npz_data)
                print(f"Saving to {filename}")
                npz_data.clear()
                running_total_verts = 0
                npz_number += 1
                filename = output_folder / f"saved_mesh_{npz_number:04d}.npz"

            if current_obj_num_verts > MAX_NUM_VERTS:
                print(f"WARNING: Object {object_name} is very large, with {current_obj_num_verts} vertices.")

        else:
            is_instance = item["is_instance"]
            if is_instance:
                instance_ids_set = frozenset(item["instance_ids"])
                mesh_id = get_mesh_id_if_cached(object_name, current_obj_num_verts, instance_ids_set, previous_frame_mesh_id_mapping)
                if mesh_id is None:
                    mesh_id = uuid4().hex[:12]
                current_frame_mesh_id_mapping[object_name][(current_obj_num_verts, instance_ids_set)] = mesh_id
            else:
                mesh_id = str(hex(int_hash(object_name)))[:12]

            if "indices" in item:
                npz_data[f"{mesh_id}_indices"] = item["indices"]
                npz_data[f"{mesh_id}_loop_totals"] = item["loop_totals"]
                npz_data[f"{mesh_id}_masktag"] = item["masktag"]
            else:
                npz_data[f"{mesh_id}_radii"] = item["radii"]
            assert f"{mesh_id}_vertices" not in npz_data
            npz_data[f"{mesh_id}_vertices"] = item["vertex_lookup"]
            matrices = np.asarray(item["matrices"], dtype=np.float32)
            npz_data[f"{mesh_id}_transformations"] = matrices
            instance_ids_array = np.asarray(item["instance_ids"], dtype=np.int32)
            assert np.unique(instance_ids_array, axis=0).shape == instance_ids_array.shape
            assert instance_ids_array.shape[1] == 3
            npz_data[f"{mesh_id}_instance_ids"] = instance_ids_array
            obj = bpy.data.objects[object_name]
            json_val = {"filename": filename.name, "mesh_id": mesh_id, "object_name": object_name, "num_verts": current_obj_num_verts, "children": [],
            "object_type": obj.type, "num_instances": matrices.shape[0], "object_idx": object_names_mapping[object_name]}
            if obj.type == "MESH":
                json_val['num_verts'] = len(obj.data.vertices)
                json_val['num_faces'] = len(obj.data.polygons)
                json_val['materials'] = obj.material_slots.keys()
                json_val['unapplied_modifiers'] = obj.modifiers.keys()
            if not is_instance:
                non_aa_bbox = np.asarray([(obj.matrix_world @ mathutils.Vector(v)) for v in obj.bound_box], dtype=np.float32)
                json_val["instance_bbox"] = calc_aa_bbox(non_aa_bbox).tolist()
                # Todo add chain up parents
            else:
                combined_bbox, instance_bbox = calc_instance_bbox(matrices, item["vertex_lookup"])
                json_val.update({"bbox": combined_bbox.tolist(), "instance_bbox": instance_bbox.tolist()})
            for child_obj in obj.children:
                if child_obj.name not in object_names_mapping:
                    object_names_mapping[child_obj.name] = len(object_names_mapping) + 1
                json_val["children"].append(object_names_mapping[child_obj.name])
            json_data.append(json_val)
            running_total_verts += current_obj_num_verts


    if len(npz_data) > 0:
        np.savez(filename, **npz_data)
        print(f"Saving to {filename}")

    for obj in bpy.data.objects:
        if obj.type not in {"MESH", "CURVES", "CAMERA"}:
            object_name = obj.name
            if object_name not in object_names_mapping:
                object_names_mapping[object_name] = len(object_names_mapping) + 1
            non_aa_bbox = np.asarray([(obj.matrix_world @ mathutils.Vector(v)) for v in obj.bound_box])
            json_val = {"object_name": object_name, "object_type": obj.type, "children": [],
            "bbox": calc_aa_bbox(non_aa_bbox).tolist(), "object_idx": object_names_mapping[object_name]}
            for child_obj in obj.children:
                if child_obj.name not in object_names_mapping:
                    object_names_mapping[child_obj.name] = len(object_names_mapping) + 1
                json_val["children"].append(object_names_mapping[child_obj.name])
            json_data.append(json_val)

    # Save JSON
    (output_folder / "saved_mesh.json").write_text(json.dumps(json_data, indent=4))
