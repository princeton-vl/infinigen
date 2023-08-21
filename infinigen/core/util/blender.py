# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alex Raistrick, Zeyu Ma, Lahav Lipson, Hei Law, Lingjie Mei


from collections import defaultdict
import pdb
from math import prod
from contextlib import nullcontext
import logging

from pathlib import Path
import gin

import bpy
import mathutils
import os
import re
import json
from uuid import uuid4
import bmesh
import numpy as np
import trimesh
from itertools import chain, product
from tqdm import tqdm
import cv2

from .math import lerp  # for other people to import from this file
from . import math as mutil
from .logging import Suppress
from infinigen.core.nodes.node_info import DATATYPE_FIELDS, DATATYPE_DIMS

logger = logging.getLogger(__name__)

def deep_clone_obj(obj, keep_modifiers=False, keep_materials=False):
    new_obj = obj.copy()
    new_obj.data = obj.data.copy()
    if not keep_modifiers:
        for mod in new_obj.modifiers:
            new_obj.modifiers.remove(mod)
    if not keep_materials:
        while len(new_obj.data.materials) > 0:
            new_obj.data.materials.pop()
    bpy.context.collection.objects.link(new_obj)
    return new_obj
    
def get_all_bpy_data_targets():
    D = bpy.data
    return [
        D.objects, D.collections, D.movieclips, D.particles,
        D.meshes, D.curves, D.armatures, D.node_groups
    ]

class ViewportMode:

    def __init__(self, obj, mode):
        self.obj = obj
        self.mode = mode

    def __enter__(self):
        self.orig_active = bpy.context.active_object
        bpy.context.view_layer.objects.active = self.obj
        self.orig_mode = bpy.context.object.mode
        bpy.ops.object.mode_set(mode=self.mode)

    def __exit__(self, *args):
        bpy.context.view_layer.objects.active = self.obj
        bpy.ops.object.mode_set(mode=self.orig_mode)
        bpy.context.view_layer.objects.active = self.orig_active


class CursorLocation:

    def __init__(self, loc):
        self.loc = loc
        self.saved = None

    def __enter__(self):
        self.saved = bpy.context.scene.cursor.location
        bpy.context.scene.cursor.location = self.loc

    def __exit__(self, *_):
        bpy.context.scene.cursor.location = self.saved


class SelectObjects:

    def __init__(self, objects, active=0):
        self.objects = list(objects) if hasattr(objects, '__iter__') else [objects]
        self.active = active

        self.saved_objs = None
        self.saved_active = None

    def __enter__(self):
        self.saved_objects = list(bpy.context.selected_objects)
        self.saved_active = bpy.context.active_object
        select_none()
        select(self.objects)

        if len(self.objects):
            if isinstance(self.active, int):
                bpy.context.view_layer.objects.active = self.objects[self.active]
            else:
                bpy.context.view_layer.objects.active = self.active

    def __exit__(self, *_):

        # our saved selection / active objects may have been deleted, update them to only include valid ones
        def enforce_not_deleted(o):
            try:
                return o if o.name in bpy.data.objects else None
            except ReferenceError:
                return None
            
        self.saved_objects = [enforce_not_deleted(o) for o in self.saved_objects]
        self.saved_objects = [o for o in self.saved_objects if o is not None]

        select_none()
        select(self.saved_objects)
        if self.saved_active is not None:
            bpy.context.view_layer.objects.active = enforce_not_deleted(self.saved_active)


class DisableModifiers:

    def __init__(self, objs, keep=[]):
        self.objs = objs if isinstance(objs, list) else [objs]
        self.keep = keep
        self.modifiers_disabled = []

    def __enter__(self):
        for o in self.objs:
            for m in o.modifiers:
                if not m.show_viewport or m in self.keep:
                    continue
                self.modifiers_disabled.append(m)
                m.show_viewport = False

    def __exit__(self, *_):
        for m in self.modifiers_disabled:
            m.show_viewport = True

class EnableParentCollections:

    def __init__(self, objs, target_key='hide_viewport', target_value=False):
        self.objs = objs
        self.target_key = target_key
        self.target_value = target_value

    def __enter__(self):
        self.enable_cols = set(chain.from_iterable([o.users_collection for o in self.objs]))
        self.enable_cols_startstate = [getattr(c, self.target_key) for c in self.enable_cols]

        for c in self.enable_cols:
            setattr(c, self.target_key, self.target_value)

    def __exit__(self, *_, **__):
        for c, s in zip(self.enable_cols, self.enable_cols_startstate):
            setattr(c, self.target_key, s)

class TemporaryObject:

    def __init__(self, obj):
        self.obj = obj

    def __enter__(self):
        return self.obj

    def __exit__(self, *_):
        if self.obj.name in bpy.data.objects:
            delete(self.obj)


def garbage_collect(targets, keep_in_use=True, keep_names=None, verbose=False):
    if keep_names is None:
        keep_names = [[]] * len(targets)

    for t, orig in zip(targets, keep_names):
        for o in t:
            if keep_in_use and o.users > 0:
                continue
            if o.name in orig:
                continue
            if '(no gc)' in o.name:
                continue
            if verbose:
                print(f'Garbage collecting {o} from {t}')
            t.remove(o)


class GarbageCollect:

    def __init__(self, targets=None, keep_in_use=True, keep_orig=True, verbose=False):
        self.targets = targets or get_all_bpy_data_targets()
        self.keep_in_use = keep_in_use
        self.keep_orig = keep_orig
        self.verbose = verbose

    def __enter__(self):
        self.names = [set(o.name for o in t) for t in self.targets]

    def __exit__(self, *_):
        garbage_collect(self.targets, keep_in_use=self.keep_in_use, keep_names=self.names, verbose=self.verbose)


def select_none():
    if bpy.context.active_object is not None:
        bpy.context.active_object.select_set(False)
    for obj in bpy.context.selected_objects:
        obj.select_set(False)


def select(objs):
    select_none()
    if not isinstance(objs, list):
        objs = [objs]
    for o in objs:
        o.select_set(True)


def delete(objs):
    if not isinstance(objs, list):
        objs = [objs]
    select_none()
    select(objs)
    with Suppress():
        bpy.ops.object.delete()


def delete_collection(collection):
    if collection.name in bpy.data.collections:
        objects = collection.objects
        bpy.data.collections.remove(collection)
        for o in objects:
            delete_collection(o)
    else:
        delete(collection)


def traverse_children(obj, fn):
    fn(obj)
    for obj in obj.children:
        fn(obj)


def iter_object_tree(obj):
    yield obj
    for c in obj.children:
        yield from iter_object_tree(c)


def get_collection(name, reuse=True):
    if reuse and name in bpy.data.collections:
        return bpy.data.collections[name]
    else:
        col = bpy.data.collections.new(name=name)
        bpy.context.scene.collection.children.link(col)
        return col


def unlink(obj):
    if not isinstance(obj, list):
        obj = [obj]
    for o in obj:
        for c in list(bpy.data.collections) + [bpy.context.scene.collection]:
            if o.name in c.objects:
                c.objects.unlink(o)


def put_in_collection(obj, collection, exclusive=True):
    if exclusive:
        unlink(obj)
    collection.objects.link(obj)


def group_in_collection(objs, name: str, reuse=True, **kwargs):
    '''
    objs: List of (None | Blender Object | List[Blender Object])
    '''

    collection = get_collection(name, reuse=reuse)

    for obj in objs:
        if obj is None:
            continue
        if not isinstance(obj, list):
            obj = [obj]
        for child in obj:
            traverse_children(child, lambda obj: put_in_collection(obj, collection, **kwargs))

    return collection


def group_toplevel_collections(keyword, hide_viewport=False, hide_render=False, reuse=True):
    scenecol = bpy.context.scene.collection
    matches = [c for c in scenecol.children if c.name.startswith(keyword) and keyword != c.name]

    parent = get_collection(keyword, reuse=reuse)
    if not parent.name in scenecol.children:
        scenecol.children.link(parent)

    for c in matches:
        scenecol.children.unlink(c)
        parent.children.link(c)

    parent.hide_viewport = hide_viewport
    parent.hide_render = hide_render


def spawn_empty(name, disp_type='PLAIN_AXES', s=0.1):
    empty = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(empty)
    empty.empty_display_size = s
    empty.empty_display_type = disp_type
    return empty


def spawn_point_cloud(name, pts, edges=None):
    if edges is None:
        edges = []

    mesh = bpy.data.meshes.new(name=name)
    mesh.from_pydata(pts, edges, [])
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def spawn_vert(name='vert'):
    return spawn_point_cloud(name, np.zeros((1, 3)))


def spawn_line(name, pts):
    idxs = np.arange(len(pts))
    edges = np.stack([idxs[:-1], idxs[1:]], axis=-1)
    return spawn_point_cloud(name, pts, edges=edges)

def spawn_plane(**kwargs):
    name = kwargs.pop('name', None)
    bpy.ops.mesh.primitive_plane_add(
        enter_editmode=False,
        align='WORLD',
        **kwargs
    )
    obj = bpy.context.active_object
    if name is not None:
        obj.name = name
    return obj

def spawn_cube(**kwargs):
    name = kwargs.pop('name', None)
    bpy.ops.mesh.primitive_cube_add(
        enter_editmode=False,
        align='WORLD',
        **kwargs
    )
    obj = bpy.context.active_object
    if name is not None:
        obj.name = name
    return obj

def clear_scene(keep=[], targets=None, materials=True):
    D = bpy.data
    if targets is None:
        targets = get_all_bpy_data_targets()

    if materials:
        targets.append(D.materials)

    for t in targets:
        if t in keep:
            continue
        for o in t:
            if o in keep or o.name in keep:
                continue
            t.remove(o)

    with Suppress():
        bpy.ops.ptcache.free_bake_all()


def spawn_capsule(rad, height, us=32, vs=16):
    mesh = bpy.data.meshes.new('Capsule')
    obj = bpy.data.objects.new('Capsule', mesh)
    bpy.context.collection.objects.link(obj)

    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=us, v_segments=vs, diameter=2 * rad)

    for v in bm.verts:
        if v.co.z > 0:
            v.co.z += height

    bm.to_mesh(mesh)
    bm.free()

    select_none()
    obj.select_set(True)
    bpy.ops.object.shade_smooth()

    return obj


def to_mesh(object, context=bpy.context):
    deg = context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(object.evaluated_get(deg), depsgraph=deg)

    new_obj = bpy.data.objects.new(object.name + "_mesh", me)
    context.collection.objects.link(new_obj)

    for o in context.selected_objects:
        o.select_set(False)

    new_obj.matrix_world = object.matrix_world
    new_obj.select_set(True)
    context.view_layer.objects.active = new_obj

    return new_obj


def get_camera_res():
    d = np.array([bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y],
                 dtype=np.float32)
    d *= bpy.context.scene.render.resolution_percentage / 100.0
    return d


def set_geomod_inputs(mod, inputs: dict):
    assert mod.type == 'NODES'
    for k, v in inputs.items():
        soc = mod.node_group.inputs[k]
        if isinstance(soc.default_value, (float, int)):
            v = type(soc.default_value)(v)

        try:
            mod[soc.identifier] = v
        except TypeError as e:
            print(f'Error incurred while assigning {v} with {type(v)=} to {soc.identifier=} of {mod.name=}')
            raise e


def modify_mesh(obj, type, apply=True, name=None, return_mod=False, ng_inputs=None, show_viewport=None,
                **kwargs) -> bpy.types.Object:
    if name is None:
        name = f'modify_mesh({type}, **{kwargs})'
    if show_viewport is None:
        show_viewport = not apply

    mod = obj.modifiers.new(name, type)
    mod.show_viewport = show_viewport

    if mod is None:
        raise ValueError(f'modifer.new() returned None, ensure {obj.type=} is valid for modifier {type=}')

    for k, v in kwargs.items():
        setattr(mod, k, v)
    if ng_inputs is not None:
        assert type == 'NODES'
        assert 'node_group' in kwargs
        set_geomod_inputs(mod, ng_inputs)

    if apply:
        apply_modifiers(obj, mod=mod)

    if return_mod:
        return obj, mod if not apply else None
    else:
        return obj

def constrain_object(obj, type, **kwargs):
    c = obj.constraints.new(type=type)
    for k, v in kwargs.items():
        setattr(c, k, v)
    return c

def apply_transform(obj, loc=False, rot=True, scale=True):
    with SelectObjects(obj):
        bpy.ops.object.transform_apply(location=loc, rotation=rot, scale=scale)


def import_mesh(path, **kwargs):
    path = Path(path)

    ext = path.parts[-1].split('.')[-1]
    ext = ext.lower().strip()

    funcs = {
        'obj': bpy.ops.import_scene.obj,
        'fbx': bpy.ops.import_scene.fbx,
        'stl': bpy.ops.import_mesh.stl,
        'ply': bpy.ops.import_mesh.ply}

    if ext not in funcs:
        raise ValueError(
            f'butil.import_mesh does not yet support extension {ext}, please contact the developer')

    select_none()
    with Suppress():
        funcs[ext](filepath=str(path), **kwargs)

    if len(bpy.context.selected_objects) > 1:
        print(
            f"Warning: {ext.upper()} Import produced {len(bpy.context.selected_objects)} objects, "
            f"but only the first is returned by import_obj")
    return bpy.context.selected_objects[0]


def boolean(objs, mode='UNION', verbose=False):
    keep, *rest = list(objs)

    if verbose:
        rest = tqdm(rest, desc=f'butil.boolean({keep.name}..., {mode=})')

    with SelectObjects(keep):
        for target in rest:
            if len(target.modifiers) != 0:
                raise ValueError(
                    f'Attempted to boolean() with {target=} which still has {len(target.modifiers)=}')

            mod = keep.modifiers.new(type='BOOLEAN', name='butil.boolean()')
            mod.operation = mode
            mod.object = target
            bpy.ops.object.modifier_apply(modifier=mod.name)

    return keep


def split_object(obj, mode='LOOSE'):
    select_none()
    select(obj)
    bpy.ops.mesh.separate(type=mode)
    return list(bpy.context.selected_objects)


def move_modifier(obj, mod, i):
    with SelectObjects(obj):
        bpy.ops.object.modifier_move_to_index(modifier=mod.name, index=i)


def join_objects(objs, check_attributes=False):
    if check_attributes:
        # make sure objs[0] has slots to recieve all the attributes of objs[1:]
        join_target = objs[0]
        for obj in objs:
            for att in obj.data.attributes:
                if att.name in join_target.data.attributes:
                    target_att = join_target.data.attributes[att.name]
                    assert att.data_type == target_att.data_type
                    assert att.domain == target_att.domain
                else:
                    join_target.data.attributes.new(att.name, att.data_type, att.domain)

    select(objs)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    return bpy.context.active_object

def clear_mesh(obj):
    with ViewportMode(obj, mode='EDIT'):
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.delete(type='VERT')

def apply_modifiers(obj, mod=None, quiet=True):
    if mod is None:
        mod = list(obj.modifiers)
    if not isinstance(mod, list):
        mod = [mod]
    for i, v in enumerate(mod):
        if isinstance(v, str):
            mod[i] = obj.modifiers[v]
    con = Suppress() if quiet else nullcontext()
    with SelectObjects(obj), con:
        for m in mod:
            try:
                bpy.ops.object.modifier_apply(modifier=m.name)
            except RuntimeError as e:
                if m.type == 'NODES':
                    logging.warning(f'apply_modifers on {obj.name=} {m.name=} raised {e}, ignoring and returning empty mesh for pre-3.5 compatibility reasons')
                    bpy.ops.object.modifier_remove(modifier=m.name)
                    clear_mesh(obj)
                else:
                    raise e


def recalc_normals(obj, inside=False):
    with ViewportMode(obj, mode='EDIT'):
        bpy.ops.mesh.select_all()
        bpy.ops.mesh.normals_make_consistent(inside=inside)


def save_blend(path, autopack=False, verbose=False):
    if verbose:
        print(f"Saving .blend to {path} ({'with' if autopack else 'without'} textures)")

    with Suppress():
        if autopack:
            bpy.ops.file.autopack_toggle()
        bpy.ops.wm.save_as_mainfile(filepath=str(path))
        if autopack:
            bpy.ops.file.autopack_toggle()


def joined_kd(objs, include_origins=False):
    if not isinstance(objs, list):
        objs = objs
    objs = [o for o in objs if o.type == 'MESH']

    size = sum(len(o.data.vertices) for o in objs)
    if include_origins:
        size += len(objs)
    kd = mathutils.kdtree.KDTree(size)

    i = 0
    for o in objs:
        for v in o.data.vertices:
            assert i < size
            kd.insert(o.matrix_world @ v.co, i)
            i += 1
        if include_origins:
            kd.insert(o.location, i)
            i += 1

    kd.balance()

    return kd

def make_instances_real():
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if len(obj.particle_systems) == 0:
            continue

        obj.select_set(True)
        bpy.ops.object.duplicates_make_real()
        obj.select_set(False)
    bpy.ops.object.select_all(action='DESELECT')


# faces are required to be triangles now
def objectdata_from_VF(vertices, faces):
    new_mesh = bpy.data.meshes.new("")
    new_mesh.vertices.add(len(vertices))
    new_mesh.vertices.foreach_set("co", vertices.reshape(-1).astype(np.float32))
    new_mesh.polygons.add(len(faces))
    new_mesh.loops.add(len(faces) * 3)
    new_mesh.polygons.foreach_set("loop_total", np.ones(len(faces), np.int32) * 3)
    new_mesh.polygons.foreach_set("loop_start", np.arange(len(faces), dtype=np.int32) * 3)
    new_mesh.polygons.foreach_set("vertices", faces.reshape(-1).astype(np.int32))
    new_mesh.update(calc_edges=True)
    return new_mesh


def object_from_VF(vertices, faces, name):
    new_mesh = objectdata_from_VF(vertices, faces)
    new_object = bpy.data.objects.new(name, new_mesh)
    new_object.rotation_euler = (0, 0, 0)
    return new_object


def object_from_trimesh(mesh, name, material=None):
    if name in bpy.data.objects.keys():
        print("replacing original object")
        delete(bpy.data.objects[name])
    new_object = object_from_VF(mesh.vertices, mesh.faces, name)
    for attr_name in mesh.vertex_attributes:
        attr_name_ls = attr_name.lstrip("_")  # this is because of trimesh bug
        if mesh.vertex_attributes[attr_name].ndim == 1 or mesh.vertex_attributes[attr_name].shape[1] == 1:
            type_key = "FLOAT"
        elif mesh.vertex_attributes[attr_name].shape[1] == 3:
            type_key = "FLOAT_VECTOR"
        elif mesh.vertex_attributes[attr_name].shape[1] == 4:
            type_key = "FLOAT_COLOR"
        else:
            raise Exception(f"attribute of shape {mesh.vertex_attributes[attr_name].shape} not supported")
        new_object.data.attributes.new(name=attr_name_ls, type=type_key, domain='POINT')
        new_object.data.attributes[attr_name_ls].data.foreach_set(DATATYPE_FIELDS[type_key],
                                                                  mesh.vertex_attributes[attr_name].reshape(
                                                                      -1).astype(np.float32))
    if material is not None:
        new_object.data.materials.append(material)
    return new_object


def object_to_vertex_attributes(obj):
    vertex_attributes = {}
    for attr in obj.data.attributes.keys():
        type_key = obj.data.attributes[attr].data_type
        tmp = np.zeros(len(obj.data.vertices) * DATATYPE_DIMS[type_key], dtype=np.float32)
        obj.data.attributes[attr].data.foreach_get(DATATYPE_FIELDS[type_key], tmp)
        vertex_attributes[attr] = tmp.reshape((len(obj.data.vertices), -1))
    return vertex_attributes


def object_to_trimesh(obj):
    verts_bpy = obj.data.vertices
    faces_bpy = obj.data.polygons
    verts = np.zeros((len(verts_bpy) * 3), dtype=float)
    verts_bpy.foreach_get("co", verts)
    faces = np.zeros((len(faces_bpy) * 3), dtype=np.int32)
    faces_bpy.foreach_get("vertices", faces)
    faces = faces.reshape((-1, 3))
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    vertex_attributes = object_to_vertex_attributes(obj)
    mesh.vertex_attributes.update(vertex_attributes)
    return mesh

def blender_internal_attr(a):
    if hasattr(a, 'name'):
        a = a.name
    if a.startswith('.'):
        return True
    if a in ['material_index', 'uv_map', 'UVMap']:
        return True
    return False

def merge_by_distance(obj, face_size):
    with SelectObjects(obj), ViewportMode(obj, mode='EDIT'), Suppress():
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.remove_doubles(threshold=face_size)

def origin_set(objs, mode, **kwargs):
    with SelectObjects(objs):
        bpy.ops.object.origin_set(type=mode, **kwargs)
        
def apply_geo(obj):
    with SelectObjects(obj):
        for m in obj.modifiers:
            m.show_viewport = False
        for m in obj.modifiers:
            if m.type == 'NODES':
                bpy.ops.object.modifier_apply(modifier=m.name)

def avg_approx_vol(objects):
    return np.mean([prod(list(o.dimensions)) for o in objects])

def parent_to(a, b, type='OBJECT', keep_transform=False, no_inverse=False, no_transform=False):
    select_none()
    with SelectObjects([a, b], active=1):
        if no_inverse:
            bpy.ops.object.parent_no_inverse_set(keep_transform=keep_transform)
        else:
            bpy.ops.object.parent_set(type=type, keep_transform=keep_transform)

    if no_transform:
        a.location = (0,0,0)
        a.rotation_euler = (0,0,0)

    assert a.parent is b

def apply_matrix_world(obj, verts: np.array):
    return mutil.dehomogenize(mutil.homogenize(verts) @ np.array(obj.matrix_world).T)

def surface_area(obj: bpy.types.Object):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    area = sum(f.calc_area() for f in bm.faces)
    bm.free()
    return area

def approve_all_drivers():

    # 'Touch' every driver in the file so that blender trusts them

    n = 0

    for o in bpy.data.objects:
        if o.animation_data is None:
            continue
        for d in o.animation_data.drivers:
            d.driver.expression = d.driver.expression
            n += 1

    logging.warning(f'Re-initialized {n} as trusted. Do not run infinigen on untrusted blend files. ')

def count_objects():
    count = 0
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH": 
            continue
        count +=1
    return count

def count_objects():
    count = 0
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH": continue
        count +=1
    return count

def count_instance():
    depsgraph = bpy.context.evaluated_depsgraph_get()
    return len([inst for inst in depsgraph.object_instances if inst.is_instance])
    
    
def bounds(obj):
    bbox = np.array(obj.bound_box)
    return bbox.min(axis=0), bbox.max(axis=0)

def create_noise_plane(size=50, cuts=10, std=3, levels=3):
    bpy.ops.mesh.primitive_grid_add(size=size, x_subdivisions=cuts, y_subdivisions=cuts)
    obj = bpy.context.active_object

    for v in obj.data.vertices:
        v.co[2] = v.co[2] + np.random.normal(0, std)

    return modify_mesh(obj, 'SUBSURF', levels=levels)