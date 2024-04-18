# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import pdb
import warnings
import logging

import bpy
import mathutils

import numpy as np
import gin

from infinigen.core.util import blender as butil
from infinigen.core.nodes.nodegroups import transfer_attributes
from infinigen.core.util.blender import deep_clone_obj


logger = logging.getLogger(__name__)

IS_COARSE = False # Global VARIABLE, set by infinigen_examples/generate_nature.py and used only for whether to emit warnings

@gin.configurable
def scatter_res_distance(dist=4):
    return dist

@gin.configurable
def target_face_size(obj, camera=None, global_multiplier=1, global_clip_min=0.003, global_clip_max=1):

    if camera is None:
        camera = bpy.context.scene.camera
    if camera is None:
        return global_clip_min

    if isinstance(obj, bpy.types.Object):
        if IS_COARSE:
            logger.warn(f'target_face_size({obj.name=}) is using the cameras location which is unsafe for {IS_COARSE=}')
        bbox = np.array([obj.matrix_world @ mathutils.Vector(v) for v in obj.bound_box])
        dists = np.linalg.norm(bbox - np.array(camera.location), axis=-1)
        eval_point = bbox[dists.argmin()]
        dist = np.linalg.norm(eval_point - camera.location)
    elif hasattr(obj, '__len__') and len(obj) == 3:
        if IS_COARSE:
            logger.warn(f'target_face_size({obj.name=}) is using the cameras location which is unsafe for {IS_COARSE=}')
        eval_point = mathutils.Vector(obj)
        dist = np.linalg.norm(eval_point - camera.location)
    elif isinstance(obj, (float, int)):
        dist = obj
    else:
        raise ValueError(f'target_face_size() could not handle {obj=}, {type(obj)=}')

    if camera is None:
        camera = bpy.context.scene.camera
    if camera is None:
        return global_clip_min  # raise ValueError(f'Please add a camera; attempted to   #
        # detail.target_face_size() but {bpy.context.scene.camera=}')
    camd = camera.data

    scene = bpy.context.scene
    mm_to_meter = 0.001
    f_m = mm_to_meter * camd.lens
    sensor_dims = mm_to_meter * np.array([camd.sensor_width, camd.sensor_height])
    pixel_shape = (scene.render.resolution_percentage / 100) * np.array(
        [scene.render.resolution_x, scene.render.resolution_y])

    pixel_dims = (sensor_dims / pixel_shape) * (dist / f_m)

    res = min(pixel_dims)

    return np.clip(global_multiplier * res, global_clip_min, global_clip_max)


def remesh_with_attrs(obj, face_size, apply=True, min_remesh_size=None, attributes=None):

    logger.debug(f'remesh_with_attrs on {obj.name=} with {face_size=:.4f} {attributes=}')

    temp_copy = deep_clone_obj(obj)
    
    remesh_size = face_size if min_remesh_size is None else max(face_size, min_remesh_size)
    butil.modify_mesh(obj, type='REMESH', apply=True, voxel_size=remesh_size)

    transfer_attributes.transfer_all(source=temp_copy, target=obj, attributes=attributes, uvs=True)
    bpy.data.objects.remove(temp_copy, do_unlink=True)

    if remesh_size > face_size:
        subdivide_to_face_size(obj, remesh_size, face_size, apply=True)

    return obj


def sharp_remesh_with_attrs(obj, face_size, apply=True, min_remesh_size=None, attributes=None):
    temp_copy = deep_clone_obj(obj)

    remesh_size = face_size if min_remesh_size is None else max(face_size, min_remesh_size)
    butil.modify_mesh(obj, 'REMESH', apply=apply, mode='SHARP',
                      octree_depth=int(np.ceil(np.log2((max(obj.dimensions) + .01) / remesh_size))))

    transfer_attributes.transfer_all(source=temp_copy, target=obj, attributes=attributes, uvs=True)
    bpy.data.objects.remove(temp_copy, do_unlink=True)

    return obj


def subdivide_to_face_size(obj, from_facesize, to_facesize, apply=True, max_levels=6):
    if to_facesize > from_facesize:
        logger.warn(f'subdivide_to_facesize recieved {from_facesize=} < {to_facesize=}. Subdivision cannot increase facesize')
        return None
    levels = int(np.ceil(np.log2(from_facesize/to_facesize)))
    if max_levels is not None and levels > max_levels:
        logger.warn(f'subdivide_to_facesize({obj.name=}, {from_facesize=:.6f}, {to_facesize=:.6f}) attempted {levels=}, clamping to {max_levels=}')
        levels = max_levels
    logger.debug(f'subdivide_to_face_size applying {levels=} of subsurf to {obj.name=}')
    _, mod = butil.modify_mesh(obj, 'SUBSURF', apply=apply, 
                    levels=levels, render_levels=levels, return_mod=True)
    return mod # None if apply=True

def merged_by_distance_col(col, face_size, inplace=False):
    if not inplace:
        with butil.SelectObjects(list(col.objects)):
            bpy.ops.object.duplicate()
            col = butil.group_in_collection(list(bpy.context.selected_objects),
                                            name=col.name + f'.detail({face_size:.5f})', reuse=False)

    for obj in col.objects:
        butil.merge_by_distance(obj, face_size)

    return col


def min_max_edgelen(mesh):
    verts = np.array([v.co for v in mesh.vertices])
    edges = np.array([e.vertices for e in mesh.edges])
    lens = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=-1)
    lens = np.sort(lens)
    if len(lens) <= 4:
        return lens[0], lens[-1]
    else:
        return lens[len(lens) // 4], lens[-len(lens) // 4]


def adapt_mesh_resolution(obj, face_size, method, approx=0.2, **kwargs):
    
    assert obj.type == 'MESH'
    assert 0 <= approx and approx <= 0.5

    logger.debug(f'adapt_mesh_resolution on {obj.name} with {method=} to {face_size=:.6f}')

    if len(obj.data.polygons) == 0:
        logger.debug(f'Ignoring adapt_mesh_resolution on {obj.name=} due to no polygons')
        return

    lmin, lmax = min_max_edgelen(obj.data)

    if method == 'subdivide':
        if lmax > face_size:
            subdivide_to_face_size(obj, from_facesize=lmax, to_facesize=face_size, **kwargs)
    elif method == 'subdiv_by_area':
        areas = np.zeros(len(obj.data.polygons))
        obj.data.polygons.foreach_get('area', areas)
        approx_facesize = np.sqrt(np.percentile(areas, q=1-approx))
        if approx_facesize > face_size:
            subdivide_to_face_size(obj, from_facesize=approx_facesize, to_facesize=face_size, **kwargs)
        else:
            logger.debug(f'No subdivision necessary on {obj.name=} {approx_facesize} < {face_size}')
    elif method == 'merge_down':
        if lmin < face_size:
            butil.merge_by_distance(obj, face_size)
    elif method == 'remesh':
        remesh_with_attrs(obj, face_size, **kwargs)
    elif method == 'sharp_remesh':
        sharp_remesh_with_attrs(obj, face_size, **kwargs)
    else:
        raise ValueError(f'Unrecognized adapt_mesh_resolution(..., {method=})')