# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import logging

import bpy
from mathutils.bvhtree import BVHTree

import numpy as np
from tqdm import trange

from infinigen.core.util import blender as butil, camera as cam_util, math
from infinigen.core.util.logging import Suppress
from infinigen.core.placement.camera import get_sensor_coords
from infinigen.core import surface

def raycast_visiblity_mask(obj, cam, start=None, end=None, verbose=True):

    bvh = BVHTree.FromObject(obj, bpy.context.evaluated_depsgraph_get())

    if start is None:
        start = bpy.context.scene.frame_start
    if end is None:
        end = bpy.context.scene.frame_end

    mask = np.zeros(len(obj.data.vertices), dtype=bool)
    rangeiter = trange if verbose else range
    for i in rangeiter(start, end+1):
        bpy.context.scene.frame_set(i)
        invworld = obj.matrix_world.inverted()
        sensor_coords, pix_it = get_sensor_coords(cam)
        for x,y in pix_it:
            direction = (sensor_coords[y,x] - cam.matrix_world.translation).normalized()
            origin = cam.matrix_world.translation
            _, _, index, dist = bvh.ray_cast(invworld @ origin, invworld.to_3x3() @ direction)
            if dist is None:
                continue
            for vi in obj.data.polygons[index].vertices:
                mask[vi] = True

    return mask

def select_vertmask(obj, mask):
    for i, v in enumerate(obj.data.vertices):
        v.select = mask[i]
    for f in obj.data.polygons:
        f.select = any(mask[vi] for vi in f.vertices)

def duplicate_mask(obj, mask, dilate=0, invert=False):
    butil.select_none()
    with butil.ViewportMode(obj, mode='EDIT'):
        bpy.ops.mesh.select_all(action='DESELECT')
    select_vertmask(obj, mask)
    with butil.ViewportMode(obj, mode='EDIT'):
        for _ in range(dilate):
            bpy.ops.mesh.select_more()
        bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='FACE')
        if invert:
            bpy.ops.mesh.select_all(action='INVERT')
        with Suppress():
            bpy.ops.mesh.duplicate_move()    
            try:
                bpy.ops.mesh.separate(type='SELECTED')
                return bpy.context.selected_objects[-1]
            except RuntimeError:
                return butil.spawn_point_cloud('duplicate_mask', [], [])

def split_inview(
    obj: bpy.types.Object, cam, vis_margin, 
    raycast=False, dilate=0, dist_max=1e7, 
    outofview=True, verbose=False, 
    print_areas=False, hide_render=None, suffix=None,
    **kwargs
):

    assert obj.type == 'MESH'
    assert cam.type == 'CAMERA'

    bpy.context.view_layer.update()
    verts = np.zeros((len(obj.data.vertices), 3))
    obj.data.vertices.foreach_get('co', verts.reshape(-1))
    verts = butil.apply_matrix_world(obj, verts)

    dists, vis_dists = cam_util.min_dists_from_cam_trajectory(verts, cam, verbose=verbose, **kwargs)

    vis_mask = vis_dists < vis_margin
    dist_mask = dists < dist_max
    mask = vis_mask * dist_mask

    logging.debug(f'split_inview {vis_mask.mean()=:.2f} {dist_mask.mean()=:.2f} {mask.mean()=:.2f}')

    if raycast:
        mask *= raycast_visiblity_mask(obj, cam)
    
    inview = duplicate_mask(obj, mask, dilate=dilate)

    if outofview:
        outview = duplicate_mask(obj, mask, dilate=dilate, invert=True) 
    else:
        outview = butil.spawn_point_cloud('duplicate_mask', [], [])

    if print_areas:
        sa_in = butil.surface_area(inview)
        sa_out = butil.surface_area(outview)
        print(f'split {obj.name=} into inview area {sa_in:.2f} and outofview area {sa_out:.2f}')

    inview.name = obj.name + '.inview'
    outview.name = obj.name + '.outofview'

    if suffix is not None:
        inview.name += '_' + suffix
        outview.name += '_' + suffix

    if hide_render is not None:
        inview.hide_render = hide_render
        outview.hide_render = hide_render

    return inview, outview, dists, vis_dists
