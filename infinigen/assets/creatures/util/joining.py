# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import bpy
import logging

import numpy as np

from infinigen.assets.creatures.util import tree, join_smoothing

from infinigen.assets.creatures.util import rigging as creature_rigging

from infinigen.core import surface
from infinigen.core.placement import detail
from infinigen.core.util import blender as butil
from infinigen.core.util.logging import Suppress, Timer

logger = logging.getLogger(__name__)

def compute_joining_effects(genome, parts):

    '''
    Compute all joining curves between parts 
    (only those with some bridge or smooth rad specified in attachment params),
    and compute any bridge parts requested. 

    ASSUMES: All parts have same matrix_world, and are triangles only
    '''

    inter_curves, bridge_objs = {}, []

    g_items = enumerate(tree.iter_items(genome.parts, postorder=True))
    part_items = tree.iter_parent_child(parts, postorder=True)
    for (i, genome), (parent, part) in zip(g_items, part_items):
        
        if genome.att is None:
            continue

        br, sr = genome.att.bridge_rad, genome.att.smooth_rad
        if not br > 0 and not sr > 0:
            continue

        logger.debug(f'Computing joining geometry for {i=} with {br=} and {sr=}')

        try:
            inter = join_smoothing.compute_intersection_curve(
                parent.obj, part.obj, parent.bvh(), part.bvh())
            inter.name = 'intersection_curve'
        except ValueError as e:
            logger.warning(f'join_smoothing.compute_intersection_curve for threw {e}, skipping')
            inter = None
        
        if inter is not None and len(inter.data.vertices) < 4:
            logger.warning(f'join_smoothing.compute_intersection_curve found too few verts, skipping')
            inter = None

        if br > 0 and inter is not None:
            b = join_smoothing.create_bevel_connection(
                parent.obj, part.obj, parent.bvh(), part.bvh(), 
                width=br, intersection_curve=inter, segments=5)
            b.name = part.obj.name + '.bevel_connector'
            b.parent = parent.obj
            bridge_objs.append(b)

        inter_curves[i] = inter

    return inter_curves, bridge_objs

def select_large_component(o, thresh=0.95, tries=5):
    
    with butil.ViewportMode(o, 'EDIT'):
        bpy.ops.mesh.select_all(action="DESELECT")

    r = 0
    for i in range(tries):
        o.data.vertices[r].select = False
        r = np.random.randint(len(o.data.vertices))
        o.data.vertices[r].select = True
        
        with butil.ViewportMode(o, 'EDIT'):
            bpy.ops.mesh.select_mode(type='VERT')
            bpy.ops.mesh.select_linked()

        pct = np.array([v.select for v in o.data.vertices]).mean()
        if pct > thresh:
            return pct

    return 0

def join_and_rig_parts(
    root, parts, genome, face_size, postprocess_func,
    adaptive_resolution=True, adapt_mode='remesh', min_remesh_size=0.01, 
    smooth_joins=True, smooth_attrs=False,
    rigging=False, constraints=False, rig_before_subdiv=False,
    materials=True, roll='GLOBAL_POS_Y',
    **_
):

    body_parts = [o for o in root.children if o.type == 'MESH']
    extras = [o for o in butil.iter_object_tree(root) if not o in body_parts and o is not root]

    if rigging:
        logger.debug(f'Computing creature rig')
        arma, ik_targets = creature_rigging.creature_rig(root, genome, parts, constraints=constraints, roll=roll)
        arma.show_in_front=True

    with butil.SelectObjects(extras):
        bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')

    with butil.SelectObjects(body_parts), butil.CursorLocation(root.location), Suppress():
        # must convert to all transforms applied & triangles only, 
        # in case we want to do join_smoothing
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.remove_doubles(threshold=0.01)
        bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode='OBJECT')
    
        # bvhs no longer valid due to transform / triangulate
        for p in parts:
            p._bvh = None

    if smooth_joins:
        inter_curves, bridge_objs = compute_joining_effects(genome, parts)
        body_parts += bridge_objs

    logger.debug(f'Joining {len(body_parts)=}')
    joined = butil.join_objects(body_parts, check_attributes=False)
    body_parts = [joined]
    joined.parent = root

    for o in extras:
        o.parent = root
    for p in parts:
        p.obj = None # deleted by join, should not be referenced    

    def rig():
        with Timer(f'Computing creature rig weights'):
            with butil.SelectObjects(body_parts + extras, active=-1), Suppress():
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.mesh.remove_doubles(threshold=0.001)
                bpy.ops.object.mode_set(mode="OBJECT")
            with butil.SelectObjects(body_parts + extras + [arma], active=-1):
                bpy.ops.object.parent_set(type='ARMATURE_AUTO')
            arma.parent = root

    if rigging and rig_before_subdiv:
        rig()

    if adaptive_resolution:

        if adapt_mode == 'remesh':
            butil.modify_mesh(joined, 'SUBSURF', levels=1)

        logger.debug(f'Adapting {joined.name=}')
        detail.adapt_mesh_resolution(joined, 
            face_size=max(face_size, min_remesh_size), 
            method=adapt_mode, apply=True)

        # remeshing can create outlier islands that mess with rigging. Clear them out
        percent = select_large_component(joined, thresh=0.9)
        if percent < 0.99: 
            logger.warning(f'Creature had largest component {percent=}')
        else:
            with butil.ViewportMode(joined, 'EDIT'):
                bpy.ops.mesh.select_all(action='INVERT')
                bpy.ops.mesh.delete(type='VERT')
        
        #for e in extras:
        #    detail.adapt_mesh_resolution(e, face_size=face_size, method='subdivide', apply=True)

    # Apply smoothing around any intersection curves found before remeshing
    if adaptive_resolution and smooth_joins:
        assert 'inter_curves' in locals()
        for i, g in enumerate(tree.iter_items(genome.parts, postorder=True)):
            if g.att is None or g.att.smooth_rad == 0: continue
            if not (l := inter_curves.get(i)): continue
            logger.debug(f'Smoothing mesh geometry around {i, l}')
            join_smoothing.smooth_around_line(joined, l, g.att.smooth_rad)

    # Cleanup any remaining join-smoothing-curves
    if smooth_joins and 'inter_curves' in locals():
        for o in inter_curves.values():
            if o is None:
                continue
            butil.delete(o)
            
    if adaptive_resolution and smooth_attrs:
        for attr in joined.data.attributes.keys():
            if butil.blender_internal_attr(attr):
                continue
            logger.debug(f'Smoothing attr {attr}')
            surface.smooth_attribute(joined, attr, iters=10)

    if materials:
        logger.debug(f'Applying postprocess func')
        with butil.DisableModifiers(body_parts):
            postprocess_func(body_parts, extras, genome.postprocess_params)
        
        logger.debug(f'Finalizing material geomods')
        for o in body_parts:
            for m in o.modifiers:
                if m.type == 'NODES':
                    butil.apply_modifiers(o, mod=m)

    if rigging and not rig_before_subdiv:
        rig()

    if not rigging:
        arma = None
        ik_targets = None

    return joined, extras, arma, ik_targets
