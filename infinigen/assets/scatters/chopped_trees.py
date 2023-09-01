# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import pdb
import logging

import bpy
import mathutils

import numpy as np
from numpy.random import uniform, normal
from tqdm import tqdm, trange

from infinigen.core.util import blender as butil
from infinigen.core.util.math import rotate_match_directions, randomspacing
from infinigen.assets.creatures.util.geometry.metaballs import plusx_cylinder_unwrap

from infinigen.core.nodes import node_utils
from infinigen.core.placement.instance_scatter import scatter_instances
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core import surface
from infinigen.assets.materials import wood

from infinigen.core.placement.detail import remesh_with_attrs, target_face_size, scatter_res_distance

from infinigen.assets.trees.generate import GenericTreeFactory, random_species

logger = logging.getLogger(__name__)

def approx_settle_transform(obj, samples=200):
    assert obj.type == 'MESH'

    if len(obj.data.vertices) < 3 or len(obj.data.polygons) == 0:
        return

    with butil.SelectObjects(obj):
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

    # sample random planes and find the normal of the biggest one
    verts = np.empty((len(obj.data.vertices), 3))
    obj.data.vertices.foreach_get('co', verts.reshape(-1))
    verts = np.stack([verts[np.random.choice(np.arange(len(verts)), samples)] for _ in range(3)], axis=0)
    ups = np.cross(verts[0] - verts[1], verts[0] - verts[2], axis=-1)
    best = np.linalg.norm(ups, axis=-1).argmax()

    # rotate according to that axis
    rot_mat = rotate_match_directions(ups[best].reshape(1, 3), np.array([0, 0, 1]).reshape(1, 3))[0]
    obj.rotation_euler = mathutils.Matrix(rot_mat).to_euler()

    with butil.SelectObjects(obj):
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    
    return obj

def chop_object(
    obj, n, cutter_size, 
    max_tilt=15, thickness=0.03
):

    assert obj.type == 'MESH'

    bbox = np.array([obj.matrix_world @ mathutils.Vector(v) for v in obj.bound_box])

    def cutter(t):
        butil.select_none()
        z = butil.lerp(bbox[:, -1].min(), bbox[:, -1].max(), t)
        loc = (*bbox[:,:-1].mean(axis=0), z)
        bpy.ops.mesh.primitive_plane_add(size=cutter_size, location=loc)
        cut = bpy.context.active_object
        cut.name = f'cutter({t:.2f})'

        butil.modify_mesh(cut, 'SOLIDIFY', thickness=thickness)
        butil.recalc_normals(cut, inside=False)

        if uniform() < 0.95:
            cut.rotation_euler = np.deg2rad(uniform(-max_tilt, max_tilt, 3))
        else:
            # vertical chopper to break things up
            cut.location += mathutils.Vector(normal(0, 0.5, 3))
            cut.rotation_euler = np.deg2rad((uniform([-max_tilt, 50, 0], [max_tilt, 80, 360])))

        return cut

    cutters = [cutter(t) for t in randomspacing(0.05, 0.85, n, margin=uniform(0.1, 0.4))]
    chopped = butil.boolean([obj] + cutters, mode='DIFFERENCE', verbose=True)
    butil.delete(cutters)

    chopped_list = butil.split_object(chopped, mode='LOOSE')
    for obj in chopped_list:
        bpy.context.view_layer.objects.active = obj
        bpy.context.object.active_material_index = len(obj.material_slots) - 1
        bpy.ops.object.material_slot_remove() # remove the default white mat
            
    return chopped_list

def chopped_tree_collection(species_seed, n, boolean_res_mult=5):

    objs = []

    (genome, _, _), _ = random_species(season='winter')
    factory = GenericTreeFactory(species_seed, genome, realize=True,
                                child_col=None, trunk_surface=surface.NoApply, 
                                decimate_placeholder_levels=0, 
                                coarse_mesh_placeholder=True)
    trees = [factory.spawn_placeholder(i,(0,0,0),(0,0,0)) for i in range(n)]

    bark = surface.registry('bark')
    face_size = target_face_size(scatter_res_distance())

    attr_name = 'original_surface'
    for t in trees:
        butil.delete(list(t.children))
        remesh_with_attrs(t, face_size=boolean_res_mult*face_size) # lower res for efficiency
    surface.write_attribute(trees, lambda nw: 1, attr_name, data_type='FLOAT', apply=True)

    for i, tree in enumerate(trees):
         
        n_chops = np.random.randint(3, 6)
        cutter_size = max(tree.dimensions[:-1])
        chopped = chop_object(tree, n=n_chops, cutter_size=cutter_size)

        for j, o in enumerate(chopped):

            if (
                len(o.data.vertices) < 10 or 
                max(o.dimensions) < 0.1 or 
                max(o.dimensions) > cutter_size * 0.8
            ):
                logger.debug(f'filtering {i, j} with {len(o.data.vertices)=}, {o.dimensions=}')
                butil.delete(o)
                chopped[j] = None
                continue

            o.name = f'chopped_tree({species_seed}, {i}, {j})'
            chopped[j] = remesh_with_attrs(o, face_size=face_size)

        chopped = [o for o in chopped if o is not None]


        def selection(nw):
            orig = nw.new_node(Nodes.NamedAttribute, [attr_name], attrs=dict(data_type='FLOAT'))
            return nw.compare('GREATER_THAN', orig, 0.9999) # some interp will happen for some reason, clamp it
        bark.apply(chopped, selection=selection)
        for o in chopped:
            butil.apply_modifiers(o)
            approx_settle_transform(o)
            o.location = (0,0,0)
            o.parent = None
        objs += chopped
    
    return butil.group_in_collection(objs, 'assets:chopped_tree', reuse=False)

def apply(obj, species_seed=None, selection=None, n_trees=1, **kwargs):

    assert obj is not None
    if species_seed is None:
        species_seed = np.random.randint(1e6)
    
    col = chopped_tree_collection(species_seed, n=n_trees)
    col.hide_viewport = True

    scatter_obj = scatter_instances(
        base_obj=obj, collection=col,
        scale=1, scale_rand=0.5, scale_rand_axi=0.15, 
        ground_offset=0.1, density=0.7,
        selection=selection)

    return scatter_obj, col
