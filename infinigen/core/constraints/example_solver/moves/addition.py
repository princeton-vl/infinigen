# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

from dataclasses import dataclass, field
import numpy as np
import typing
import logging
import gin

from pprint import pprint

import bpy
from mathutils import Vector, Matrix
import trimesh

from infinigen.assets.utils import bbox_from_mesh

from infinigen.core.constraints.example_solver.state_def import State, ObjectState

from infinigen.core.constraints import (
    constraint_language as cl,
    usage_lookup
)

from infinigen.core.util import blender as butil
from infinigen.core.constraints.constraint_language.util import (
    delete_obj,
    meshes_from_names
)
from infinigen.core.constraints.example_solver.geometry import(
    validity
)

from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.placement.factory import AssetFactory

from infinigen.core.constraints.example_solver.geometry import dof, parse_scene
from . import moves
from .reassignment import pose_backup, restore_pose_backup
from time import time
# from line_profiler import LineProfiler


logger = logging.getLogger(__name__)

GLOBAL_GENERATOR_SINGLETON_CACHE = {}

def sample_rand_placeholder(gen_class: type[AssetFactory]):


    if singleton_gen and gen_class in GLOBAL_GENERATOR_SINGLETON_CACHE:
        gen = GLOBAL_GENERATOR_SINGLETON_CACHE[gen_class]
    else:
        fac_seed = np.random.randint(1e7)
        gen = gen_class(fac_seed)
        if singleton_gen:
            GLOBAL_GENERATOR_SINGLETON_CACHE[gen_class] = gen

    inst_seed = np.random.randint(1e7)

        new_obj = gen.spawn_placeholder(inst_seed, loc=(0,0,0), rot=(0,0,0))
        new_obj = gen.spawn_asset(inst_seed, loc=(0,0,0), rot=(0,0,0))
        new_obj = bbox_from_mesh.bbox_mesh_from_hipoly(gen, inst_seed, use_pholder=True)
    else:
        new_obj = bbox_from_mesh.bbox_mesh_from_hipoly(gen, inst_seed)

    if new_obj.type != 'MESH':
        raise ValueError(f'Addition created {new_obj.name=} with type {new_obj.type}') 
    if len(new_obj.data.polygons) == 0:
        raise ValueError(f'Addition created {new_obj.name=} with 0 faces') 

    butil.put_in_collection(
        list(butil.iter_object_tree(new_obj)), 
        butil.get_collection(f'placeholders')
    )
    parse_scene.preprocess_obj(new_obj)
    tagging.tag_canonical_surfaces(new_obj)

    return new_obj, gen

@dataclass
class Addition(moves.Move):
    
    """ Move which generates an object and adds it to the scene with certain relations
    """

    gen_class: typing.Any
    relation_assignments: list
    temp_force_tags: set

    _new_obj: bpy.types.Object = None
    

    def __repr__(self):
        return f'{self.__class__.__name__}({self.gen_class.__name__}, {len(self.relation_assignments)} relations)'

    def apply(self, state: State):
        
        target_name, = self.names
        assert target_name not in state.objs

        self._new_obj, gen = sample_rand_placeholder(self.gen_class)

        parse_scene.add_to_scene(state.trimesh_scene, self._new_obj, preprocess=True)

        tags = self.temp_force_tags.union(usage_lookup.usages_of_factory(gen.__class__))

        assert isinstance(self._new_obj, bpy.types.Object)
        objstate = ObjectState(
            obj=self._new_obj,
            generator=gen,
            tags=tags,
            relations=self.relation_assignments
        )

        state.objs[target_name] = objstate
        success = dof.try_apply_relation_constraints(state, target_name)
        logger.debug(f'{self} {success=}')
        return success

    def revert(self, state: State):
        to_delete = list(butil.iter_object_tree(self._new_obj))
        delete_obj(state.trimesh_scene, [a.name for a in to_delete])

        new_name, = self.names
        del state.objs[new_name]

@dataclass
class Resample(moves.Move):

    """ Move which replaces an existing object with a new one from the same generator
    """

    align_corner: int = None

    _backup_gen = None
    _backup_obj = None
    _backup_poseinfo = None

    def apply(self, state: State):

        assert len(self.names) == 1
        target_name = self.names[0]

        os = state.objs[target_name]
        self._backup_gen = os.generator
        self._backup_obj = os.obj
        self._backup_poseinfo = pose_backup(os)

        scene = state.trimesh_scene
        scene.graph.transforms.remove_node(os.obj.name)
        scene.delete_geometry(os.obj.name + '_mesh')

        os.obj, os.generator = sample_rand_placeholder(os.generator.__class__)

        if self.align_corner is not None:
            c_old = self._backup_obj.bound_box[self.align_corner]
            c_new = os.obj.bound_box[self.align_corner]
            raise NotImplementedError(f'{self.align_corner=}')
            
        parse_scene.add_to_scene(state.trimesh_scene, os.obj, preprocess=True)        
        dof.apply_relations_surfacesample(state, target_name)
        
        return validity.check_post_move_validity(state, target_name)
        
    def revert(self, state: State):

        target_name, = self.names

        os = state.objs[target_name]
        delete_obj(state.trimesh_scene, os.obj.name)

        os.obj = self._backup_obj
        os.generator = self._backup_gen
        parse_scene.add_to_scene(state.trimesh_scene, os.obj, preprocess=False)
        restore_pose_backup(state, target_name, self._backup_poseinfo)

    def accept(self, state: State):
        butil.delete(list(butil.iter_object_tree(self._backup_obj)))
