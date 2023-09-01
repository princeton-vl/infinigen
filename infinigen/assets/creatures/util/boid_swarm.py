# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


from functools import partial
import logging

import bpy

import numpy as np
from numpy.random import uniform as U, normal as N

from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement import particles, animation_policy

from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import random_general

logger = logging.getLogger(__name__)


def creature_col_to_particle_col(col, name, prefix='particleassets'):
    roots = [o for o in col.objects if o.parent is None]
    objs = []
    
    for root in roots:
        first, *rest = [o for o in butil.iter_object_tree(root) if o.type == 'MESH']
        if len(rest):
            logger.warning(f'{col.name=} had {root.name=} with multiple child meshes, taking {first.name=}, but it should have been joined with {rest}')
        objs.append(first)
        root.location.z -= 100 # we will have hide_render=False so make sure the base asset is not visible

    col = butil.group_in_collection(objs, name=f'{prefix}:{name}', exclusive=False)
    col.hide_viewport = True
    col.hide_render = False
    return col

class BoidSwarmFactory(AssetFactory):

    def __init__(
        self, factory_seed, child_col, settings, bvh, 
        collider_col, volume=0.1, coarse=False
    ):
        super().__init__(factory_seed, coarse)
        
        self.collider_col = collider_col
        self.settings = settings
        self.bvh = bvh
        self.target_child_volume = volume

        self.col = creature_col_to_particle_col(child_col, name=f'{self}.children')

    def create_placeholder(self, loc, rot, **kwargs) -> bpy.types.Object:
        
        p = butil.spawn_cube(size=4)
        p.location = loc
        p.rotation_euler = rot

        speed_keys = ['land_speed_max', 'air_speed_max', 'climb_speed_max']
        speed = max(self.settings['boids_settings'].get(k, 0) for k in speed_keys)
        step_size_range = speed * 3 * N(1, 0.1) * np.array([0.5, 1.5])
        policy = animation_policy.AnimPolicyRandomForwardWalk(forward_vec=(1, 0, 0),
            speed=speed*N(1, 0.1), step_range=step_size_range, yaw_dist=("normal", 0, 70))
        animation_policy.animate_trajectory(p, self.bvh, policy, retry_rotation=True, max_full_retries=20)

        return p

    def create_asset(self, placeholder, **params) -> bpy.types.Object:

        assert self.col is not None

        size = self.settings['particle_size']
        size_random = self.settings['size_random']
        avg_size = (size + size_random*size)/2
        child_vol = butil.avg_approx_vol(self.col.objects)
        count = random_general(self.target_child_volume) / float(avg_size**3 * child_vol)
        self.settings['count'] = int(count)

        emitter, system = particles.particle_system(
            emitter=placeholder, subject=self.col, 
            collision_collection=self.collider_col,
            settings=self.settings)

        for r in system.settings.boids.states[0].rules:
            if r.type in ['GOAL', 'FOLLOW_LEADER']:
                r.object = placeholder

        logger.info(f'Baking {emitter.name=} with {self.col.name=}')
        particles.bake(emitter, system)

        emitter.hide_render = False
        
        return emitter

    
