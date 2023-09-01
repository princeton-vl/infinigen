# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import bpy

import logging

import numpy as np
from numpy.random import uniform as U, normal as N, randint
import gin

from infinigen.assets.creatures.util import genome
from infinigen.assets.creatures.util.genome import Joint
from infinigen.assets.creatures import parts
from infinigen.assets.creatures.util.creature_util import offset_center
from infinigen.assets.creatures.util import creature, hair as creature_hair, joining
from infinigen.assets.creatures.util.animation import run_cycle as creature_animation
from infinigen.assets.creatures.util.boid_swarm import BoidSwarmFactory

from infinigen.core.placement.factory import AssetFactory, make_asset_collection
from infinigen.core.util.math import lerp, clip_gaussian, FixedSeed
from infinigen.core import surface
import infinigen.assets.materials.chitin
from infinigen.assets.utils.tag import tag_object, tag_nodegroup


logger = logging.getLogger(__name__)

def insect_hair_params():

    mat_roughness = U(0.7, 1)

    length = U(0.01, 0.04)
    puff = U(0.7, 1)

    return {    
        'density': 4000,
        'clump_n': 1,
        'avoid_features_dist': 0.01,
        'grooming': {
            'Length MinMaxScale': np.array((length, length * N(2, 0.5), U(15, 60)), dtype=np.float32),
            'Puff MinMaxScale': np.array((puff, puff * N(3, 0.5), U(15, 60)), dtype=np.float32),
            'Combing': U(0, 0.2),
            'Strand Random Mag': 0.0,
            'Strand Perlin Mag': 0.0,
            'Strand Perlin Scale': U(15, 45),
            'Tuft Spread': 0.0,
            'Tuft Clumping': 0.0,
            'Root Radius': 0.001,
            'Post Clump Noise Mag': 0,
            'Hair Length Pct Min': U(0.7, 1)
        },
        'material': {
            'Roughness': mat_roughness,
            'Radial Roughness': mat_roughness + N(0, 0.07),
            'Random Roughness': 0,
            'IOR': 1.55
        }
    }

def beetle_postprocessing(body_parts, extras, params):
    
    main_template = surface.registry.sample_registry(params['surface_registry'])
    main_template.apply(body_parts)

def beetle_genome():

    fac = parts.generic_nurbs.NurbsBody(prefix='body_insect', tags=['body', 'rigid'], var=2)
    if U() < 0.5:
        n = len(fac.params['proportions'])
        noise = U(1, 3, n)
        noise[-n//3:] = 1
        fac.params['proportions'] *= noise
    
    body = genome.part(fac)

    l = fac.params['proportions'].sum () * fac.params['length']

    leg_fac = parts.leg.InsectLeg()
    n_leg_pairs = int(np.clip(l * clip_gaussian(3, 2, 2, 6), 2, 15))
    leg_fac.params['length_rad1_rad2'][0] /= n_leg_pairs / 1.8
    splay = U(30, 60)
    for t in np.linspace(0.15, 0.6, n_leg_pairs):
        for side in [-1, 1]:
            leg = genome.part(leg_fac)
            xrot = lerp(70, -100, t)
            genome.attach(leg, body, coord=(t, splay/180, 1), 
                joint=Joint((xrot, 5, 90)), side=side)

    head = genome.part(parts.generic_nurbs.NurbsHead(prefix='head_insect', tags=['head', 'rigid']))
    genome.attach(head, body, coord=(1, 0, 0), joint=Joint((0, -15, 0)))

    if U() < 0.7:
        mandible_fac = parts.head_detail.InsectMandible()
        rot = np.array((120, 20, 80)) * N(1, 0.15)
        for side in [-1, 1]:
            genome.attach(genome.part(mandible_fac), head, coord=(0.75, 0.5, 0.1), 
                joint=Joint(rot), side=side)

    return genome.CreatureGenome(
        parts=body,
        postprocess_params=dict(
            surface_registry=[
                (infinigen.assets.materials.chitin, 1)
            ],
            hair=insect_hair_params()
        )
    )

@gin.configurable
class BeetleFactory(AssetFactory):

    def __init__(self, factory_seed=None, bvh=None, coarse=False, animation_mode=None, **kwargs):
        super().__init__(factory_seed, coarse)
        self.bvh = bvh
        self.animation_mode = animation_mode

    def create_asset(self, i, hair=False, **kwargs):
        genome = beetle_genome()
        root, parts = creature.genome_to_creature(genome, name=f'beetle({self.factory_seed}, {i})')
        tag_object(root, 'beetle')
        offset_center(root)
        joined, extras, arma, ik_targets = joining.join_and_rig_parts(root, parts, genome,
            rigging=(self.animation_mode is not None),
            postprocess_func=beetle_postprocessing, **kwargs)
        if self.animation_mode == 'walk_cycle':
            creature_animation.animate_run(root, arma, ik_targets, steps_per_sec=N(2, 0.2))
        if hair and U() < 0.5:
            creature_hair.configure_hair(joined, root, genome.postprocess_params['hair'])
        return root
    
class AntSwarmFactory(BoidSwarmFactory):

    def ant_swarm_settings(self, mode=None):

        boids_settings = dict(
            use_flight = False,
            use_land = True,
            use_climb = True,

            land_speed_max = U(0.5, 2),
            land_acc_max = U(0.7, 1),
            land_personal_space = 0.05,
            land_jump_speed = U(0, 0.05),

            bank = 0,
            pitch = 0,
            
            rule_fuzzy = U(0.6, 0.9)
        )

        if mode is None:
            mode = np.random.choice(['queues', 'goal_swarm', 'random_swarm'])
            logger.debug(f'Randomly chose ant_swarm_settings {mode=}')

        if mode == 'queues':
            boids_settings['rules'] = [
                dict(type='FOLLOW_LEADER', use_line=True, queue_count=100, distance=0.0),
            ]
        elif mode == 'goal_swarm':
            boids_settings['rules'] = [
                dict(type='SEPARATE'),
                dict(type='GOAL', use_predict=True),
                dict(type='FLOCK')
            ]
        elif mode == 'random_swarm':
            boids_settings['rules'] = [
                dict(type='SEPARATE'),
                dict(type='AVERAGE_SPEED'),
                dict(type='FLOCK')
            ]
        else:
            raise ValueError(f'Unrecognized {mode=}')

        return dict(      
            particle_size=U(0.02, 0.1),
            size_random=U(0.3, 0.7),

            use_rotation_instance=True,
            lifetime=bpy.context.scene.frame_end - bpy.context.scene.frame_start,
            warmup_frames=1, emit_duration=0, # all particles appear immediately
            emit_from='VOLUME',
            mass = 2,
            use_multiply_size_mass=True,
            boids_settings=boids_settings
        )

    def __init__(self, factory_seed, bvh, coarse=False):
        with FixedSeed(factory_seed):
            settings = self.ant_swarm_settings()
            col = make_asset_collection(BeetleFactory(factory_seed=randint(1e7), animation_mode='walk_cycle'), n=1)
        super(AntSwarmFactory, self).__init__(
            factory_seed, child_col=col, 
            collider_col=bpy.data.collections.get('colliders'),
            settings=settings, bvh=bvh,
            volume=N(0.1, 0.015),
            coarse=coarse
    )