# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: 
# - Alexander Raistrick: regular bird, hair params
# - Beining Han: adapt to create flying bird


import pdb
import bpy
import mathutils

import numpy as np
from numpy.random import normal as N, uniform as U
import gin

from infinigen.assets.creatures.util import genome
from infinigen.assets.creatures.util.genome import Joint
from infinigen.assets.creatures import parts

from infinigen.assets.creatures.util.creature_util import euler

import infinigen.assets.materials.basic_bsdf
import infinigen.assets.materials.spot_sparse_attr
import infinigen.assets.materials.reptile_brown_circle_attr
import infinigen.assets.materials.reptile_two_color_attr
import infinigen.assets.materials.bird

from infinigen.assets.materials import bone, tongue, eyeball, beak
from infinigen.core.util.math import clip_gaussian, FixedSeed
from infinigen.core.util.random import random_general as rg
from infinigen.core.util import blender as butil
from infinigen.core import surface

from infinigen.core.placement.factory import AssetFactory
from infinigen.assets.creatures.util.creature_util import offset_center
from infinigen.assets.creatures.util import creature, hair as creature_hair, joining
from infinigen.assets.creatures.util.animation.driver_wiggle import animate_wiggle_bones
from infinigen.assets.creatures.util.animation import idle, run_cycle
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

from infinigen.core.placement import animation_policy

def bird_hair_params(flying=True):

    length = U(0.01, 0.025) if flying else U(0.03, 0.06)
    puff = U(0.03, 0.2)

    return {
        'density': 70000,
        'clump_n': 10,
        'avoid_features_dist': 0.02,
        'grooming': {
            'Length MinMaxScale': np.array((length, length * N(2, 0.5), U(15, 60)), dtype=np.float32),
            'Puff MinMaxScale': np.array((puff, puff * N(1.5, 0.5), U(15, 60)), dtype=np.float32),
            'Combing': U(0.6, 1),
            'Strand Random Mag': 0.0,
            'Strand Perlin Mag': U(0, 0.003),
            'Strand Perlin Scale': 30.0,
            'Tuft Spread': 0.01,
            'Tuft Clumping': U(0.5, 1),
            'Root Radius': 0.006,
            'Post Clump Noise Mag': 0.001,
            'Hair Length Pct Min': U(0.5, 0.9)
        },
        'material': {
            'Roughness': U(0, 0.4),
            'Radial Roughness': U(0.1, 0.3),
            'Random Roughness': U(0, 0.2),
            'IOR': 1.55
        }
    }


def bird_postprocessing(body_parts, extras, params):
    
    get_extras = lambda k: [o for o in extras if k in o.name]

    main_template = surface.registry.sample_registry(params['surface_registry'])
    main_template.apply(body_parts + get_extras('BodyExtra') + get_extras('Feather'))

    tongue.apply(get_extras('Tongue'))
    bone.apply(get_extras('Teeth') + get_extras('Claws'))
    eyeball.apply(get_extras('Eyeball'), shader_kwargs={"coord": "X"})
    beak.apply(get_extras('Beak'))

def duck_genome(mode):
    body_lrr = np.array((0.85, 0.25, 0.38)) * N(1, 0.2) * N(1, 0.2, 3)
    body_fac = parts.generic_nurbs.NurbsBody(prefix='body_bird', tags=['body', 'rigid'], var=U(0.3, 1))
    body = genome.part(body_fac)
    l = body_fac.params['length'][0]

    tail = genome.part(parts.wings.BirdTail())
    genome.attach(tail, body, coord=(0.2, 1, 0.5), joint=Joint(rest=(0, 170 * N(1, 0.1), 0)))

    shoulder_bounds = np.array([[-20, -20, -20], [20, 20, 20]])
    foot_fac = parts.foot.Foot({
        'length_rad1_rad2': np.array((l * 0.1, 0.025, 0.04)) * N(1, 0.1) * N(1, 0.1, 3),
        'Toe Length Rad1 Rad2': np.array((l * N(0.4, 0.07), 0.03, 0.02)) * N(1, 0.1) * N(1, 0.1, 3),
        'Toe Splay': 35 * N(1, 0.2),
        'Toebean Radius': 0.03 * N(1, 0.1),
        'Toe Rotate': (0., -1.57, 0.),
        'Claw Curl Deg': 12 * N(1, 0.2),
        'Claw Pct Length Rad1 Rad2': np.array((0.13, 0.64, 0.05)) * N(1, 0.1) * N(1, 0.1, 3),
        'Thumb Pct': np.array((0.61, 1.17, 1.5)) * N(1, 0.1) * N(1, 0.1, 3),
        'Toe Curl Scalar': 0.34 * N(1, 0.2)
    }, bald=True)

    leg_fac = parts.leg.BirdLeg({'length_rad1_rad2': (l * 0.5 * N(1, 0.05), 0.09 * N(1, 0.1), 0.06 * N(1, 0.1))})
    leg_coord = (N(0.5, 0.05), N(0.7, 0.05), N(0.95, 0.05))
    for side in [-1, 1]:
        leg = genome.attach(genome.part(foot_fac), genome.part(leg_fac), coord=(0.9, 0, 0), joint=Joint(rest=(0,0,0)))
        genome.attach(leg, body, coord=leg_coord, joint=Joint(rest=(0, 90, 0), bounds=shoulder_bounds), side=side)

    extension = U(0.7, 1) if mode == 'flying' else U(0.01, 0.1)
    wing_len = l * 0.5 * clip_gaussian(1.2, 0.7, 0.5, 2.5)
    wing_fac = parts.wings.BirdWing({
        'length_rad1_rad2': np.array((wing_len, 0.1 * N(1, 0.1), 0.02 * N(1, 0.2))),
        'Extension': extension
    })

    
    wing_coord = (N(0.7, 0.02), 110/180 * N(1, 0.1), 0.95)
    if wing_fac.params['Extension'] > 0.5:
        wing_rot = (90, 0, 90)
    else:
        wing_rot = (90, 40, 90)
    for side in [-1, 1]:
        wing = genome.part(wing_fac)
        genome.attach(wing, body, coord=wing_coord, joint=Joint(rest=wing_rot), side=side)

    head_fac = parts.head.BirdHead()
    head = genome.part(head_fac)

    beak = genome.part(parts.beak.BirdBeak())
    genome.attach(beak, head, coord=(0.75, 0, 0.5), joint=Joint(rest=(0, 0, 0)))

    eye_fac = parts.eye.MammalEye({'Radius': N(0.03, 0.005)})
    t, splay = U(0.6, 0.85), U(80, 110)/180
    r = 0.85
    rot = np.array([0, 0, 90]) * N(1, 0.1, 3)
    for side in [-1, 1]:
        eye = genome.part(eye_fac)
        genome.attach(eye, head, coord=(t, splay, r), joint=Joint(rest=(0,0,0)), rotation_basis='normal', side=side)

    genome.attach(head, body, coord=(1, 0, 0), joint=Joint(rest=(0, 0, 0)))

    return genome.CreatureGenome(
        parts=body,
        postprocess_params=dict(
            animation=dict(), 
            hair=bird_hair_params(flying=False),
            surface_registry=[
                (infinigen.assets.materials.spot_sparse_attr, 4),
                (infinigen.assets.materials.reptile_brown_circle_attr, 0.5),
                (infinigen.assets.materials.reptile_two_color_attr, 0.5),
                (infinigen.assets.materials.bird, 5)
            ]
        ) 
    )

def flying_bird_genome(mode):

    body_lrr = np.array((0.95, 0.13, 0.18)) * N(1.0, 0.05, size=(3,))
    body = genome.part(parts.body.BirdBody({'length_rad1_rad2': body_lrr}))
    l = body_lrr[0]

    tail = genome.part(parts.wings.FlyingBirdTail())
    genome.attach(tail, body, coord=(U(0.08, 0.15), 1, 0.5), joint=Joint(rest=(0, 180 * N(1, 0.1), 0)))

    shoulder_bounds = np.array([[-20, -20, -20], [20, 20, 20]])
    foot_fac = parts.foot.Foot({
        'length_rad1_rad2': np.array((l * 0.2, 0.01, 0.02)) *  N(1, 0.1, 3),
        'Toe Length Rad1 Rad2': np.array((l * N(0.4, 0.02), 0.02, 0.01)) * N(1, 0.1) * N(1, 0.1, 3),
        'Toe Splay': 8 * N(1, 0.2),
        'Toe Rotate': (0., -N(0.55, 0.1), 0.),
        'Toebean Radius': 0.01 * N(1, 0.1),
        'Claw Curl Deg': 12 * N(1, 0.2),
        'Claw Pct Length Rad1 Rad2': np.array((0.13, 0.64, 0.05)) * N(0.5, 0.05) * N(1, 0.1, 3),
        'Thumb Pct': np.array((0.4, 0.5, 0.75)) * N(1, 0.1) * N(1, 0.1, 3),
        'Toe Curl Scalar': 0.34 * N(1, 0.2)
    }, bald=True)

    leg_fac = parts.leg.BirdLeg({'length_rad1_rad2': (l * 0.5 * N(1, 0.05), 0.04 * N(1, 0.1), 0.02 * N(1, 0.1)),
                'Thigh Rad1 Rad2 Fullness': np.array((0.12, 0.04, 1.26)) * N(1, 0.1, 3),
            'Shin Rad1 Rad2 Fullness': np.array((0.1, 0.04, 5.0)) * N(1, 0.1, 3)})
    leg_coord = (N(0.5, 0.05), N(0.2, 0.04), N(0.8, 0.05))
    for side in [-1, 1]:
        leg = genome.attach(genome.part(foot_fac), genome.part(leg_fac), coord=(0.9, 0, 0), joint=Joint(rest=(0, 0, 0)))
        genome.attach(leg, body, coord=leg_coord, joint=Joint(rest=(0, U(135, 175), 0), bounds=shoulder_bounds), side=side)

    extension = U(0.8, 1)
    wing_len = l * clip_gaussian(1.0, 0.2, 0.6, 1.5) * 0.8
    wing_fac = parts.wings.FlyingBirdWing({
        'length_rad1_rad2': np.array((wing_len, U(0.08, 0.15), 0.02 * N(1, 0.2))),
        'Extension': extension,
        'feather_density': U(25, 40)
    })

    wing_coord = (N(0.68, 0.02), 150 / 180 * N(1, 0.1), 0.8)
    if wing_fac.params['Extension'] > 0.5:
        wing_rot = (90, 0, 90)
    else:
        wing_rot = (90, 40, 90)
    for side in [-1, 1]:
        wing = genome.part(wing_fac)
        genome.attach(wing, body, coord=wing_coord, joint=Joint(rest=wing_rot), side=side)

    head_fac = parts.head.FlyingBirdHead()
    head = genome.part(head_fac)

    beak = genome.part(parts.beak.FlyingBirdBeak())
    genome.attach(beak, head, coord=(0.85, 0, 0.5), joint=Joint(rest=(0, 0, 0)))

    eye_fac = parts.eye.MammalEye({'Radius': N(0.02, 0.005)})
    t, splay = U(0.7, 0.85), U(80, 110) / 180
    r = 0.85
    rot = np.array([0, 0, 90]) * N(1, 0.1, 3)
    for side in [-1, 1]:
        eye = genome.part(eye_fac)
        genome.attach(eye, head, coord=(t, splay, r), joint=Joint(rest=(0, 0, 0)), rotation_basis='normal', side=side)

    genome.attach(head, body, coord=(U(0.84, 0.85), 0, U(1.05, 1.15)), joint=Joint(rest=(0, N(18, 5), 0)))

    return genome.CreatureGenome(
        parts=body,
        postprocess_params=dict(
            animation=dict(), 
            hair=bird_hair_params(flying=True),
            surface_registry=[
                #(infinigen.assets.materials.spot_sparse_attr, 4),
                #(infinigen.assets.materials.reptile_brown_circle_attr, 0.5),
                #(infinigen.assets.materials.reptile_two_color_attr, 0.5),
                (infinigen.assets.materials.bird, 5)
            ]
        ) 
    )

@gin.configurable
class BirdFactory(AssetFactory):

    def __init__(self, factory_seed=None, coarse=False, bvh=None, animation_mode=None, **kwargs):
        super().__init__(factory_seed, coarse)
        self.bvh = bvh
        self.animation_mode = animation_mode

    def create_asset(self, i, placeholder, hair=True, **kwargs):

        dynamic = self.animation_mode is not None

        genome = duck_genome(mode=self.animation_mode)
        root, parts = creature.genome_to_creature(genome, name=f'bird({self.factory_seed}, {i})')
        tag_object(root, 'bird')
        offset_center(root)
        joined, extras, arma, ik_targets = joining.join_and_rig_parts(root, parts, genome,
            rigging=dynamic,
            postprocess_func=bird_postprocessing, **kwargs)
        
        joined_extras = butil.join_objects(extras)
        joined_extras.parent = joined

        butil.parent_to(root, placeholder, no_inverse=True)

        if hair:
            creature_hair.configure_hair(joined, root, genome.postprocess_params['hair'])
        if dynamic:
            if self.animation_mode == 'run':
                run_cycle.animate_run(root, arma, ik_targets)
            elif self.animation_mode == 'idle':
                idle.snap_iks_to_floor(ik_targets, self.bvh)
                idle.idle_body_noise_drivers(ik_targets, wing_mag=U(0, 0.3))
            elif self.animation_mode == 'swim':
                spine = [b for b in arma.pose.bones if 'Body' in b.name]
                tail = [b for b in arma.pose.bones if 'Tail' in b.name]
                animate_wiggle_bones(arma=arma, bones=tail, mag_deg=U(0, 30), freq=U(0.5, 2))
            else:
                raise ValueError(f'Unrecognized mode {self.animation_mode=}')
        return root
   
@gin.configurable
class FlyingBirdFactory(AssetFactory):

    max_expected_radius = 1
    max_distance = 40

    def __init__(self, factory_seed=None, coarse=False, bvh=None, animation_mode=None, altitude=("uniform", 15, 30)):
        super().__init__(factory_seed, coarse)
        self.animation_mode = animation_mode
        self.altitude = altitude
        self.bvh = bvh
        with FixedSeed(factory_seed):
            self.policy = animation_policy.AnimPolicyRandomForwardWalk(
                    forward_vec=(1, 0, 0), speed=U(7, 15), 
                    step_range=(5, 40), yaw_dist=("normal", 0, 15))

    def create_placeholder(self, i, loc, rot):
        
        p = butil.spawn_cube(size=3)
        p.name = f"{self}.create_placeholder({i})"
        p.location = loc
        p.rotation_euler = rot

        if self.bvh is None:
            return p

        altitude = rg(self.altitude)
        p.location.z += altitude
        curve = animation_policy.policy_create_bezier_path(p, self.bvh, self.policy, retry_rotation=True, max_full_retries=30, fatal=True)
        curve.name = f'animhelper:{self}.create_placeholder({i}).path'

        # animate the placeholder to the APPROX location of the snake, so the camera can follow itcurve.location = (0, 0, 0)
        run_cycle.follow_path(p, curve, use_curve_follow=True, offset=0,
            duration=bpy.context.scene.frame_end-bpy.context.scene.frame_start)
        p.rotation_euler.z += np.pi / 2
        curve.data.twist_mode = 'Z_UP'
        curve.data.driver_add('eval_time').driver.expression = 'frame'

        return p

    def create_asset(self, i,  placeholder, hair=True, animate=False,**kwargs):

        genome = flying_bird_genome(self.animation_mode)
        root, parts = creature.genome_to_creature(genome, name=f'flying_bird({self.factory_seed}, {i})')
        joined, extras, arma, ik_targets = joining.join_and_rig_parts(root, parts, genome,
            rigging=self.animation_mode is not None, postprocess_func=bird_postprocessing, **kwargs)
        
        joined_extras = butil.join_objects(extras)
        joined_extras.parent = joined

        if hair:
            creature_hair.configure_hair(joined, root, genome.postprocess_params['hair'])
        if self.animation_mode is not None:
            if self.animation_mode == 'idle':
                idle.idle_body_noise_drivers(ik_targets, body_mag=0.0, foot_motion_chance=1.0, head_benddown=0)
            else:
                raise ValueError(f'Unrecognized {self.animation_mode=}')
        
        return root