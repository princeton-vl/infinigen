# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: 
# - Hongyu Wen: primary author
# - Alexander Raistrick: snake curve following animation


import pdb
import gin
import logging

import bpy

import numpy as np
from numpy.random import normal as N, uniform as U

from infinigen.assets.creatures.util import genome
from infinigen.assets.creatures.util.genome import Joint
from infinigen.assets.creatures import parts

from infinigen.assets.creatures.util.creature_util import euler

import infinigen.assets.materials.spot_sparse_attr
import infinigen.assets.materials.snake_scale
import infinigen.assets.materials.snake_shaders
import infinigen.assets.materials.bird
import infinigen.assets.materials.scale

from infinigen.assets.creatures.util import creature, joining, animation as creature_animation
from infinigen.core.util import blender as butil

from infinigen.assets.materials import bone, tongue, eyeball, nose, horn
from infinigen.core import surface

from infinigen.core.placement.factory import AssetFactory

from infinigen.assets.creatures.util import creature, joining

from infinigen.core.util.math import clip_gaussian, FixedSeed
from infinigen.core.util import blender as butil
from infinigen.core.util.random import random_general

from infinigen.assets.creatures.util.animation import curve_slither
from infinigen.core.placement import animation_policy

from infinigen.assets.creatures.util.animation.run_cycle import follow_path


def dinosaur():
    open_mouth = U() > 0
    # body_size = {
    #     'scale_x': 20 + N(0, 2),
    #     'scale_y': 1,
    #     'scale_z': 1,
    # }
    body_fac = parts.body_tube.ReptileBody(type='dinosaur_body')
    body = genome.part(body_fac)
    shoulder_bounds = np.array([[-20, -20, -20], [20, 20, 20]])

    # fleg_fac = parts.reptile_detail.LizardFrontLeg()
    # toe_fac = parts.reptile_detail.LizardToe()
    # for side in [-1, 1]:
    #     leg = genome.part(fleg_fac)
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.97, 0.5, 0.6), joint=Joint(rest=(0,0,40)))
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.98, 0.5, 0.3), joint=Joint(rest=(0,0,13)))
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.98, 0.5, -0.3), joint=Joint(rest=(0,0,-13)))
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.97, 0.5, -0.6), joint=Joint(rest=(0,0,-40)))
    #     genome.attach(leg, body, coord=(U(0.75, 0.77), 0.5, 0.7), joint=Joint(rest=(0, 0, 110), bounds=shoulder_bounds), side=side)

    # bleg_fac = parts.reptile_detail.LizardBackLeg()
    # for side in [-1, 1]:
    #     leg = genome.part(bleg_fac)
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.97, 0.5, 0.6), joint=Joint(rest=(0,0,40)))
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.98, 0.5, 0.3), joint=Joint(rest=(0,0,13)))
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.98, 0.5, -0.3), joint=Joint(rest=(0,0,-13)))
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.97, 0.5, -0.6), joint=Joint(rest=(0,0,-40)))
    #     genome.attach(leg, body, coord=(U(0.81, 0.83), 0.5, 0.6), joint=Joint(rest=(0, 0, 80), bounds=shoulder_bounds), side=side)

    # neck_fac = parts.reptile_neck.ReptileNeck()
    # neck = genome.part(neck_fac)
    # genome.attach(neck, body, coord=(0.1, 0, 0.2), joint=Joint(rest=(180, 180, 0)), rotation_basis='global', bridge_rad=0.2, smooth_rad=0.1)

    
    # head_size = {
    #     'scale_x': 0.8 + N(0, 0.02),
    #     'scale_y': 0.3 + N(0, 0.02),
    # }
    # head_fac = parts.reptile_detail.ReptileUpperHead(head_size)
    # head = genome.part(head_fac)
    # genome.attach(head, neck, coord=(0.88, 0, 0.2), joint=Joint(rest=(180, 180, 180)), rotation_basis='global', bridge_rad=0.2, smooth_rad=0.1)

    # eye_fac = parts.eye.MammalEye({'Radius': N(0.03, 0.005)})
    # t, splay = U(0.7, 0.7), 100/180
    # r = 1
    # rot = np.array([0, 0, 90]) * N(1, 0.1, 3)
    # for side in [-1, 1]:
    #     eye = genome.part(eye_fac)
    #     genome.attach(eye, head, coord=(t, splay, r), joint=Joint(rest=(0,0,0)), rotation_basis='normal', side=side)

    # # teeth
    # horn_fac = parts.horn.Horn({'depth_of_ridge': 0, 'length': U(0.2, 0.3), 'rad1': U(0.4, 0.4), 'rad2': U(0.3, 0.3), 'thickness': U(0.04, 0.08), 'height': 0})
    # t, splay = U(0.67, 0.7), 60/180
    # for side in [-1, 1]:
    #     horn = genome.part(horn_fac)
    #     genome.attach(horn, head, coord=(t, splay, 0.8), joint=Joint(rest=(30, 130, -20)), rotation_basis='global', side=side)


    # jaw_fac = parts.reptile_detail.ReptileLowerHead(head_size)
    # jaw = genome.part(jaw_fac)
    # genome.attach(jaw, neck, coord=(0.88, 0, 0.1), joint=Joint(rest=(180, 170, 180)), rotation_basis='global', bridge_rad=0.1, smooth_rad=0.1)

    return genome.CreatureGenome(
        parts=body,
        postprocess_params=dict(
            animation=dict(), 
            surface_registry=[
                (infinigen.assets.materials.snake_scale, 1),
            ]
        ) 
    )

def lizard_genome():
    open_mouth = U() > 0
    # body_fac = parts.reptile_detail.ReptileBody(type='lizard', n_bones=15, shoulder_ik_ts=[0.0, 0.3, 0.6, 1.0])
    # body = genome.part(body_fac)
    # shoulder_bounds = np.array([[-20, -20, -20], [20, 20, 20]])

    # fleg_fac = parts.reptile_detail.LizardFrontLeg()
    # toe_fac = parts.reptile_detail.LizardToe()
    # for side in [-1, 1]:
    #     leg = genome.part(fleg_fac)
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.97, 0.5, 0.6), joint=Joint(rest=(0,0,40)))
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.98, 0.5, 0.3), joint=Joint(rest=(0,0,13)))
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.98, 0.5, -0.3), joint=Joint(rest=(0,0,-13)))
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.97, 0.5, -0.6), joint=Joint(rest=(0,0,-40)))
    #     genome.attach(leg, body, coord=(U(0.05, 0.07), 0.5, 0.4), joint=Joint(rest=(0, 0, 90), bounds=shoulder_bounds), side=side, bridge_rad=0.1, smooth_rad=0.1)

    # bleg_fac = parts.reptile_detail.LizardBackLeg()
    # for side in [-1, 1]:
    #     leg = genome.part(bleg_fac)
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.97, 0.5, 0.6), joint=Joint(rest=(0,0,40)))
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.98, 0.5, 0.3), joint=Joint(rest=(0,0,13)))
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.98, 0.5, -0.3), joint=Joint(rest=(0,0,-13)))
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.97, 0.5, -0.6), joint=Joint(rest=(0,0,-40)))
    #     genome.attach(leg, body, coord=(U(0.21, 0.23), 0.5, 0.6), joint=Joint(rest=(0, 0, 80), bounds=shoulder_bounds), side=side, bridge_rad=0.1, smooth_rad=0.1)
    
    head_size = {
        'scale_x': 0.8 + N(0, 0.02),
        'scale_y': 0.3 + N(0, 0.02),
    }
    head_fac = parts.reptile_detail.ReptileUpperHead(head_size)
    head = genome.part(head_fac)
    # genome.attach(head, body, coord=(0.01, 0, 0.2), joint=Joint(rest=(180, 180, 0)), rotation_basis='global', bridge_rad=0.2, smooth_rad=0.1)

    # eye_fac = parts.eye.MammalEye({'Radius': N(0.03, 0.005)})
    # t, splay = U(0.7, 0.7), 100/180
    # r = 1
    # rot = np.array([0, 0, 90]) * N(1, 0.1, 3)
    # for side in [-1, 1]:
    #     eye = genome.part(eye_fac)
    #     genome.attach(eye, head, coord=(t, splay, r), joint=Joint(rest=(0,0,0)), rotation_basis='normal', side=side)

    # # teeth
    # horn_fac = parts.horn.Horn({'depth_of_ridge': 0, 'length': U(0.2, 0.3), 'rad1': U(0.4, 0.4), 'rad2': U(0.3, 0.3), 'thickness': U(0.04, 0.08), 'height': 0})
    # t, splay = U(0.67, 0.7), 60/180
    # for side in [-1, 1]:
    #     horn = genome.part(horn_fac)
    #     genome.attach(horn, head, coord=(t, splay, 0.8), joint=Joint(rest=(30, 130, -20)), rotation_basis='global', side=side)


    # jaw_fac = parts.reptile_detail.ReptileLowerHead(head_size)
    # jaw = genome.part(jaw_fac)
    # genome.attach(jaw, body, coord=(0.01, 0, 0.1), joint=Joint(rest=(180, 150, 0)), rotation_basis='global', bridge_rad=0.1, smooth_rad=0.1)

    return genome.CreatureGenome(
        parts=head,   
        postprocess_params=dict(
            anim=lizard_run_params(),
            surface_registry=[
                (infinigen.assets.materials.snake_scale, 1),
            ]
        ) 
    )

def snake_genome():
    open_mouth = U() > 0

    w_mod = N(1, 0.05)
    h_mod = N(1, 0.05)

    body_fac = parts.reptile_detail.ReptileBody(type='snake', n_bones=15, shoulder_ik_ts=[0.0, 0.3, 0.6, 1.0], mod=(1, w_mod, h_mod))

    body = genome.part(body_fac)

    head_size = {
        'scale_x': 0.8 + N(0, 0.02),
        'scale_y': 0.3 + N(0, 0.02),
    }

    head_fac = parts.reptile_detail.ReptileUpperHead(head_size, mod=(1, w_mod, h_mod))
    head = genome.part(head_fac)
    genome.attach(head, body, coord=(0.01, 0, 0.2), joint=Joint(rest=(180, 180, 0)), rotation_basis='global', bridge_rad=0.2, smooth_rad=0.1)

    eye_fac = parts.eye.MammalEye({'Radius': N(0.03, 0.005)})
    t, splay = U(0.7, 0.7), 100/180
    r = 1
    rot = np.array([0, 0, 90]) * N(1, 0.1, 3)
    for side in [-1, 1]:
        eye = genome.part(eye_fac)
        genome.attach(eye, head, coord=(t, splay, r), joint=Joint(rest=(0,0,0)), rotation_basis='normal', side=side)

    # teeth
    horn_fac = parts.horn.Horn({'depth_of_ridge': 0, 'length': U(0.2, 0.3), 'rad1': U(0.4, 0.4), 'rad2': U(0.3, 0.3), 'thickness': U(0.04, 0.08), 'height': 0})
    t, splay = U(0.67, 0.7), 60/180
    for side in [-1, 1]:
        horn = genome.part(horn_fac)
        genome.attach(horn, head, coord=(t, splay, 0.8), joint=Joint(rest=(30, 130, -20)), rotation_basis='global', side=side)

    jaw_fac = parts.reptile_detail.ReptileLowerHead(head_size, mod=(1, w_mod, h_mod))
    jaw = genome.part(jaw_fac)
    mouth_open_deg = 0
    genome.attach(jaw, body, coord=(0.01, 0, 0.15), joint=Joint(rest=(180, 180 - mouth_open_deg, 0)), rotation_basis='global', bridge_rad=0.1, smooth_rad=0.1)

    return genome.CreatureGenome(
        parts=body,   
        postprocess_params=dict(
            anim=snake_swim_params(),
            surface_registry=[
                (infinigen.assets.materials.snake_scale, 1),
            ]
        ) 
    )

def chameleon_genome():
    open_mouth = U() > 0

    body_fac = parts.chameleon.Chameleon()
    body = genome.part(body_fac)

    return genome.CreatureGenome(
        parts=body,   
        postprocess_params=dict(
            anim=snake_swim_params(),
            surface_registry=[
                (infinigen.assets.materials.snake_scale, 1),
            ]
        ) 
    )

def frog_genome():
    #body_fac = parts.reptile_detail.ReptileHeadBody(params={'open_mouth': False}, type='frog')
    #body = genome.part(body_fac)
    #shoulder_bounds = np.array([[-20, -20, -20], [20, 20, 20]])
    # open_mouth = U() > 0
    # body_fac = parts.body_tube.ReptileBody(type='frog_body')
    # body = genome.part(body_fac)
    # shoulder_bounds = np.array([[-20, -20, -20], [20, 20, 20]])

    # fleg_fac = parts.reptile_detail.LizardFrontLeg(type='frog')
    # toe_fac = parts.reptile_detail.LizardToe()
    # for side in [-1, 1]:
    #     leg = genome.part(fleg_fac)
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.97, 0.5, 0.6), joint=Joint(rest=(0,0,40)))
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.98, 0.5, 0.3), joint=Joint(rest=(0,0,13)))
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.98, 0.5, -0.3), joint=Joint(rest=(0,0,-13)))
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.97, 0.5, -0.6), joint=Joint(rest=(0,0,-40)))
    #     genome.attach(leg, body, coord=(U(0.5, 0.55), 0.45, 0.9), joint=Joint(rest=(0, 0, 110), bounds=shoulder_bounds), side=side)

    # bleg_fac = parts.reptile_detail.LizardBackLeg(type='frog')
    # for side in [-1, 1]:
    #     leg = genome.part(bleg_fac)
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.97, 0.5, 0.6), joint=Joint(rest=(0,0,40)))
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.98, 0.5, 0.3), joint=Joint(rest=(0,0,13)))
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.98, 0.5, -0.3), joint=Joint(rest=(0,0,-13)))
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.97, 0.5, -0.6), joint=Joint(rest=(0,0,-40)))
    #     genome.attach(leg, body, coord=(U(0.70, 0.75), 0.45, 0.8), joint=Joint(rest=(0, 0, 50), bounds=shoulder_bounds), side=side)
    
    head_size = {
        'scale_y': 0.4 + N(0, 0.02),
    }
    head_fac = parts.reptile_detail.ReptileUpperHead(head_size)
    head = genome.part(head_fac)
    # genome.attach(head, body, coord=(0.1, 0.5, 0.5), joint=Joint(rest=(180, 180, 0)), rotation_basis='global')

    # eye_fac = parts.eye.MammalEye({'Radius': N(0.03, 0.005)})
    # t, splay = U(0.7, 0.7), 100/180
    # r = 1
    # rot = np.array([0, 0, 90]) * N(1, 0.1, 3)
    # for side in [-1, 1]:
    #     eye = genome.part(eye_fac)
    #     genome.attach(eye, head, coord=(t, splay, r), joint=Joint(rest=(0,0,0)), rotation_basis='normal', side=side)

    # jaw_fac = parts.reptile_detail.ReptileLowerHead(head_size)
    # jaw = genome.part(jaw_fac)
    # genome.attach(jaw, body, coord=(0.1, 0.5, 0.5), joint=Joint(rest=(180, 170, 0)), rotation_basis='global')

    return genome.CreatureGenome(
        parts=head,
        postprocess_func=reptile_postprocessing,
        postprocess_params=dict(
            animation=dict(
                mode='swim',
                speed_m_s=0.5
            ), 
            surface_registry=[
                (infinigen.assets.materials.snake_scale, 1),
            ]
        ) 
    )

def snake_swim_params():
    swim_freq = 2 * clip_gaussian(1, 0.3, 0.1, 2)
    swim_mag = N(200, 3)
    return dict(
        swim_mag=swim_mag,
        swim_freq=swim_freq,
        flipper_freq = 2 * clip_gaussian(1, 0.5, 0.1, 3) * swim_freq,
        flipper_mag = 0.25 * N(1, 0.1) * swim_mag,
        flipper_var = U(0, 0.2),
    )

def chameleon_eye_params():
    swim_freq = 0.2 * clip_gaussian(1, 0.3, 0.1, 2)
    swim_mag = N(20, 3)
    return dict(
        swim_mag=swim_mag,
        swim_freq=swim_freq,
        flipper_freq = 2 * clip_gaussian(1, 0.5, 0.1, 3) * swim_freq,
        flipper_mag = 0.25 * N(1, 0.1) * swim_mag,
        flipper_var = U(0, 0.2),
    )


def animate_snake_swim(root, arma, params, ik_targets):
    spine = [b for b in arma.pose.bones if 'Body' in b.name]
    creature_animation.animate_wiggle_bones(
        arma=arma, bones=spine, fixed_head=False, off=1/2,
        mag_deg=params['swim_mag'], freq=params['swim_freq'], wavelength=U(0.2, 0.4))
    
def animate_chameleon_eye(root, arma, params, ik_targets):
    spine = [b for b in arma.pose.bones if 'Eye' in b.name]
    creature_animation.animate_wiggle_bones(
        arma=arma, bones=spine, fixed_head=False, off=1/2,
        mag_deg=params['swim_mag'], freq=params['swim_freq'], wavelength=U(0.2, 0.4))
    
def lizard_run_params():
    swim_freq = 1 * clip_gaussian(1, 0.3, 0.1, 2)
    swim_mag = N(50, 3)
    return dict(
        swim_mag=swim_mag,
        swim_freq=swim_freq,
        flipper_freq = 2 * clip_gaussian(1, 0.5, 0.1, 3) * swim_freq,
        flipper_mag = 0.25 * N(1, 0.1) * swim_mag,
        flipper_var = U(0, 0.2),
    )

def animate_lizard_run(root, arma, params, ik_targets):
    spine = [b for b in arma.pose.bones if 'Body' in b.name]
    creature_animation.animate_wiggle_bones(
        arma=arma, bones=spine, fixed_head=False, off=1/2,
        mag_deg=params['swim_mag'], freq=params['swim_freq'], wavelength=U(1, 1.2))
    
    spine = [b for b in arma.pose.bones if 'FrontLeg' in b.name]
    print(spine)
    creature_animation.animate_running_front_leg(
        arma=arma, bones=spine, fixed_head=False, off=1/2,
        mag_deg=params['swim_mag'], freq=params['swim_freq'], wavelength=U(1, 1.2))

    spine = [b for b in arma.pose.bones if 'BackLeg' in b.name]
    print(spine)
    creature_animation.animate_running_back_leg(
        arma=arma, bones=spine, fixed_head=False, off=0,
        mag_deg=params['swim_mag'], freq=params['swim_freq'], wavelength=U(1, 1.2))
    # creature_animation.animate_run(root, arma, ik_targets)

def reptile_postprocessing(body_parts, extras, params):
    get_extras = lambda k: [o for o in extras if k in o.name]
    main_template = surface.registry.sample_registry(params['surface_registry'])
    body = body_parts + get_extras('BodyExtra')
    main_template.apply(body)

    tongue.apply(get_extras('Tongue'))
    bone.apply(get_extras('Horn'))
    eyeball.apply(get_extras('Eyeball'), shader_kwargs={"coord": "X"})
    nose.apply(get_extras('Nose'))

def chameleon_postprocessing(body_parts, extras, params):
    get_extras = lambda k: [o for o in extras if k in o.name]
    main_template = surface.registry.sample_registry(params['surface_registry'])
    body = body_parts + get_extras('BodyExtra')
    main_template.apply(body)

    #chameleon_eye.apply(get_extras('Eye'))

@gin.configurable
class LizardFactory(AssetFactory):

    max_distance = 40

    def __init__(self, factory_seed, bvh=None, coarse=False):
        super().__init__(factory_seed, coarse)
        self.bvh = bvh

    def create_asset(self, i, animate=False, rigging=False, cloth=False, **kwargs):    
        genome = lizard_genome()
        root, parts = creature.genome_to_creature(genome, name=f'lizard({self.factory_seed}, {i})')
        
        joined, extras, arma, ik_targets = joining.join_and_rig_parts(root, parts, genome,
            postprocess_func=reptile_postprocessing, adapt_mode='remesh', rigging=rigging, **kwargs)
        if animate and arma is not None:
            pass 
        else:
            joined = butil.join_objects([joined] + extras)
            
        return root
    
@gin.configurable
class FrogFactory(AssetFactory):

    max_distance = 40

    def __init__(self, factory_seed, bvh=None, coarse=False):
        super().__init__(factory_seed, coarse)
        self.bvh = bvh

    def create_asset(self, i, animate=False, rigging=False, simulate=False, **kwargs):
        
        genome = frog_genome()
        root, parts = creature.genome_to_creature(genome, name=f'frog({self.factory_seed}, {i})')
        
        joined, extras, arma, ik_targets = joining.join_and_rig_parts(root, parts, genome,
            postprocess_func=reptile_postprocessing, adapt_mode='remesh', rigging=rigging, **kwargs)
        if animate and arma is not None:
            pass 
        if simulate:
            pass
        else:
            joined = butil.join_objects([joined] + extras)
            
        return root
    
@gin.configurable
class SnakeFactory(AssetFactory):

    max_distance = 40

    def __init__(self, factory_seed, bvh=None, coarse=False, snake_length=('uniform', 0.5, 3), **kwargs):
        super().__init__(factory_seed, coarse)
        self.bvh = bvh
        with FixedSeed(factory_seed):
            self.snake_length = random_general(snake_length)
            self.policy = animation_policy.AnimPolicyRandomForwardWalk(
                forward_vec=(1, 0, 0), speed=min(self.snake_length, 2)*U(0.5, 1), 
                step_range=(0.2, 0.2), yaw_dist=("uniform", -7, 7)) # take very small steps, to avoid clipping into convex surfaces

    def create_placeholder(self, i, loc, rot, **kwargs):
        p = butil.spawn_cube(size=self.snake_length)
        p.location = loc
        p.rotation_euler = rot

        if self.bvh is None:
            return p
        
        curve = animation_policy.policy_create_bezier_path(p, self.bvh, self.policy, eval_offset=(0, 0, 0.5), retry_rotation=True)
        curve.name = f'animhelper:{self}.create_placeholder({i}).path'

        slither_curve = butil.deep_clone_obj(curve)
        curve_slither.add_curve_slithers(slither_curve, snake_length=self.snake_length)

        if slither_curve.type != 'CURVE':
            logging.warning(f'{self.__class__.__name__} created invalid path {curve.name} with {curve.type=}')
            return p

        curve_slither.snap_curve_to_floor(slither_curve, self.bvh)
        butil.parent_to(curve, slither_curve, keep_transform=True)

        # animate the placeholder to the APPROX location of the snake, so the camera can follow it
        follow_path(p, curve, use_curve_follow=True, offset=0,
                    duration=bpy.context.scene.frame_end-bpy.context.scene.frame_start)
        curve.data.driver_add('eval_time').driver.expression = 'frame'

        return p

    def create_asset(self, i, placeholder, **kwargs):

        genome = snake_genome()
        root, parts = creature.genome_to_creature(genome, name=f'snake({self.factory_seed}, {i})')

        joined, extras, arma, ik_targets = joining.join_and_rig_parts(root, parts, genome,
            postprocess_func=reptile_postprocessing, adaptive_resolution=False, rigging=False, **kwargs)

        joined = butil.join_objects([joined] + extras)

        s = self.snake_length / 20 # convert to real units. existing code averages 20m length
        joined.scale = (s, s, s)
        butil.apply_transform(joined, scale=True)

        if len(placeholder.constraints) and placeholder.constraints[0].type == 'FOLLOW_PATH':
            curve = placeholder.constraints[0].target.parent
            assert curve.type == 'CURVE', curve.type
            if len(curve.data.splines[0].points) > 3:
                
                orig_len = curve.data.splines[0].calc_length()
                
                joined.parent = None
                curve_slither.slither_along_path(joined, curve, speed=self.policy.speed, orig_len=orig_len)

                root.parent = butil.spawn_empty('snake_parent_temp') # so AssetFactory.spawn_asset doesnt attempt to parent
                butil.parent_to(joined, root, keep_transform=True)

        return joined

@gin.configurable
class ChameleonFactory(AssetFactory):
    max_distance = 40

    def __init__(self, factory_seed, bvh=None, coarse=False, **kwargs):
        super().__init__(factory_seed, coarse)
        self.bvh = bvh

    def create_placeholder(self, i, loc, rot, **kwargs):
        p = butil.spawn_cube(size=1)
        p.location = loc
        p.rotation_euler = rot

        return p

    def create_asset(self, i, placeholder, **kwargs):

        genome = chameleon_genome()
        root, parts = creature.genome_to_creature(genome, name=f'snake({self.factory_seed}, {i})')

        joined, extras, arma, ik_targets = joining.join_and_rig_parts(root, parts, genome,
            postprocess_func=reptile_postprocessing, adaptive_resolution=False, rigging=False, **kwargs)

        joined = butil.join_objects([joined] + extras)

        return root