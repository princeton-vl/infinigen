import pdb
import gin


import numpy as np
from numpy.random import normal as N, uniform as U

from assets.creatures import genome
from assets.creatures.genome import Joint
from assets.creatures import parts

from assets.creatures.creature_util import euler

import surfaces.templates.basic_bsdf
import surfaces.templates.spot_sparse_attr
import surfaces.templates.snake_scale
import surfaces.templates.snake_shaders
import surfaces.templates.bird
import surfaces.templates.scale

from assets.creatures import creature, generate as creature_gen, animation as creature_animation
from assets.creatures import cloth_sim
from util import blender as butil

from surfaces.templates import bone, tongue, eyeball, nose, horn
from surfaces import surface




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
                (surfaces.templates.snake_scale, 1),
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
                (surfaces.templates.snake_scale, 1),
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

    return genome.CreatureGenome(
        parts=body,   
        postprocess_params=dict(
            anim=snake_swim_params(),
            surface_registry=[
                (surfaces.templates.snake_scale, 1),
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
                (surfaces.templates.snake_scale, 1),
            ]
        ) 
    )

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
                (surfaces.templates.snake_scale, 1),
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
    main_template = surfaces.surface.registry.sample_registry(params['surface_registry'])
    body = body_parts + get_extras('BodyExtra')
    main_template.apply(body)

    tongue.apply(get_extras('Tongue'))
    bone.apply(get_extras('Horn'))
    eyeball.apply(get_extras('Eyeball'), shader_kwargs={"coord": "X"})
    nose.apply(get_extras('Nose'))

def chameleon_postprocessing(body_parts, extras, params):
    get_extras = lambda k: [o for o in extras if k in o.name]
    main_template = surfaces.surface.registry.sample_registry(params['surface_registry'])
    body = body_parts + get_extras('BodyExtra')
    main_template.apply(body)

    chameleon_eye.apply(get_extras('Eye'))

@gin.configurable

    max_distance = 40

        super().__init__(factory_seed, coarse)

    def create_asset(self, i, animate=False, rigging=False, cloth=False, **kwargs):    
        
        joined, extras, arma, ik_targets = creature_gen.join_and_rig_parts(root, parts, genome,
        if animate and arma is not None:
            
        else:
            joined = butil.join_objects([joined] + extras)
            
        return root

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

        joined, extras, arma, ik_targets = creature_gen.join_and_rig_parts(root, parts, genome,


        return root