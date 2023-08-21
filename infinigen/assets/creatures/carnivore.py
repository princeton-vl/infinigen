# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import numpy as np
from numpy.random import uniform as U, normal as N
import gin

from infinigen.assets.creatures.util import genome
from infinigen.assets.creatures.util.genome import Joint
from infinigen.assets.creatures import parts
from infinigen.core.util.math import clip_gaussian

import infinigen.assets.materials.tiger_attr
import infinigen.assets.materials.giraffe_attr
import infinigen.assets.materials.spot_sparse_attr
from infinigen.core import surface

from infinigen.assets.materials import bone, tongue, eyeball, nose

from infinigen.core.placement.factory import AssetFactory
from infinigen.assets.creatures.util.creature_util import offset_center
from infinigen.assets.creatures.util import creature, joining
from infinigen.assets.creatures.util import hair as creature_hair, cloth_sim
from infinigen.assets.creatures.util.animation import idle, run_cycle

from infinigen.assets.utils.tag import tag_object, tag_nodegroup

from infinigen.core.util import blender as butil


def tiger_hair_params():

    mat_roughness = U(0.4, 0.7)

    length = clip_gaussian(0.022, 0.03, 0.01, 0.1)
    puff = U(0.14, 0.4)

    return {    
        'density': 500000,
        'clump_n': np.random.randint(5, 70),
        'avoid_features_dist': 0.01,
        'grooming': {
            'Length MinMaxScale': np.array((length, length * N(2, 0.5), U(15, 60)), dtype=np.float32),
            'Puff MinMaxScale': np.array((puff, puff * N(3, 0.5), U(15, 60)), dtype=np.float32),
            'Combing': U(0.7, 1),
            'Strand Random Mag': 0.0,
            'Strand Perlin Mag': U(0, 0.006),
            'Strand Perlin Scale': U(15, 45),
            'Tuft Spread': N(0.01, 0.002),
            'Tuft Clumping': U(0.2, 0.8),
            'Root Radius': 0.001,
            'Post Clump Noise Mag': 0.0005 * N(1, 0.15),
            'Hair Length Pct Min': U(0.5, 0.9)
        },
        'material': {
            'Roughness': mat_roughness,
            'Radial Roughness': mat_roughness + N(0, 0.07),
            'Random Roughness': 0,
            'IOR': 1.55
        }
    }

def tiger_skin_sim_params():
    return {
        'bending_stiffness_max': 450.0, 
        'compression_stiffness_max': 80.0, 
        'goal_spring': 0.8, 
        'pin_stiffness': 1, 
        'shear_stiffness': 15.0, 
        'shear_stiffness_max': 80.0, 
        'tension_stiffness_max': 80.0, 
        'uniform_pressure_force': 5.0, 
        'use_pressure': True, 
    }

def tiger_postprocessing(body_parts, extras, params):

    get_extras = lambda k: [o for o in extras if k in o.name]

    main_template = surface.registry.sample_registry(params['surface_registry'])
    main_template.apply(body_parts + get_extras('BodyExtra'))

    tongue.apply(get_extras('Tongue'))
    bone.apply(get_extras('Teeth') + get_extras('Claws'))
    eyeball.apply(get_extras('Eyeball'), shader_kwargs={"coord": "X"})
    nose.apply(get_extras('Nose'))

def tiger_genome():

    body_fac = parts.generic_nurbs.NurbsBody(prefix='body_feline', tags=['body'], var=0.7, temperature=0.2)
    body_fac.params['thetas'][-3] *= N(1, 0.1)
    body = genome.part(body_fac)

    tail = genome.part(parts.tail.Tail())
    genome.attach(tail, body, coord=(0.07, 1, 1), joint=Joint(rest=(N(0, 10), 180, 0)))
 
    if U() < 0.5:

        head_length_rad1_rad2 = np.array((0.36, 0.20, 0.18)) * N(1, 0.1, 3)
        head_fac = parts.head.CarnivoreHead({'length_rad1_rad2': head_length_rad1_rad2})
        head = genome.part(head_fac)

        jaw_pct = np.array((1.05, 0.55, 0.5))
        jaw = genome.part(parts.head.CarnivoreJaw({'length_rad1_rad2': head_length_rad1_rad2 * jaw_pct}))
        genome.attach(jaw, head, coord=(0.2 * N(1, 0.1), 0, 0.35 * N(1, 0.1)), joint=Joint(rest=(0, U(10, 35), 0), pose=(0,0,0)))

    else:
        head_fac = parts.generic_nurbs.NurbsHead(prefix='head_carnivore', tags=['head'], var=0.5)
        head = genome.part(head_fac)

        headl = head_fac.params['length'][0]
        head_length_rad1_rad2 = np.array((headl, 0.20, 0.18)) * N(1, 0.1, 3)
        
        jaw_pct = np.array((0.7, 0.55, 0.5))
        jaw = genome.part(parts.head.CarnivoreJaw({'length_rad1_rad2': head_length_rad1_rad2 * jaw_pct}))   
        genome.attach(jaw, head, coord=(0.12, 0, 0.3 * N(1, 0.1)), joint=Joint(rest=(0, U(10, 35), 0), pose=(0,0,0)))

        eye_fac = parts.eye.MammalEye({'Radius': N(0.027, 0.009)})
        eye_t, splay = U(0.61, 0.64), U(90, 140)/180
        r = U(0.8, 0.9)
        rot = np.array([0, 0, 0])
        for side in [-1, 1]:
            eye = genome.part(eye_fac)
            genome.attach(eye, head, coord=(eye_t, splay, r), joint=Joint(rest=rot), rotation_basis='normal', side=side)

    nose = genome.part(parts.head_detail.CatNose())
    genome.attach(nose, head, coord=(U(0.9, 0.96), 1, U(0.5, 0.7)), joint=Joint(rest=(0, 20, 0)))

    ear_fac = parts.head_detail.CatEar()
    t, splay = N(0.33, 0.07), U(100, 150)/180
    rot = np.array([-20, -10, -23]) + N(0, 4, 3)
    for side in [-1, 1]:
        ear = genome.part(ear_fac)
        genome.attach(ear, head, coord=(t, splay, 1), joint=Joint(rest=rot), rotation_basis='normal', side=side)

    neck_t = 0.7
    shoulder_bounds = np.array([[-20, -20, -20], [20, 20, 20]])
    splay = clip_gaussian(130, 7, 90, 130)/180
    shoulder_t = clip_gaussian(0.12, 0.05, 0.08, 0.12)
    params = {'length_rad1_rad2': np.array((1.6, 0.1, 0.05)) * N(1, (0.15, 0.05, 0.05), 3)}

    foot_fac = parts.foot.Foot()
    backleg_fac = parts.leg.QuadrupedBackLeg(params=params)
    for side in [-1, 1]:
        back_leg = genome.attach(genome.part(foot_fac), genome.part(backleg_fac), coord=(0.9, 0, 0), joint=Joint(rest=(0, 0, 0)))
        genome.attach(back_leg, body, coord=(shoulder_t, splay, 1.2), 
            joint=Joint(rest=(0, 90, 0), bounds=shoulder_bounds), 
            rotation_basis='global', side=side)#, smooth_rad=0.06)#, bridge_rad=0.1)

    frontleg_fac = parts.leg.QuadrupedFrontLeg(params=params)
    for side in [-1, 1]:
        front_leg = genome.attach(genome.part(foot_fac), genome.part(frontleg_fac), coord=(0.9, 0, 0), joint=Joint(rest=(0, 0, 0)))
        genome.attach(front_leg, body, coord=(neck_t - shoulder_t, splay, 0.8), 
            joint=Joint(rest=(0, 90, 0)), rotation_basis='global', side=side)#, smooth_rad=0.06)#, bridge_rad=0.1)

    #neck_lrr = np.array((body_lrr[0], body_lrr[-1], body_lrr[-1])) * np.array((0.45, 0.5, 0.25)) * N(1, 0.05, 3)
    #neck = genome.part(parts.head.Neck({'length_rad1_rad2': neck_lrr}))
    genome.attach(head, body, coord=(N(0.97, 0.01), 0, 0), 
        joint=Joint(rest=(0, N(20, 5), 0)), 
        rotation_basis='global')#, bridge_rad=0.1)
    #genome.attach(neck, body, coord=(0.8, 0, 0.1), joint=Joint(rest=(0, -N(15, 2), 0)))

    return genome.CreatureGenome(
        parts=body,
        postprocess_params=dict(
            hair=tiger_hair_params(),
            skin=tiger_skin_sim_params(),
            surface_registry=[
                (infinigen.assets.materials.tiger_attr, 3),
                (infinigen.assets.materials.giraffe_attr, 0.2),
                (infinigen.assets.materials.spot_sparse_attr, 2)
            ]
        ) 
    )

@gin.configurable
class CarnivoreFactory(AssetFactory):

    def __init__(self, factory_seed=None, bvh=None, coarse=False, animation_mode=None, **kwargs):
        super().__init__(factory_seed, coarse)
        self.bvh = bvh
        self.animation_mode = animation_mode

    def create_placeholder(self, **kwargs):
        return butil.spawn_cube(size=4)

    def create_asset(self, i, placeholder, hair=True, simulate=False, **kwargs):
        
        genome = tiger_genome()
        root, parts = creature.genome_to_creature(genome, name=f'carnivore({self.factory_seed}, {i})')
        tag_object(root, 'carnivore')
        offset_center(root)
        
        dynamic = self.animation_mode is not None

        joined, extras, arma, ik_targets = joining.join_and_rig_parts(
            root, parts, genome, rigging=dynamic,
            postprocess_func=tiger_postprocessing, **kwargs)
        
        butil.parent_to(root, placeholder, no_inverse=True)

        if hair:
            creature_hair.configure_hair(
                joined, root, genome.postprocess_params['hair'])
        if dynamic:
            if self.animation_mode == 'run':
                run_cycle.animate_run(root, arma, ik_targets)
            elif self.animation_mode == 'idle':
                idle.snap_iks_to_floor(ik_targets, self.bvh)
                idle.idle_body_noise_drivers(ik_targets)
            elif self.animation_mode == 'tpose':
                pass
            else:
                raise ValueError(f'Unrecognized mode {self.animation_mode=}')
        if simulate:
            rigidity = surface.write_vertex_group(
                joined, cloth_sim.local_pos_rigity_mask, apply=True)
            cloth_sim.bake_cloth(joined, genome.postprocess_params['skin'], 
                attributes=dict(vertex_group_mass=rigidity))

        return root