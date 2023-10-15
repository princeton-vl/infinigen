# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import colorsys
from collections import defaultdict

import bpy
import gin
import numpy as np
from numpy.random import uniform

from infinigen.assets.creatures.util.creature import genome_to_creature
from infinigen.assets.creatures.util.joining import join_and_rig_parts
from infinigen.assets.creatures.util.genome import CreatureGenome, Joint, attach, part
from infinigen.assets.creatures.parts.crustacean.antenna import LobsterAntennaFactory, SpinyLobsterAntennaFactory
from infinigen.assets.creatures.parts.crustacean.claw import CrabClawFactory, LobsterClawFactory
from infinigen.assets.creatures.parts.crustacean.eye import CrustaceanEyeFactory
from infinigen.assets.creatures.parts.crustacean.fin import CrustaceanFinFactory
from infinigen.assets.creatures.parts.crustacean.leg import CrabLegFactory, LobsterLegFactory
from infinigen.assets.creatures.parts.crustacean.body import CrabBodyFactory, LobsterBodyFactory
from infinigen.assets.creatures.parts.crustacean.tail import CrustaceanTailFactory
from infinigen.assets.utils.decorate import assign_material, read_material_index, write_material_index
from infinigen.assets.utils.misc import build_color_ramp, log_uniform
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.surface import read_attr_data, shaderfunc_to_material
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed

n_legs = 4
n_limbs = 5
n_side_fin = 2


def crustacean_genome(sp):
    body_fac = sp['body_fn']()
    obj = part(body_fac)
    # Add legs
    leg_x_length = sp['leg_x_length'](body_fac.params)
    leg_x_lengths = np.sort(uniform(.6, 1, 4))[:: -1] * leg_x_length
    leg_angle = sp['leg_angle']
    x_legs = sp['x_legs']
    leg_joints_x, leg_joints_y, leg_joints_z = sp['leg_joint']

    shared_leg_params = ['bottom_flat', 'bottom_cutoff']
    leg_fn = sp['leg_fn']
    leg_params = {k: v for k, v in leg_fn().params.items() if k in shared_leg_params}
    leg_fac = [leg_fn({**leg_params, 'x_length': leg_x_lengths[i]}) for i in range(n_legs)]
    for i in range(n_legs):
        for side in [1, -1]:
            attach(part(leg_fac[i]), obj, (x_legs[i + 1], leg_angle, .99),
                   Joint((leg_joints_x[i], leg_joints_y[i], leg_joints_z[i])), side=side)
    # Add claws
    claw_angle = sp['claw_angle']
    claw_fn = sp['claw_fn']
    claw_fac = claw_fn({'x_length': sp['claw_x_length'](body_fac.params)})

    for side in [1, -1]:
        attach(part(claw_fac), obj, (x_legs[0] + sp['x_claw_offset'], claw_angle, .99), Joint(sp['claw_joint']),
               side=side)
    # Add tails
    tail_fac = sp['tail_fn']
    if tail_fac is not None:
        shared_params = ['bottom_shift', 'bottom_cutoff', 'top_shift', 'top_cutoff', 'y_length', 'z_length']
        tail_fac = tail_fac({**{k: v for k, v in body_fac.params.items() if k in shared_params},
                                'x_length': sp['tail_x_length'](body_fac.params),
                            })
        tail = part(tail_fac)
        attach(tail, obj, (0, 0, 0), Joint((0, 0, 180)))
        fin_fn = sp['fin_fn']
        if fin_fn is not None:
            fin_fn = sp['fin_fn']
            x_fins = sp['x_fins']
            fin_joints_x, fin_joints_y, fin_joints_z = sp['fin_joints']
            fin_x_length = sp['fin_x_length'](body_fac.params)
            fin_x_lengths = np.sort(uniform(.6, 1, 4))[:: -1] * fin_x_length
            fin_fac = [fin_fn({'x_length': fin_x_lengths[i]}) for i in range(n_side_fin + 1)]

            for i in range(n_side_fin):
                for side in [1, -1]:
                    attach(part(fin_fac[i]), tail, (x_fins[i], .5, .99),
                           Joint((fin_joints_x[i], fin_joints_y[i], fin_joints_z[i])), side=side)
            attach(part(fin_fac[-1]), tail, (.99, .5, .9), Joint((0, 0, 0)))

    # Add eyes
    x_eye = sp['x_eye']
    eye_angle = sp['eye_angle']
    eye_joint_x, eye_joint_y, eye_joint_z = sp['eye_joint']
    eye_fac = CrustaceanEyeFactory()
    for side in [1, -1]:
        attach(part(eye_fac), obj, (x_eye, eye_angle, .99), Joint((eye_joint_x, eye_joint_y, eye_joint_z)),
               side=side)
    # Add antenna
    antenna_fn = sp['antenna_fn']
    if antenna_fn is not None:
        x_antenna = sp['x_antenna']
        antenna_angle = sp['antenna_angle']
        antenna_fac = antenna_fn({'x_length': sp['antenna_x_length'](body_fac.params)})
        for side in [1, -1]:
            attach(part(antenna_fac), obj, (x_antenna, antenna_angle, .99), Joint(sp['antenna_joint']),
                   side=side)

    anim_params = {k: v for k, v in sp.items() if 'curl' in k or 'rot' in k}
    anim_params['freq'] = sp['freq']
    postprocess_params = dict(material={'base_hue': sp['base_hue']}, anim=anim_params)
    return CreatureGenome(obj, postprocess_params)


def build_base_hue():
    if uniform(0, 1) < .6:
        return uniform(0, .05)
    else:
        return uniform(.4, .45)


def shader_crustacean(nw: NodeWrangler, params):
    value_shift = log_uniform(2, 10)
    base_hue = params['base_hue']
    bright_color = *colorsys.hsv_to_rgb(base_hue, uniform(.8, 1.), log_uniform(.02, .05) * value_shift), 1
    dark_color = *colorsys.hsv_to_rgb((base_hue + uniform(-.05, .05)) % 1, uniform(.8, 1.),
                                      log_uniform(.01, .02) * value_shift), 1
    light_color = *colorsys.hsv_to_rgb(base_hue, uniform(.0, .4), log_uniform(.2, 1.)), 1
    specular = uniform(.6, .8)
    specular_tint = uniform(0, 1)
    clearcoat = uniform(.2, .8)
    roughness = uniform(.1, .3)
    metallic = uniform(.6, .8)
    x, y, z = nw.separate(nw.new_node(Nodes.NewGeometry).outputs['Position'])
    color = build_color_ramp(nw, nw.new_node(Nodes.MapRange, [
        nw.new_node(Nodes.MusgraveTexture, [nw.combine(x, nw.math('ABSOLUTE', y), z)],
                    input_kwargs={'Scale': log_uniform(5, 8)}), -1, 1, 0, 1]), [.0, .3, .7, 1.],
                             [bright_color, bright_color, dark_color, dark_color], )
    ratio = nw.new_node(Nodes.Attribute, attrs={'attribute_name': 'ratio'}).outputs['Fac']
    color = nw.new_node(Nodes.MixRGB, [ratio, light_color, color])
    bsdf = nw.new_node(Nodes.PrincipledBSDF, input_kwargs={
        'Base Color': color,
        'Metallic': metallic,
        'Roughness': roughness,
        'Specular': specular,
        'Specular Tint': specular_tint,
        'Clearcoat': clearcoat
    })
    return bsdf


def shader_eye(nw: NodeWrangler):
    return nw.new_node(Nodes.PrincipledBSDF, input_kwargs={'Base Color': (0.1, 0.1, 0.1, 1), 'Specular': 0})


def crustacean_postprocessing(body_parts, extras, params):
    tag_list = ['body', 'claw', 'leg']
    materials = [shaderfunc_to_material(shader_crustacean, params['material']) for _, t in enumerate(tag_list)]
    tag_list.append('eye')
    materials.append(shaderfunc_to_material(shader_eye))
    assign_material(body_parts + extras, materials)

    for part in body_parts:
        material_indices = read_material_index(part)
        for i, tag_name in enumerate(tag_list):
            if f'tag_{tag_name}' in part.data.attributes.keys():
                part.data.attributes.active = part.data.attributes[f'tag_{tag_name}']
                with butil.SelectObjects(part):
                    bpy.ops.geometry.attribute_convert(domain='FACE')
                has_tag = read_attr_data(part, f'tag_{tag_name}', 'FACE')
                material_indices[np.nonzero(has_tag)[0]] = i
        write_material_index(part, material_indices)
    for extra in extras:
        material_indices = read_material_index(extra)
        material_indices.fill(tag_list.index('claw'))
        write_material_index(extra, material_indices)


def animate_crustacean_move(arma, params):
    groups = defaultdict(list)
    for bone in arma.pose.bones.values():
        groups[(bone.bone['factory_class'], bone.bone['index'])].append(bone)
    for (factory_name, part_id), bones in groups.items():
        eval(factory_name).animate_bones(arma, bones, params)


@gin.configurable
class CrustaceanFactory(AssetFactory):
    max_expected_radius = 1
    max_distance = 40

    def __init__(self, factory_seed, coarse=False, **_):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.species_params = {
                'lobster': self.lobster_params,
                'crab': self.crab_params,
                'spiny_lobster': self.spiny_lobster_params
            }
            self.species = np.random.choice(list(self.species_params.keys()))

    def create_asset(self, i, animate=True, rigging=True, cloth=False, **kwargs):
        genome = crustacean_genome(self.species_params[self.species]())
        root, parts = genome_to_creature(genome, name=f'crustacean({self.factory_seed}, {i})')
        for p in parts:
            if p.obj.name.split("=")[-1] == "CrustaceanEyeFactor":
                assign_material(p.obj, shaderfunc_to_material(shader_eye))
        joined, extras, arma, ik_targets = join_and_rig_parts(root, parts, genome,
                                                              postprocess_func=crustacean_postprocessing,
                                                              rigging=rigging, min_remesh_size=.005,
                                                              face_size=kwargs['face_size'],
                                                              roll='GLOBAL_POS_Z')
        if animate and arma is not None:
            animate_crustacean_move(arma, genome.postprocess_params['anim'])
        else:
            butil.join_objects([joined] + extras)
        return root

    def crab_params(self):
        base_leg_curl = uniform(-np.pi * .15, np.pi * .15)
        return {
            'body_fn': CrabBodyFactory,
            'leg_fn': CrabLegFactory,
            'claw_fn': CrabClawFactory,
            'tail_fn': None,
            'antenna_fn': None,
            'fin_fn': None,
            'leg_x_length': lambda p: p['y_length'] * log_uniform(2., 3.),
            'claw_x_length': lambda p: p['y_length'] * log_uniform(1.5, 1.8),
            'tail_x_length': lambda p: 0,
            'antenna_x_length': lambda p: 0,
            'fin_x_length': lambda p: 0,
            'x_legs': (np.linspace(uniform(.08, .1), uniform(.55, .6), n_limbs) + np.arange(n_limbs) * .02)[
            ::-1],
            'leg_angle': uniform(.42, .44),
            'leg_joint': (
            np.sort(uniform(-5, 5, n_legs))[::1 if uniform(0, 1) > .5 else -1], np.sort(uniform(0, 10, n_legs)),
            np.sort(uniform(65, 105, n_legs) + uniform(-8, 8)) + np.arange(n_legs) * 2),
            'x_claw_offset': uniform(.08, .1),
            'claw_angle': uniform(.44, .46),
            'claw_joint': (uniform(-50, -40), uniform(-20, 20), uniform(10, 20)),
            'x_eye': uniform(.92, .96),
            'eye_angle': uniform(.8, .85),
            'eye_joint': (0, uniform(-60, -0), uniform(10, 70)),
            'x_antenna': 0,
            'antenna_angle': 0,
            'antenna_joint': (0, 0, 0),
            'x_fins': 0,
            'fin_joints': ([0] * n_side_fin, [0] * n_side_fin, [0] * n_side_fin),
            'leg_rot': (uniform(np.pi * .8, np.pi * 1.1), 0, 0),
            'leg_curl': (
                (-np.pi * 1.1, -np.pi * .7), 0, (base_leg_curl - np.pi * .02, base_leg_curl + np.pi * .02)),
            'claw_curl': ((-np.pi * .2, np.pi * .1), 0, (-np.pi * .1, np.pi * .1)),
            'claw_lower_curl': ((-np.pi * .1, np.pi * .1), 0, 0),
            'tail_curl': (0, 0, 0),
            'antenna_curl': (0, 0, 0),
            'base_hue': build_base_hue(),
            'freq': 1 / log_uniform(100, 200),
        }

    def lobster_params(self):
        base_leg_curl = uniform(-np.pi * .4, np.pi * .4)
        return {
            'body_fn': LobsterBodyFactory,
            'leg_fn': LobsterLegFactory,
            'claw_fn': LobsterClawFactory,
            'tail_fn': CrustaceanTailFactory,
            'antenna_fn': LobsterAntennaFactory,
            'fin_fn': CrustaceanFinFactory,
            'leg_x_length': lambda p: p['x_length'] * log_uniform(.6, .8),
            'claw_x_length': lambda p: p['x_length'] * log_uniform(1.2, 1.5),
            'tail_x_length': lambda p: p['x_length'] * log_uniform(1.2, 1.8),
            'antenna_x_length': lambda p: p['x_length'] * log_uniform(1.6, 3.),
            'fin_x_length': lambda p: p['y_length'] * log_uniform(1.2, 2.5),
            'x_legs': (np.linspace(.05, uniform(.2, .25), n_limbs) + np.arange(n_limbs) * .02)[::-1],
            'leg_angle': uniform(.3, .35),
            'leg_joint': (
            uniform(-5, 5, n_legs), uniform(0, 10, n_legs), np.sort(uniform(95, 110, n_legs) + uniform(-8, 8))),
            'x_claw_offset': uniform(.08, .1),
            'claw_angle': uniform(.4, .5),
            'claw_joint': (uniform(-80, -70), uniform(-10, 10), uniform(10, 20)),
            'x_eye': uniform(.8, .88),
            'eye_angle': uniform(.8, .85),
            'eye_joint': (0, uniform(-60, -0), uniform(10, 70)),
            'x_antenna': uniform(.76, .8),
            'antenna_angle': uniform(.6, .7),
            'antenna_joint': (uniform(70, 110), uniform(-40, -30), uniform(20, 40)),
            'x_fins': np.sort(uniform(.85, .95, n_side_fin)),
            'fin_joints': (
                np.sort(uniform(0, 30, n_side_fin))[::1 if uniform(0, 1) < .5 else -1], [0] * n_side_fin,
                np.sort(uniform(10, 30, n_side_fin))),
            'leg_rot': (uniform(np.pi * .8, np.pi * 1.1), 0, 0),
            'leg_curl': (
                (-np.pi * 1.1, -np.pi * .7), 0, (base_leg_curl - np.pi * .02, base_leg_curl + np.pi * .02)),
            'claw_curl': ((-np.pi * .1, np.pi * .2), 0, 0),
            'claw_lower_curl': ((-np.pi * .1, np.pi * .1), 0, 0),
            'tail_curl': ((-np.pi * .6, 0), 0, 0),
            'antenna_curl': ((np.pi * .1, np.pi * .3), 0, (0, np.pi * .8)),
            'base_hue': build_base_hue(),
            'freq': 1 / log_uniform(400, 500),
        }

    def spiny_lobster_params(self):
        lobster_params = self.lobster_params()
        leg_joint_x, leg_joint_y, leg_joint_z = lobster_params['leg_joint']
        leg_joint_z_min = np.min(leg_joint_z) + uniform(-10, -5)
        return {**lobster_params,
            'antenna_fn': SpinyLobsterAntennaFactory,
            'claw_fn': LobsterLegFactory,
            'claw_x_length': lobster_params['leg_x_length'],
            'claw_angle': lobster_params['leg_angle'],
            'claw_joint': (uniform(10, 40), uniform(0, 10), leg_joint_z_min),
            'x_antenna': uniform(.7, .75),
            'antenna_angle': uniform(.4, .5),
        }


@gin.configurable
class CrabFactory(CrustaceanFactory):
    def __init__(self, factory_seed, coarse=False, **_):
        super().__init__(factory_seed, coarse)
        self.species = 'crab'


@gin.configurable
class LobsterFactory(CrustaceanFactory):
    def __init__(self, factory_seed, coarse=False, **_):
        super().__init__(factory_seed, coarse)
        self.species = 'lobster'


@gin.configurable
class SpinyLobsterFactory(CrustaceanFactory):
    def __init__(self, factory_seed, coarse=False, **_):
        super().__init__(factory_seed, coarse)
        self.species = 'spiny_lobster'
