# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this
# source tree.

# Authors: Lingjie Mei
# Date Signed: April 13 2023 

import colorsys

import bpy
import numpy as np
from numpy.random import uniform

from assets.creatures.animation.driver_repeated import repeated_driver
from assets.utils.decorate import assign_material, read_co, subsurface2face_size, write_co
from assets.utils.draw import make_circular_interp
import util.blender as butil
from placement.factory import AssetFactory
from assets.utils.diff_growth import build_diff_growth
from assets.utils.object import mesh2obj, data2mesh
from assets.utils.mesh import polygon_angles
from nodes.node_wrangler import NodeWrangler, Nodes
from surfaces import surface
from assets.utils.misc import build_color_ramp, log_uniform
from util.math import FixedSeed
from assets.utils.tag import tag_object, tag_nodegroup


class SeaweedFactory(AssetFactory):

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.base_hue = uniform(.0, .1) if uniform(0, 1) < .5 else uniform(.3, .4)
            self.material = surface.shaderfunc_to_material(self.shader_seaweed, self.base_hue)
            self.freq = 1 / log_uniform(200, 500)

    def create_asset(self, face_size=0.01, **params):
        growth_vec = 0, 0, uniform(4., 6.)
        inhibit_shell = uniform(.6, .8)
        obj = SeaweedFactory.differential_growth_make(fac_noise=2., inhibit_shell=inhibit_shell,
                                                      repulsion_radius=1., growth_vec=growth_vec, dt=.25,
                                                      max_polygons=2e3)

        obj.scale = [2 / max(obj.dimensions)] * 3
        obj.scale[-1] *= uniform(1.5, 2)
        obj.location[-1] -= .02
        butil.apply_transform(obj, loc=True)
        f_scale = make_circular_interp(1, 6, 2, log_uniform)
        x, y, z = read_co(obj).T
        scale = f_scale(np.arctan2(y, x) + np.pi)
        co = np.stack([scale * x, scale * y, z], -1)
        write_co(obj, co)
        subsurface2face_size(obj, face_size / 2)
        butil.modify_mesh(obj, 'TRIANGULATE')
        texture = bpy.data.textures.new(name='seaweed', type='STUCCI')
        texture.noise_scale = .1
        butil.modify_mesh(obj, 'DISPLACE', True, strength=.02, texture=texture)
        assign_material(obj, self.material)
        self.animate_bend(obj)
        tag_object(obj, 'seaweed')
        return obj

    def animate_bend(self, obj):
        obj, mod = butil.modify_mesh(obj, 'SIMPLE_DEFORM', False, deform_method='BEND', deform_axis='Y',
                                     return_mod=True)
        driver = mod.driver_add('angle').driver
        start_angle = uniform(-np.pi / 6, 0)
        driver.expression = repeated_driver(start_angle, start_angle + uniform(np.pi / 4, np.pi / 2), self.freq)

    @staticmethod
    def differential_growth_make(**kwargs):
        n_base = 24
        angles = polygon_angles(n_base)
        vertices = np.block([[np.cos(angles), 0], [np.sin(angles), 0], [np.zeros(n_base + 1)]]).T
        faces = np.stack([np.arange(n_base), np.roll(np.arange(n_base), 1), np.full(n_base, n_base)]).T
        obj = mesh2obj(data2mesh(vertices, [], faces, 'diff_growth'))

        boundary = obj.vertex_groups.new(name='Boundary')
        boundary.add(list(range(n_base)), 1.0, 'REPLACE')
        build_diff_growth(obj, boundary.index, **kwargs)
        return obj

    @staticmethod
    def geo_seaweed_waves(nw: NodeWrangler):
        translation_scale = uniform(0., .25)
        expand_scale = uniform(.2, .3)
        geometry = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
        x, y, z = nw.separate(nw.new_node(Nodes.InputPosition))
        angle = np.random.uniform(0, 2 * np.pi)
        displacement = nw.scale(nw.add(nw.scale(nw.combine(np.cos(angle), np.sin(angle), 0),
                                                nw.scalar_multiply(nw.musgrave(10), translation_scale)),
                                       nw.scale(nw.combine(x, y, 0), expand_scale)), z)
        geometry = nw.new_node(Nodes.SetPosition, input_kwargs={'Geometry': geometry, 'Offset': displacement})
        nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geometry})

    @staticmethod
    def shader_seaweed(nw: NodeWrangler, base_hue=.3):
        h_perturb = uniform(-.1, .1)
        s_perturb = uniform(-.1, -.0)
        v_perturb = log_uniform(1., 2)

        def map_perturb(h, s, v):
            return *colorsys.hsv_to_rgb(h + h_perturb, s + s_perturb, v / v_perturb), 1.

        subsurface_ratio = .01
        roughness = .8
        mix_ratio = uniform(.2, .4)
        specular = .2

        color_1 = map_perturb(base_hue, uniform(.6, .8), .25)
        color_2 = map_perturb(base_hue - uniform(.05, .1), uniform(.6, .8), .15)
        cr = build_color_ramp(nw, nw.musgrave(uniform(5, 10)), [0, .3, .7, 1.],
                              [color_1, color_1, color_2, color_2])

        principled_bsdf = nw.new_node(Nodes.PrincipledBSDF, input_kwargs={
            'Base Color': cr,
            'Subsurface': subsurface_ratio,
            'Subsurface Radius': (.01, .01, .01),
            'Subsurface Color': map_perturb(base_hue, .6, .2),
            'Roughness': roughness,
            'Specular': specular
        })

        translucent_bsdf = nw.new_node(Nodes.TransparentBSDF, input_kwargs={'Color': cr})

        mix_shader = nw.new_node(Nodes.MixShader, [mix_ratio, principled_bsdf, translucent_bsdf])
        return mix_shader
