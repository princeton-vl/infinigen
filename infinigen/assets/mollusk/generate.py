# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import colorsys

import bpy
import numpy as np
from numpy.random import uniform

import infinigen.core.util.blender as butil
from .base import BaseMolluskFactory
from .shell import ShellBaseFactory, ScallopBaseFactory, ClamBaseFactory, MusselBaseFactory
from .snail import SnailBaseFactory, ConchBaseFactory, AugerBaseFactory, VoluteBaseFactory, NautilusBaseFactory
from infinigen.assets.utils.misc import build_color_ramp, log_uniform
from ..utils.decorate import assign_material, subsurface2face_size
from infinigen.core.nodes.node_wrangler import NodeWrangler, Nodes
from infinigen.core import surface
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util.math import FixedSeed
from infinigen.assets.utils.tag import tag_object, tag_nodegroup


class MolluskFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False, factory_method=None):
        super(MolluskFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.factory_methods = [SnailBaseFactory, ShellBaseFactory]
            weights = np.array([1] * len(self.factory_methods))
            self.weights = weights / weights.sum()
            if factory_method is None:
                factory_method = np.random.choice(self.factory_methods, p=self.weights)
            self.factory: BaseMolluskFactory = factory_method(factory_seed, coarse)

            base_hue = self.build_base_hue()
            self.material = surface.shaderfunc_to_material(self.shader_mollusk, base_hue, self.factory.ratio,
                                                           self.factory.x_scale, self.factory.z_scale,
                                                           self.factory.distortion)

    def create_asset(self, face_size=0.01, **params):
        obj = self.factory.create_asset(**params)
        self.decorate_mollusk(obj, face_size)
        return obj

    def decorate_mollusk(self, obj, face_size):
        subsurface2face_size(obj, face_size)
        butil.modify_mesh(obj, 'SOLIDIFY', True, thickness=.005)
        t = np.random.choice(['STUCCI', 'MARBLE'])
        texture = bpy.data.textures.new(name='mollusk', type=t)
        texture.noise_scale = log_uniform(.1, .2)
        butil.modify_mesh(obj, 'DISPLACE', strength=self.factory.noise_strength, mid_level=0, texture=texture)
        assign_material(obj, self.material)
        tag_object(obj, 'mollusk')
        return obj

    @staticmethod
    def build_base_hue():
        if uniform(0, 1) < .4:
            return uniform(0, .2)
        else:
            return uniform(.05, .12)

    @staticmethod
    def shader_mollusk(nw: NodeWrangler, base_hue, ratio=0, x_scale=2, z_scale=1, distortion=5):
        roughness = uniform(.2, .8)
        specular = .3
        value_scale = log_uniform(1, 20)
        saturation_scale = log_uniform(.4, 1)

        def dark_color():
            return *colorsys.hsv_to_rgb(base_hue + uniform(-.06, .06), uniform(.6, 1.) * saturation_scale,
                                        .005 * value_scale ** 1.5), 1

        def light_color():
            return *colorsys.hsv_to_rgb(base_hue + uniform(-.06, .06), uniform(.6, 1.) * saturation_scale,
                                        .05 * value_scale), 1

        def color_fn(dark_prob):
            return dark_color() if uniform(0, 1) < dark_prob else light_color()

        vector = nw.new_node(Nodes.Attribute, attrs={'attribute_name': 'vector'}).outputs['Vector']
        n = np.random.randint(3, 5)
        texture_0 = nw.new_node(Nodes.WaveTexture,
                                input_kwargs={'Vector': vector, 'Distortion': distortion, 'Scale': x_scale},
                                attrs={'wave_profile': 'SAW', 'bands_direction': 'X'})
        cr_0 = build_color_ramp(nw, texture_0, np.sort(uniform(0, 1, n)), [color_fn(.4) for _ in range(n)])
        texture_1 = nw.new_node(Nodes.WaveTexture,
                                input_kwargs={'Vector': vector, 'Distortion': distortion, 'Scale': z_scale},
                                attrs={'wave_profile': 'SAW', 'bands_direction': 'Z'})
        cr_1 = build_color_ramp(nw, texture_1, np.sort(uniform(0, 1, n)), [color_fn(.4) for _ in range(n)])
        principled_bsdf = nw.new_node(Nodes.PrincipledBSDF, input_kwargs={
            'Base Color': nw.new_node(Nodes.MixRGB, [ratio, cr_0, cr_1]),
            'Specular': specular,
            'Roughness': roughness
        })
        return principled_bsdf


class ScallopFactory(MolluskFactory):
    def __init__(self, factory_seed, coarse=False):
        super(ScallopFactory, self).__init__(factory_seed, coarse, ScallopBaseFactory)


class ClamFactory(MolluskFactory):
    def __init__(self, factory_seed, coarse=False):
        super(ClamFactory, self).__init__(factory_seed, coarse, ClamBaseFactory)


class MusselFactory(MolluskFactory):
    def __init__(self, factory_seed, coarse=False):
        super(MusselFactory, self).__init__(factory_seed, coarse, MusselBaseFactory)


class ConchFactory(MolluskFactory):
    def __init__(self, factory_seed, coarse=False):
        super(ConchFactory, self).__init__(factory_seed, coarse, ConchBaseFactory)


class AugerFactory(MolluskFactory):
    def __init__(self, factory_seed, coarse=False):
        super(AugerFactory, self).__init__(factory_seed, coarse, AugerBaseFactory)


class VoluteFactory(MolluskFactory):
    def __init__(self, factory_seed, coarse=False):
        super(VoluteFactory, self).__init__(factory_seed, coarse, VoluteBaseFactory)


class NautilusFactory(MolluskFactory):
    def __init__(self, factory_seed, coarse=False):
        super(NautilusFactory, self).__init__(factory_seed, coarse, NautilusBaseFactory)
