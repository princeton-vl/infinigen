# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import colorsys

import bpy
import numpy as np
from numpy.random import uniform

import infinigen.core.util.blender as butil
from .base import BaseCactusFactory
from .globular import GlobularBaseCactusFactory
from .columnar import ColumnarBaseCactusFactory
from .pricky_pear import PrickyPearBaseCactusFactory
from .kalidium import KalidiumBaseCactusFactory
from infinigen.assets.cactus import spike
from infinigen.assets.utils.misc import build_color_ramp, log_uniform
from infinigen.assets.utils.decorate import assign_material, join_objects
from infinigen.core.nodes.node_wrangler import NodeWrangler, Nodes
from infinigen.core.placement.detail import remesh_with_attrs
from infinigen.core import surface
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util.math import FixedSeed
from infinigen.assets.utils.tag import tag_object

class CactusFactory(AssetFactory):
    
    def __init__(self, factory_seed, coarse=False, factory_method=None):
        super(CactusFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.factory_methods = [GlobularBaseCactusFactory, ColumnarBaseCactusFactory,
                PrickyPearBaseCactusFactory, KalidiumBaseCactusFactory]
            weights = np.array([1] * len(self.factory_methods))
            self.weights = weights / weights.sum()
            if factory_method is None:
                with FixedSeed(self.factory_seed):
                    factory_method = np.random.choice(self.factory_methods, p=self.weights)
            self.factory: BaseCactusFactory = factory_method(factory_seed, coarse)
            base_hue = uniform(.2, .4)
            self.material = surface.shaderfunc_to_material(self.shader_cactus, base_hue)

    def create_asset(self, face_size=0.01, realize=True, **params):
        obj = self.factory.create_asset(**params)
        remesh_with_attrs(obj, face_size)

        if self.factory.noise_strength > 0:
            t = np.random.choice(['STUCCI', 'MARBLE'])
            texture = bpy.data.textures.new(name='coral', type=t)
            texture.noise_scale = log_uniform(.1, .15)
            butil.modify_mesh(obj, 'DISPLACE', True, strength=self.factory.noise_strength, mid_level=0,
                              texture=texture)
        assign_material(obj, self.material)
        if face_size <= .05 and self.factory.density > 0:
            t = spike.apply(obj, self.factory.points_fn, self.factory.base_radius, realize)
            obj = join_objects([obj, t])

        tag_object(obj, 'cactus')
        return obj

    @staticmethod
    def shader_cactus(nw: NodeWrangler, base_hue):
        shift = uniform(-.15, .15)
        bright_color = *colorsys.hsv_to_rgb((base_hue + shift) % 1, 1., .02), 1
        dark_color = *colorsys.hsv_to_rgb(base_hue, .8, .01), 1
        fresnel_color = *colorsys.hsv_to_rgb((base_hue - uniform(.05, .1)) % 1, .9, uniform(.3, .5)), 1
        specular = .25

        fresnel = nw.scalar_multiply(nw.new_node(Nodes.Fresnel), log_uniform(.6, 1.))
        color = build_color_ramp(nw, nw.musgrave(log_uniform(10, 50)), [.0, .3, .7, 1.],
                                 [dark_color, dark_color, bright_color, bright_color])
        color = nw.new_node(Nodes.MixRGB, [fresnel, color, fresnel_color])
        noise_texture = nw.new_node(Nodes.NoiseTexture, input_kwargs={'Scale': 50})
        roughness = nw.build_float_curve(noise_texture, [(0, .5), (1, .8)])
        bsdf = nw.new_node(Nodes.PrincipledBSDF,
                           input_kwargs={'Base Color': color, 'Roughness': roughness, 'Specular': specular})
        return bsdf


class GlobularCactusFactory(CactusFactory):
    def __init__(self, factory_seed, coarse=False):
        super(GlobularCactusFactory, self).__init__(factory_seed, coarse, GlobularBaseCactusFactory)


class ColumnarCactusFactory(CactusFactory):
    def __init__(self, factory_seed, coarse=False):
        super(ColumnarCactusFactory, self).__init__(factory_seed, coarse, ColumnarBaseCactusFactory)


class PrickyPearCactusFactory(CactusFactory):
    def __init__(self, factory_seed, coarse=False):
        super(PrickyPearCactusFactory, self).__init__(factory_seed, coarse, PrickyPearBaseCactusFactory)


class KalidiumCactusFactory(CactusFactory):

    def __init__(self, factory_seed, coarse=False):
        super(KalidiumCactusFactory, self).__init__(factory_seed, coarse, KalidiumBaseCactusFactory)
