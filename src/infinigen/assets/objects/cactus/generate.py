# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import bpy
import numpy as np
from numpy.random import uniform

import infinigen.core.util.blender as butil
from infinigen.assets.objects.cactus import spike
from infinigen.assets.utils.misc import assign_material
from infinigen.assets.utils.object import join_objects
from infinigen.core import surface, tagging
from infinigen.core.nodes.node_utils import build_color_ramp
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.detail import remesh_with_attrs
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util.color import hsv2rgba
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform

from .base import BaseCactusFactory
from .columnar import ColumnarBaseCactusFactory
from .globular import GlobularBaseCactusFactory
from .kalidium import KalidiumBaseCactusFactory
from .pricky_pear import PrickyPearBaseCactusFactory


class CactusFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False, factory_method=None):
        super(CactusFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.factory_methods = [
                GlobularBaseCactusFactory,
                ColumnarBaseCactusFactory,
                PrickyPearBaseCactusFactory,
            ]  # , KalidiumBaseCactusFactory]
            weights = np.array([1] * len(self.factory_methods))
            self.weights = weights / weights.sum()
            if factory_method is None:
                with FixedSeed(self.factory_seed):
                    factory_method = np.random.choice(
                        self.factory_methods, p=self.weights
                    )
            self.factory: BaseCactusFactory = factory_method(factory_seed, coarse)
            base_hue = uniform(0.2, 0.4)
            self.material = surface.shaderfunc_to_material(self.shader_cactus, base_hue)

    def create_asset(self, face_size=0.01, realize=True, **params):
        obj = self.factory.create_asset(**params)

        remesh_with_attrs(obj, face_size)

        if self.factory.noise_strength > 0:
            t = np.random.choice(["STUCCI", "MARBLE"])
            texture = bpy.data.textures.new(name="coral", type=t)
            texture.noise_scale = log_uniform(0.1, 0.15)
            butil.modify_mesh(
                obj,
                "DISPLACE",
                True,
                strength=self.factory.noise_strength,
                mid_level=0,
                texture=texture,
            )

        assign_material(obj, self.material)

        if face_size <= 0.05 and self.factory.density > 0:
            t = spike.apply(
                obj, self.factory.points_fn, self.factory.base_radius, realize
            )

            tagging.tag_object(obj, "cactus_spike")
            obj = join_objects([obj, t])

        tagging.tag_object(obj, "cactus")

        return obj

    @staticmethod
    def shader_cactus(nw: NodeWrangler, base_hue):
        shift = uniform(-0.15, 0.15)
        bright_color = hsv2rgba((base_hue + shift) % 1, 1.0, 0.02)
        dark_color = hsv2rgba(base_hue, 0.8, 0.01)
        fresnel_color = hsv2rgba(
            (base_hue - uniform(0.05, 0.1)) % 1, 0.9, uniform(0.3, 0.5)
        )
        specular = 0.25

        fresnel = nw.scalar_multiply(nw.new_node(Nodes.Fresnel), log_uniform(0.6, 1.0))
        color = build_color_ramp(
            nw,
            nw.musgrave(log_uniform(10, 50)),
            [0.0, 0.3, 0.7, 1.0],
            [dark_color, dark_color, bright_color, bright_color],
        )
        color = nw.new_node(Nodes.MixRGB, [fresnel, color, fresnel_color])
        noise_texture = nw.new_node(Nodes.NoiseTexture, input_kwargs={"Scale": 50})
        roughness = nw.build_float_curve(noise_texture, [(0, 0.5), (1, 0.8)])
        bsdf = nw.new_node(
            Nodes.PrincipledBSDF,
            input_kwargs={
                "Base Color": color,
                "Roughness": roughness,
                "Specular IOR Level": specular,
            },
        )
        return bsdf


class GlobularCactusFactory(CactusFactory):
    def __init__(self, factory_seed, coarse=False):
        super(GlobularCactusFactory, self).__init__(
            factory_seed, coarse, GlobularBaseCactusFactory
        )


class ColumnarCactusFactory(CactusFactory):
    def __init__(self, factory_seed, coarse=False):
        super(ColumnarCactusFactory, self).__init__(
            factory_seed, coarse, ColumnarBaseCactusFactory
        )


class PrickyPearCactusFactory(CactusFactory):
    def __init__(self, factory_seed, coarse=False):
        super(PrickyPearCactusFactory, self).__init__(
            factory_seed, coarse, PrickyPearBaseCactusFactory
        )


class KalidiumCactusFactory(CactusFactory):
    def __init__(self, factory_seed, coarse=False):
        super(KalidiumCactusFactory, self).__init__(
            factory_seed, coarse, KalidiumBaseCactusFactory
        )
