# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import colorsys

from numpy.random import uniform

from infinigen.assets.utils.object import join_objects, origin2lowest
from infinigen.core import surface
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_utils import build_color_ramp
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform

from .cap import MushroomCapFactory
from .stem import MushroomStemFactory


class MushroomGrowthFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.base_hue = self.build_base_hue()
            self.material_func = lambda: surface.shaderfunc_to_material(
                self.shader_mushroom, self.base_hue
            )
            self.cap_factory = MushroomCapFactory(
                factory_seed, self.base_hue, self.material_func, coarse
            )
            self.stem_factory = MushroomStemFactory(
                factory_seed, self.cap_factory.inner_radius, self.material_func, coarse
            )

    @staticmethod
    def build_base_hue():
        if uniform(0, 1) < 0.4:
            return uniform(0, 1)
        else:
            return uniform(0.02, 0.15)

    def create_asset(self, **params):
        cap = self.cap_factory(**params)
        stem = self.stem_factory(**params)
        obj = join_objects([cap, stem])
        origin2lowest(obj)
        return cap

    @staticmethod
    def shader_mushroom(nw: NodeWrangler, base_hue):
        roughness = 0.8
        front_color = (
            *colorsys.hsv_to_rgb(
                (base_hue + uniform(-0.1, 0.1)) % 1,
                uniform(0.1, 0.3),
                log_uniform(0.02, 0.5),
            ),
            1,
        )
        back_color = (
            *colorsys.hsv_to_rgb(
                (base_hue + uniform(-0.1, 0.1)) % 1,
                uniform(0.1, 0.3),
                log_uniform(0.02, 0.5),
            ),
            1,
        )

        x, y, z = nw.separate(nw.new_node(Nodes.TextureCoord).outputs["Generated"])
        musgrave = nw.new_node(
            Nodes.MapRange,
            [
                nw.new_node(
                    Nodes.MusgraveTexture,
                    [nw.combine(x, y, nw.scalar_multiply(uniform(5, 10), z))],
                    input_kwargs={"Scale": 200},
                ),
                -1,
                1,
                0,
                1,
            ],
        )

        color = build_color_ramp(
            nw,
            musgrave,
            [0, 0.3, 0.7, 1],
            [front_color, front_color, back_color, back_color],
        )
        principled_bsdf = nw.new_node(
            Nodes.PrincipledBSDF,
            input_kwargs={"Base Color": color, "Roughness": roughness},
        )
        return principled_bsdf
