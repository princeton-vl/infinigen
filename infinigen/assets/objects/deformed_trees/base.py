# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Lingjie Mei


import colorsys

from numpy.random import uniform

from infinigen.assets.objects.trees.generate import GenericTreeFactory, random_species
from infinigen.core import surface
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.surface import NoApply
from infinigen.core.util.color import hsv2rgba
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform


class BaseDeformedTreeFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super(BaseDeformedTreeFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            (tree_params, _, _), _ = random_species()
            tree_params.skinning.update({"Scaling": 0.2})
            self.base_factory = GenericTreeFactory(
                factory_seed, tree_params, None, NoApply, coarse
            )
            self.trunk_surface = surface.registry("bark")
            self.base_hue = uniform(0.02, 0.08)
            self.material = surface.shaderfunc_to_material(
                self.shader_rings, self.base_hue
            )

    def build_tree(self, i, distance, **kwargs):
        return self.base_factory.spawn_asset(i=i, distance=distance)

    @staticmethod
    def geo_xyz(nw: NodeWrangler):
        geometry = nw.new_node(
            Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
        )
        for name, component in zip(
            "xyz", nw.separate(nw.new_node(Nodes.InputPosition))
        ):
            geometry = nw.new_node(
                Nodes.StoreNamedAttribute,
                input_kwargs={"Geometry": geometry, "Name": name, "Value": component},
            )
        nw.new_node(Nodes.GroupOutput, input_kwargs={"Geometry": geometry})

    @staticmethod
    def shader_rings(nw: NodeWrangler, base_hue):
        position = nw.combine(
            *map(
                lambda n: nw.new_node(Nodes.Attribute, attrs={"attribute_name": n}),
                "xyz",
            )
        )
        ratio = nw.new_node(
            Nodes.WaveTexture,
            [position],
            input_kwargs={"Scale": uniform(10, 20), "Distortion": uniform(4, 10)},
            attrs={"wave_type": "RINGS", "rings_direction": "Z", "wave_profile": "SAW"},
        )
        bright_color = hsv2rgba(base_hue, uniform(0.4, 0.8), log_uniform(0.2, 0.8))
        dark_color = (
            *colorsys.hsv_to_rgb(
                (base_hue + uniform(-0.02, 0.02)) % 1,
                uniform(0.4, 0.8),
                log_uniform(0.02, 0.05),
            ),
            1.0,
        )
        color = nw.new_node(Nodes.MixRGB, [ratio, dark_color, bright_color])
        principled_bsdf = nw.new_node(
            Nodes.PrincipledBSDF, input_kwargs={"Base Color": color}
        )
        return principled_bsdf

    def create_asset(self, face_size, **params):
        raise NotImplementedError
