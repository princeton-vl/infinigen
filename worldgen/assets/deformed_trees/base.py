# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
# Date Signed: April 13 2023 

import colorsys

from numpy.random import uniform

from assets.trees import TreeFactory
from assets.trees.generate import GenericTreeFactory, random_species
from assets.utils.misc import log_uniform
from nodes.node_info import Nodes
from nodes.node_wrangler import NodeWrangler
from placement.factory import AssetFactory
from surfaces import surface
from surfaces.surface import NoApply
from util.math import FixedSeed


class BaseDeformedTreeFactory(AssetFactory):

    def __init__(self, factory_seed, coarse=False):
        super(BaseDeformedTreeFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            (tree_params, _, _), _ = random_species()
            tree_params.skinning.update({'Scaling': .2})
            self.base_factory = GenericTreeFactory(factory_seed, tree_params, None, NoApply, coarse)
            self.trunk_surface = surface.registry('bark')
            self.base_hue = uniform(.02, .08)
            self.material = surface.shaderfunc_to_material(self.shader_rings, self.base_hue)

    def create_placeholder(self, **kwargs):
        return self.base_factory.create_placeholder(**kwargs)

    def build_tree(self, face_size, **params):
        return self.base_factory.create_asset(**params, face_size=face_size, child_merge_dist=0)

    @staticmethod
    def geo_xyz(nw: NodeWrangler):
        geometry = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
        for name, component in zip('xyz', nw.separate(nw.new_node(Nodes.InputPosition))):
            geometry = nw.new_node(Nodes.StoreNamedAttribute, [geometry, name, None, component])
        nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geometry})

    @staticmethod
    def shader_rings(nw: NodeWrangler, base_hue):
        position = nw.combine(*map(lambda n: nw.new_node(Nodes.Attribute, attrs={'attribute_name': n}), 'xyz'))
        ratio = nw.new_node(Nodes.WaveTexture, [position],
                            input_kwargs={'Scale': uniform(10, 20), 'Distortion': uniform(4, 10)},
                            attrs={'wave_type': 'RINGS', 'rings_direction': 'Z', 'wave_profile': 'SAW'})
        bright_color = *colorsys.hsv_to_rgb(base_hue, uniform(.4, .8), log_uniform(.2, .8)), 1.
        dark_color = *colorsys.hsv_to_rgb((base_hue + uniform(-.02, .02)) % 1, uniform(.4, .8),
                                          log_uniform(.02, .05)), 1.
        color = nw.new_node(Nodes.MixRGB, [ratio, dark_color, bright_color])
        principled_bsdf = nw.new_node(Nodes.PrincipledBSDF, input_kwargs={'Base Color': color})
        return principled_bsdf

    def create_asset(self, face_size, **params):
        raise NotImplementedError
