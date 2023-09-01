# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo


import bpy
import mathutils
import numpy as np
from numpy.random import uniform, normal, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category, hsv2rgba
from infinigen.core import surface

from infinigen.core.util.math import FixedSeed
from infinigen.core.util import blender as butil
from infinigen.core.placement.factory import AssetFactory

from infinigen.assets.fruits.general_fruit import FruitFactoryGeneralFruit
from infinigen.assets.fruits.apple import FruitFactoryApple
from infinigen.assets.fruits.pineapple import FruitFactoryPineapple
from infinigen.assets.fruits.starfruit import FruitFactoryStarfruit
from infinigen.assets.fruits.strawberry import FruitFactoryStrawberry
from infinigen.assets.fruits.blackberry import FruitFactoryBlackberry
from infinigen.assets.fruits.coconuthairy import FruitFactoryCoconuthairy
from infinigen.assets.fruits.coconutgreen import FruitFactoryCoconutgreen
from infinigen.assets.fruits.durian import FruitFactoryDurian

fruit_names = {'Apple': FruitFactoryApple, 
               'Pineapple': FruitFactoryPineapple,
               'Starfruit': FruitFactoryStarfruit,
               'Strawberry': FruitFactoryStrawberry,
               'Blackberry': FruitFactoryBlackberry,
               'Coconuthairy': FruitFactoryCoconuthairy,
               'Coconutgreen': FruitFactoryCoconutgreen,
               'Durian': FruitFactoryDurian,
               }

class FruitFactoryCompositional(FruitFactoryGeneralFruit):
    def __init__(self, factory_seed, scale=1.0, coarse=False):
        super(FruitFactoryCompositional, self).__init__(factory_seed, scale=scale, coarse=coarse)

        self.name = 'compositional'
        self.factories = {}

        for name, factory in fruit_names.items():
            self.factories[name] = factory(factory_seed, scale, coarse)

        with FixedSeed(factory_seed):
            self.cross_section_source = np.random.choice(list(fruit_names.keys()))
            self.shape_source = np.random.choice(list(fruit_names.keys()))
            self.surface_source = np.random.choice(list(fruit_names.keys()))
            self.stem_source = np.random.choice(list(fruit_names.keys()))

    def sample_cross_section_params(self, surface_resolution=256):
        return self.factories[self.cross_section_source].sample_cross_section_params(surface_resolution)

    def sample_shape_params(self, surface_resolution=256):
        return self.factories[self.shape_source].sample_shape_params(surface_resolution)

    def sample_surface_params(self):
        return self.factories[self.surface_source].sample_surface_params()

    def sample_stem_params(self):
        return self.factories[self.stem_source].sample_stem_params()


