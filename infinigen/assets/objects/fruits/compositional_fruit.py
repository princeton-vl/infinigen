# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo


import numpy as np

from infinigen.assets.objects.fruits.apple import FruitFactoryApple
from infinigen.assets.objects.fruits.blackberry import FruitFactoryBlackberry
from infinigen.assets.objects.fruits.coconutgreen import FruitFactoryCoconutgreen
from infinigen.assets.objects.fruits.coconuthairy import FruitFactoryCoconuthairy
from infinigen.assets.objects.fruits.durian import FruitFactoryDurian
from infinigen.assets.objects.fruits.general_fruit import FruitFactoryGeneralFruit
from infinigen.assets.objects.fruits.pineapple import FruitFactoryPineapple
from infinigen.assets.objects.fruits.starfruit import FruitFactoryStarfruit
from infinigen.assets.objects.fruits.strawberry import FruitFactoryStrawberry
from infinigen.core.util.math import FixedSeed

fruit_names = {
    "Apple": FruitFactoryApple,
    "Pineapple": FruitFactoryPineapple,
    "Starfruit": FruitFactoryStarfruit,
    "Strawberry": FruitFactoryStrawberry,
    "Blackberry": FruitFactoryBlackberry,
    "Coconuthairy": FruitFactoryCoconuthairy,
    "Coconutgreen": FruitFactoryCoconutgreen,
    "Durian": FruitFactoryDurian,
}


class FruitFactoryCompositional(FruitFactoryGeneralFruit):
    def __init__(self, factory_seed, scale=1.0, coarse=False):
        super(FruitFactoryCompositional, self).__init__(
            factory_seed, scale=scale, coarse=coarse
        )

        self.name = "compositional"
        self.factories = {}

        for name, factory in fruit_names.items():
            self.factories[name] = factory(factory_seed, scale, coarse)

        with FixedSeed(factory_seed):
            self.cross_section_source = np.random.choice(list(fruit_names.keys()))
            self.shape_source = np.random.choice(list(fruit_names.keys()))
            self.surface_source = np.random.choice(list(fruit_names.keys()))
            self.stem_source = np.random.choice(list(fruit_names.keys()))

    def sample_cross_section_params(self, surface_resolution=256):
        return self.factories[self.cross_section_source].sample_cross_section_params(
            surface_resolution
        )

    def sample_shape_params(self, surface_resolution=256):
        return self.factories[self.shape_source].sample_shape_params(surface_resolution)

    def sample_surface_params(self):
        return self.factories[self.surface_source].sample_surface_params()

    def sample_stem_params(self):
        return self.factories[self.stem_source].sample_stem_params()
