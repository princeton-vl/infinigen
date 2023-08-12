# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import bpy
import numpy as np
from numpy.random import uniform

from .veratrum import VeratrumMonocotFactory
from .banana import BananaMonocotFactory, TaroMonocotFactory
from .agave import AgaveMonocotFactory
from .grasses import GrassesMonocotFactory, MaizeMonocotFactory, WheatMonocotFactory
from .growth import MonocotGrowthFactory
from .tussock import TussockMonocotFactory
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util.math import FixedSeed
from ..utils.decorate import join_objects
from ..utils.mesh import polygon_angles
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

class MonocotFactory(AssetFactory):
    max_cluster = 10

    def create_asset(self, i, **params) -> bpy.types.Object:
        params['decorate'] = True
        if self.factory.is_grass:
            n = np.random.randint(1, 6)
            angles = polygon_angles(n, np.pi / 4, np.pi * 2)
            radius = uniform(.08, .16, n)
            monocots = [self.factory.create_asset(**params, i=j + i * self.max_cluster) for j in range(n)]
            for m, a, r in zip(monocots, angles, radius):
                m.location = r * np.cos(a), r * np.sin(a), 0
            obj = join_objects(monocots)
            tag_object(obj, 'monocot')
            return obj
        else:
            m = self.factory.create_asset(**params)
            tag_object(m, 'monocot')
            return m

    def __init__(self, factory_seed, coarse=False, factory_method=None, grass=None):
        super(MonocotFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            grass_factory = [TussockMonocotFactory, GrassesMonocotFactory, WheatMonocotFactory,
                MaizeMonocotFactory]
            nongrass_factory = [AgaveMonocotFactory, BananaMonocotFactory, TaroMonocotFactory,
                VeratrumMonocotFactory]
            # noinspection PyTypeChecker
            self.factory_methods = grass_factory + nongrass_factory if grass is None else grass_factory if \
                grass else nongrass_factory
            weights = np.array([1] * len(self.factory_methods))
            self.weights = weights / weights.sum()
            if factory_method is None:
                with FixedSeed(self.factory_seed):
                    factory_method = np.random.choice(self.factory_methods, p=self.weights)
            self.factory: MonocotGrowthFactory = factory_method(factory_seed, coarse)
