# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.
# Authors: Lingjie Mei


import bpy
import numpy as np

from infinigen.assets.deformed_trees import FallenTreeFactory, HollowTreeFactory, RottenTreeFactory
from infinigen.assets.deformed_trees.truncated import TruncatedTreeFactory
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util.math import FixedSeed
from infinigen.assets.utils.tag import tag_object, tag_nodegroup


class DeformedTreeFactory(AssetFactory):

    def __init__(self, factory_seed, coarse=False):
        super(DeformedTreeFactory, self).__init__(factory_seed, coarse)
        self.maker_factories = [FallenTreeFactory, RottenTreeFactory, TruncatedTreeFactory, HollowTreeFactory]
        self.weights = np.array([1, 1, 1, 1])
        with FixedSeed(factory_seed):
            self.maker_factory = np.random.choice(self.maker_factories, p=self.weights / self.weights.sum())
            self.maker = self.maker_factory(factory_seed, coarse)

    def create_asset(self, **params):
        return self.maker.create_asset(**params)
