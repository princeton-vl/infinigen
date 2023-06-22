# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
# Date Signed: April 13 2023 

import numpy as np

from assets.deformed_trees import FallenTreeFactory, HollowTreeFactory, RottenTreeFactory
from assets.deformed_trees.truncated import TruncatedTreeFactory
from placement.factory import AssetFactory
from util.math import FixedSeed
from assets.utils.tag import tag_object, tag_nodegroup

class DeformedTreeFactory(AssetFactory):

    def __init__(self, factory_seed, coarse=False):
        super(DeformedTreeFactory, self).__init__(factory_seed, coarse)
        self.makers = [FallenTreeFactory, RottenTreeFactory, TruncatedTreeFactory, HollowTreeFactory]
        self.weights = np.array([1, 1, 1, 1])
        with FixedSeed(factory_seed):
            self.maker = np.random.choice(self.makers, p=self.weights / self.weights.sum())

    def create_asset(self, **params):
        return self.maker(**params)
