# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import bpy
import numpy as np

from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util.math import FixedSeed

from .cantilever import CantileverStaircaseFactory
from .curved import CurvedStaircaseFactory
from .l_shaped import LShapedStaircaseFactory
from .spiral import SpiralStaircaseFactory
from .straight import StraightStaircaseFactory
from .u_shaped import UShapedStaircaseFactory


class StaircaseFactory(AssetFactory):
    factories = [
        StraightStaircaseFactory,
        LShapedStaircaseFactory,
        UShapedStaircaseFactory,
        SpiralStaircaseFactory,
        CurvedStaircaseFactory,
        CantileverStaircaseFactory,
    ]
    probs = np.array([4, 3, 3, 1, 2, 2])

    def __init__(self, factory_seed, coarse=False):
        super(StaircaseFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            base_factory_fn = np.random.choice(
                self.factories, p=self.probs / self.probs.sum()
            )
            self.base_factory = base_factory_fn(self.factory_seed)

    def create_asset(self, **params) -> bpy.types.Object:
        return self.base_factory.create_asset(**params)

    def finalize_assets(self, assets):
        self.base_factory.finalize_assets(assets)
