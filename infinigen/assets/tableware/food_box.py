# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.materials import text
from infinigen.assets.utils.object import new_cube
from infinigen.assets.utils.uv import wrap_six_sides
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform
from infinigen.core.util import blender as butil


class FoodBoxFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            dimensions = np.sort(log_uniform(.05, .3, 3)).tolist()
            self.dimensions = np.array([dimensions[1], dimensions[0], dimensions[2]])
            self.surface = text.Text(self.factory_seed)
            self.texture_shared = uniform() < .4
        
    def create_placeholder(self, **params):
        obj = new_cube()
        obj.scale = self.dimensions / 2
        butil.apply_transform(obj)
        return obj

    def create_asset(self, placeholder, **params) -> bpy.types.Object:
        obj = butil.copy(placeholder)
        wrap_six_sides(obj, self.surface, self.texture_shared)
        butil.modify_mesh(obj, 'BEVEL', width=.001)
        return obj
