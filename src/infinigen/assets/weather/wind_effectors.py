# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Alexander Raistrick


import bpy
import gin
import numpy as np

from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import FixedSeed, random_general


@gin.configurable
class WindEffector(AssetFactory):
    def __init__(self, factory_seed, strength):
        super().__init__(factory_seed)
        with FixedSeed(factory_seed):
            self.strength = random_general(strength)

    def create_asset(self, **kwargs):
        bpy.ops.object.effector_add(type="WIND")
        wind = bpy.context.active_object

        yaw = np.random.uniform(0, 360)
        wind.rotation_euler = np.deg2rad((90, 0, yaw))

        wind.field.strength = self.strength
        wind.field.flow = 0

        return wind


@gin.configurable
class TurbulenceEffector(AssetFactory):
    def __init__(self, factory_seed, strength, noise, size=1, flow=0):
        super().__init__(factory_seed)
        with FixedSeed(factory_seed):
            self.strength = random_general(strength)
            self.noise = random_general(noise)
            self.size = random_general(size)
            self.flow = random_general(flow)

    def create_asset(self, **kwargs):
        bpy.ops.object.effector_add(type="TURBULENCE")
        wind = bpy.context.active_object
        wind.field.strength = self.strength
        wind.field.noise = self.noise
        wind.field.flow = self.flow
        wind.field.size = self.size

        return wind
