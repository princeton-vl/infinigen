# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.materials import glass
from infinigen.assets.objects.tableware.base import TablewareFactory
from infinigen.assets.utils.draw import spin
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform


class WineglassFactory(TablewareFactory):
    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.x_end = 0.25
            self.z_length = log_uniform(0.6, 2.0)
            self.z_cup = uniform(0.3, 0.6) * self.z_length
            self.z_mid = self.z_cup + uniform(0.3, 0.5) * (self.z_length - self.z_cup)
            self.x_neck = log_uniform(0.01, 0.02)
            self.x_top = self.x_end * log_uniform(1, 1.4)
            self.x_mid = self.x_top * log_uniform(0.9, 1.2)
            self.has_guard = False
            self.thickness = uniform(0.01, 0.03)
            self.surface = glass
            self.scale = log_uniform(0.1, 0.3)

    def create_asset(self, **params) -> bpy.types.Object:
        z_bottom = self.z_length * log_uniform(0.01, 0.05)
        x_anchors = (
            self.x_end,
            self.x_end / 2,
            self.x_neck,
            self.x_neck,
            self.x_mid,
            self.x_top,
        )
        z_anchors = 0, z_bottom / 2, z_bottom, self.z_cup, self.z_mid, self.z_length
        anchors = x_anchors, np.zeros_like(x_anchors), z_anchors
        obj = spin(anchors, [0, 1, 2, 3])
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness)
        obj.scale = [self.scale] * 3
        butil.apply_transform(obj)

        with butil.SelectObjects(obj):
            bpy.ops.object.shade_smooth()

        return obj
