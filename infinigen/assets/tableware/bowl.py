# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.tableware.base import TablewareFactory
from infinigen.assets.utils.decorate import subsurf, set_shade_smooth
from infinigen.assets.utils.draw import spin
from infinigen.assets.utils.object import new_bbox
from infinigen.core.util.random import log_uniform
from infinigen.core.util.math import FixedSeed
from infinigen.core.util import blender as butil


class BowlFactory(TablewareFactory):
    allow_transparent = True

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.x_end = .5
            self.z_length = log_uniform(.4, .8)
            self.z_bottom = log_uniform(.02, .05)
            self.x_bottom = uniform(.2, .3) * self.x_end
            self.x_mid = uniform(.8, .95) * self.x_end
            self.has_guard = False
            self.thickness = uniform(.01, .03)
            self.has_inside = uniform(0, 1) < .5
            self.scale = log_uniform(.15, .4)
        self.edge_wear = None

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        radius = self.x_end * self.scale
        return new_bbox(-radius, radius, -radius, radius, 0, self.z_length * self.scale)

    def create_asset(self, **params) -> bpy.types.Object:
        x_anchors = 0, self.x_bottom, self.x_bottom + 1e-3, self.x_bottom, self.x_mid, self.x_end
        z_anchors = 0, 0, 0, self.z_bottom, self.z_length / 2, self.z_length
        anchors = x_anchors, np.zeros_like(x_anchors), z_anchors
        obj = spin(anchors, [2, 3], 16, 64)
        subsurf(obj, 1)
        self.solidify_with_inside(obj, self.thickness)
        obj.scale = [self.scale] * 3
        butil.apply_transform(obj)
        subsurf(obj, 1)
        set_shade_smooth(obj)
        return obj
