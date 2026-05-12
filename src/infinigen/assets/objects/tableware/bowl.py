# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.objects.tableware.base import TablewareFactory
from infinigen.assets.utils.decorate import set_shade_smooth, subsurf
from infinigen.assets.utils.draw import spin
from infinigen.assets.utils.object import new_bbox
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform


class BowlFactory(TablewareFactory):
    allow_transparent = True

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.x_end = 0.5
            self.z_length = log_uniform(0.4, 0.8)
            self.z_bottom = log_uniform(0.02, 0.05)
            self.x_bottom = uniform(0.2, 0.3) * self.x_end
            self.x_mid = uniform(0.8, 0.95) * self.x_end
            self.has_guard = False
            self.has_inside = uniform(0, 1) < 0.5
            self.scale = log_uniform(0.15, 0.4)
            self.thickness = uniform(0.01, 0.03) * self.scale
        self.edge_wear = None

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        radius = self.x_end * self.scale
        return new_bbox(-radius, radius, -radius, radius, 0, self.z_length * self.scale)

    def create_asset(self, **params) -> bpy.types.Object:
        x_anchors = (
            0,
            self.x_bottom,
            self.x_bottom + 1e-3,
            self.x_bottom,
            self.x_mid,
            self.x_end,
        )
        z_anchors = 0, 0, 0, self.z_bottom, self.z_length / 2, self.z_length
        anchors = np.array(x_anchors) * self.scale, 0, np.array(z_anchors) * self.scale
        obj = spin(anchors, [2, 3])
        self.solidify_with_inside(obj, self.thickness)
        butil.modify_mesh(
            obj, "BEVEL", width=self.thickness / 2, segments=np.random.randint(2, 5)
        )
        subsurf(obj, 1)
        set_shade_smooth(obj)
        return obj
