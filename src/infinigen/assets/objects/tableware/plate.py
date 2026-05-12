# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.objects.tableware.base import TablewareFactory
from infinigen.assets.utils.decorate import subsurf
from infinigen.assets.utils.draw import spin
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform


class PlateFactory(TablewareFactory):
    allow_transparent = True

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.x_end = 0.5
            self.z_length = log_uniform(0.05, 0.2)
            self.x_mid = uniform(0.3, 1.0) * self.x_end
            self.z_mid = uniform(0.3, 0.8) * self.z_length
            self.has_guard = False
            self.pre_level = 1
            self.has_inside = uniform(0, 1) < 0.2
            self.scale = log_uniform(0.2, 0.4)
            self.thickness = uniform(0.01, 0.03) * self.scale
            self.scratch = self.edge_wear = None

    def create_asset(self, **params) -> bpy.types.Object:
        x_anchors = 0, self.x_mid, self.x_mid, self.x_end
        z_anchors = 0, 0, self.z_mid, self.z_length
        anchors = np.array(x_anchors) * self.scale, 0, np.array(z_anchors) * self.scale
        obj = spin(anchors, [1, 2])
        butil.modify_mesh(
            obj, "SUBSURF", render_levels=self.pre_level, levels=self.pre_level
        )
        self.solidify_with_inside(obj, self.thickness)
        subsurf(obj, 1)
        return obj
