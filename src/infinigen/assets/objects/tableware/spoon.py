# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.utils.decorate import subsurf, write_co
from infinigen.assets.utils.object import new_grid
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform

from .base import TablewareFactory


class SpoonFactory(TablewareFactory):
    x_end = 0.15
    is_fragile = True

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.x_length = log_uniform(0.2, 0.8)
            self.y_length = log_uniform(0.06, 0.12)
            self.z_depth = log_uniform(0.08, 0.25)
            self.z_offset = uniform(0.0, 0.05)
            self.thickness = log_uniform(0.008, 0.015)
            self.has_guard = uniform(0, 1) < 0.4
            self.guard_type = "round" if uniform(0, 1) < 0.6 else "double"
            self.guard_depth = log_uniform(0.2, 1.0) * self.thickness
            self.scale = log_uniform(0.15, 0.25)

    def create_asset(self, **params) -> bpy.types.Object:
        x_anchors = np.array(
            [
                log_uniform(0.07, 0.25),
                0,
                -0.08,
                -0.12,
                -self.x_end,
                -self.x_end - self.x_length,
                -self.x_end - self.x_length * log_uniform(1.2, 1.4),
            ]
        )
        y_anchors = np.array(
            [
                self.y_length * log_uniform(0.1, 0.8),
                self.y_length * log_uniform(1.0, 1.2),
                self.y_length * log_uniform(0.6, 1.0),
                self.y_length * log_uniform(0.2, 0.4),
                log_uniform(0.01, 0.02),
                log_uniform(0.02, 0.05),
                log_uniform(0.01, 0.02),
            ]
        )
        z_anchors = np.array(
            [
                0,
                0,
                0,
                0,
                self.z_offset,
                self.z_offset + uniform(-0.02, 0.04),
                self.z_offset + uniform(-0.02, 0),
            ]
        )
        obj = new_grid(x_subdivisions=len(x_anchors) - 1, y_subdivisions=2)
        x = np.concatenate([x_anchors] * 3)
        y = np.concatenate([y_anchors, np.zeros_like(y_anchors), -y_anchors])
        z = np.concatenate([z_anchors] * 3)
        x[len(x_anchors)] += 0.02
        z[len(x_anchors) + 1] = -self.z_depth
        write_co(obj, np.stack([x, y, z], -1))
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness)
        subsurf(obj, 1)

        def selection(nw, x):
            return nw.compare("LESS_THAN", x, -self.x_end)

        if self.guard_type == "double":
            selection = self.make_double_sided(selection)
        self.add_guard(obj, selection)
        subsurf(obj, 2)
        obj.scale = [self.scale] * 3
        butil.apply_transform(obj)
        return obj
