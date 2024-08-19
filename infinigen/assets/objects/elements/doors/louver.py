# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import numpy as np
from numpy.random import uniform

from infinigen.assets.utils.decorate import write_attribute, write_co
from infinigen.assets.utils.object import new_cube, new_plane
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform

from .panel import PanelDoorFactory


class LouverDoorFactory(PanelDoorFactory):
    def __init__(self, factory_seed, coarse=False, constants=None):
        super(LouverDoorFactory, self).__init__(factory_seed, coarse, constants)
        with FixedSeed(self.factory_seed):
            self.x_subdivisions = 1
            self.y_subdivisions = np.clip(np.random.binomial(5, 0.4), 1, None)
            self.has_panel = uniform() < 0.7
            self.has_upper_panel = uniform() < 0.5
            self.louver_width = uniform(0.002, 0.004)
            self.louver_margin = uniform(0.02, 0.03)
            self.louver_size = log_uniform(0.05, 0.1)
            self.louver_angle = uniform(np.pi / 4.5, np.pi / 3.5)
            self.has_louver = True

    def louver(self, obj, panel):
        x_min, x_max, y_min, y_max = panel["dimension"]
        cutter = new_cube(location=(1, 1, 1))
        butil.apply_transform(cutter, loc=True)
        write_attribute(cutter, 1, "louver", "FACE")
        cutter.location = (
            x_min - self.louver_margin,
            -self.louver_width,
            y_min - self.louver_margin,
        )
        cutter.scale = [
            (x_max - x_min) / 2 + self.louver_margin,
            self.depth / 2 + self.louver_width,
            (y_max - y_min) / 2 + self.louver_margin,
        ]
        butil.apply_transform(cutter, loc=True)
        butil.modify_mesh(obj, "BOOLEAN", object=cutter, operation="DIFFERENCE")

        hole = new_cube(location=(1, 1, 1))
        butil.apply_transform(hole, loc=True)
        write_attribute(hole, 1, "louver", "FACE")
        hole.location = x_min, -self.louver_width * 2, y_min
        hole.scale = (
            (x_max - x_min) / 2,
            self.depth / 2 + self.louver_width * 2,
            (y_max - y_min) / 2,
        )
        butil.apply_transform(hole, loc=True)
        butil.modify_mesh(cutter, "BOOLEAN", object=hole, operation="DIFFERENCE")
        butil.delete(hole)

        louver = new_plane()
        x = x_min, x_max, x_min, x_max
        y = 0, 0, self.depth, self.depth
        y_upper = y_min + self.depth * np.tan(self.louver_angle)
        z = y_min, y_min, y_upper, y_upper
        write_co(louver, np.stack([x, y, z], -1))
        butil.modify_mesh(louver, "SOLIDIFY", thickness=self.louver_width, offset=0)
        butil.modify_mesh(
            louver,
            "ARRAY",
            use_relative_offset=False,
            use_constant_offset=True,
            constant_offset_displace=(0, 0, self.louver_size),
            count=int(np.ceil((y_max - y_min) / self.louver_size) + 0.5),
        )
        write_attribute(louver, 1, "louver", "FACE")
        return [cutter, louver]

    def make_panels(self):
        panels = super(LouverDoorFactory, self).make_panels()
        if len(panels) == 1:
            panels[0]["func"] = self.louver
        elif len(panels) == 2:
            if not self.has_panel:
                panels[0]["func"] = self.louver
            panels[1]["func"] = self.louver
        else:
            if self.has_upper_panel:
                panels = [panels[0], panels[-1]]
            else:
                panels = [panels[0]]
            for panel in panels:
                panel["func"] = self.louver
        return panels
