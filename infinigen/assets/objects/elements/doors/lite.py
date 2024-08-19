# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import numpy as np
from numpy.random import uniform

from infinigen.core.util.math import FixedSeed

from .panel import PanelDoorFactory


class LiteDoorFactory(PanelDoorFactory):
    def __init__(self, factory_seed, coarse=False, constants=None):
        super(LiteDoorFactory, self).__init__(factory_seed, coarse, constants)
        with FixedSeed(self.factory_seed):
            r = uniform()
            subdivide_glass = False
            if r <= 1 / 6:
                dimension = 0, 1, uniform(0.4, 0.6), 1
                subdivide_glass = True
            elif r <= 1 / 3:
                dimension = 0, 1, 0, 1
                subdivide_glass = True
            elif r <= 1 / 2:
                dimension = 0, uniform(0.3, 0.4), uniform(0.4, 0.6), 1
            elif r <= 2 / 3:
                dimension = 0, uniform(0.3, 0.4), uniform(0.4, 0.6), 1
            elif r <= 5 / 6:
                dimension = 0, 1, 0, 1
            else:
                x = uniform(0.3, 0.35)
                dimension = x, 1 - x, uniform(0.7, 0.8), 1
            self.x_min, self.x_max, self.y_min, self.y_max = dimension
            if subdivide_glass:
                self.x_subdivisions = np.random.choice([1, 3])
                self.y_subdivisions = int(
                    self.height / self.width * self.x_subdivisions
                ) + np.random.randint(-1, 2)
            else:
                self.x_subdivisions = 1
                self.y_subdivisions = 1
            self.has_glass = True

    def make_panels(self):
        x_range = (
            np.linspace(self.x_min, self.x_max, self.x_subdivisions + 1)
            * (self.width - self.panel_margin * 2)
            + self.panel_margin
        )
        y_range = (
            np.linspace(self.y_min, self.y_max, self.y_subdivisions + 1)
            * (self.height - self.panel_margin * 2)
            + self.panel_margin
        )
        panels = []
        for x_min, x_max in zip(x_range[:-1], x_range[1:]):
            for y_min, y_max in zip(y_range[:-1], y_range[1:]):
                panels.append(
                    {
                        "dimension": (x_min, x_max, y_min, y_max),
                        "func": self.bevel,
                        "attribute_name": "glass",
                    }
                )
        return panels
