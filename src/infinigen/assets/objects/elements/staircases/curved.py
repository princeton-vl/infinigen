# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Lingjie Mei
# - Karhan Kayan: fix constants

import numpy as np

from infinigen.assets.objects.elements.staircases.straight import (
    StraightStaircaseFactory,
)
from infinigen.assets.utils.decorate import read_co, write_co
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform


class CurvedStaircaseFactory(StraightStaircaseFactory):
    support_types = (
        "weighted_choice",
        (2, "single-rail"),
        (2, "double-rail"),
        (4, "side"),
        (4, "solid"),
        (4, "hole"),
    )

    handrail_types = "weighted_choice", (2, "horizontal-post"), (2, "vertical-post")

    def __init__(self, factory_seed, coarse=False, constants=None):
        self.full_angle, self.radius, self.theta = 0, 0, 0
        super(CurvedStaircaseFactory, self).__init__(factory_seed, coarse, constants)
        with FixedSeed(self.factory_seed):
            self.has_spiral = True

    def build_size_config(self):
        while True:
            self.full_angle = np.random.randint(1, 5) * np.pi / 2
            self.n = np.random.randint(13, 21)
            self.step_height = self.constants.wall_height / self.n
            self.theta = self.full_angle / self.n
            self.step_length = self.step_height * log_uniform(1, 1.5)
            self.step_width = log_uniform(0.9, 1.5)
            self.radius = self.step_length / self.theta
            if self.radius / self.step_width > 1.5:
                break

    def make_spiral(self, obj):
        x, y, z = read_co(obj).T
        u = x + self.radius - self.step_width
        t = y / self.step_length * self.theta
        write_co(obj, np.stack([u * np.cos(t), u * np.sin(t), z], -1))

    def unmake_spiral(self, obj):
        co = read_co(obj)
        x, y, z = co.T
        u = np.linalg.norm(co[:, :2], axis=-1)
        t = np.arctan2(y, x)
        margins, ts = [], []
        for o in np.linspace(0, np.pi * 2, 8):
            t_ = (t - o) % (np.pi * 2) + o
            margins.append(np.max(t_) - np.min(t_))
            ts.append(t_)
        t = ts[np.argmin(margins)]
        x = u - self.radius + self.step_width
        y = t * self.step_length / self.theta
        co = np.stack([x, y, z], -1)
        write_co(obj, co)
        return obj

    @property
    def upper(self):
        return np.pi / 2 + self.full_angle
