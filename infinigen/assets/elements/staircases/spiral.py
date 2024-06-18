# Copyright (c) Princeton University.

import numpy as np
from numpy.random import uniform

from infinigen.assets.elements.staircases.curved import CurvedStaircaseFactory
from infinigen.assets.utils.decorate import read_co, remove_vertices, write_attribute
from infinigen.core.constraints.example_solver.room import constants
from infinigen.core.util.random import log_uniform
from infinigen.assets.utils.nodegroup import geo_radius
from infinigen.assets.utils.object import new_line, separate_loose
from infinigen.core import surface
from infinigen.core.util.math import FixedSeed
import infinigen.core.util.blender as butil


class SpiralStaircaseFactory(CurvedStaircaseFactory):
    support_types = 'column'

    def __init__(self, factory_seed, coarse=False):
        super(SpiralStaircaseFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            self.column_radius = self.radius - self.step_width + uniform(.05, .08)
            self.has_column = True
            self.handrail_alphas = [1 - self.handrail_offset / self.step_width]

    def build_size_config(self):
        while True:
            self.full_angle = np.random.randint(1, 5) * np.pi / 2
            self.n = np.random.randint(13, 21)
            self.theta = self.full_angle / self.n
            self.step_length = self.step_height * log_uniform(1, 1.2)
            self.radius = self.step_length / self.theta
            if .9 < self.radius < 1.5:
                self.step_width = self.radius * uniform(.9, .95)
                break

    def make_column(self):
        obj = new_line(self.n, self.step_height * self.n + self.post_height)
        obj.rotation_euler[1] = - np.pi / 2
        butil.apply_transform(obj)
        surface.add_geomod(obj, geo_radius, apply=True, input_args=[self.column_radius, 16])
        write_attribute(obj, 1, 'steps', 'FACE')
        return obj

    def unmake_spiral(self, obj):
        obj = super().unmake_spiral(obj)
        x, y, z = read_co(obj).T
        margin = .1
        if (x >= 0).sum() >= (x <= 0).sum():
            remove_vertices(obj, lambda x, y, z: x < margin)
        else:
            remove_vertices(obj, lambda x, y, z: x > -margin)
        obj = separate_loose(obj)
        return obj
