# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import numpy as np
from numpy.random import uniform

from infinigen.assets.objects.creatures.parts.utils.draw import make_segments
from infinigen.assets.objects.creatures.util.animation.driver_repeated import (
    bend_bones_lerp,
)
from infinigen.assets.objects.creatures.util.creature import Part, PartFactory
from infinigen.assets.objects.creatures.util.genome import Joint
from infinigen.assets.utils.decorate import read_co
from infinigen.assets.utils.object import join_objects
from infinigen.core.surface import write_attr_data
from infinigen.core.util.random import log_uniform


class CrabLegFactory(PartFactory):
    tags = ["leg"]

    def make_part(self, params) -> Part:
        x_length = params["x_length"]
        segments, x_cuts = self.make_segments(params)
        obj = join_objects(segments)

        skeleton = np.zeros((2, 3))
        skeleton[1, 0] = x_length
        joints = {x: Joint(rest=(0, 0, 0)) for x in x_cuts[1:-1]}
        return Part(skeleton, obj, joints=joints)

    def make_segments(self, params):
        x_cuts = [0, params["x_mid_first"], params["x_mid_second"], 1]
        y_cuts = [1, params["y_mid_first"], params["y_mid_second"], 0.01]

        def x_anchors(u, v):
            return u, u + 0.01, (u + v) / 2, v - 0.01, v

        def y_anchors(u, v):
            return u * 0.9, u, (u + v) / 2 * params["y_expand"], v, v * 0.9

        segments = make_segments(x_cuts, y_cuts, x_anchors, y_anchors, params)
        for obj in segments:
            z = read_co(obj).T[-1]
            write_attr_data(
                obj,
                "ratio",
                1 + np.where(z > 0, 0, uniform(0.8, 1.5) * z / params["y_length"]),
            )
        return segments, x_cuts

    def sample_params(self):
        x_length = uniform(0.8, 1.2)
        y_length = uniform(0.025, 0.035)
        z_length = y_length * uniform(1.0, 1.5)
        x_mid_first = uniform(0.3, 0.4)
        x_mid_second = uniform(0.6, 0.7)
        y_mid_first = uniform(0.7, 1.0)
        y_mid_second = y_mid_first / 2 * uniform(1.1, 1.3)
        y_expand = uniform(1.1, 1.3)
        noise_strength = uniform(0.005, 0.01)
        noise_scale = log_uniform(5, 10)
        bottom_shift = uniform(0.3, 0.5)
        bottom_cutoff = uniform(0.2, 0.5)
        top_shift = uniform(0.2, 0.4)
        top_cutoff = uniform(0.6, 0.8)
        return {
            "x_length": x_length,
            "y_length": y_length,
            "z_length": z_length,
            "x_mid_first": x_mid_first,
            "x_mid_second": x_mid_second,
            "y_mid_first": y_mid_first,
            "y_mid_second": y_mid_second,
            "y_expand": y_expand,
            "noise_strength": noise_strength,
            "noise_scale": noise_scale,
            "bottom_shift": bottom_shift,
            "bottom_cutoff": bottom_cutoff,
            "top_shift": top_shift,
            "top_cutoff": top_cutoff,
        }

    @staticmethod
    def animate_bones(arma, bones, params):
        bend_bones_lerp(
            arma, bones, params["leg_curl"], params["freq"], rot=params["leg_rot"]
        )


class LobsterLegFactory(CrabLegFactory):
    def sample_params(self):
        y_length = uniform(0.01, 0.015)
        z_length = y_length * log_uniform(1, 1.2)
        return {
            **super(LobsterLegFactory, self).sample_params(),
            "y_length": y_length,
            "z_length": z_length,
        }
