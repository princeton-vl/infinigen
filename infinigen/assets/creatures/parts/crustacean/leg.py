# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import numpy as np
from numpy.random import uniform

from infinigen.assets.creatures.util.animation.driver_repeated import bend_bones_lerp
from infinigen.assets.creatures.util.creature import Part, PartFactory
from infinigen.assets.creatures.util.genome import Joint
from infinigen.assets.creatures.parts.utils.draw import make_segments
from infinigen.assets.utils.decorate import join_objects, read_co
from infinigen.assets.utils.misc import log_uniform
from infinigen.core.surface import write_attr_data


class CrabLegFactory(PartFactory):
    tags = ['leg']

    def make_part(self, params) -> Part:
        x_length = params['x_length']
        segments, x_cuts = self.make_segments(params)
        obj = join_objects(segments)

        skeleton = np.zeros((2, 3))
        skeleton[1, 0] = x_length
        joints = {x: Joint(rest=(0, 0, 0)) for x in x_cuts[1:-1]}
        return Part(skeleton, obj, joints=joints)

    def make_segments(self, params):
        x_cuts = [0, params['x_mid_first'], params['x_mid_second'], 1]
        y_cuts = [1, params['y_mid_first'], params['y_mid_second'], .01]
        x_anchors = lambda u, v: (u, u + 1e-2, (u + v) / 2, v - 1e-2, v)
        y_anchors = lambda u, v: (u * .9, u, (u + v) / 2 * params['y_expand'], v, v * .9)
        segments = make_segments(x_cuts, y_cuts, x_anchors, y_anchors, params)
        for obj in segments:
            z = read_co(obj).T[-1]
            write_attr_data(obj, 'ratio', 1 + np.where(z > 0, 0, uniform(.8, 1.5) * z / params['y_length']))
        return segments, x_cuts

    def sample_params(self):
        x_length = uniform(.8, 1.2)
        y_length = uniform(.025, .035)
        z_length = y_length * uniform(1., 1.5)
        x_mid_first = uniform(.3, .4)
        x_mid_second = uniform(.6, .7)
        y_mid_first = uniform(.7, 1.)
        y_mid_second = y_mid_first / 2 * uniform(1.1, 1.3)
        y_expand = uniform(1.1, 1.3)
        noise_strength = uniform(.005, .01)
        noise_scale = log_uniform(5, 10)
        bottom_shift = uniform(.3, .5)
        bottom_cutoff = uniform(.2, .5)
        top_shift = uniform(.2, .4)
        top_cutoff = uniform(.6, .8)
        return {
            'x_length': x_length,
            'y_length': y_length,
            'z_length': z_length,
            'x_mid_first': x_mid_first,
            'x_mid_second': x_mid_second,
            'y_mid_first': y_mid_first,
            'y_mid_second': y_mid_second,
            'y_expand': y_expand,
            'noise_strength': noise_strength,
            'noise_scale': noise_scale,
            'bottom_shift': bottom_shift,
            'bottom_cutoff': bottom_cutoff,
            'top_shift': top_shift,
            'top_cutoff': top_cutoff,
        }

    @staticmethod
    def animate_bones(arma, bones, params):
        bend_bones_lerp(arma, bones, params['leg_curl'], params['freq'], rot=params['leg_rot'])


class LobsterLegFactory(CrabLegFactory):
    def sample_params(self):
        y_length = uniform(.01, .015)
        z_length = y_length * log_uniform(1, 1.2)
        return {**super(LobsterLegFactory, self).sample_params(), 'y_length': y_length, 'z_length': z_length}
