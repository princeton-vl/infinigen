# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import numpy as np
from numpy.random import uniform
from scipy.interpolate import interp1d

from infinigen.assets.creatures.util.animation.driver_repeated import bend_bones_lerp
from infinigen.assets.creatures.util.creature import Part, PartFactory
from infinigen.assets.creatures.util.genome import Joint
from infinigen.assets.creatures.parts.utils.draw import make_segments
from infinigen.assets.utils.decorate import join_objects, read_co
from infinigen.assets.utils.misc import log_uniform
from infinigen.core.surface import write_attr_data


class CrustaceanTailFactory(PartFactory):
    tags = ['body']

    def make_part(self, params) -> Part:
        x_length = params['x_length']
        segments, x_cuts = self.make_segments(params)
        obj = join_objects(segments)

        skeleton = np.zeros((2, 3))
        skeleton[1, 0] = x_length
        joints = {x: Joint(rest=(0, 0, 0)) for x in x_cuts[1:]}
        return Part(skeleton, obj, joints=joints)

    def make_segments(self, params):
        n = params['n_segments']
        decay = np.exp(np.log(params['x_decay']) / n)
        x_cuts = np.cumsum(decay ** np.arange(n))
        x_cuts = [0, *x_cuts / x_cuts[-1]]
        y_cuts_scale = interp1d([0, 1 / 3, 2 / 3, 1], [1 / params['shell_ratio'], params['y_midpoint_first'],
            params['y_midpoint_second'], .1], fill_value='extrapolate')
        y_cuts = y_cuts_scale(x_cuts)
        x_anchors = lambda u, v: (u, (u + v) / 2, v)
        y_anchors = lambda u, v: (u, np.sqrt(u * v), v * params['shell_ratio'])
        segments = make_segments(x_cuts, y_cuts, x_anchors, y_anchors, params)
        height = uniform(.5, 1.)
        for obj in segments:
            z = read_co(obj).T[-1]
            write_attr_data(obj, 'ratio', 1 + np.where(z > 0, 0, height * z / params['y_length']))
        return segments, x_cuts

    def sample_params(self):
        x_length = uniform(1., 1.5)
        y_length = uniform(.15, .2)
        z_length = y_length * uniform(1, 1.2)
        y_expand = uniform(1.1, 1.3)
        y_midpoint_first = uniform(.85, .95)
        y_midpoint_second = uniform(.7, .8)
        noise_strength = uniform(.01, .02)
        noise_scale = log_uniform(10, 20)
        bottom_shift = uniform(.3, .5)
        bottom_cutoff = uniform(.2, .5)
        top_shift = 0
        top_cutoff = 1
        n_segments = np.random.randint(6, 10)
        x_decay = log_uniform(.2, .3)
        shell_ratio = uniform(1.05, 1.08)
        fin_x_length = uniform(.5, .8)
        return {
            'x_length': x_length,
            'y_length': y_length,
            'z_length': z_length,
            'y_expand': y_expand,
            'noise_strength': noise_strength,
            'noise_scale': noise_scale,
            'bottom_shift': bottom_shift,
            'bottom_cutoff': bottom_cutoff,
            'top_shift': top_shift,
            'top_cutoff': top_cutoff,
            'n_segments': n_segments,
            'x_decay': x_decay,
            'shell_ratio': shell_ratio,
            'y_midpoint_first': y_midpoint_first,
            'y_midpoint_second': y_midpoint_second,
            'fin_x_length': fin_x_length,
        }

    @staticmethod
    def animate_bones(arma, bones, params):
        bend_bones_lerp(arma, bones, params['tail_curl'], params['freq'])
