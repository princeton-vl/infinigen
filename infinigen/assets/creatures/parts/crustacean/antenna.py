# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import numpy as np
from numpy.random import uniform

from infinigen.assets.creatures.util.animation.driver_repeated import bend_bones_lerp
from infinigen.assets.creatures.util.creature import Part
from infinigen.assets.creatures.util.genome import Joint
from infinigen.assets.creatures.parts.crustacean.leg import CrabLegFactory
from infinigen.assets.utils.decorate import displace_vertices, join_objects
from infinigen.assets.utils.misc import log_uniform


class LobsterAntennaFactory(CrabLegFactory):
    tag = ['claw']

    def make_part(self, params) -> Part:
        x_length, z_length = params['x_length'], params['z_length']
        segments, x_cuts = self.make_segments(params)
        displace_vertices(segments[-1], lambda x, y, z: (
            0, 0, params['antenna_bend'] * (x / x_length - x_cuts[-2]) ** 2 * params['z_length']))
        obj = join_objects(segments)

        skeleton = np.zeros((2, 3))
        skeleton[1, 0] = x_length
        joints = {x: Joint(rest=(0, 0, 0)) for x in x_cuts[1:]}
        return Part(skeleton, obj, joints=joints)

    @staticmethod
    def animate_bones(arma, bones, params):
        bend_bones_lerp(arma, bones, params['antenna_curl'], params['freq'])

    def sample_params(self):
        y_length = uniform(.01, .015)
        z_length = y_length * log_uniform(1, 1.2)
        x_mid_first = uniform(.1, .15)
        x_mid_second = uniform(.25, .3)
        antenna_bend = uniform(2, 5)
        return {**super().sample_params(),
            'y_length': y_length,
            'z_length': z_length,
            'x_mid_first': x_mid_first,
            'x_mid_second': x_mid_second,
            'antenna_bend': antenna_bend,
        }


class SpinyLobsterAntennaFactory(LobsterAntennaFactory):
    tag = ['claw']

    def sample_params(self):
        y_length = uniform(.05, .08)
        z_length = y_length * log_uniform(1, 1.2)
        return {**super().sample_params(), 'y_length': y_length, 'z_length': z_length}
