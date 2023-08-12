# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import numpy as np
from numpy.random import uniform

from infinigen.assets.creatures.util.creature import Part, PartFactory
from infinigen.assets.utils.draw import leaf
from infinigen.core.surface import write_attr_data
from infinigen.core.util import blender as butil


class CrustaceanFinFactory(PartFactory):
    tags = ['body']

    def make_part(self, params) -> Part:
        x_length, y_length, x_tip, y_mid = map(params.get, ['x_length', 'y_length', 'x_tip', 'y_mid'])
        x_anchors = 0, x_tip / 2, x_tip, 1
        y_anchors = 0, y_mid, 1, 0
        obj = leaf(np.array(x_anchors) * x_length, np.array(y_anchors) * y_length)
        butil.modify_mesh(obj, 'SOLIDIFY', thickness=.01, offset=0.)
        write_attr_data(obj, 'ratio', np.ones(len(obj.data.vertices)))
        skeleton = np.zeros((2, 3))
        skeleton[1, 0] = x_length
        return Part(skeleton, obj)

    def sample_params(self):
        x_length = uniform(.15, .3)
        y_length = x_length * uniform(.3, .4)
        x_tip = uniform(.7, .8)
        y_mid = uniform(.6, .8)
        return {'x_length': x_length, 'y_length': y_length, 'x_tip': x_tip, 'y_mid': y_mid}
