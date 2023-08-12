# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import numpy as np
from numpy.random import uniform

from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.util.math import FixedSeed, int_hash


def scatter_lower(nw: NodeWrangler, height_range=(.5, 2), fill_range=(.0, .8), noise_scale=.4):
    height = uniform(*height_range)
    middle = height * uniform(*fill_range)
    lower = nw.bernoulli(nw.build_float_curve(nw.separate(nw.new_node(Nodes.InputPosition))[-1],
                                              [(0, 1), (middle, 1), (height, 0)]))
    compare = nw.compare('GREATER_THAN', lower,
                         nw.new_node(Nodes.NoiseTexture, input_kwargs={'Scale': noise_scale}), )
    return compare


def scatter_upward(nw: NodeWrangler, normal_thresh=np.pi * .75, noise_scale=.4, noise_thresh=.3):
    compare = nw.compare('GREATER_THAN', nw.new_node(Nodes.NoiseTexture, input_kwargs={'Scale': noise_scale}),
                         noise_thresh)
    upward = nw.compare_direction('LESS_THAN', nw.new_node(Nodes.InputNormal), (0, 0, 1), normal_thresh)
    return nw.boolean_math('AND', compare, upward)
