# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from collections.abc import Iterable

import numpy as np
from numpy.random import uniform

from infinigen.core.util.color import hsv2rgba, rgb2hsv
from infinigen.core.util.random import random_general as rg, log_uniform
from . import (
    brushed_metal, galvanized_metal, grained_and_polished_metal, hammered_metal,
    metal_basic,
)
from .. import common
from ..bark_random import hex_to_rgb


def apply(obj, selection=None, metal_color=None, **kwargs):
    color = sample_metal_color(metal_color)
    shader = get_shader()
    common.apply(obj, shader, selection, base_color=color, **kwargs)


def get_shader():
    return np.random.choice(
        [brushed_metal.shader_brushed_metal, galvanized_metal.shader_galvanized_metal,
            grained_and_polished_metal.shader_grained_metal,
            hammered_metal.shader_hammered_metal]
    )


plain_colors = 'weighted_choice', (.5, 0xfdd017), (1, 0xc0c0c0), (1, 0x8c7853), (.5, 0xb87333), (.5, 0xb5a642), (
    1, 0xbdbaae), (1, 0xa9acb6), (1, 0xb6afa9)
natural_colors = 'weighted_choice', (1, 0xc0c0c0), (1, 0x8c7853), (1, 0xbdbaae), (1, 0xa9acb6), (1, 0xb6afa9)


def sample_metal_color(metal_color=None, **kwargs):
    match metal_color:
        case np.ndarray():
            return metal_color
        case 'plain':
            h, s, v = rgb2hsv(hex_to_rgb(rg(plain_colors))[:-1])
            return hsv2rgba(h + uniform(-.1, .1), s + uniform(-.1, .1), v * log_uniform(.5, .2))
        case 'natural':
            h, s, v = rgb2hsv(hex_to_rgb(rg(natural_colors))[:-1])
            return hsv2rgba(h + uniform(-.1, .1), s + uniform(-.1, .1), v * log_uniform(.5, .2))
        case 'bw':
            return hsv2rgba(uniform(0, 1), uniform(.0, .2), log_uniform(.01, .2))
        case 'bw+natural':
            return sample_metal_color('bw') if uniform() < .5 else sample_metal_color('natural')
        case _:
            if uniform() < .2:
                return sample_metal_color('natural')
            return hsv2rgba(uniform(0, 1), uniform(.3, .6), log_uniform(.02, .5))
