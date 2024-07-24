# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick, Yiming Zuo, Lingjie Mei, Lahav Lipson


import colorsys

import gin
import mathutils
import numpy as np

from infinigen.core.util.math import int_hash


def hsv2rgba(hsv, *args):
    # hsv is a len-3 tuple or array
    c = mathutils.Color()
    if len(args) > 0:
        hsv = hsv, *args
    c.hsv = np.array([hsv[0] % 1, hsv[1], hsv[2]])
    rgba = list(c) + [1]
    return np.array(rgba)


def rgb2hsv(rgb, *args):
    # hsv is a len-3 tuple or array
    c = mathutils.Color()
    if len(args) > 0:
        rgb = rgb, *args
    c.r, c.g, c.b = rgb
    return np.array(c.hsv)


def srgb_to_linearrgb(c):
    if c < 0:
        return 0
    elif c < 0.04045:
        return c / 12.92
    else:
        return ((c + 0.055) / 1.055) ** 2.4


def hex2rgba(h, alpha=1):
    r = (h & 0xFF0000) >> 16
    g = (h & 0x00FF00) >> 8
    b = h & 0x0000FF
    return tuple([srgb_to_linearrgb(c / 0xFF) for c in (r, g, b)] + [alpha])

def hex2rgb(h, alpha=1):
    r = (h & 0xFF0000) >> 16
    g = (h & 0x00FF00) >> 8
    b = h & 0x0000FF
    return tuple([srgb_to_linearrgb(c / 0xFF) for c in (r, g, b)] + [alpha])


@gin.configurable
def random_color_mapping(color_tuple, scene_seed, hue_stddev):
    r, g, b, a = color_tuple
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    color_hash = int_hash((int(h * 1e3), scene_seed))
    h = np.random.RandomState(color_hash).normal(h, hue_stddev) % 1.0
    return colorsys.hsv_to_rgb(h, s, v) + (a,)
