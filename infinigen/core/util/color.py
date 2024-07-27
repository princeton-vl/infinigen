# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick, Yiming Zuo, Lingjie Mei, Lahav Lipson


import colorsys
from dataclasses import dataclass

import gin
import mathutils
import numpy as np

from infinigen.core.util.math import int_hash


@dataclass
class ChannelScheme:
    args: list
    dist: str = "uniform"
    clip: tuple = (0, 1)
    wrap: bool = False

    def sample(self):
        if self.dist == "log_uniform":
            v = np.exp(np.random.uniform(np.log(self.args[0]), np.log(self.args[1])))
        else:
            v = getattr(np.random, self.dist)(*self.args)
        if self.wrap:
            v = np.mod(v, 1)
        if self.clip is not None:
            v = np.clip(v, *self.clip)
        return v


def U(min, max, **kwargs):
    return ChannelScheme([min, max], dist="uniform", **kwargs)


def LU(min, max, **kwargs):
    return ChannelScheme([min, max], dist="log_uniform", **kwargs)


def N(m, std, **kwargs):
    return ChannelScheme([m, std], dist="normal", **kwargs)


HSV_RANGES = {
    "petal": (N(0.95, 1.2, wrap=True), U(0.2, 0.85), U(0.2, 0.75)),
    "gem": (U(0, 1), U(0.85, 0.85), U(0.5, 1)),
    "greenery": (U(0.25, 0.33), N(0.65, 0.03), U(0.1, 0.45)),
    "yellowish": (N(0.15, 0.005, wrap=True), N(0.95, 0.02), N(0.9, 0.02)),
    "red": (N(0.0, 0.05, wrap=True), N(0.9, 0.03), N(0.6, 0.05)),
    "pink": (N(0.88, 0.06, wrap=True), N(0.6, 0.05), N(0.8, 0.05)),
    "white": (N(0.0, 0.06, wrap=True), U(0.0, 0.2, clip=[0, 1]), N(0.95, 0.02)),
    "fog": (U(0, 1), U(0, 0.2), U(0.8, 1)),
    "water": (U(0.2, 0.6), N(0.5, 0.1), U(0.7, 1)),
    "darker_water": (U(0.2, 0.6), N(0.5, 0.1), U(0.2, 0.3)),
    "under_water": (U(0.5, 0.7), U(0.7, 0.95), U(0.7, 1)),
    "eye_schlera": (U(0.05, 0.15), U(0.2, 0.8), U(0.05, 0.5)),
    "eye_pupil": (U(0, 1), U(0.1, 0.9), U(0.1, 0.9)),
    "beak": (U(0, 0.13), U(0, 0.9), U(0.1, 0.6)),
    "fur": (U(0, 0.11), U(0.5, 0.95), U(0.02, 0.9)),
    "pine_needle": (
        N(0.05, 0.02, wrap=True),
        U(0.5, 0.93),
        U(0.045, 0.4),
    ),
    "wet_sand": (
        U(0.05, 0.1),
        U(0.65, 0.7),
        U(0.05, 0.15),
    ),
    "dry_sand": (
        U(0.05, 0.1),
        U(0.65, 0.7),
        U(0.15, 0.25),
    ),
    "leather": (
        U(0.04, 0.07),
        U(0.80, 1.0),
        U(0.1, 0.6),
    ),
    "concrete": (
        U(0.0, 1.0),
        U(0.02, 0.12),
        LU(0.1, 0.9),
    ),
    "textile": (
        U(0, 1),
        U(0.15, 0.7),
        U(0.1, 0.3),
    ),
    "fabric": (U(0, 1), U(0.3, 0.8), U(0.6, 0.9)),
    # 'dirt': ('uniform', [], []),
    # 'rock': ('uniform', [], []),
    # 'creature_fur': ('normal', [0.89, 0.6, 0.2], []),
    # 'creature_scale': ('uniform', [], []),
    # 'wood': ('uniform', [], []),
}


def color_category(name):
    if name not in HSV_RANGES:
        raise ValueError(
            f"color_category did not recognize {name=}, options are {HSV_RANGES.keys()=}"
        )
    schemes = HSV_RANGES[name]
    assert len(schemes) == 3
    hsv = [s.sample() for s in schemes]
    return hsv2rgba(hsv)


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


@gin.configurable
def random_color_mapping(color_tuple, scene_seed, hue_stddev):
    r, g, b, a = color_tuple
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    color_hash = int_hash((int(h * 1e3), scene_seed))
    h = np.random.RandomState(color_hash).normal(h, hue_stddev) % 1.0
    return colorsys.hsv_to_rgb(h, s, v) + (a,)
