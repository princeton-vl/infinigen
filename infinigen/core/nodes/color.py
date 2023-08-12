# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick, Yiming Zuo, Lingjie Mei, Lahav Lipson


from dataclasses import dataclass

import bpy
import mathutils

import numpy as np
import colorsys
import gin

from infinigen.core.util.math import int_hash

@dataclass
class ChannelScheme:
    args: list
    dist: str = 'uniform'
    clip: tuple = (0, 1)
    wrap: bool = False

    def sample(self):
        v = getattr(np.random, self.dist)(*self.args)
        if self.wrap:
            v = np.mod(v, 1)
        if self.clip is not None:
            v = np.clip(v, *self.clip)
        return v

U = lambda min, max, **kwargs: ChannelScheme([min, max], dist='uniform', **kwargs)
N = lambda m, std, **kwargs: ChannelScheme([m, std], dist='normal', **kwargs)

HSV_RANGES = {
    'petal': (
        N(0.95, 1.2, wrap=True), 
        U(0.2, 0.85), 
        U(0.2, 0.75)),
    'gem': (
        U(0, 1), 
        U(0.85, 0.85), 
        U(0.5, 1)),
    'greenery': (
        U(0.25, 0.33), 
        N(0.65, 0.03), 
        U(0.1, 0.45)
    ),
    'yellowish': (
        N(0.15, 0.005, wrap=True), 
        N(0.95, 0.02), 
        N(0.9, 0.02)
    ),
    'red': (
        N(0.0, 0.05, wrap=True), 
        N(0.9, 0.03), 
        N(0.6, 0.05)
    ),
    'pink': (
        N(0.88, 0.06, wrap=True), 
        N(0.6, 0.05), 
        N(0.8, 0.05)
    ),
    'white': (
        N(0.0, 0.06, wrap=True), 
        U(0.0, 0.2, clip=[0, 1]), 
        N(0.95, 0.02)
    ),
    'fog': (
        U(0, 1),
        U(0, 0.2),
        U(0.8, 1)
    ),
    'water': (
        U(0.2, 0.6),
        N(0.5, 0.1),
        U(0.7, 1)
    ),
    'darker_water': (
        U(0.2, 0.6),
        N(0.5, 0.1),
        U(0.2, 0.3)
    ),
    'under_water': (
        U(0.5, 0.7),
        U(0.7, 0.95),
        U(0.7, 1)
    ),
    'eye_schlera': (
        U(0.05, 0.15),
        U(0.2, 0.8),
        U(0.05, 0.5)
    ),
    'eye_pupil': (
        U(0, 1),
        U(0.1, 0.9),
        U(0.1, 0.9)
    ),
    'beak': (
        U(0, 0.13),
        U(0, 0.9),
        U(0.1, 0.6)
    ),
    'fur': (
        U(0, 0.11),
        U(0.5, 0.95),
        U(0.02, 0.9)
    ),
    'pine_needle': (
        N(0.05, 0.02, wrap=True),
        U(0.5, 0.93),
        U(0.045, 0.4),
    ),
    'wet_sand': (
        U(0.05, 0.1),
        U(0.65, 0.7),
        U(0.05, 0.15),
    ),
    'dry_sand': (
        U(0.05, 0.1),
        U(0.65, 0.7),
        U(0.15, 0.25),
    ),
    #'dirt': ('uniform', [], []),
    #'rock': ('uniform', [], []),
    #'creature_fur': ('normal', [0.89, 0.6, 0.2], []),
    #'creature_scale': ('uniform', [], []),
    #'wood': ('uniform', [], []),
}

def color_category(name):
    if not name in HSV_RANGES:
        raise ValueError(f'color_category did not recognize {name=}, options are {HSV_RANGES.keys()=}')
    schemes = HSV_RANGES[name]
    assert len(schemes) == 3
    hsv = [s.sample() for s in schemes]
    return hsv2rgba(hsv)

def hsv2rgba(hsv):
    # hsv is a len-3 tuple or array
    c = mathutils.Color()
    c.hsv = list(hsv)
    rgba = list(c) + [1]
    return np.array(rgba)

@gin.configurable
def random_color_mapping(color_tuple, scene_seed, hue_stddev):
    r,g,b,a = color_tuple
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    color_hash = int_hash((int(h*1e3), scene_seed))
    h = np.random.RandomState(color_hash).normal(h, hue_stddev) % 1.0
    return colorsys.hsv_to_rgb(h, s, v) + (a,)
