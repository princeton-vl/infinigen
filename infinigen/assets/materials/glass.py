# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import colorsys

from numpy.random import uniform

from infinigen.core.util.color import hsv2rgba
from infinigen.assets.materials import common
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler


    if color is None:


def apply(obj, selection=None, clear=False, **kwargs):
    color = get_glass_color(clear)
    common.apply(obj, shader_glass, selection, color, **kwargs)

def get_glass_color(clear):
    if uniform(0, 1) < .5:
        color = 1, 1, 1, 1
    else:
        color = hsv2rgba(uniform(0, 1), .01 if clear else uniform(.05, .25), 1)
    return color
