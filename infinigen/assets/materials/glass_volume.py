# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo, Lingjie Mei

from numpy.random import uniform

from infinigen.core.util.color import hsv2rgba
from infinigen.assets.materials import common
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler

def shader_glass_volume(nw: NodeWrangler, color=None, density=100.0, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler
    if color is None:
        if uniform(0, 1) < .3:
            color = 1, 1, 1, 1
        else:
            color = hsv2rgba(uniform(0, 1), uniform(.5, .9), uniform(.6, .9))


    volume_absorption = nw.new_node('ShaderNodeVolumeAbsorption',

    material_output = nw.new_node(Nodes.MaterialOutput,

def apply(obj, selection=None, **kwargs):
    common.apply(obj, shader_glass_volume, selection, **kwargs)
