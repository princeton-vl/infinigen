# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Hongyu Wen

from numpy.random import uniform as U

from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.util.color import hsv2rgba

# used in ceiling lights and tv


def shader_black(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    color = hsv2rgba(U(0.45, 0.55), U(0, 0.1), U(0, 1))
    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF, input_kwargs={"Base Color": color}
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": principled_bsdf},
        attrs={"is_active_output": True},
    )
