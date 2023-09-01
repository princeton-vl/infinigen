# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan
# Acknowledgement: This file draws inspiration from https://www.youtube.com/watch?v=zyIJQHlFQs0 by PolyFjord

import bpy
import mathutils
from numpy.random import uniform, normal as N, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface
from infinigen.core.util.random import random_color_neighbour



def blackbody_shader(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    volume_info = nw.new_node("ShaderNodeVolumeInfo")

    colorramp = nw.new_node(
        Nodes.ColorRamp, input_kwargs={"Fac": volume_info.outputs["Flame"]}
    )
    colorramp.color_ramp.interpolation = "B_SPLINE"
    colorramp.color_ramp.elements.new(0)
    colorramp.color_ramp.elements[0].position = 0.2455 + 0.01 * N()
    colorramp.color_ramp.elements[0].color = [0.0000, 0.0000, 0.0000, 1.0000]
    colorramp.color_ramp.elements[1].position = 0.2818 + 0.01 * N()
    colorramp.color_ramp.elements[1].color = [1.0000, 1.0000, 1.0000, 1.0000]
    colorramp.color_ramp.elements[2].position = 0.5864 + 0.01 * N()
    colorramp.color_ramp.elements[2].color = [0.0000, 0.0000, 0.0000, 1.0000]

    colorramp_1 = nw.new_node(
        Nodes.ColorRamp, input_kwargs={"Fac": volume_info.outputs["Density"]}
    )
    colorramp_1.color_ramp.interpolation = "B_SPLINE"
    colorramp_1.color_ramp.elements[0].position = 0.3636 + 0.01 * N()
    colorramp_1.color_ramp.elements[0].color = [1.0000, 1.0000, 1.0000, 1.0000]
    colorramp_1.color_ramp.elements[1].position = 0.6409 + 0.01 * N()
    colorramp_1.color_ramp.elements[1].color = [0.0000, 0.0000, 0.0000, 1.0000]

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: colorramp.outputs["Color"], 1: colorramp_1.outputs["Color"]},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply, 1: 8626.6650 + 20 * N()},
        attrs={"operation": "MULTIPLY"},
    )

    principled_volume = nw.new_node(
        Nodes.PrincipledVolume,
        input_kwargs={
            "Color": random_color_neighbour((0.3568, 0.3568, 0.3568, 1.0000),0.1,0.1,0.1),
            "Density": 15.0000 + N(),
            "Blackbody Intensity": multiply_1,
        },
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Volume": principled_volume},
        attrs={"is_active_output": True},
    )


def apply(obj, selection=None, **kwargs):
    surface.add_material(obj, blackbody_shader, selection=selection)