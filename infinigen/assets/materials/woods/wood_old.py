# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Mingzhe Wang, Lingjie Mei

import math as ma

import numpy as np
from numpy.random import uniform

from infinigen.assets.materials import common
from infinigen.assets.materials.utils.surface_utils import (
    sample_color,
    sample_range,
    sample_ratio,
)
from infinigen.core import surface
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.util.random import log_uniform


def shader_wood_old(nw: NodeWrangler, scale=1, offset=None, rotation=None, **kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    texture_coordinate_1 = nw.new_node(Nodes.TextureCoord)

    rotation = (
        uniform(0, ma.pi * 2, 3)
        if rotation is None
        else surface.eval_argument(nw, rotation)
    )
    mapping_2 = nw.new_node(
        Nodes.Mapping,
        input_kwargs={
            "Vector": texture_coordinate_1.outputs["Object"],
            "Location": surface.eval_argument(nw, offset),
            "Rotation": rotation,
        },
    )

    mapping_1 = nw.new_node(
        Nodes.Mapping,
        input_kwargs={
            "Vector": mapping_2,
            "Scale": np.array(
                [log_uniform(2, 4), log_uniform(8, 16), log_uniform(2, 4)]
            )
            * scale,
        },
    )

    musgrave_texture_2 = nw.new_node(
        Nodes.MusgraveTexture,
        input_kwargs={"Vector": mapping_1, "Scale": 2.0},
        attrs={"musgrave_dimensions": "4D"},
    )
    musgrave_texture_2.inputs["W"].default_value = sample_range(0, 5)
    musgrave_texture_2.inputs["Scale"].default_value = sample_ratio(2.0, 3 / 4, 4 / 3)

    noise_texture_1 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={"Vector": musgrave_texture_2, "W": 0.7, "Scale": 10.0},
        attrs={"noise_dimensions": "4D"},
    )
    noise_texture_1.inputs["W"].default_value = sample_range(0, 5)
    noise_texture_1.inputs["Scale"].default_value = sample_ratio(5, 0.5, 2)

    colorramp_2 = nw.new_node(
        Nodes.ColorRamp, input_kwargs={"Fac": noise_texture_1.outputs["Fac"]}
    )
    colorramp_2.color_ramp.elements.new(0)
    colorramp_2.color_ramp.elements[0].position = 0.1727
    colorramp_2.color_ramp.elements[0].color = (0.1567, 0.0162, 0.0017, 1.0)
    colorramp_2.color_ramp.elements[1].position = 0.4364
    colorramp_2.color_ramp.elements[1].color = (0.2908, 0.1007, 0.0148, 1.0)
    colorramp_2.color_ramp.elements[2].position = 0.5864
    colorramp_2.color_ramp.elements[2].color = (0.0814, 0.0344, 0.0125, 1.0)
    colorramp_2.color_ramp.elements[0].position += sample_range(-0.05, 0.05)
    colorramp_2.color_ramp.elements[1].position += sample_range(-0.1, 0.1)
    colorramp_2.color_ramp.elements[2].position += sample_range(-0.05, 0.05)
    for e in colorramp_2.color_ramp.elements:
        sample_color(e.color, offset=0.04)

    colorramp_4 = nw.new_node(
        Nodes.ColorRamp, input_kwargs={"Fac": noise_texture_1.outputs["Fac"]}
    )
    colorramp_4.color_ramp.elements[0].position = 0.0
    colorramp_4.color_ramp.elements[0].color = (0.4855, 0.4855, 0.4855, 1.0)
    colorramp_4.color_ramp.elements[1].position = 1.0
    colorramp_4.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    principled_bsdf_1 = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": colorramp_2.outputs["Color"],
            "Roughness": colorramp_4.outputs["Color"],
        },
        attrs={"subsurface_method": "BURLEY"},
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput, input_kwargs={"Surface": principled_bsdf_1}
    )


def apply(obj, selection=None, scale=1, **kwargs):
    common.apply(obj, shader_wood_old, selection, scale=scale, **kwargs)
