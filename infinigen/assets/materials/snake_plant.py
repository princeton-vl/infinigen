# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Beining Han
# Acknowledgements: This file draws inspiration from https://www.youtube.com/watch?v=sHr8LjfX09c

import bpy
import bpy
import mathutils
from numpy.random import uniform as U, normal as N, randint, choice
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import hsv2rgba
from infinigen.core import surface
import numpy as np
import colorsys


def shader_snake_plant(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    r = 2.0 * np.random.choice([0, 1], p=(0.4, 0.6))
    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: 0.001, 1: r},
                           attrs={'operation': 'MULTIPLY'})

    colorramp_1 = nw.new_node(Nodes.ColorRamp,
                              input_kwargs={'Fac': multiply})
    e = U(0.34, 0.42)
    colorramp_1.color_ramp.elements[0].position = e
    colorramp_1.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    colorramp_1.color_ramp.elements[1].position = e + 0.02
    colorramp_1.color_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)

    texture_coordinate_1 = nw.new_node(Nodes.TextureCoord)

    mapping_1 = nw.new_node(Nodes.Mapping,
                            input_kwargs={'Vector': texture_coordinate_1.outputs["Object"]})

    noise_texture = nw.new_node(Nodes.NoiseTexture,
                                input_kwargs={'Vector': mapping_1, 'Scale': U(0.2, 1.0), 'Roughness': 1.0})

    multiply_1 = nw.new_node(Nodes.VectorMath,
                             input_kwargs={0: noise_texture.outputs["Fac"], 1: (1.0, 1.0, 0.6)},
                             attrs={'operation': 'MULTIPLY'})

    add = nw.new_node(Nodes.VectorMath,
                      input_kwargs={0: multiply_1.outputs["Vector"], 1: mapping_1})

    wave_texture = nw.new_node(Nodes.WaveTexture,
                               input_kwargs={'Vector': add.outputs["Vector"], 'Scale': U(1.0, 2.5),
                                             'Distortion': U(2.0, 4.5),
                                             'Detail Scale': U(2.0, 8.0), 'Detail Roughness': 2.0},
                               attrs={'bands_direction': 'Z'})

    w = U(0.2, 0.7)
    greater_than = nw.new_node(Nodes.Math,
                               input_kwargs={0: wave_texture.outputs["Fac"], 1: w},
                               attrs={'operation': 'GREATER_THAN'})

    mapping_2 = nw.new_node(Nodes.Mapping,
                            input_kwargs={'Vector': texture_coordinate_1.outputs["Object"], 'Scale': (7.0, 7.0, 0.05)})

    noise_texture_1 = nw.new_node(Nodes.NoiseTexture,
                                  input_kwargs={'Vector': mapping_2, 'Scale': U(20.0, 40.0)})

    multiply_2 = nw.new_node(Nodes.Math,
                             input_kwargs={0: greater_than, 1: noise_texture_1.outputs["Fac"]},
                             attrs={'operation': 'MULTIPLY'})

    colorramp_8 = nw.new_node(Nodes.ColorRamp,
                              input_kwargs={'Fac': multiply_2})
    colorramp_8.color_ramp.elements[0].position = 0.2318
    colorramp_8.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_8.color_ramp.elements[1].position = U(0.55, 0.75)
    colorramp_8.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    r = 0.6 + (w - 0.2) * 0.6
    colorramp_4 = nw.new_node(Nodes.ColorRamp,
                              input_kwargs={'Fac': wave_texture.outputs["Fac"]})
    colorramp_4.color_ramp.elements[0].position = 0.6 + (w - 0.2) * 0.6
    colorramp_4.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_4.color_ramp.elements[1].position = np.minimum(1.0, r + U(0.02, 0.15))
    colorramp_4.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    mix_1 = nw.new_node(Nodes.MixRGB,
                        input_kwargs={'Fac': U(0.8, 1.0), 'Color1': colorramp_8.outputs["Color"],
                                      'Color2': colorramp_4.outputs["Color"]},
                        attrs={'blend_type': 'ADD'})

    c = [U(0.28, 0.36), U(0.35, 0.80), U(0.20, 0.45)]
    colorramp_3 = nw.new_node(Nodes.ColorRamp,
                              input_kwargs={'Fac': mix_1})
    colorramp_3.color_ramp.elements[0].position = 0.0
    colorramp_3.color_ramp.elements[0].color = hsv2rgba(c)
    colorramp_3.color_ramp.elements[1].position = 1.0
    c[2] = U(0.03, 0.07)
    c[1] = U(0.5, 0.8)
    c[0] += N(0, 0.015)
    colorramp_3.color_ramp.elements[1].color = hsv2rgba(c)

    mix = nw.new_node(Nodes.MixRGB,
                      input_kwargs={'Fac': colorramp_1.outputs["Color"], 'Color1': (*colorsys.hsv_to_rgb(
                          *[U(0.16, 0.23), U(0.8, 0.95), U(0.35, 0.8)]), 1.0),
                                    'Color2': colorramp_3.outputs["Color"]})

    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
                                  input_kwargs={'Base Color': mix, 'Roughness': U(8.0, 15.0), 'Clearcoat Roughness': 0.0})

    material_output = nw.new_node(Nodes.MaterialOutput,
                                  input_kwargs={'Surface': principled_bsdf})


def apply(obj, selection=None, **kwargs):
    surface.add_material(obj, shader_snake_plant, selection=selection)
