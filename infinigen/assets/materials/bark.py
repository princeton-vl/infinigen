# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Mingzhe Wang

import os, sys
import numpy as np
from numpy.random import uniform as U, normal as N
import math as ma
from infinigen.assets.materials.utils.surface_utils import clip, sample_range, sample_ratio, sample_color, geo_voronoi_noise
import bpy
import mathutils
from numpy.random import uniform, normal
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core import surface
import random


def shader_bark(nw, rand=False, **input_kwargs):

    texture_coordinate = nw.new_node(Nodes.TextureCoord)
    
    mapping = nw.new_node(Nodes.Mapping,
        input_kwargs={'Vector': texture_coordinate.outputs["Object"]})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': mapping, 'Detail': 16.0, 'Roughness': 0.62})
    if rand:
        sample_max = input_kwargs['noise_scale_max'] if 'noise_scale_max' in input_kwargs else 3
        sample_min = input_kwargs['noise_scale_min'] if 'noise_scale_min' in input_kwargs else 1/sample_max
        noise_texture.inputs["Scale"].default_value = sample_ratio(noise_texture.inputs["Scale"].default_value, sample_min, sample_max)

    colorramp_1 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': noise_texture.outputs["Fac"]})
    colorramp_1.color_ramp.elements[0].position = 0.627
    colorramp_1.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_1.color_ramp.elements[1].position = 0.63
    colorramp_1.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    attribute = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'offset'})
    
    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Color1': noise_texture.outputs["Fac"], 'Color2': attribute.outputs["Color"]},
        attrs={'blend_type': 'MULTIPLY'})
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': mix})
    colorramp.color_ramp.elements[0].position = 0.0
    colorramp.color_ramp.elements[0].color = (0.02, 0.0091, 0.0016, 1.0)
    colorramp.color_ramp.elements[1].position = 1.0
    colorramp.color_ramp.elements[1].color = (0.2243, 0.1341, 0.1001, 1.0)
    for e in colorramp.color_ramp.elements:
        sample_color(e.color)
        #print(e.color[0], e.color[1], e.color[2])

    mix_1 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp_1.outputs["Color"], 'Color1': colorramp.outputs["Color"], 'Color2': (0.0897, 0.052, 0.0149, 1.0)})
    if rand:
        for i in range(3):
            mix_1.inputs[7].default_value[i] = (colorramp.color_ramp.elements[0].color[i] + colorramp.color_ramp.elements[1].color[i]) / 2

    colorramp_2 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': noise_texture.outputs["Fac"]})
    colorramp_2.color_ramp.elements[0].position = 0.0
    colorramp_2.color_ramp.elements[0].color = (0.5173, 0.5173, 0.5173, 1.0)
    colorramp_2.color_ramp.elements[1].position = 1.0
    colorramp_2.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': mix_1, 'Roughness': colorramp_2.outputs["Color"]},
        attrs={'subsurface_method': 'BURLEY'})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

def geo_bark(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
    
    position = nw.new_node(Nodes.InputPosition)
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 5.0000
    
    multiply = nw.new_node(Nodes.VectorMath, input_kwargs={0: position, 1: value}, attrs={'operation': 'MULTIPLY'})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': multiply.outputs["Vector"]})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture, 
        input_kwargs={'Scale': N(10, 2), 'W': U(-10, 10)},
        attrs={'noise_dimensions': '4D'},
    )
    
    subtract = nw.new_node(Nodes.Math, input_kwargs={0: noise_texture.outputs["Fac"]}, attrs={'operation': 'SUBTRACT'})
    
    multiply_1 = nw.new_node(Nodes.Math, input_kwargs={0: subtract, 1: 3.0000}, attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz.outputs["Y"], 1: multiply_1})
    
    multiply_2 = nw.new_node(Nodes.Math, input_kwargs={0: add, 1: 1.0000}, attrs={'operation': 'MULTIPLY'})
    
    fract = nw.new_node(Nodes.Math, input_kwargs={0: multiply_2}, attrs={'operation': 'FRACT'})
    
    subtract_1 = nw.new_node(Nodes.Math, input_kwargs={0: fract}, attrs={'operation': 'SUBTRACT'})
    
    multiply_3 = nw.new_node(Nodes.Math, input_kwargs={0: subtract_1, 1: subtract_1}, attrs={'operation': 'MULTIPLY'})
    
    multiply_4 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_3, 1: 4.0000}, attrs={'operation': 'MULTIPLY'})
    
    normal = nw.new_node(Nodes.InputNormal)
    
    multiply_5 = nw.new_node(Nodes.VectorMath, input_kwargs={0: multiply_4, 1: normal}, attrs={'operation': 'MULTIPLY'})
    
    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = 0.0100
    
    multiply_6 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: multiply_5.outputs["Vector"], 1: value_1},
        attrs={'operation': 'MULTIPLY'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Offset': multiply_6.outputs["Vector"]})
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': set_position, 1: multiply_4},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': capture_attribute.outputs["Geometry"], 'Attribute': capture_attribute.outputs["Attribute"]},
        attrs={'is_active_output': True})

def apply(obj, geo_kwargs=None, shader_kwargs=None, **kwargs):
    surface.add_geomod(obj, geo_bark, apply=False, input_kwargs=geo_kwargs, attributes=['offset'])
    surface.add_material(obj, shader_bark, reuse=False, input_kwargs=shader_kwargs)