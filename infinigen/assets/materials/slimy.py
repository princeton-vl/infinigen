# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Mingzhe Wang
# Acknowledgement: This file draws inspiration from https://blender.stackexchange.com/questions/111219/slime-effect-material

import os, sys
import numpy as np
import math as ma
from infinigen.assets.materials.utils.surface_utils import clip, sample_range, sample_ratio, sample_color, geo_voronoi_noise
import bpy
import mathutils
from numpy.random import uniform, normal
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core import surface
import random

def shader_slimy(nw, rand=False, **input_kwargs):
    texture_coordinate = nw.new_node(Nodes.TextureCoord)
    
    value = nw.new_node(Nodes.Value)
    value.outputs["Value"].default_value = input_kwargs['scale'] if 'scale' in input_kwargs else 0.5

    mapping = nw.new_node(Nodes.Mapping,
        input_kwargs={'Vector': texture_coordinate.outputs["Generated"], 'Scale': value})
    
    musgrave_texture = nw.new_node(Nodes.MusgraveTexture,
        input_kwargs={'Vector': mapping, 'Scale': 6.4},
        attrs={'musgrave_dimensions': '4D'})
    if rand:
        musgrave_texture.inputs["W"].default_value = sample_range(-5, 5)

    colorramp_1 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': musgrave_texture})
    colorramp_1.color_ramp.elements[0].position = 0.0399
    colorramp_1.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_1.color_ramp.elements[1].position = 0.2464
    colorramp_1.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    noise_texture_2 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': mapping, 'Scale': 7.6, 'Distortion': 3.0},
        attrs={'noise_dimensions': '4D'})
    if rand:
        noise_texture_2.inputs["W"].default_value = sample_range(-5, 5)
    
    colorramp_4 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': noise_texture_2.outputs["Fac"]})
    colorramp_4.color_ramp.elements[0].position = 0.3554
    colorramp_4.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_4.color_ramp.elements[1].position = 1.0
    colorramp_4.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    mix_1 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Color1': colorramp_1.outputs["Color"], 'Color2': colorramp_4.outputs["Color"]})
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': (0.6605, 0.0279, 0.0359, 1.0), 'Subsurface': 0.2, 'Subsurface Color': (0.4621, 0.0213, 0.0265, 1.0), 'Specular': 0.8591, 'Roughness': mix_1})
    if rand:
        sample_color(principled_bsdf.inputs['Base Color'].default_value)
        sample_color(principled_bsdf.inputs['Subsurface Color'].default_value)

    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

def geo_slimy(nw, rand=False, **input_kwargs):

    group_input = nw.new_node(Nodes.GroupInput)

    position = nw.new_node(Nodes.InputPosition)
    
    value = nw.new_node(Nodes.Value)
    value.outputs["Value"].default_value = input_kwargs['scale'] if 'scale' in input_kwargs else 0.2
    
    vector_math = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: position, 1: value},
        attrs={'operation': 'MULTIPLY'})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': vector_math.outputs["Vector"], 'Distortion': 2.0},
        attrs={'noise_dimensions': '4D'})
    if rand:
        noise_texture.inputs['W'].default_value = sample_range(-5, 5)
        noise_texture.inputs['Scale'].default_value = sample_range(3, 7)
        noise_texture.inputs['Distortion'].default_value = sample_range(1, 4)
    
    wave_texture = nw.new_node(Nodes.WaveTexture,
        input_kwargs={'Vector': noise_texture.outputs["Fac"]})
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': wave_texture.outputs["Fac"]})
    colorramp.color_ramp.elements[0].position = 0.0
    colorramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp.color_ramp.elements[1].position = 1.0
    colorramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    math = nw.new_node(Nodes.Math,
        input_kwargs={0: 1.0, 1: colorramp.outputs["Color"]},
        attrs={'operation': 'SUBTRACT'})
    
    normal = nw.new_node(Nodes.InputNormal)
    
    vector_math_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: math, 1: normal},
        attrs={'operation': 'MULTIPLY'})
    
    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs["Value"].default_value = uniform(0.005, 0.02) if rand else 0.015

    vector_math_2 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: vector_math_1.outputs["Vector"], 1: value_1},
        attrs={'operation': 'MULTIPLY'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input, 'Offset': vector_math_2.outputs["Vector"]})
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={
            'Geometry': set_position, 
            1: math},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': capture_attribute.outputs["Geometry"], 'Attribute': capture_attribute.outputs["Attribute"]})

def apply(obj, geo_kwargs=None, shader_kwargs=None, **kwargs):
    surface.add_geomod(obj, geo_slimy, apply=False, input_kwargs=geo_kwargs, attributes=['offset'])
    surface.add_material(obj, shader_slimy, reuse=False, input_kwargs=shader_kwargs)
