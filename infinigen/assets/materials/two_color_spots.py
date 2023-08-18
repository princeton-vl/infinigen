# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Mingzhe Wang


import os, sys
import numpy as np
import math as ma
from infinigen.assets.materials.utils.surface_utils import clip, sample_range, sample_ratio, sample_color, geo_voronoi_noise
import bpy
import mathutils
from numpy.random import uniform, normal
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core import surface

def shader_two_color_spots(nw, rand=True, **input_kwargs):
    attribute = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'offset'})
    
    mix_2 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': attribute.outputs["Fac"], 'Color1': (1.0, 0.2397, 0.0028, 1.0), 'Color2': (0.4915, 0.4636, 0.3855, 1.0)})
    if rand:
        sample_color(mix_2.inputs[6].default_value)
        sample_color(mix_2.inputs[7].default_value)

    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': mix_2},
        attrs={'subsurface_method': 'BURLEY'})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

def geo_two_color_spots(nw, rand=True, **input_kwargs):

    group_input = nw.new_node(Nodes.GroupInput)

    position = nw.new_node(Nodes.InputPosition)
    
    scale = nw.new_node(Nodes.Value)
    scale.outputs["Value"].default_value = input_kwargs['scale'] if 'scale' in input_kwargs else 0.2

    vector_math = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: position, 1: scale},
        attrs={'operation': 'MULTIPLY'})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': vector_math.outputs["Vector"], 'Scale': 10.0, 'Detail': 10.0})
    
    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Color1': noise_texture.outputs["Color"], 'Color2': vector_math.outputs["Vector"]})
    if rand:
        mix.inputs["Factor"].default_value = sample_range(0.5, 0.9)

    voronoi_texture = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': mix},
        attrs={'voronoi_dimensions': '4D'})
    if rand:
        voronoi_texture.inputs["W"].default_value = sample_range(-5, 5)
        voronoi_texture.inputs['Scale'].default_value = sample_range(5, 20)
    
    math_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: voronoi_texture.outputs["Distance"], 1: voronoi_texture.outputs["Distance"]},
        attrs={'operation': 'MULTIPLY'})
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': math_1})
    colorramp.color_ramp.elements.new(1)
    colorramp.color_ramp.elements[0].position = 0.0
    colorramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp.color_ramp.elements[1].position = 0.257
    colorramp.color_ramp.elements[1].color = (0.4793, 0.4793, 0.4793, 1.0)
    colorramp.color_ramp.elements[2].position = 0.514
    colorramp.color_ramp.elements[2].color = (1.0, 1.0, 1.0, 1.0)
    if rand:
        color = sample_range(0.45, 0.7)
        for i in range(3):
            colorramp.color_ramp.elements[1].color[i] = color
    
    voronoi_texture_1 = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': mix, 'Scale': 5}, 
        attrs={'voronoi_dimensions': '4D'})
    if rand:
        voronoi_texture_1.inputs["W"].default_value = sample_range(-5, 5)
        voronoi_texture_1.inputs['Scale'].default_value = sample_range(5, 20)
    
    math_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: voronoi_texture_1.outputs["Distance"], 1: voronoi_texture_1.outputs["Distance"]},
        attrs={'operation': 'MULTIPLY'})
    
    mix_1 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.82, 'Color1': colorramp.outputs["Color"], 'Color2': math_2},
        attrs={'blend_type': 'BURN'})
    
    vector_math_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: (1.0, 1.0, 1.0), 1: mix_1},
        attrs={'operation': 'SUBTRACT'})
    
    normal = nw.new_node(Nodes.InputNormal)
    
    vector_math_2 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: vector_math_1.outputs["Vector"], 1: normal},
        attrs={'operation': 'MULTIPLY'})
    
    offsetscale = nw.new_node(Nodes.Value)
    offsetscale.outputs["Value"].default_value = input_kwargs['offsetscale'] if 'offsetscale' in input_kwargs else 0.1
    
    vector_math_3 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: vector_math_2.outputs["Vector"], 1: offsetscale},
        attrs={'operation': 'MULTIPLY'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input, 'Offset': vector_math_3.outputs["Vector"]})
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': set_position, 1: mix_1},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': capture_attribute.outputs["Geometry"], 'Attribute': capture_attribute.outputs["Attribute"]})

def apply(obj, geo_kwargs=None, shader_kwargs=None, **kwargs):
    surface.add_geomod(obj, geo_two_color_spots, apply=False, input_kwargs=geo_kwargs, attributes=['offset'])
    surface.add_material(obj, shader_two_color_spots, reuse=False, input_kwargs=shader_kwargs)
