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

def shader_spot(nw, rand=True, **input_kwargs):

    texture_coordinate = nw.new_node(Nodes.TextureCoord)
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': texture_coordinate.outputs["Object"]})
    
    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.7, 'Color1': noise_texture.outputs["Color"], 'Color2': texture_coordinate.outputs["Object"]})
    if rand:
        mix.inputs["Factor"].default_value = sample_range(0.5, 0.9)

    scale = nw.new_node(Nodes.Value)
    scale.outputs["Value"].default_value = input_kwargs['scale'] if 'scale' in input_kwargs else 2
    
    mapping = nw.new_node(Nodes.Mapping,
        input_kwargs={'Vector': mix, 'Scale': scale})
    if rand:
        for i in range(3):
            mapping.inputs['Location'].default_value[i] = sample_range(-1, 1)
            mapping.inputs['Rotation'].default_value[i] = sample_range(0, 2*ma.pi)

    spot1_1 = nw.new_node(Nodes.Value)
    spot1_1.outputs["Value"].default_value = 7.5
    
    noise_texture_1 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': mapping, 'Scale': spot1_1})
    
    mix_7 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.95, 'Color1': noise_texture_1.outputs["Color"], 'Color2': mapping})
    
    mapping_1 = nw.new_node(Nodes.Mapping,
        input_kwargs={'Vector': mix_7})
    if rand:
        for i in range(3):
            mapping_1.inputs['Scale'].default_value[i] = sample_range(0.8, 1.2)
    
    spot2_size = nw.new_node(Nodes.Value)
    spot2_size.outputs["Value"].default_value = sample_range(1, 3) if rand else 1.5

    voronoi_texture_2 = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': mapping_1, 'Scale': spot2_size},
        attrs={'voronoi_dimensions': '4D'})
    if rand:
        voronoi_texture_2.inputs['W'].default_value = sample_range(-5, 5)

    math_4 = nw.new_node(Nodes.Math,
        input_kwargs={0: voronoi_texture_2.outputs["Distance"], 1: voronoi_texture_2.outputs["Distance"]},
        attrs={'operation': 'MULTIPLY'})
    
    colorramp_1 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': math_4})
    colorramp_1.color_ramp.elements.new(1)
    colorramp_1.color_ramp.elements[0].position = 0.1409
    colorramp_1.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_1.color_ramp.elements[1].position = 0.2295
    colorramp_1.color_ramp.elements[1].color = (0.0563, 0.0563, 0.0563, 1.0)
    colorramp_1.color_ramp.elements[2].position = 0.5227
    colorramp_1.color_ramp.elements[2].color = (1.0, 1.0, 1.0, 1.0)
    if rand:
        colorramp_1.color_ramp.elements[1].position = sample_range(0.18, 0.23)

    math_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: spot2_size, 1: 10.0},
        attrs={'operation': 'MULTIPLY'})
    
    voronoi_texture_3 = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': mapping_1, 'Scale': math_2},
        attrs={'voronoi_dimensions': '4D'})
    if rand:
        voronoi_texture_3.inputs['W'].default_value = sample_range(-5, 5)

    math_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: voronoi_texture_3.outputs["Distance"], 1: voronoi_texture_3.outputs["Distance"]},
        attrs={'operation': 'MULTIPLY'})
    
    mix_4 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.4467, 'Color1': colorramp_1.outputs["Color"], 'Color2': math_3},
        attrs={'blend_type': 'BURN'})
    
    spot2 = nw.new_node(Nodes.Value)
    spot2.outputs["Value"].default_value = 1

    noise_texture_2 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': mapping, 'Scale': spot2})
    
    mix_3 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.875, 'Color1': noise_texture_2.outputs["Color"], 'Color2': mapping})
    
    mapping_2 = nw.new_node(Nodes.Mapping,
        input_kwargs={'Vector': mix_3})
    if rand:
        for i in range(3):
            mapping_2.inputs['Scale'].default_value[i] = sample_range(0.8, 1.2)

    spot1_size = nw.new_node(Nodes.Value)
    spot1_size.outputs["Value"].default_value = sample_range(1, 3) if rand else 1.5

    voronoi_texture = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': mapping_2, 'W': 1.0, 'Scale': spot1_size},
        attrs={'voronoi_dimensions': '4D'})
    if rand:
        voronoi_texture.inputs['W'].default_value = sample_range(-5, 5)

    math_5 = nw.new_node(Nodes.Math,
        input_kwargs={0: voronoi_texture.outputs["Distance"], 1: voronoi_texture.outputs["Distance"]},
        attrs={'operation': 'MULTIPLY'})
    
    colorramp_2 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': math_5})
    colorramp_2.color_ramp.elements.new(1)
    colorramp_2.color_ramp.elements[0].position = 0.0
    colorramp_2.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_2.color_ramp.elements[1].position = 0.2523
    colorramp_2.color_ramp.elements[1].color = (0.4798, 0.4798, 0.4798, 1.0)
    colorramp_2.color_ramp.elements[2].position = 0.5136
    colorramp_2.color_ramp.elements[2].color = (1.0, 1.0, 1.0, 1.0)
    if rand:
        colorramp_2.color_ramp.elements[1].position = sample_range(0.18, 0.32)

    value = nw.new_node(Nodes.Value)
    value.outputs["Value"].default_value = sample_range(2, 8) if rand else 5

    voronoi_texture_1 = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': mapping_2, 'Scale': value},
        attrs={'voronoi_dimensions': '4D'})
    if rand:
        voronoi_texture_1.inputs['W'].default_value = sample_range(-5, 5)

    math_6 = nw.new_node(Nodes.Math,
        input_kwargs={0: voronoi_texture_1.outputs["Distance"], 1: voronoi_texture_1.outputs["Distance"]},
        attrs={'operation': 'MULTIPLY'})
    
    mix_1 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.82, 'Color1': colorramp_2.outputs["Color"], 'Color2': math_6},
        attrs={'blend_type': 'BURN'})
    
    math = nw.new_node(Nodes.Math,
        input_kwargs={0: mix_4, 1: mix_1},
        attrs={'operation': 'LESS_THAN'})
    
    rgb = nw.new_node(Nodes.RGB)
    sample_color(rgb.outputs['Color'].default_value)

    color1 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': mix_1, 'Color1': (1.0, 0.0223, 0.0, 1.0), 'Color2': rgb})
    sample_color(color1.inputs[6].default_value)

    color2 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': mix_4, 'Color1': (0.0021, 0.0021, 0.0144, 1.0), 'Color2': rgb})
    sample_color(color2.inputs[6].default_value)

    mix_6 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': math, 'Color1': color1, 'Color2': color2})
    
    attribute = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'offset'})
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': attribute.outputs["Color"]})
    colorramp.color_ramp.elements[0].position = 0.0
    colorramp.color_ramp.elements[0].color = (0.3036, 0.3036, 0.3036, 1.0)
    colorramp.color_ramp.elements[1].position = 1.0
    colorramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': mix_6, 'Roughness': colorramp.outputs["Color"]},
        attrs={'subsurface_method': 'BURLEY'})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

def apply(obj, geo_kwargs=None, shader_kwargs=None, **kwargs):
    surface.add_geomod(obj, geo_voronoi_noise, apply=False, input_kwargs=geo_kwargs, attributes=['offset'])
    surface.add_material(obj, shader_spot, reuse=False, input_kwargs=shader_kwargs)
