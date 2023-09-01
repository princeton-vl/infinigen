# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Mingzhe Wang


import os, sys
import numpy as np
import math as ma
from infinigen.assets.materials.utils.surface_utils import clip, sample_range, sample_ratio, sample_color, geo_voronoi_noise
import bpy
import mathutils
from numpy.random import uniform, normal, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface
import random

def shader_fin_regular(nw: NodeWrangler, rand=True, **input_kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    attribute = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'Bump'})
    
    colorramp_2 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': attribute.outputs["Color"]})
    colorramp_2.color_ramp.elements[0].position = 0.0227
    colorramp_2.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_2.color_ramp.elements[1].position = 0.1432
    colorramp_2.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Scale': 20.0},
        attrs={'noise_dimensions': '4D'})
    if rand:
        noise_texture.inputs['W'].default_value = sample_range(-2, 2)

    colorramp_1 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': noise_texture.outputs["Fac"]})
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements[0].position = 0.0
    colorramp_1.color_ramp.elements[0].color = (0.0288, 0.0301, 0.0266, 1.0)
    colorramp_1.color_ramp.elements[1].position = 0.6727
    colorramp_1.color_ramp.elements[1].color = (0.0909, 0.083, 0.0803, 1.0)
    colorramp_1.color_ramp.elements[2].position = 1.0
    colorramp_1.color_ramp.elements[2].color = (0.0969, 0.0765, 0.0439, 1.0)
    if rand:
        for i in range(3):
            for e in colorramp_1.color_ramp.elements:
                e.color[i] = sample_range(0, 0.15)

    noise_texture_1 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Scale': 10.0},
        attrs={'noise_dimensions': '4D'})
    if rand:
        noise_texture_1.inputs['W'].default_value = sample_range(-2, 2)

    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': noise_texture_1.outputs["Fac"]})
    colorramp.color_ramp.elements[0].position = 0.0045
    colorramp.color_ramp.elements[0].color = (0.1512, 0.1236, 0.0977, 1.0)
    colorramp.color_ramp.elements[1].position = 0.5364
    colorramp.color_ramp.elements[1].color = (0.0322, 0.0275, 0.0275, 1.0)
    if rand:
        for i in range(3):
            for e in colorramp_1.color_ramp.elements:
                e.color[i] = sample_range(0, 0.15)

    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp_2.outputs["Color"], 'Color1': colorramp_1.outputs["Color"], 'Color2': colorramp.outputs["Color"]})
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': mix})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

def shader_fin_gold(nw: NodeWrangler, rand=True, **input_kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    attribute = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'Bump'})
    
    colorramp_2 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': attribute.outputs["Vector"]})
    colorramp_2.color_ramp.elements[0].position = 0.0
    colorramp_2.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_2.color_ramp.elements[1].position = 0.7977
    colorramp_2.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    attribute_2 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'BumpMask'})
    
    colorramp_8 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': attribute_2.outputs["Color"]})
    colorramp_8.color_ramp.elements[0].position = 0.0727
    colorramp_8.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_8.color_ramp.elements[1].position = 1.0
    colorramp_8.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    if rand:
        colorramp_8.color_ramp.elements[0].position = sample_range(0.05, 0.15)

    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: colorramp_2.outputs["Color"], 1: colorramp_8.outputs["Color"]},
        attrs={'operation': 'MULTIPLY'})
    
    colorramp_5 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': multiply})
    colorramp_5.color_ramp.elements.new(0)
    colorramp_5.color_ramp.elements[0].position = 0.0
    colorramp_5.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_5.color_ramp.elements[1].position = 0.1443
    colorramp_5.color_ramp.elements[1].color = (0.5, 0.5, 0.5, 1.0)
    colorramp_5.color_ramp.elements[2].position = 0.6977
    colorramp_5.color_ramp.elements[2].color = (1.0, 1.0, 1.0, 1.0)
    
    colorramp_7 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': attribute_2.outputs["Color"]})
    colorramp_7.color_ramp.elements.new(0)
    colorramp_7.color_ramp.elements[0].position = 0.0
    colorramp_7.color_ramp.elements[0].color = (0.4063, 0.4063, 0.4063, 1.0)
    colorramp_7.color_ramp.elements[1].position = 0.3659
    colorramp_7.color_ramp.elements[1].color = (0.124, 0.124, 0.124, 1.0)
    colorramp_7.color_ramp.elements[2].position = 1.0
    colorramp_7.color_ramp.elements[2].color = (0.0, 0.0, 0.0, 1.0)
    if rand:
        colorramp_7.color_ramp.elements[1].position = sample_range(0.2, 0.8)

    colorramp_4 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': colorramp_7.outputs["Color"]})
    colorramp_4.color_ramp.elements.new(0)
    colorramp_4.color_ramp.elements[0].position = 0.0
    colorramp_4.color_ramp.elements[0].color = (1.0, 0.8, 0.6, 1.0)
    colorramp_4.color_ramp.elements[1].position = 0.1682
    colorramp_4.color_ramp.elements[1].color = (1.0, 0.9, 0.8, 1.0)
    colorramp_4.color_ramp.elements[2].position = 1.0
    colorramp_4.color_ramp.elements[2].color = (1.0, 1.0, 1.0, 1.0)
    if rand:
        sample_color(colorramp_4.color_ramp.elements[0].color, offset=0.03)
        sample_color(colorramp_4.color_ramp.elements[1].color, offset=0.03)

    transparent_bsdf = nw.new_node(Nodes.TransparentBSDF,
        input_kwargs={'Color': colorramp_4.outputs["Color"]})
    
    translucent_bsdf = nw.new_node(Nodes.TranslucentBSDF,
        input_kwargs={'Color': (1.0, 0.7354, 0.4708, 1.0)})

    mix_shader_1 = nw.new_node(Nodes.MixShader,
        input_kwargs={'Fac': 0.1, 1: transparent_bsdf, 2: translucent_bsdf})
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': attribute.outputs["Vector"]})
    colorramp.color_ramp.elements[0].position = 0.0
    colorramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp.color_ramp.elements[1].position = 0.1273
    colorramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    noise_texture_1 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Scale': 10.0},
        attrs={'noise_dimensions': '4D'})
    if rand:
        noise_texture_1.inputs['W'].default_value = sample_range(-2, 2)

    colorramp_3 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': noise_texture_1.outputs["Fac"]})
    colorramp_3.color_ramp.elements[0].position = 0.3568
    colorramp_3.color_ramp.elements[0].color = (0.8258, 0.1192, 0.0, 1.0)
    colorramp_3.color_ramp.elements[1].position = 1.0
    colorramp_3.color_ramp.elements[1].color = (0.1679, 0.0, 0.0048, 1.0)
    if rand:
        sample_color(colorramp_3.color_ramp.elements[0].color, offset=0.05)
        sample_color(colorramp_3.color_ramp.elements[1].color, offset=0.05)

    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp.outputs["Color"], 'Color1': (1.0, 0.5473, 0.2571, 1.0), 'Color2': colorramp_3.outputs["Color"]})

    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': mix, 'Roughness': 1.0})
    
    mix_shader = nw.new_node(Nodes.MixShader,
        input_kwargs={'Fac': colorramp_5.outputs["Color"], 1: mix_shader_1, 2: principled_bsdf})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': mix_shader})

def apply(obj, geo_kwargs={}, shader_kwargs={}, **kwargs):
    if 'goldfish' in shader_kwargs:
        if shader_kwargs['goldfish']:
            shader = shader_fin_gold
        else:
            shader = shader_fin_regular
    else:
        if random.random() < 0.5:
            shader = shader_fin_gold
        else:
            shader = shader_fin_regular
    surface.add_material(obj, shader, input_kwargs=shader_kwargs)