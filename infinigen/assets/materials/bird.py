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


@node_utils.to_nodegroup('nodegroup_l_inear', singleton=False, type='ShaderNodeTree')
def nodegroup_l_inear(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'Vector', (0.0, 0.0, 0.0)),
            ('NodeSocketFloat', 'CoffX', 0.5),
            ('NodeSocketFloat', 'CoffZ', 0.5)])
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group_input.outputs["Vector"]})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: group_input.outputs["CoffX"]},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: group_input.outputs["CoffZ"]},
        attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply, 1: multiply_1})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Value': add})

@node_utils.to_nodegroup('nodegroup_head_neck', singleton=False, type='ShaderNodeTree')
def nodegroup_head_neck(nw: NodeWrangler, rand=True, kind='duck'):
    # Code generated using version 2.4.3 of the node_transpiler

    texture_coordinate = nw.new_node(Nodes.TextureCoord)
    
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketColor', 'Color1', (0.046, 0.5, 0.0, 1.0)),
            ('NodeSocketFloat', 'W', 6.0)])
    
    noise_texture_1 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': texture_coordinate.outputs["Generated"], 'W': group_input.outputs["W"], 'Scale': 2.0},
        attrs={'noise_dimensions': '4D'})
    if rand:
        noise_texture_1.inputs['W'].default_value = sample_range(-2, 2)
    
    subtract = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: noise_texture_1.outputs["Fac"], 1: (0.0, 0.0, 0.0)},
        attrs={'operation': 'SUBTRACT'})
    
    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.2, 'Color1': texture_coordinate.outputs["Generated"], 'Color2': subtract.outputs["Vector"]})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': mix})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: 0.05})
    
    reroute = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': add})
    
    colorramp_4 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': reroute})
    colorramp_4.color_ramp.elements.new(0)
    colorramp_4.color_ramp.elements.new(0)
    colorramp_4.color_ramp.elements[0].position = 0.83
    colorramp_4.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_4.color_ramp.elements[1].position = 0.831
    colorramp_4.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    colorramp_4.color_ramp.elements[2].position = 0.834
    colorramp_4.color_ramp.elements[2].color = (1.0, 1.0, 1.0, 1.0)
    colorramp_4.color_ramp.elements[3].position = 0.835
    colorramp_4.color_ramp.elements[3].color = (0.0, 0.0, 0.0, 1.0)
    
    colorramp_3 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': reroute})
    colorramp_3.color_ramp.elements[0].position = 0.83
    colorramp_3.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_3.color_ramp.elements[1].position = 0.84
    colorramp_3.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    attribute_2 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'tag_head'})
    
    add_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: colorramp_3.outputs["Color"], 1: attribute_2.outputs["Color"]})
    
    colorramp_1 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': add_1})
    colorramp_1.color_ramp.elements[0].position = 0.4545
    colorramp_1.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_1.color_ramp.elements[1].position = 0.5455
    colorramp_1.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'W': 1.7, 'Scale': 3.0},
        attrs={'noise_dimensions': '4D'})
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': noise_texture.outputs["Fac"]})
    colorramp.color_ramp.elements[0].position = 0.5077
    colorramp.color_ramp.elements[0].color = (0.0063, 0.017, 0.005, 1.0)
    colorramp.color_ramp.elements[1].position = 1.0
    colorramp.color_ramp.elements[1].color = (0.0018, 0.0571, 0.0, 1.0)
    if rand:
        if kind == 'duck':
            sample_color(colorramp.color_ramp.elements[0].color, keep_sum=True)
            for i in range(3):
                colorramp.color_ramp.elements[1].color[i] = colorramp.color_ramp.elements[0].color[i]+0.005
        elif kind == 'eagle':
            colorramp.color_ramp.elements[0].color = (0.265, 0.265, 0.265, 1.0)
            sample_color(colorramp.color_ramp.elements[0].color, offset=0.05)

    mix_1 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp_1.outputs["Color"], 'Color1': group_input.outputs["Color1"], 'Color2': colorramp.outputs["Color"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Color': mix_1})

def shader_bird_body(nw: NodeWrangler, rand=True, kind='duck', **input_kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    attribute = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'tag_tail'})
    
    attribute_3 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'tag_body'})
    
    attribute_5 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'tag_leg'})
    
    attribute_6 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'tag_wing'})
    
    attribute_4 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'tag_foot'})
    
    mix_3 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': attribute_4.outputs["Color"], 'Color1': (0.0, 0.0, 0.0, 1.0), 'Color2': (0.0225, 0.0055, 0.0024, 1.0)})
    
    mix_8 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': attribute_6.outputs["Color"], 'Color1': mix_3, 'Color2': (0.008, 0.008, 0.008, 1.0)})
    
    texture_coordinate_2 = nw.new_node(Nodes.TextureCoord)
    
    noise_texture_2 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': texture_coordinate_2.outputs["Generated"]})
    
    mix_6 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.2, 'Color1': texture_coordinate_2.outputs["Generated"], 'Color2': noise_texture_2.outputs["Color"]})
    
    group_2 = nw.new_node(nodegroup_l_inear().name,
        input_kwargs={'Vector': mix_6, 'CoffX': 0.1, 'CoffZ': 1.0})
    if rand:
        if random.random() < 0.5:
            group_2.inputs['CoffX'].default_value = sample_range(-0.1, 0.1)
        else:
            group_2.inputs['CoffX'].default_value = sample_range(0.1, 0.8)

    add = nw.new_node(Nodes.Math,
        input_kwargs={0: group_2, 1: 0.1})
    
    colorramp_3 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': add})
    colorramp_3.color_ramp.elements[0].position = 0.4159
    colorramp_3.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    colorramp_3.color_ramp.elements[1].position = 0.6886
    colorramp_3.color_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)
    
    noise_texture_4 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Scale': 20.0})
    
    colorramp_5 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': noise_texture_4.outputs["Fac"]})
    colorramp_5.color_ramp.elements[0].position = 0.3341
    colorramp_5.color_ramp.elements[0].color = (0.0079, 0.0062, 0.0063, 1.0)
    colorramp_5.color_ramp.elements[1].position = 0.9932
    colorramp_5.color_ramp.elements[1].color = (0.0302, 0.0264, 0.0262, 1.0)
    if rand:
        if kind == 'duck':
            for i in range(3):
                colorramp_5.color_ramp.elements[0].color[i] = sample_range(0, 0.2)
                colorramp_5.color_ramp.elements[1].color[i] = sample_range(0, 0.2)
        elif kind == 'eagle':
            for i in range(3):
                colorramp_5.color_ramp.elements[0].color[i] = sample_range(0, 0.01)
                colorramp_5.color_ramp.elements[0].position = sample_range(0.5, 0.6)
                colorramp_5.color_ramp.elements[1].color[i] = sample_range(0, 0.1)
                
    noise_texture_1 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Scale': 10.0},
        attrs={'noise_dimensions': '4D'})
    if rand:
        noise_texture_1.inputs['W'].default_value = sample_range(-2, 2)
        x = random.random()
        if x < 0.3:
            noise_texture_1.inputs['Scale'].default_value = 1
        if x > 0.7:
            noise_texture_1.inputs['Scale'].default_value = 50

    colorramp_1 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': noise_texture_1.outputs["Color"]})
    colorramp_1.color_ramp.elements[0].position = 0.4614
    colorramp_1.color_ramp.elements[0].color = (0.1, 0.1, 0.1, 1.0)
    colorramp_1.color_ramp.elements[1].position = 1.0
    colorramp_1.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    if rand:
        if kind == 'eagle':
            for i in range(3):
                colorramp_1.color_ramp.elements[0].color[i] = sample_range(0, 0.01)
                colorramp_1.color_ramp.elements[0].position = sample_range(0.5, 0.6)
                colorramp_1.color_ramp.elements[1].color[i] = sample_range(0, 0.1)

    mix_5 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp_3.outputs["Color"], 'Color1': colorramp_5.outputs["Color"], 'Color2': colorramp_1.outputs["Color"]})
    
    mix_7 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': attribute_5.outputs["Color"], 'Color1': mix_8, 'Color2': mix_5})
    
    texture_coordinate = nw.new_node(Nodes.TextureCoord)
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': texture_coordinate.outputs["Generated"]})
    
    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.2, 'Color1': texture_coordinate.outputs["Generated"], 'Color2': noise_texture.outputs["Color"]})
    
    group_1 = nw.new_node(nodegroup_l_inear().name,
        input_kwargs={'Vector': mix, 'CoffX': 0.6})
    if rand:
        group_1.inputs['CoffX'].default_value = sample_range(0, 0.08)
        group_1.inputs['CoffZ'].default_value = 1.1 - group_1.inputs['CoffX'].default_value

    add_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_1, 1: -0.02})
    if rand:
        add_1.inputs[1].default_value = sample_range(-0.07, 0.03)

    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': add_1})
    colorramp.color_ramp.elements[0].position = 0.6295
    colorramp.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    colorramp.color_ramp.elements[1].position = 0.7068
    colorramp.color_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)
    
    noise_texture_3 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Scale': 20.0},
        attrs={'noise_dimensions': '4D'})
    if rand:
        noise_texture_3.inputs['W'].default_value = sample_range(-2, 2)

    colorramp_4 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': noise_texture_3.outputs["Fac"]})
    colorramp_4.color_ramp.elements[0].position = 0.4636
    colorramp_4.color_ramp.elements[0].color = (0.0112, 0.0053, 0.0047, 1.0)
    colorramp_4.color_ramp.elements[1].position = 1.0
    colorramp_4.color_ramp.elements[1].color = (0.0231, 0.0128, 0.0121, 1.0)
    
    if rand:
        if kind == 'duck':
            sample_color(colorramp_4.color_ramp.elements[0].color, keep_sum=True)
            sample_color(colorramp_4.color_ramp.elements[1].color, keep_sum=True)
        if kind == 'eagle':
            for i in range(3):
                colorramp_4.color_ramp.elements[0].color[i] = sample_range(0, 0.01)
                colorramp_4.color_ramp.elements[0].position = sample_range(0.5, 0.6)
                colorramp_4.color_ramp.elements[1].color[i] = sample_range(0, 0.1)

    mix_1 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp.outputs["Color"], 'Color1': colorramp_4.outputs["Color"], 'Color2': mix_5})
    
    mix_2 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': attribute_3.outputs["Color"], 'Color1': mix_7, 'Color2': mix_1})
    
    mix_4 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': attribute.outputs["Color"], 'Color1': mix_2, 'Color2': (0.0, 0.0, 0.0, 1.0)})
    
    group = nw.new_node(nodegroup_head_neck(rand=rand, kind=kind).name,
        input_kwargs={'Color1': mix_4, 'W': 0.5})
    if rand:
        group.inputs['W'].default_value = sample_range(-2, 2)

    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': group, 'Subsurface IOR': 0.0, 'Specular': 0.0, 'Roughness': 1.0})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})


def shader_bird_feather(nw: NodeWrangler, rand=True, kind='duck', tail=False, **input_kwargs):
    # Code generated using version 2.4.3 of the node_transpiler
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'W': 1.6},
        attrs={'noise_dimensions': '4D'})
    if rand:
        noise_texture.inputs['W'].default_value = sample_range(-2, 2)

    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': noise_texture.outputs["Fac"]})
    colorramp.color_ramp.elements[0].position = 0.377
    colorramp.color_ramp.elements[0].color = (0.02, 0.02, 0.02, 1.0)
    colorramp.color_ramp.elements[1].position = 1.0
    colorramp.color_ramp.elements[1].color = (0.0061, 0.0058, 0.0059, 1.0)
    if rand:
        if kind == 'duck':
            x = sample_range(0.02, 0.15)
            for i in range(3):
                colorramp.color_ramp.elements[1].color[i] = x
        elif kind == 'eagle':
            if tail:
                colorramp.color_ramp.elements[0].color = (0.265, 0.265, 0.265, 1.0)
                sample_color(colorramp.color_ramp.elements[0].color, offset=0.05)
                colorramp.color_ramp.elements[1].color = (0.007, 0.007, 0.007, 1.0)
            else: 
                colorramp.color_ramp.elements[0].color = (0.012861, 0.006847, 0.004, 1.0)
                sample_color(colorramp.color_ramp.elements[0].color, offset=0.003)
                colorramp.color_ramp.elements[1].color = (0.154963, 0.081816, 0.042745, 1.0)
                sample_color(colorramp.color_ramp.elements[1].color, offset=0.005)
                colorramp.color_ramp.elements[0].position = sample_range(0.56, 0.62)

    texture_coordinate = nw.new_node(Nodes.TextureCoord)

    mapping = nw.new_node(Nodes.Mapping,
                          input_kwargs={'Vector': texture_coordinate.outputs["Object"]})

    wave_texture = nw.new_node(Nodes.WaveTexture,
                               input_kwargs={'Vector': mapping, 'Scale': 5.00, 'Distortion': 10.0000, 'Detail': 10.0000,
                                             'Detail Roughness': 2.0000})

    colorramp2 = nw.new_node(Nodes.ColorRamp,
                            input_kwargs={'Fac': wave_texture.outputs["Color"]})
    colorramp2.color_ramp.elements[0].position = 0.0955
    colorramp2.color_ramp.elements[0].color = [0.0000, 0.0000, 0.0000, 1.0000]
    colorramp2.color_ramp.elements[1].position = 0.6364
    colorramp2.color_ramp.elements[1].color = [1.0000, 1.0000, 1.0000, 1.0000]

    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.5 if tail else 1.0, 'Color1': colorramp2.outputs["Color"], 'Color2': colorramp.outputs["Color"]})

    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
    input_kwargs={'Base Color': (mix, "Result"), 'Specular': 0.0, 'Roughness': 1.0},
        attrs={'subsurface_method': 'BURLEY'})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

def shader_wave_feather(nw: NodeWrangler, **input_kwargs):
    # Code generated using version 2.5.1 of the node_transpiler

    texture_coordinate = nw.new_node(Nodes.TextureCoord)

    mapping = nw.new_node(Nodes.Mapping,
                          input_kwargs={'Vector': texture_coordinate.outputs["Object"]})

    wave_texture = nw.new_node(Nodes.WaveTexture,
                               input_kwargs={'Vector': mapping, 'Scale': 5.00, 'Distortion': 10.0000, 'Detail': 10.0000,
                                             'Detail Roughness': 2.0000})

    colorramp = nw.new_node(Nodes.ColorRamp,
                            input_kwargs={'Fac': wave_texture.outputs["Color"]})
    colorramp.color_ramp.elements[0].position = 0.0955
    colorramp.color_ramp.elements[0].color = [0.0000, 0.0000, 0.0000, 1.0000]
    colorramp.color_ramp.elements[1].position = 0.6364
    colorramp.color_ramp.elements[1].color = [1.0000, 1.0000, 1.0000, 1.0000]

    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
                                  input_kwargs={'Base Color': colorramp.outputs["Color"]})

    material_output = nw.new_node(Nodes.MaterialOutput,
                                  input_kwargs={'Surface': principled_bsdf})

def shader_bird_beak(nw: NodeWrangler, rand=True, **input_kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'W': 1.1, 'Scale': 20.0, 'Roughness': 0.5142},
        attrs={'noise_dimensions': '4D'})
    if rand:
        noise_texture.inputs['W'].default_value = sample_range(-2, 2)

    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': noise_texture.outputs["Fac"]})
    colorramp.color_ramp.interpolation = "EASE"
    colorramp.color_ramp.elements[0].position = 0.3815
    colorramp.color_ramp.elements[0].color = (0.2773, 0.271, 0.047, 1.0)
    colorramp.color_ramp.elements[1].position = 0.7736
    colorramp.color_ramp.elements[1].color = (0.141, 0.146, 0.007, 1.0)
    if rand:
        sample_color(colorramp.color_ramp.elements[0].color, keep_sum=True)
        sample_color(colorramp.color_ramp.elements[1].color, keep_sum=True)

    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': colorramp.outputs["Color"], 'Roughness': 0.3408})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

def shader_bird_eyeball(nw: NodeWrangler, rand=True, **input_kwargs):
    # Code generated using version 2.4.3 of the node_transpiler
    '''
    texture_coordinate = nw.new_node(Nodes.TextureCoord)
    
    mapping = nw.new_node(Nodes.Mapping,
        input_kwargs={'Vector': texture_coordinate.outputs["Generated"], 'Rotation': (0.0, 0.0, -0.5236)})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': mapping})
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': separate_xyz.outputs["X"]})
    colorramp.color_ramp.interpolation = "CONSTANT"
    colorramp.color_ramp.elements.new(0)
    colorramp.color_ramp.elements[0].position = 0.0
    colorramp.color_ramp.elements[0].color = (0.4489, 0.3077, 0.1451, 1.0)
    colorramp.color_ramp.elements[1].position = 0.0837
    colorramp.color_ramp.elements[1].color = (0.6744, 0.0691, 0.3627, 1.0)
    colorramp.color_ramp.elements[2].position = 0.1909
    colorramp.color_ramp.elements[2].color = (0.0, 0.0, 0.0, 1.0)
    '''
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': (0.0, 0.0, 0.0, 1.0), 'Roughness': 0.0})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

def shader_bird_claw(nw: NodeWrangler, rand=True, **input_kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': (0.0091, 0.0091, 0.0091, 1.0), 'Specular': 0.0, 'Roughness': 0.4409})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

def apply(objs, shader_kwargs={}, **kwargs):
    x = random.random()
    if x < 0.4:
        kind = 'eagle'
    else:
        kind = 'duck'
    shader_kwargs['kind'] = kind
    if not isinstance(objs, list):
        objs = [objs]
    for obj in objs:
        if "Tail" in obj.name:
            shader_kwargs['tail'] = True
            surface.add_material(obj, shader_bird_feather, input_kwargs=shader_kwargs)
        else:
            shader_kwargs['tail'] = False
        if "Body" in obj.name:
            surface.add_material(obj, shader_bird_body, input_kwargs=shader_kwargs)
        if "Feather" in obj.name and "Tail" not in obj.name:
            surface.add_material(obj, shader_bird_feather, input_kwargs=shader_kwargs)
        if "Claw" in obj.name:
            surface.add_material(obj, shader_bird_claw, input_kwargs=shader_kwargs)
        if "Eyeball" in obj.name:
            surface.add_material(obj, shader_bird_eyeball, input_kwargs=shader_kwargs)
        if "Beak" in obj.name:
            surface.add_material(obj, shader_bird_beak, input_kwargs=shader_kwargs)

if __name__ == "__main__":
    for i in range(10):
        bpy.ops.wm.open_mainfile(filepath='dev_scene_test_bird.blend')
        objs = [
            "creature(98047, 0).parts(0, factory=NurbsBody)",
            "creature(98047, 0).parts(1).extra(TailFeathers, 1)",
            "creature(98047, 0).parts(3).extra(Claws_0.001, 3)",
            "creature(98047, 0).parts(3).extra(Claws_1.001, 3)",
            "creature(98047, 0).parts(3).extra(Claws_2.001, 3)",
            "creature(98047, 0).parts(3).extra(Claws_3.001, 3)",
            "creature(98047, 0).parts(3).extra(Claws_4.001, 3)",
            "creature(98047, 0).parts(5).extra(Claws_0, 5)",
            "creature(98047, 0).parts(5).extra(Claws_1, 5)",
            "creature(98047, 0).parts(5).extra(Claws_2, 5)",
            "creature(98047, 0).parts(5).extra(Claws_3, 5)",
            "creature(98047, 0).parts(5).extra(Claws_4, 5)",
            "creature(98047, 0).parts(6).extra(Feathers.001, 6)",
            "creature(98047, 0).parts(7).extra(Feathers.002, 7)",
            "creature(98047, 0).parts(8).extra(LeftEye, 8)",
            "creature(98047, 0).parts(8).extra(RightEye, 8)",
            "creature(98047, 0).parts(9).extra(BeakLower, 9)",
            "creature(98047, 0).parts(9).extra(BeakUpper, 9)",
        ]
        objs = [bpy.data.objects[x] for x in objs]
        apply(objs)
        fn_blend = os.path.join(os.path.abspath(os.curdir), 'dev_scene_eagle.blend')
        fn = os.path.join(os.path.abspath(os.curdir), 'test_bird%d.jpg'%(i))
        bpy.ops.wm.save_as_mainfile(filepath=fn_blend)
        bpy.context.scene.render.filepath = fn
        bpy.context.scene.render.image_settings.file_format='JPEG'
        bpy.ops.render.render(write_still=True)
        