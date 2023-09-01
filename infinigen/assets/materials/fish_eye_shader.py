# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Mingzhe Wang
# Acknowledgement: This file draws inspiration from https://www.youtube.com/watch?v=EfNzAaqKHXQ by PixelicaCG, https://www.youtube.com/watch?v=JcHX4AT1vtg by CGCookie and https://www.youtube.com/watch?v=E0JyyWeptSA by CGRogue

import os, sys
import numpy as np
import math as ma
from infinigen.assets.materials.utils.surface_utils import clip, sample_range, sample_ratio, sample_color, geo_voronoi_noise
import bpy
import mathutils
from numpy.random import uniform as U, normal as N, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface

@node_utils.to_nodegroup('nodegroup_rotate2_d_002', singleton=False, type='ShaderNodeTree')
def nodegroup_rotate2_d_002(nw: NodeWrangler, rand=True, **input_kwargs):
    # Code generated using version 2.6.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'Value1', 0.5000),
            ('NodeSocketFloat', 'Value2', 0.5000),
            ('NodeSocketFloat', 'Value3', 0.5000)])
    
    multiply = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Value3"], 1: 0.0175}, attrs={'operation': 'MULTIPLY'}) # pretty sure Value3 is the right one here
    
    sine = nw.new_node(Nodes.Math, input_kwargs={0: multiply}, attrs={'operation': 'SINE'})
    
    multiply_1 = nw.new_node(Nodes.Math, input_kwargs={0: sine, 1: group_input}, attrs={'operation': 'MULTIPLY'})
    
    cosine = nw.new_node(Nodes.Math, input_kwargs={0: multiply}, attrs={'operation': 'COSINE'})
    
    multiply_2 = nw.new_node(Nodes.Math, input_kwargs={0: group_input, 1: cosine}, attrs={'operation': 'MULTIPLY'})
    
    subtract = nw.new_node(Nodes.Math, input_kwargs={0: multiply_1, 1: multiply_2}, attrs={'operation': 'SUBTRACT'})
    
    multiply_3 = nw.new_node(Nodes.Math, input_kwargs={0: group_input, 1: cosine}, attrs={'operation': 'MULTIPLY'})
    
    multiply_4 = nw.new_node(Nodes.Math, input_kwargs={0: group_input, 1: sine}, attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.Math, input_kwargs={0: multiply_3, 1: multiply_4})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Value': subtract, 'Value1': add}, attrs={'is_active_output': True})

def shader_eyeball_fish(nw: NodeWrangler):
    # Code generated using version 2.6.3 of the node_transpiler

    attribute_1 = nw.new_node(Nodes.Attribute, attrs={'attribute_name': 'EyeballPosition'})
    
    mapping = nw.new_node(Nodes.Mapping, input_kwargs={'Vector': attribute_1.outputs["Color"]})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture, input_kwargs={'Vector': mapping, 'Scale': 50.0000})
    
    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.0200, 'Color1': mapping, 'Color2': noise_texture.outputs["Color"]})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': mix})
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.0000
    
    group = nw.new_node(nodegroup_rotate2_d_002().name,
        input_kwargs={0: separate_xyz.outputs["Y"], 'Value2': separate_xyz.outputs["Z"], 2: value})
    
    multiply = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz.outputs["Z"], 1: 0.3000}, attrs={'operation': 'MULTIPLY'})
    
    multiply_1 = nw.new_node(Nodes.Math, input_kwargs={0: multiply, 1: multiply}, attrs={'operation': 'MULTIPLY'})
    
    multiply_2 = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz.outputs["Y"], 1: 0.8000}, attrs={'operation': 'MULTIPLY'})
    
    multiply_3 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_2, 1: multiply_2}, attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.Math, input_kwargs={0: multiply_1, 1: multiply_3})
    
    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: add, 1: 0.6300})
    
    colorramp = nw.new_node(Nodes.ColorRamp, input_kwargs={'Fac': add_1})
    colorramp.color_ramp.elements[0].position = 0.6400
    colorramp.color_ramp.elements[0].color = [1.0000, 1.0000, 1.0000, 1.0000]
    colorramp.color_ramp.elements[1].position = 0.6591
    colorramp.color_ramp.elements[1].color = [0.0000, 0.0000, 0.0000, 1.0000]
    
    mapping_1 = nw.new_node(Nodes.Mapping,
        input_kwargs={'Vector': attribute_1.outputs["Color"], 'Scale': (1.0000, 100.0000, 1.0000)})
    
    mix_1 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.3000, 'Color1': mapping_1, 'Color2': attribute_1.outputs["Color"]})
    
    noise_texture_1 = nw.new_node(Nodes.NoiseTexture, input_kwargs={'Vector': mix_1, 'Scale': 10.0000})
    
    mix_2 = nw.new_node(Nodes.MixRGB, input_kwargs={'Fac': 0.7000, 'Color1': noise_texture_1.outputs["Fac"], 'Color2': mix_1})
    
    voronoi_texture = nw.new_node(Nodes.VoronoiTexture, input_kwargs={'Vector': mix_2, 'Scale': 20.0000})
    
    multiply_4 = nw.new_node(Nodes.Math,
        input_kwargs={0: voronoi_texture.outputs["Distance"], 1: voronoi_texture.outputs["Distance"]},
        attrs={'operation': 'MULTIPLY'})
    
    mapping_2 = nw.new_node(Nodes.Mapping,
        input_kwargs={'Vector': attribute_1.outputs["Color"], 'Scale': (20.0000, 1.0000, 1.0000)})
    
    mix_3 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.3000, 'Color1': mapping_2, 'Color2': attribute_1.outputs["Color"]})
    
    noise_texture_2 = nw.new_node(Nodes.NoiseTexture, input_kwargs={'Vector': mix_3, 'Scale': 10.0000})
    
    mix_4 = nw.new_node(Nodes.MixRGB, input_kwargs={'Fac': 0.7000, 'Color1': noise_texture_2.outputs["Fac"], 'Color2': mix_3})
    
    voronoi_texture_1 = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': mix_4, 'W': U(-10, 10), 'Scale': 1.0000},
        attrs={'voronoi_dimensions': '4D'})
    
    multiply_5 = nw.new_node(Nodes.Math,
        input_kwargs={0: voronoi_texture_1.outputs["Distance"], 1: voronoi_texture_1.outputs["Distance"]},
        attrs={'operation': 'MULTIPLY'})
    
    mix_5 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 1.0000, 'Color1': multiply_4, 'Color2': multiply_5},
        attrs={'blend_type': 'OVERLAY'})
    
    bright_contrast = nw.new_node('ShaderNodeBrightContrast', input_kwargs={'Color': mix_5, 'Bright': 0.6000, 'Contrast': 1.5000})
    
    scale1 = U(0.65, 1.2)

    multiply_6 = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz.outputs["Z"], 1: scale1}, attrs={'operation': 'MULTIPLY'})
    
    multiply_7 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_6, 1: multiply_6}, attrs={'operation': 'MULTIPLY'})
    
    multiply_8 = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz.outputs["Y"], 1: scale1}, attrs={'operation': 'MULTIPLY'})
    
    multiply_9 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_8, 1: multiply_8}, attrs={'operation': 'MULTIPLY'})
    
    add_2 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_7, 1: multiply_9})
    
    add_3 = nw.new_node(Nodes.Math, input_kwargs={0: add_2})
    
    colorramp_1 = nw.new_node(Nodes.ColorRamp, input_kwargs={'Fac': add_3})
    colorramp_1.color_ramp.elements[0].position = 0.6159
    colorramp_1.color_ramp.elements[0].color = [1.0000, 1.0000, 1.0000, 1.0000]
    colorramp_1.color_ramp.elements[1].position = 0.6591
    colorramp_1.color_ramp.elements[1].color = [0.0000, 0.0000, 0.0000, 1.0000]
    
    colorramp_2 = nw.new_node(Nodes.ColorRamp, input_kwargs={'Fac': colorramp_1.outputs["Color"]})
    colorramp_2.color_ramp.elements[0].position = 0.0295
    colorramp_2.color_ramp.elements[0].color = [0.0000, 0.0000, 0.0000, 1.0000]
    colorramp_2.color_ramp.elements[1].position = 0.0523
    colorramp_2.color_ramp.elements[1].color = [1.0000, 1.0000, 1.0000, 1.0000]
    
    add_4 = nw.new_node(Nodes.Math,
        input_kwargs={0: bright_contrast, 1: colorramp_2.outputs["Color"]},
        attrs={'use_clamp': True})
    
    scale2 = U(0.6, 0.8)
    multiply_10 = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz.outputs["Z"], 1: scale2}, attrs={'operation': 'MULTIPLY'})
    
    multiply_11 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_10, 1: multiply_10}, attrs={'operation': 'MULTIPLY'})
    
    multiply_12 = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz.outputs["Y"], 1: scale2}, attrs={'operation': 'MULTIPLY'})
    
    multiply_13 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_12, 1: multiply_12}, attrs={'operation': 'MULTIPLY'})
    
    add_5 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_11, 1: multiply_13})
    
    add_6 = nw.new_node(Nodes.Math, input_kwargs={0: add_5, 1: 0.1800})
    
    colorramp_3 = nw.new_node(Nodes.ColorRamp, input_kwargs={'Fac': add_6})
    colorramp_3.color_ramp.elements[0].position = 0.4773
    colorramp_3.color_ramp.elements[0].color = [1.0000, 1.0000, 1.0000, 1.0000]
    colorramp_3.color_ramp.elements[1].position = 0.6659
    colorramp_3.color_ramp.elements[1].color = [0.0000, 0.0000, 0.0000, 1.0000]
    
    attribute_2 = nw.new_node(Nodes.Attribute, attrs={'attribute_name': 'EyeballPosition'})
    
    noise_texture_3 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': attribute_2.outputs["Color"], 'W': U(-10, 10), 'Scale': 0.5000},
        attrs={'noise_dimensions': '4D'})
    
    colorramp_4 = nw.new_node(Nodes.ColorRamp, input_kwargs={'Fac': noise_texture_3.outputs["Color"]})
    colorramp_4.color_ramp.interpolation = "CARDINAL"
    colorramp_4.color_ramp.elements[0].position = 0.3704
    colorramp_4.color_ramp.elements[0].color = [0.9570, 0.9247, 0.2801, 1.0000]
    colorramp_4.color_ramp.elements[1].position = 0.5455
    colorramp_4.color_ramp.elements[1].color = [1.0000, 0.6872, 0.5327, 1.0000]
    sample_color(colorramp_4.color_ramp.elements[0].color)
    sample_color(colorramp_4.color_ramp.elements[1].color)


    mix_6 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp_3.outputs["Color"], 'Color1': (0.7384, 0.5239, 0.2703, 1.0000), 'Color2': colorramp_4.outputs["Color"]})
    sample_color(mix_6.inputs[6].default_value)

    mix_7 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp_1.outputs["Color"], 'Color1': mix_6, 'Color2': (0.0000, 0.0000, 0.0000, 1.0000)})
    
    mix_8 = nw.new_node(Nodes.MixRGB, input_kwargs={'Fac': add_4, 'Color1': (0.0000, 0.0000, 0.0000, 1.0000), 'Color2': mix_7})
    
    mix_9 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp.outputs["Color"], 'Color1': mix_8, 'Color2': (0.0000, 0.0000, 0.0000, 1.0000)})
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF, input_kwargs={'Base Color': mix_8, 'Specular': 0.0000, 'Roughness': 0.0000})
    
    glossy_bsdf = nw.new_node('ShaderNodeBsdfGlossy')
    
    mix_shader_1 = nw.new_node(Nodes.MixShader, input_kwargs={'Fac': 0.0200, 1: principled_bsdf, 2: glossy_bsdf})
    
    material_output = nw.new_node(Nodes.MaterialOutput, input_kwargs={'Surface': mix_shader_1}, attrs={'is_active_output': True})


def apply(obj, selection=None, **kwargs):
    surface.add_material(obj, shader_eyeball_fish, selection=selection)