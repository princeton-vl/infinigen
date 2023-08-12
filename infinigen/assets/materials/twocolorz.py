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

def shader_twocolorz(nw, rand=True, **input_kwargs):

    texture_coordinate = nw.new_node(Nodes.TextureCoord)
    
    mapping = nw.new_node(Nodes.Mapping,
        input_kwargs={'Vector': texture_coordinate.outputs["Generated"]})
    if rand:
        for i in range(2):
            # do not change Z
            mapping.inputs['Location'].default_value[i] = sample_range(-2, 2)

    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': mapping})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': mapping, 'Scale': 10.0, 'Detail': 3.0, 'Distortion': 0.5})
    if rand:
        for k in ['Scale', 'Detail', 'Distortion', 'Roughness']:
            noise_texture.inputs[k].default_value = sample_ratio(noise_texture.inputs[k].default_value, 1/3, 3)

    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Color1': separate_xyz.outputs["Z"], 'Color2': noise_texture.outputs["Fac"]},
        attrs={'blend_type': 'MULTIPLY'})
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': mix})
    colorramp.color_ramp.elements[0].position = 0.2182
    colorramp.color_ramp.elements[0].color = (0.2059, 1.0, 0.1039, 1.0)
    colorramp.color_ramp.elements[1].position = 0.5364
    colorramp.color_ramp.elements[1].color = (1.0, 0.0047, 0.2941, 1.0)
    if rand:
        pos_max = [0.4, 0.8]
        colorramp.color_ramp.elements[0].position = sample_range(0, pos_max[0])
        _min = (pos_max[1] - colorramp.color_ramp.elements[0].position) / 3 + colorramp.color_ramp.elements[0].position
        colorramp.color_ramp.elements[1].position = sample_range(_min, pos_max[1])
        sample_color(colorramp.color_ramp.elements[0].color)
        sample_color(colorramp.color_ramp.elements[1].color)

    attribute = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'offset'})
    
    colorramp_1 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': attribute.outputs["Color"]})
    colorramp_1.color_ramp.elements[0].position = 0.0
    colorramp_1.color_ramp.elements[0].color = (0.2634, 0.2634, 0.2634, 1.0)
    colorramp_1.color_ramp.elements[1].position = 1.0
    colorramp_1.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': colorramp.outputs["Color"], 'Specular': 0.0, 'Roughness': colorramp_1.outputs["Color"]})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

def apply(obj, geo_kwargs=None, shader_kwargs=None, **kwargs):
    surface.add_geomod(obj, geo_voronoi_noise, apply=False, input_kwargs=geo_kwargs, attributes=['offset'])
    surface.add_material(obj, shader_twocolorz, reuse=False, input_kwargs=shader_kwargs)
