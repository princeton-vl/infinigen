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

from infinigen.assets.creatures.util.nodegroups.shader import nodegroup_color_mask

def shader_two_color(nw: NodeWrangler, rand=True, **input_kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    group = nw.new_node(nodegroup_color_mask().name)
    
    texture_coordinate = nw.new_node(Nodes.TextureCoord)
    
    mapping = nw.new_node(Nodes.Mapping,
        input_kwargs={'Vector': texture_coordinate.outputs["Generated"], 'Rotation': (0.5236, -0.6807, 0.0)})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': mapping, 'Scale': 3.0, 'Detail': 10.0, 'Distortion': 0.5})
    
    noise_texture_1 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': mapping, 'Scale': 20.0, 'Detail': 50.0, 'Distortion': 0.5},
        attrs={'noise_dimensions': '4D'})
    
    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.5667, 'Color1': noise_texture.outputs["Fac"], 'Color2': noise_texture_1.outputs["Fac"]},
        attrs={'blend_type': 'MULTIPLY'})
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 2.0
    
    multiply = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: mix, 1: value},
        attrs={'operation': 'MULTIPLY'})
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': multiply.outputs["Vector"]})
    colorramp.color_ramp.elements.new(0)
    colorramp.color_ramp.elements[0].position = 0.0
    colorramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp.color_ramp.elements[1].position = 0.1909
    colorramp.color_ramp.elements[1].color = (0.1064, 0.1064, 0.1064, 1.0)
    colorramp.color_ramp.elements[2].position = 1.0
    colorramp.color_ramp.elements[2].color = (1.0, 1.0, 1.0, 1.0)
    
    colorramp_1 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': colorramp.outputs["Color"]})
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements[0].position = 0.0
    colorramp_1.color_ramp.elements[0].color = (0.0285, 0.023, 0.0151, 1.0)
    colorramp_1.color_ramp.elements[1].position = 0.0491
    colorramp_1.color_ramp.elements[1].color = (0.0283, 0.0244, 0.0155, 1.0)
    colorramp_1.color_ramp.elements[2].position = 0.1986
    colorramp_1.color_ramp.elements[2].color = (0.0346, 0.0307, 0.0184, 1.0)
    colorramp_1.color_ramp.elements[3].position = 0.257
    colorramp_1.color_ramp.elements[3].color = (0.0577, 0.0527, 0.029, 1.0)
    colorramp_1.color_ramp.elements[4].position = 0.2804
    colorramp_1.color_ramp.elements[4].color = (0.1082, 0.1029, 0.0678, 1.0)
    colorramp_1.color_ramp.elements[5].position = 0.2874
    colorramp_1.color_ramp.elements[5].color = (0.0406, 0.0361, 0.019, 1.0)
    colorramp_1.color_ramp.elements[6].position = 0.3481
    colorramp_1.color_ramp.elements[6].color = (0.0322, 0.0273, 0.0158, 1.0)
    colorramp_1.color_ramp.elements[7].position = 0.5023
    colorramp_1.color_ramp.elements[7].color = (0.0293, 0.0234, 0.0131, 1.0)
    colorramp_1.color_ramp.elements[8].position = 0.6472
    colorramp_1.color_ramp.elements[8].color = (0.0309, 0.0232, 0.0122, 1.0)
    colorramp_1.color_ramp.elements[9].position = 0.6729
    colorramp_1.color_ramp.elements[9].color = (0.1134, 0.0858, 0.0396, 1.0)
    colorramp_1.color_ramp.elements[10].position = 0.6846
    colorramp_1.color_ramp.elements[10].color = (0.0919, 0.0647, 0.0312, 1.0)
    colorramp_1.color_ramp.elements[11].position = 0.7126
    colorramp_1.color_ramp.elements[11].color = (0.0879, 0.0618, 0.0308, 1.0)
    colorramp_1.color_ramp.elements[12].position = 0.743
    colorramp_1.color_ramp.elements[12].color = (0.0991, 0.0682, 0.0319, 1.0)
    colorramp_1.color_ramp.elements[13].position = 0.771
    colorramp_1.color_ramp.elements[13].color = (0.1151, 0.0783, 0.0337, 1.0)
    colorramp_1.color_ramp.elements[14].position = 0.8037
    colorramp_1.color_ramp.elements[14].color = (0.1138, 0.0766, 0.0308, 1.0)
    colorramp_1.color_ramp.elements[15].position = 0.8248
    colorramp_1.color_ramp.elements[15].color = (0.1328, 0.0929, 0.0428, 1.0)
    colorramp_1.color_ramp.elements[16].position = 0.8458
    colorramp_1.color_ramp.elements[16].color = (0.0844, 0.0585, 0.0263, 1.0)
    colorramp_1.color_ramp.elements[17].position = 0.9252
    colorramp_1.color_ramp.elements[17].color = (0.0834, 0.0571, 0.0264, 1.0)
    colorramp_1.color_ramp.elements[18].position = 0.9603
    colorramp_1.color_ramp.elements[18].color = (0.0612, 0.0423, 0.0239, 1.0)
    colorramp_1.color_ramp.elements[19].position = 0.9766
    colorramp_1.color_ramp.elements[19].color = (0.0566, 0.0405, 0.0273, 1.0)
    colorramp_1.color_ramp.elements[20].position = 1.0
    colorramp_1.color_ramp.elements[20].color = (0.0356, 0.0247, 0.0168, 1.0)
    
    mix_1 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': group, 'Color1': colorramp_1.outputs["Color"], 'Color2': (1.0, 1.0, 1.0, 1.0)})
    
    attribute = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'value'})
    
    colorramp_2 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': attribute.outputs["Color"]})
    colorramp_2.color_ramp.elements[0].position = 0.0
    colorramp_2.color_ramp.elements[0].color = (0.2634, 0.2634, 0.2634, 1.0)
    colorramp_2.color_ramp.elements[1].position = 1.0
    colorramp_2.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': mix_1, 'Specular': 0.0, 'Roughness': colorramp_2.outputs["Color"]})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

def geometry_reptile_vor(nw: NodeWrangler, rand=True, **input_kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketVector', 'value', (0.0, 0.0, 0.0))])
    
    normal = nw.new_node(Nodes.InputNormal)
    
    multiply = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["value"], 1: normal},
        attrs={'operation': 'MULTIPLY'})
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.003
    
    multiply_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: multiply.outputs["Vector"], 1: value},
        attrs={'operation': 'MULTIPLY'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Offset': multiply_1.outputs["Vector"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_position})

def geometry_reptile_vor_attr(nw: NodeWrangler, rand=True, **input_kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None)])
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["Geometry"]})
    
    position = nw.new_node(Nodes.InputPosition)
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 1.0
    
    multiply = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: position, 1: value},
        attrs={'operation': 'MULTIPLY'})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': multiply.outputs["Vector"], 'Scale': 6.0, 'Detail': 15.0})
    
    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.1, 'Color1': multiply.outputs["Vector"], 'Color2': noise_texture.outputs["Fac"]},
        attrs={'blend_type': 'ADD'})
    
    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = 0.0
    
    value_2 = nw.new_node(Nodes.Value)
    value_2.outputs[0].default_value = 80.0
    
    voronoi_texture = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': mix, 'W': value_1, 'Scale': value_2},
        attrs={'voronoi_dimensions': '4D', 'feature': 'DISTANCE_TO_EDGE'})
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': voronoi_texture.outputs["Distance"]})
    colorramp.color_ramp.elements[0].position = 0.02
    colorramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp.color_ramp.elements[1].position = 0.2
    colorramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    noise_texture_1 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': multiply.outputs["Vector"], 'Scale': 100.0})
    
    colorramp_1 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': noise_texture_1.outputs["Fac"]})
    colorramp_1.color_ramp.elements[0].position = 0.1
    colorramp_1.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_1.color_ramp.elements[1].position = 0.4
    colorramp_1.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    mix_1 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp.outputs["Color"], 'Color1': colorramp.outputs["Color"], 'Color2': colorramp_1.outputs["Color"]})
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': set_position, 1: mix_1},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    voronoi_texture_1 = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': mix, 'W': value_1, 'Scale': value_2},
        attrs={'voronoi_dimensions': '4D'})
    
    capture_attribute_1 = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': capture_attribute.outputs["Geometry"], 1: voronoi_texture_1.outputs["Position"]},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': capture_attribute_1.outputs["Geometry"], 'attr1': capture_attribute.outputs["Attribute"], 'attr2': capture_attribute_1.outputs["Attribute"]})

def apply(obj, geo_kwargs=None, shader_kwargs=None, **kwargs):
    surface.add_geomod(obj, geometry_reptile_vor_attr, input_kwargs=geo_kwargs, attributes=['value', 'index'])
    surface.add_geomod(obj, geometry_reptile_vor, input_kwargs=geo_kwargs, attributes=[])
    surface.add_material(obj, shader_two_color, input_kwargs=shader_kwargs)
