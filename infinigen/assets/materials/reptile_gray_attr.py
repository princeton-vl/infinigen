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

def shader_gray(nw: NodeWrangler, rand=True, **input_kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    group = nw.new_node(nodegroup_color_mask().name)
    
    colorramp_2 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': group})
    colorramp_2.color_ramp.elements[0].position = 0.0
    colorramp_2.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_2.color_ramp.elements[1].position = 1.0
    colorramp_2.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    attribute = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'value'})
    
    attribute_1 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'index'})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': attribute_1.outputs["Color"], 'Scale': 10.0})
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': noise_texture.outputs["Color"]})
    colorramp.color_ramp.elements.new(0)
    colorramp.color_ramp.elements.new(0)
    colorramp.color_ramp.elements[0].position = 0.1
    colorramp.color_ramp.elements[0].color = (0.7802, 0.7802, 0.7802, 1.0)
    colorramp.color_ramp.elements[1].position = 0.25
    colorramp.color_ramp.elements[1].color = (0.4839, 0.4839, 0.4839, 1.0)
    colorramp.color_ramp.elements[2].position = 0.6
    colorramp.color_ramp.elements[2].color = (0.0489, 0.0488, 0.0489, 1.0)
    colorramp.color_ramp.elements[3].position = 0.8227
    colorramp.color_ramp.elements[3].color = (0.0282, 0.0252, 0.027, 1.0)
    
    texture_coordinate = nw.new_node(Nodes.TextureCoord)
    
    noise_texture_1 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': texture_coordinate.outputs["Generated"], 'Scale': 200.0})
    
    colorramp_1 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': noise_texture_1.outputs["Fac"]})
    colorramp_1.color_ramp.elements[0].position = 0.0
    colorramp_1.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_1.color_ramp.elements[1].position = 1.0
    colorramp_1.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp.outputs["Color"], 'Color1': colorramp.outputs["Color"], 'Color2': colorramp_1.outputs["Color"]})
    
    mix_1 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': attribute.outputs["Color"], 'Color1': (0.033, 0.033, 0.033, 1.0), 'Color2': mix})
    
    mix_2 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp_2.outputs["Color"], 'Color1': mix_1, 'Color2': (1.0, 1.0, 1.0, 1.0)})
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': mix_2, 'Roughness': 1.0})
    
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
    surface.add_material(obj, shader_gray, input_kwargs=shader_kwargs)

