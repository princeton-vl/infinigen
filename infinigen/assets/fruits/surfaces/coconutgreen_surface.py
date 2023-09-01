# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo


import bpy
import mathutils
from numpy.random import uniform, normal, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface

from infinigen.assets.fruits.fruit_utils import nodegroup_add_dent, nodegroup_surface_bump
from infinigen.assets.fruits.surfaces.surface_utils import nodegroup_stripe_pattern

def shader_coconut_green_shader(nw: NodeWrangler, basic_color, bottom_color):
    # Code generated using version 2.4.3 of the node_transpiler

    texture_coordinate_1 = nw.new_node(Nodes.TextureCoord)
    
    noise_texture_1 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': texture_coordinate_1.outputs["Object"], 'Scale': 1.0, 'Detail': 10.0, 'Roughness': 0.7})
    
    separate_rgb = nw.new_node(Nodes.SeparateColor,
        input_kwargs={'Color': noise_texture_1.outputs["Color"]})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_rgb.outputs["Green"], 1: 0.4, 2: 0.7, 3: 0.48, 4: 0.52},
        attrs={'interpolation_type': 'SMOOTHSTEP'})
    
    map_range_2 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_rgb.outputs["Blue"], 1: 0.4, 2: 0.7, 3: 0.6},
        attrs={'interpolation_type': 'SMOOTHSTEP'})
    
    attribute_1 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'shape_coordinate'})
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': attribute_1.outputs["Fac"]})
    colorramp.color_ramp.elements.new(0)
    colorramp.color_ramp.elements[0].position = 0.0
    colorramp.color_ramp.elements[0].color = bottom_color # (0.0908, 0.2664, 0.013, 1.0)
    colorramp.color_ramp.elements[1].position = 0.01
    colorramp.color_ramp.elements[1].color = bottom_color # (0.0908, 0.2664, 0.013, 1.0)
    colorramp.color_ramp.elements[2].position = 1.0
    colorramp.color_ramp.elements[2].color = basic_color # (0.2462, 0.4125, 0.0044, 1.0)
    
    hue_saturation_value_1 = nw.new_node('ShaderNodeHueSaturation',
        input_kwargs={'Hue': map_range_1.outputs["Result"], 'Value': map_range_2.outputs["Result"], 'Color': colorramp.outputs["Color"]})
    
    attribute_2 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'crosssection_coordinate'})
    
    group = nw.new_node(nodegroup_stripe_pattern().name,
        input_kwargs={'Color': hue_saturation_value_1, 'attribute': attribute_2.outputs["Fac"], 'seed': 10.0})
    
    group_1 = nw.new_node(nodegroup_stripe_pattern().name,
        input_kwargs={'Color': group, 'attribute': attribute_1.outputs["Fac"], 'voronoi scale': 10.0, 'voronoi randomness': 0.6446, 'seed': -10.0, 'noise amount': 0.48, 'hue min': 1.32, 'hue max': 0.9})
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': group_1, 'Specular': 0.4773, 'Roughness': 0.4455})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

@node_utils.to_nodegroup('nodegroup_coconutgreen_surface', singleton=False, type='GeometryNodeTree')
def nodegroup_coconutgreen_surface(nw: NodeWrangler, basic_color=(0.2462, 0.4125, 0.0044, 1.0), bottom_color=(0.0908, 0.2664, 0.013, 1.0)):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketFloat', 'spline parameter', 0.0),
            ('NodeSocketVector', 'spline tangent', (0.0, 0.0, 0.0)),
            ('NodeSocketFloat', 'distance to center', 0.0),
            ('NodeSocketFloat', 'cross section paramater', 0.5)])
    
    surfacebump = nw.new_node(nodegroup_surface_bump().name,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Displacement': 0.2, 'Scale': 0.5})
    
    surfacebump_1 = nw.new_node(nodegroup_surface_bump().name,
        input_kwargs={'Geometry': surfacebump, 'Displacement': 0.0, 'Scale': 10.0})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': group_input.outputs["distance to center"], 1: 0.05, 2: 0.2, 4: 0.68})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["cross section paramater"], 1: map_range.outputs["Result"]},
        attrs={'operation': 'MULTIPLY'})
    
    adddent = nw.new_node(nodegroup_add_dent(dent_control_points=[(0.0, 0.4219), (0.0977, 0.4469), (0.2273, 0.4844), (0.5568, 0.5125), (1.0, 0.5)]).name,
        input_kwargs={'Geometry': surfacebump_1, 'spline parameter': group_input.outputs["spline parameter"], 'spline tangent': group_input.outputs["spline tangent"], 'distance to center': group_input.outputs["distance to center"], 'bottom': True, 'intensity': multiply, 'max radius': 3.0})
    
    set_material_3 = nw.new_node(Nodes.SetMaterial,
        input_kwargs={'Geometry': adddent, 'Material': surface.shaderfunc_to_material(shader_coconut_green_shader, basic_color, bottom_color)})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_material_3})