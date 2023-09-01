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

def shader_black_w_noise_shader(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group = nw.new_node(nodegroup_color_noise().name,
        input_kwargs={'Scale': 10.0, 'Color': (0.0779, 0.0839, 0.0809, 1.0)})
    
    principled_bsdf_1 = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': group, 'Metallic': 0.9, 'Specular': 0.5114, 'Roughness': 0.2568})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf_1})

@node_utils.to_nodegroup('nodegroup_add_noise', singleton=False, type='ShaderNodeTree')
def nodegroup_add_noise(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'Vector', (0.0, 0.0, 0.0)),
            ('NodeSocketFloat', 'Scale', 10.0),
            ('NodeSocketVector', 'amount', (0.1, 0.26, 0.0)),
            ('NodeSocketFloat', 'seed', 0.0),
            ('NodeSocketVector', 'Noise Eval Position', None)])
    
    noise_texture_1 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': group_input.outputs["Noise Eval Position"], 'W': group_input.outputs["seed"], 'Scale': group_input.outputs["Scale"]},
        attrs={'noise_dimensions': '4D'})
    
    subtract = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: noise_texture_1.outputs["Color"], 1: (0.5, 0.5, 0.5)},
        attrs={'operation': 'SUBTRACT'})
    
    multiply = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: subtract.outputs["Vector"], 1: group_input.outputs["amount"]},
        attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: multiply.outputs["Vector"], 1: group_input.outputs["Vector"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Vector': add.outputs["Vector"]})

@node_utils.to_nodegroup('nodegroup_color_noise', singleton=False, type='ShaderNodeTree')
def nodegroup_color_noise(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    texture_coordinate = nw.new_node(Nodes.TextureCoord)
    
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'Scale', 0.8),
            ('NodeSocketColor', 'Color', (0.0147, 0.0156, 0.0152, 1.0)),
            ('NodeSocketFloat', 'Hue From Min', 0.4),
            ('NodeSocketFloat', 'Hue From Max', 0.7),
            ('NodeSocketFloat', 'Hue To Min', 0.48),
            ('NodeSocketFloat', 'Hue To Max', 0.55),
            ('NodeSocketFloat', 'Value From Min', 0.4),
            ('NodeSocketFloat', 'Value From Max', 0.78),
            ('NodeSocketFloat', 'Value To Min', -0.56),
            ('NodeSocketFloat', 'Value To Max', 1.0)])
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': texture_coordinate.outputs["Object"], 'Scale': group_input.outputs["Scale"], 'Detail': 10.0, 'Roughness': 0.7})
    
    separate_rgb = nw.new_node(Nodes.SeparateColor,
        input_kwargs={'Color': noise_texture.outputs["Color"]},
        attrs={'mode': 'HSV'})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_rgb.outputs["Green"], 1: group_input.outputs["Hue From Min"], 2: group_input.outputs["Hue From Max"], 3: group_input.outputs["Hue To Min"], 4: group_input.outputs["Hue To Max"]},
        attrs={'interpolation_type': 'SMOOTHSTEP'})
    
    map_range_2 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_rgb.outputs["Blue"], 1: group_input.outputs["Value From Min"], 2: group_input.outputs["Value From Max"], 3: group_input.outputs["Value To Min"], 4: group_input.outputs["Value To Max"]},
        attrs={'interpolation_type': 'SMOOTHSTEP'})
    
    hue_saturation_value = nw.new_node('ShaderNodeHueSaturation',
        input_kwargs={'Hue': map_range_1.outputs["Result"], 'Value': map_range_2.outputs["Result"], 'Color': group_input.outputs["Color"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Color': hue_saturation_value})