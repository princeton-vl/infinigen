# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Mingzhe Wang and Alexander Raistrick


import bpy
import mathutils
from numpy.random import uniform as U, normal as N, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface

@node_utils.to_nodegroup('nodegroup_norm_local_pos', singleton=True, type='ShaderNodeTree')
def nodegroup_norm_local_pos(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    attribute_5 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'local_pos'})
    
    attribute_6 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'skeleton_rad'})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: attribute_6.outputs["Fac"], 1: -1.0},
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Y': multiply, 'Z': multiply})
    
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'X Max', 1.0)])
    
    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': group_input.outputs["X Max"], 'Y': attribute_6.outputs["Fac"], 'Z': attribute_6.outputs["Fac"]})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Vector': attribute_5.outputs["Vector"], 7: combine_xyz_2, 8: combine_xyz_1},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Vector': map_range_1.outputs["Vector"]})

@node_utils.to_nodegroup('nodegroup_abs_y', singleton=True, type='ShaderNodeTree')
def nodegroup_abs_y(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'Vector', (0.0, 0.0, 0.0))])
    
    separate_xyz_4 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group_input.outputs["Vector"]})
    
    absolute = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_4.outputs["Y"]},
        attrs={'operation': 'ABSOLUTE'})
    
    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': separate_xyz_4.outputs["X"], 'Y': absolute, 'Z': separate_xyz_4.outputs["Z"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Vector': combine_xyz_1})

@node_utils.to_nodegroup('nodegroup_color_mask', singleton=False, type='ShaderNodeTree')
def nodegroup_color_mask(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    attribute_2 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'tag_body'})
    
    attribute_3 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'tag_leg'})
    
    attribute_4 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'tag_head'})
    
    attribute_5 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'local_pos'})
    
    group_2 = nw.new_node(nodegroup_abs_y().name,
        input_kwargs={'Vector': attribute_5.outputs["Vector"]})
    
    musgrave_texture = nw.new_node(Nodes.MusgraveTexture,
        input_kwargs={'Vector': group_2, 'W': U(1e4), 'Scale': N(7, 1), 'Detail': N(7, 1), 'Dimension': U(1.5, 3)},
        attrs={'musgrave_dimensions': '4D'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: musgrave_texture, 1: 0.69999999999999996})
    
    colorramp_4 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': add})
    colorramp_4.color_ramp.interpolation = "EASE"
    colorramp_4.color_ramp.elements[0].position = 0.0
    colorramp_4.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_4.color_ramp.elements[1].position = 0.4864
    colorramp_4.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    group = nw.new_node(nodegroup_norm_local_pos().name)
    
    separate_xyz_4 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group})
    
    colorramp_5 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': separate_xyz_4.outputs["Z"]})
    colorramp_5.color_ramp.interpolation = "EASE"
    colorramp_5.color_ramp.elements[0].position = 0.0
    colorramp_5.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_5.color_ramp.elements[1].position = 0.5318
    colorramp_5.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: colorramp_4.outputs["Color"], 1: colorramp_5.outputs["Color"]},
        attrs={'operation': 'MULTIPLY'})
    
    mix_3 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': attribute_4.outputs["Fac"], 'Color1': (1.0, 1.0, 1.0, 1.0), 'Color2': multiply})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Scale': N(14, 2)})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Vector': noise_texture.outputs["Color"], 9: (-0.10000000000000001, -0.10000000000000001, -0.10000000000000001), 10: (0.10000000000000001, 0.10000000000000001, 0.10000000000000001)},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    add_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group, 1: map_range_1.outputs["Vector"]})
    
    separate_xyz_2 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': add_1.outputs["Vector"]})
    
    colorramp_1 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': separate_xyz_2.outputs["X"]})
    colorramp_1.color_ramp.interpolation = "EASE"
    colorramp_1.color_ramp.elements[0].position = 0.3091
    colorramp_1.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_1.color_ramp.elements[1].position = 0.9773
    colorramp_1.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    colorramp_2 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': separate_xyz_2.outputs["Y"]})
    colorramp_2.color_ramp.interpolation = "EASE"
    colorramp_2.color_ramp.elements[0].position = 0.0955
    colorramp_2.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    colorramp_2.color_ramp.elements[1].position = 0.5318
    colorramp_2.color_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: colorramp_1.outputs["Color"], 1: colorramp_2.outputs["Color"]},
        attrs={'operation': 'MULTIPLY'})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: 1.0, 1: multiply_1},
        attrs={'operation': 'SUBTRACT'})
    
    mix_2 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': attribute_3.outputs["Fac"], 'Color1': mix_3, 'Color2': subtract})
    
    separate_xyz_3 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': add_1.outputs["Vector"]})
    
    add_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_3.outputs["Z"]})
    
    colorramp_3 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': add_2})
    colorramp_3.color_ramp.elements[0].position = 0.2
    colorramp_3.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_3.color_ramp.elements[1].position = 0.6136
    colorramp_3.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    mix_1 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': attribute_2.outputs["Fac"], 'Color1': mix_2, 'Color2': colorramp_3.outputs["Color"]})
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': mix_1})
    colorramp.color_ramp.elements.new(0)
    colorramp.color_ramp.elements[0].position = 0.2727
    colorramp.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    colorramp.color_ramp.elements[1].position = 0.6091
    colorramp.color_ramp.elements[1].color = (0.78220000000000001, 0.78220000000000001, 0.78220000000000001, 1.0)
    colorramp.color_ramp.elements[2].position = 0.9727
    colorramp.color_ramp.elements[2].color = (0.0, 0.0, 0.0, 1.0)
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Color': colorramp.outputs["Color"]})