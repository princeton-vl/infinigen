# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import bpy
import mathutils
from numpy.random import uniform, normal, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface

@node_utils.to_nodegroup('nodegroup_floor_ceil', singleton=False, type='GeometryNodeTree')
def nodegroup_floor_ceil(nw: NodeWrangler):
    # Code generated using version 2.6.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketFloat', 'Value', 0.0000)])
    
    float_to_integer = nw.new_node(Nodes.FloatToInt, input_kwargs={'Float': group_input.outputs["Value"]}, attrs={'rounding_mode': 'FLOOR'})
    
    float_to_integer_1 = nw.new_node(Nodes.FloatToInt,
        input_kwargs={'Float': group_input.outputs["Value"]},
        attrs={'rounding_mode': 'CEILING'})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Value"], 1: float_to_integer},
        attrs={'operation': 'SUBTRACT'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Floor': float_to_integer, 'Ceil': float_to_integer_1, 'Remainder': subtract},
        attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_clamp_or_wrap', singleton=False, type='GeometryNodeTree')
def nodegroup_clamp_or_wrap(nw: NodeWrangler):
    # Code generated using version 2.6.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketInt', 'Value', 0),
            ('NodeSocketFloat', 'Max', 0.5000),
            ('NodeSocketBool', 'Use Wrap', False)])
    
    clamp = nw.new_node(Nodes.Clamp, input_kwargs={'Value': group_input.outputs["Value"], 'Max': group_input.outputs["Max"]})
    
    wrap = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Value"], 1: group_input.outputs["Max"], 2: 0.0000},
        attrs={'operation': 'WRAP'})
    
    switch = nw.new_node(Nodes.Switch,
        input_kwargs={0: group_input.outputs["Use Wrap"], 4: clamp, 5: wrap},
        attrs={'input_type': 'INT'})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Output': switch.outputs[1]}, attrs={'is_active_output': True})


@node_utils.to_nodegroup('nodegroup_polar_to_cart', singleton=True, type='GeometryNodeTree')
def nodegroup_polar_to_cart(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'Angle', 0.5),
            ('NodeSocketFloat', 'Length', 0.0),
            ('NodeSocketVector', 'Origin', (0.0, 0.0, 0.0))])
    
    cosine = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Angle"]},
        attrs={'operation': 'COSINE'})
    
    sine = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Angle"]},
        attrs={'operation': 'SINE'})
    
    construct_unit_vector = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': cosine, 'Z': sine},
        label='Construct Unit Vector')
    
    offset_polar = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Length"], 1: construct_unit_vector, 2: group_input.outputs["Origin"]},
        label='Offset Polar',
        attrs={'operation': 'MULTIPLY_ADD'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Vector': offset_polar.outputs["Vector"]})

@node_utils.to_nodegroup('nodegroup_switch4', singleton=True, type='GeometryNodeTree')
def nodegroup_switch4(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketInt', 'Arg', 0),
            ('NodeSocketVector', 'Arg == 0', (0.0, 0.0, 0.0)),
            ('NodeSocketVector', 'Arg == 1', (0.0, 0.0, 0.0)),
            ('NodeSocketVector', 'Arg == 2', (0.0, 0.0, 0.0)),
            ('NodeSocketVector', 'Arg == 3', (0.0, 0.0, 0.0))])
    
    greater_equal = nw.new_node(Nodes.Compare,
        input_kwargs={2: group_input.outputs["Arg"], 3: 2},
        attrs={'data_type': 'INT', 'operation': 'GREATER_EQUAL'})
    
    greater_equal_1 = nw.new_node(Nodes.Compare,
        input_kwargs={2: group_input.outputs["Arg"], 3: 1},
        attrs={'data_type': 'INT', 'operation': 'GREATER_EQUAL'})
    
    switch_1 = nw.new_node(Nodes.Switch,
        input_kwargs={0: greater_equal_1, 8: group_input.outputs["Arg == 0"], 9: group_input.outputs["Arg == 1"]},
        attrs={'input_type': 'VECTOR'})
    
    greater_equal_2 = nw.new_node(Nodes.Compare,
        input_kwargs={2: group_input.outputs["Arg"], 3: 3},
        attrs={'data_type': 'INT', 'operation': 'GREATER_EQUAL'})
    
    switch_2 = nw.new_node(Nodes.Switch,
        input_kwargs={0: greater_equal_2, 8: group_input.outputs["Arg == 2"], 9: group_input.outputs["Arg == 3"]},
        attrs={'input_type': 'VECTOR'})
    
    switch = nw.new_node(Nodes.Switch,
        input_kwargs={0: greater_equal, 8: switch_1.outputs[3], 9: switch_2.outputs[3]},
        attrs={'input_type': 'VECTOR'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Output': switch.outputs[3]})

@node_utils.to_nodegroup('nodegroup_deg2_rad', singleton=True, type='GeometryNodeTree')
def nodegroup_deg2_rad(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'Deg', (0.0, 0.0, 0.0))])
    
    multiply = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Deg"], 1: (0.0175, 0.0175, 0.0175)},
        attrs={'operation': 'MULTIPLY'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Rad': multiply.outputs["Vector"]})

@node_utils.to_nodegroup('nodegroup_aspect_to_dim', singleton=True, type='GeometryNodeTree')
def nodegroup_aspect_to_dim(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'Aspect Ratio', 1.0)])
    
    greater_than = nw.new_node(Nodes.Compare,
        input_kwargs={0: group_input.outputs["Aspect Ratio"], 1: 1.0})
    
    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': group_input.outputs["Aspect Ratio"], 'Y': 1.0})
    
    divide = nw.new_node(Nodes.Math,
        input_kwargs={0: 1.0, 1: group_input.outputs["Aspect Ratio"]},
        attrs={'operation': 'DIVIDE'})
    
    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': 1.0, 'Y': divide})
    
    switch = nw.new_node(Nodes.Switch,
        input_kwargs={0: greater_than, 8: combine_xyz_1, 9: combine_xyz_2},
        attrs={'input_type': 'VECTOR'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'XY Scale': switch.outputs[3]})

@node_utils.to_nodegroup('nodegroup_vector_sum', singleton=True, type='GeometryNodeTree')
def nodegroup_vector_sum(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'Vector', (0.0, 0.0, 0.0))])
    
    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group_input.outputs["Vector"]})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["X"], 1: separate_xyz_1.outputs["Y"]})
    
    add_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: add, 1: separate_xyz_1.outputs["Z"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Sum': add_1})

@node_utils.to_nodegroup('nodegroup_vector_bezier', singleton=True, type='GeometryNodeTree')
def nodegroup_vector_bezier(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 't', 0.0),
            ('NodeSocketVector', 'a', (0.0, 0.0, 0.0)),
            ('NodeSocketVector', 'b', (0.0, 0.0, 0.0)),
            ('NodeSocketVector', 'c', (0.0, 0.0, 0.0))])
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Vector': group_input.outputs["t"], 9: group_input.outputs["a"], 10: group_input.outputs["b"]},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Vector': group_input.outputs["t"], 9: map_range.outputs["Vector"], 10: group_input.outputs["c"]},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Vector': map_range_1.outputs["Vector"]})
