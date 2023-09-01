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

from infinigen.assets.creatures.util.nodegroups.math import nodegroup_vector_bezier

@node_utils.to_nodegroup('nodegroup_comb_direction', singleton=True, type='GeometryNodeTree')
def nodegroup_comb_direction(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Surface', None),
            ('NodeSocketVector', 'Root Positiion', (0.0, 0.0, 0.0))])
    
    normal = nw.new_node(Nodes.InputNormal)
    
    surface_normal = nw.new_node(Nodes.SampleNearestSurface,
        input_kwargs={'Mesh': group_input.outputs["Surface"], 'Value': normal, 'Sample Position': group_input.outputs["Root Positiion"]},
        label='Surface Normal',
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    named_attribute = nw.new_node(Nodes.NamedAttribute,
        input_kwargs={'Name': 'skeleton_loc'},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    named_attribute_1 = nw.new_node(Nodes.NamedAttribute,
        input_kwargs={'Name': 'parent_skeleton_loc'},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    subtract = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: named_attribute.outputs["Attribute"], 1: named_attribute_1.outputs["Attribute"]},
        attrs={'operation': 'SUBTRACT'})
    
    normalize = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: subtract.outputs["Vector"]},
        attrs={'operation': 'NORMALIZE'})
    
    skeleton_tangent = nw.new_node(Nodes.SampleNearestSurface,
        input_kwargs={'Mesh': group_input.outputs["Surface"], 'Value': normalize.outputs["Vector"], 'Sample Position': group_input.outputs["Root Positiion"]},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    cross_product = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: surface_normal, 1: skeleton_tangent},
        attrs={'operation': 'CROSS_PRODUCT'})
    
    cross_product_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: surface_normal, 1: cross_product.outputs["Vector"]},
        attrs={'operation': 'CROSS_PRODUCT'})
    
    normalize_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: cross_product_1.outputs["Vector"]},
        attrs={'operation': 'NORMALIZE'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={
            'Combing Direction': normalize_1.outputs["Vector"], 
            'Surface Normal': (surface_normal, "Value"), 
            'Skeleton Tangent': skeleton_tangent
        })

@node_utils.to_nodegroup('nodegroup_hair_position', singleton=True, type='GeometryNodeTree')
def nodegroup_hair_position(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Curves', None)])
    
    position = nw.new_node(Nodes.InputPosition)
    
    index = nw.new_node(Nodes.Index)
    
    spline_length = nw.new_node(Nodes.SplineLength)
    
    snap = nw.new_node(Nodes.Math,
        input_kwargs={0: index, 1: spline_length.outputs["Point Count"]},
        attrs={'operation': 'SNAP'})
    
    hair_root_position = nw.new_node(Nodes.SampleIndex,
        input_kwargs={'Geometry': group_input.outputs["Curves"], 'Value': position, 'Index': snap},
        label='Hair Root Position',
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    position_1 = nw.new_node(Nodes.InputPosition)
    
    relative_position = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: position_1, 1: hair_root_position},
        label='Relative Position',
        attrs={'operation': 'SUBTRACT'})

    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={
            'Root Position': hair_root_position, 
            'Relative Position': relative_position.outputs["Vector"]
        })

@node_utils.to_nodegroup('nodegroup_comb_hairs', singleton=True, type='GeometryNodeTree')
def nodegroup_comb_hairs(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Curves', None),
            ('NodeSocketVector', 'Root Position', (0.0, 0.0, 0.0)),
            ('NodeSocketVector', 'Comb Dir', (0.0, 0.0, 0.0)),
            ('NodeSocketVector', 'Surface Normal', (0.0, 0.0, 0.0)),
            ('NodeSocketFloat', 'Length', 0.03),
            ('NodeSocketFloat', 'Puiff', 1.0),
            ('NodeSocketFloat', 'Comb', 1.0)])
    
    spline_parameter = nw.new_node(Nodes.SplineParameter)
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Surface Normal"], 'Scale': group_input.outputs["Comb"]},
        attrs={'operation': 'SCALE'})
    
    scale_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Comb Dir"], 'Scale': group_input.outputs["Puiff"]},
        attrs={'operation': 'SCALE'})
    
    add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: scale_1.outputs["Vector"], 1: scale.outputs["Vector"]})
    
    vectorbezier = nw.new_node(nodegroup_vector_bezier().name,
        input_kwargs={'t': spline_parameter.outputs["Factor"], 'b': scale.outputs["Vector"], 'c': add.outputs["Vector"]})
    
    length = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: add.outputs["Vector"]},
        attrs={'operation': 'LENGTH'})
    
    divide = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Length"], 1: length.outputs["Value"]},
        attrs={'operation': 'DIVIDE'})
    
    scale_2 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: vectorbezier, 'Scale': divide},
        attrs={'operation': 'SCALE'})
    
    add_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Root Position"], 1: scale_2.outputs["Vector"]})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["Curves"], 'Position': add_1.outputs["Vector"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_position})

@node_utils.to_nodegroup('nodegroup_strand_noise', singleton=False, type='GeometryNodeTree')
def nodegroup_strand_noise(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketFloat', 'Random Mag', 0.001),
            ('NodeSocketFloat', 'Perlin Mag', 1.0),
            ('NodeSocketFloat', 'Perlin Scale', 5.0)])
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Scale': group_input.outputs["Perlin Scale"], 'Detail': 10.0, 'Roughness': 1.0})
    
    subtract = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: noise_texture.outputs["Color"], 1: (0.5, 0.5, 0.5)},
        attrs={'operation': 'SUBTRACT'})
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: subtract.outputs["Vector"], 'Scale': group_input.outputs["Perlin Mag"]},
        attrs={'operation': 'SCALE'})
    
    random_value = nw.new_node(Nodes.RandomValue,
        input_kwargs={0: (-1.0, -1.0, -1.0)},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    scale_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: random_value.outputs["Value"], 'Scale': group_input.outputs["Random Mag"]},
        attrs={'operation': 'SCALE'})
    
    add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: scale.outputs["Vector"], 1: scale_1.outputs["Vector"]})
    
    add_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: add.outputs["Vector"]})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Offset': add_1.outputs["Vector"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_position})

@node_utils.to_nodegroup('nodegroup_duplicate_to_clumps', singleton=False, type='GeometryNodeTree')
def nodegroup_duplicate_to_clumps(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketVector', 'Surface Normal', (0.0, 0.0, 0.0)),
            ('NodeSocketInt', 'Amount', 3),
            ('NodeSocketFloat', 'Tuft Spread', 0.01),
            ('NodeSocketFloat', 'Tuft Clumping', 0.5)])
    
    duplicate_elements = nw.new_node(Nodes.DuplicateElements,
        attrs={'domain': 'SPLINE'},
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Amount': group_input.outputs["Amount"]})
    
    random_value = nw.new_node(Nodes.RandomValue,
        input_kwargs={0: (-1.0, -1.0, -1.0)},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: random_value.outputs["Value"], 'Scale': group_input.outputs["Tuft Spread"]},
        attrs={'operation': 'SCALE'})
    
    project = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: scale.outputs["Vector"], 1: group_input.outputs["Surface Normal"]},
        attrs={'operation': 'PROJECT'})
    
    subtract = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: scale.outputs["Vector"], 1: project.outputs["Vector"]},
        attrs={'operation': 'SUBTRACT'})
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': duplicate_elements.outputs["Geometry"], 1: subtract.outputs["Vector"]},
        attrs={'domain': 'CURVE', 'data_type': 'FLOAT_VECTOR'})
    
    spline_parameter = nw.new_node(Nodes.SplineParameter)
    
    subtract_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: 1.0, 1: group_input.outputs["Tuft Clumping"]},
        attrs={'operation': 'SUBTRACT'})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': spline_parameter.outputs["Factor"], 3: 1.0, 4: subtract_1})
    
    scale_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: capture_attribute.outputs["Attribute"], 'Scale': map_range.outputs["Result"]},
        attrs={'operation': 'SCALE'})
    
    set_position_1 = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': capture_attribute.outputs["Geometry"], 'Offset': scale_1.outputs["Vector"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_position_1})

@node_utils.to_nodegroup('nodegroup_hair_length_rescale', singleton=False, type='GeometryNodeTree')
def nodegroup_hair_length_rescale(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Curves', None),
            ('NodeSocketFloat', 'Min', 0.69999999999999996)])
    
    random_value_1 = nw.new_node(Nodes.RandomValue,
        input_kwargs={2: group_input.outputs["Min"]})
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': group_input.outputs["Curves"], 2: random_value_1.outputs[1]},
        attrs={'domain': 'CURVE'})
    
    hairposition = nw.new_node(nodegroup_hair_position().name,
        input_kwargs={'Curves': group_input.outputs["Curves"]})
    
    multiply_add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: hairposition.outputs["Relative Position"], 1: capture_attribute.outputs[2], 2: hairposition.outputs["Root Position"]},
        attrs={'operation': 'MULTIPLY_ADD'})
    
    set_position_1 = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': capture_attribute.outputs["Geometry"], 'Position': multiply_add.outputs["Vector"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_position_1})

@node_utils.to_nodegroup('nodegroup_snap_roots_to_surface', singleton=True, type='GeometryNodeTree')
def nodegroup_snap_roots_to_surface(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Target', None),
            ('NodeSocketGeometry', 'Curves', None)])
    
    hair_pos = nw.new_node(nodegroup_hair_position().name,
        input_kwargs={'Curves': group_input.outputs["Curves"]})
    
    geometry_proximity = nw.new_node(Nodes.Proximity,
        input_kwargs={'Target': group_input.outputs["Target"], 'Source Position': hair_pos.outputs["Root Position"]})
    
    add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: geometry_proximity.outputs["Position"], 1: hair_pos.outputs["Relative Position"]})
    
    set_position_2 = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["Curves"], 'Position': add.outputs["Vector"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_position_2})