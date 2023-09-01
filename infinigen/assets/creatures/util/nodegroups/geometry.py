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

@node_utils.to_nodegroup('nodegroup_symmetric_instance', singleton=True, type='GeometryNodeTree')
def nodegroup_symmetric_instance(nw: NodeWrangler):
    # Code generated using version 2.4.1 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketVector', 'Offset', (0.0, 0.0, 0.0)),
            ('NodeSocketVector', 'Reflector', (1.0, -1.0, 1.0))])
    
    multiply = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Offset"], 1: group_input.outputs["Reflector"]},
        attrs={'operation': 'MULTIPLY'})
    
    mesh_line = nw.new_node(Nodes.MeshLine,
        input_kwargs={'Count': 2, 'Start Location': group_input.outputs["Offset"], 'Offset': multiply.outputs["Vector"]},
        attrs={'mode': 'END_POINTS'})
    
    instance_on_points = nw.new_node(Nodes.InstanceOnPoints,
        input_kwargs={'Points': mesh_line, 'Instance': group_input.outputs["Geometry"]})
    
    index = nw.new_node(Nodes.Index)
    
    equal = nw.new_node(Nodes.Compare,
        input_kwargs={2: index},
        attrs={'data_type': 'INT', 'operation': 'EQUAL'})
    
    scale_instances = nw.new_node(Nodes.ScaleInstances,
        input_kwargs={'Instances': instance_on_points, 'Selection': equal})
    
    flip_faces = nw.new_node(Nodes.FlipFaces,
        input_kwargs={'Mesh': scale_instances, 'Selection': equal})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Instances': flip_faces})

@node_utils.to_nodegroup('nodegroup_symmetric_clone', singleton=True, type='GeometryNodeTree')
def nodegroup_symmetric_clone(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketVectorXYZ', 'Scale', (1.0, -1.0, 1.0))])
    
    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Scale': group_input.outputs["Scale"]})
    
    flip_faces = nw.new_node(Nodes.FlipFaces,
        input_kwargs={'Mesh': transform})
    
    join_geometry_2 = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [group_input.outputs["Geometry"], flip_faces]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Both': join_geometry_2, 'Orig': group_input.outputs["Geometry"], 'Inverted': flip_faces})


@node_utils.to_nodegroup('nodegroup_solidify', singleton=True, type='GeometryNodeTree')
def nodegroup_solidify(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Mesh', None),
            ('NodeSocketFloatDistance', 'Distance', 0.0)])
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Distance"]},
        attrs={'operation': 'MULTIPLY'})
    
    extrude_mesh = nw.new_node(Nodes.ExtrudeMesh,
        input_kwargs={'Mesh': group_input.outputs["Mesh"], 'Offset Scale': multiply, 'Individual': False})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Distance"], 1: -0.5},
        attrs={'operation': 'MULTIPLY'})
    
    extrude_mesh_1 = nw.new_node(Nodes.ExtrudeMesh,
        input_kwargs={'Mesh': group_input.outputs["Mesh"], 'Offset Scale': multiply_1, 'Individual': False})
    
    flip_faces = nw.new_node(Nodes.FlipFaces,
        input_kwargs={'Mesh': extrude_mesh_1.outputs["Mesh"]})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [extrude_mesh.outputs["Mesh"], flip_faces]})
    
    merge_by_distance = nw.new_node(Nodes.MergeByDistance,
        input_kwargs={'Geometry': join_geometry, 'Distance': 0.0})
    
    set_shade_smooth = nw.new_node(Nodes.SetShadeSmooth,
        input_kwargs={'Geometry': merge_by_distance, 'Shade Smooth': False})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_shade_smooth})

@node_utils.to_nodegroup('nodegroup_taper', singleton=True, type='GeometryNodeTree')
def nodegroup_taper(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketVector', 'Start', (1.0, 0.63, 0.72)),
            ('NodeSocketVector', 'End', (1.0, 1.0, 1.0))])
    
    position = nw.new_node(Nodes.InputPosition)
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': position})
    
    attribute_statistic = nw.new_node(Nodes.AttributeStatistic,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 2: separate_xyz.outputs["X"]})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Vector': separate_xyz.outputs["X"], 7: attribute_statistic.outputs["Min"], 8: attribute_statistic.outputs["Max"], 9: group_input.outputs["Start"], 10: group_input.outputs["End"]},
        attrs={'data_type': 'FLOAT_VECTOR', 'clamp': False})
    
    multiply = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: position, 1: map_range.outputs["Vector"]},
        attrs={'operation': 'MULTIPLY'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Position': multiply.outputs["Vector"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_position})