# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: David Yan

import bpy


def find_displacement_node(mat):
    links = mat.node_tree.links
    shader_nodes = mat.node_tree.nodes
    outputNode = shader_nodes["Material Output"]
    displacement_node = None
    for link in links:
        if link.to_node == outputNode and link.to_socket.name == "Displacement":
            displacement_node = link.from_node
            break
    return displacement_node


def convert_shader_displacement(mat: bpy.types.Material):
    mat_copy = mat.copy()
    mat_copy.name = mat.name + "_copy"

    shader_nodes = mat_copy.node_tree.nodes

    displacement_node = find_displacement_node(mat_copy)

    assert displacement_node is not None

    height = displacement_node.inputs["Height"].default_value
    mid_level = displacement_node.inputs["Midlevel"].default_value
    scale = displacement_node.inputs["Scale"].default_value

    new_scale = (height - mid_level) * scale
    shader_nodes.remove(displacement_node)

    geo_node_group = bpy.data.node_groups.new("GeometryNodes", "GeometryNodeTree")
    group_input = geo_node_group.nodes.new("NodeGroupInput")
    group_output = geo_node_group.nodes.new("NodeGroupOutput")
    geo_node_group.interface.new_socket(
        name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
    )
    geo_node_group.interface.new_socket(
        name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )
    set_pos = geo_node_group.nodes.new("GeometryNodeSetPosition")
    normal = geo_node_group.nodes.new("GeometryNodeInputNormal")
    scale = geo_node_group.nodes.new("ShaderNodeVectorMath")
    scale.operation = "SCALE"
    scale.inputs["Scale"].default_value = new_scale

    geo_node_group.links.new(group_input.outputs[0], set_pos.inputs["Geometry"])
    geo_node_group.links.new(normal.outputs["Normal"], scale.inputs["Vector"])
    geo_node_group.links.new(scale.outputs["Vector"], set_pos.inputs["Offset"])
    geo_node_group.links.new(set_pos.outputs["Geometry"], group_output.inputs[0])

    return mat_copy, geo_node_group
