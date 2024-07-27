# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: David Yan

import bpy

from infinigen.core.nodes.node_wrangler import geometry_node_group_empty_new

visited_nodes = []


def find_connected(node, origin_socket_index, tree_origin_socket=False):  # WIP, unused
    if node in visited_nodes:
        return
    visited_nodes.append(node)

    if node.type == "GROUP":
        find_connected(node.node_tree.nodes["Group Output"], origin_socket_index, True)

    if tree_origin_socket:
        for link in node.inputs[origin_socket_index].links:
            from_node = link.from_node
            find_connected(from_node, from_node.outputs[:].index(link.from_socket))
    else:
        for index, _ in enumerate(node.inputs):
            for link in node.inputs[index].links:
                from_node = link.from_node
                find_connected(from_node, from_node.outputs[:].index(link.from_socket))


def remove_unconnected(node_tree):  # WIP, unused
    nodes = node_tree.nodes
    for node in nodes:
        if node not in visited_nodes:
            nodes.remove(node)
            continue
        if node.type == "GROUP":
            remove_unconnected(node.node_tree)


def copy_nodes(shader_node_tree, geo_node_tree):  # WIP, unused
    shader_nodes = shader_node_tree.nodes
    for shader_node in shader_nodes:
        if shader_node.type == "GROUP":
            geo_node = geo_node_tree.nodes.new("GeometryNodeGroup")
            copy_nodes(geo_node.node_tree, shader_node.node_tree)
        else:
            try:
                geo_node = geo_node_tree.nodes.new(shader_node.bl_idname)
                geo_node.location = shader_node.location
                geo_node.width = shader_node.width
            except RuntimeError:
                continue


def bake_vertex_colors(obj):
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.cycles.samples = 1
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    vertColor = bpy.context.object.data.color_attributes.new(
        name="Displacement", domain="POINT", type="FLOAT_COLOR"
    )
    bpy.context.object.data.attributes.active_color = vertColor
    bpy.ops.object.bake(type="EMIT", pass_filter={"COLOR"}, target="VERTEX_COLORS")
    obj.select_set(False)


def create_modifier(obj, scale_val, apply_geo_modifier):
    modifier = obj.modifiers.new("Displacement", "NODES")
    modifier.node_group = geometry_node_group_empty_new()
    nodes = modifier.node_group.nodes
    normal = nodes.new(type="GeometryNodeInputNormal")
    attribute = nodes.new(type="GeometryNodeInputNamedAttribute")
    attribute.data_type = "FLOAT_COLOR"
    attribute.inputs[0].default_value = "Displacement"
    set_pos = nodes.new(type="GeometryNodeSetPosition")
    mult = nodes.new(type="ShaderNodeVectorMath")
    mult.operation = "MULTIPLY"
    scale = nodes.new(type="ShaderNodeVectorMath")
    scale.operation = "SCALE"
    scale.inputs["Scale"].default_value = scale_val
    output = nodes["Group Output"]
    input = nodes["Group Input"]

    modifier.node_group.links.new(input.outputs["Geometry"], set_pos.inputs["Geometry"])
    modifier.node_group.links.new(
        attribute.outputs[2], mult.inputs[0]
    )  # index 2 must be hardcoded
    modifier.node_group.links.new(normal.outputs["Normal"], mult.inputs[1])
    modifier.node_group.links.new(mult.outputs["Vector"], scale.inputs["Vector"])
    modifier.node_group.links.new(scale.outputs["Vector"], set_pos.inputs["Offset"])
    modifier.node_group.links.new(
        set_pos.outputs["Geometry"], output.inputs["Geometry"]
    )

    if apply_geo_modifier:
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier="Displacement")
        obj.select_set(False)


def convert_shader_displacement(obj, apply_geo_modifier=True):
    displaced_materials = {}

    for slot in obj.material_slots:
        mat = slot.material
        nodes = mat.node_tree.nodes
        if nodes.get("Displacement"):
            scale_val = nodes["Displacement"].inputs["Scale"].default_value
            displacement_link = nodes["Displacement"].inputs["Height"].links[0]
            displace_socket = displacement_link.from_socket
            bsdf_link = nodes["Material Output"].inputs["Surface"].links[0]
            bsdf_socket = bsdf_link.from_socket
            mat.node_tree.links.remove(displacement_link)
            mat.node_tree.links.new(
                displace_socket, nodes["Material Output"].inputs["Surface"]
            )
            displaced_materials[mat] = bsdf_socket

    if len(displaced_materials) != 0:
        bake_vertex_colors(obj)
        create_modifier(obj, scale_val, apply_geo_modifier)

    for mat in displaced_materials:
        mat = slot.material
        mat.node_tree.links.remove(nodes["Material Output"].inputs["Surface"].links[0])
        mat.node_tree.links.new(
            displaced_materials[mat], nodes["Material Output"].inputs["Surface"]
        )
