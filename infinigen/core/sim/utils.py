from typing import Any, Optional

import bpy


def get_node_by_idname(node_tree: bpy.types.NodeTree, bl_idname: str) -> bpy.types.Node:
    """
    Returns a list of nodes of the given type
    """
    return next(node for node in node_tree.nodes if node.bl_idname == bl_idname)


def is_node_group(node: bpy.types.Node) -> bool:
    """
    Returns true if the current node is a node group
    """
    return node.bl_idname == "GeometryNodeGroup"


def is_join(node: bpy.types.Node) -> bool:
    """
    Returns true if the current node is a join node
    """
    return node.bl_idname == "GeometryNodeJoinGeometry"


def is_hinge(node: bpy.types.Node) -> bool:
    """
    Returns true if the current node is a hinge joint node
    """
    return is_node_group(node) and "Hinge Joint" in node.node_tree.name


def is_sliding(node: bpy.types.Node) -> bool:
    """
    Returns true if the current node is a sliding joint node
    """
    return is_node_group(node) and "Sliding Joint" in node.node_tree.name


def is_joint(node: bpy.types.Node) -> bool:
    """
    Returns true if the current node is a joint node
    """
    return is_hinge(node) or is_sliding(node)


def is_add_metadata(node: bpy.types.Node) -> bool:
    """
    Returns true fi the node is an add metadata node.
    """
    return (
        is_node_group(node) and "Add Jointed Geometry Metadata" in node.node_tree.name
    )


def is_duplicate(node: bpy.types.Node) -> bool:
    """
    Returns true if the node is a duplicate joint node.
    """
    return is_node_group(node) and "Duplicate Joints on Parent" in node.node_tree.name


def is_switch(node: bpy.types.Node) -> bool:
    """
    Returns true if the current node is a switch or index switch node
    """
    return "Switch" in node.bl_idname


def inject_store_named_attr(
    link: bpy.types.NodeLink,
    data_type: str = "INT",
    default_name: Optional[str] = None,
    default_value: Optional[Any] = None,
):
    """
    Injects a store named attribute node of type integer between
    two nodes given the link connecting them.
    """
    parent_socket, child_socket = link.to_socket, link.from_socket

    # remove the link between parent and child node
    node_tree = link.id_data
    node_tree.links.remove(link)

    # create a new store attribute node
    store_named_attr_node = node_tree.nodes.new("GeometryNodeStoreNamedAttribute")
    store_named_attr_node.data_type = data_type
    if default_name:
        store_named_attr_node.inputs["Name"].default_value = default_name
    if default_value:
        store_named_attr_node.inputs["Value"].default_value = default_value

    # connects the new node in between the parent and child nodes
    node_tree.links.new(child_socket, store_named_attr_node.inputs["Geometry"])
    node_tree.links.new(store_named_attr_node.outputs["Geometry"], parent_socket)

    return store_named_attr_node


def find_link(
    from_socket: bpy.types.NodeSocket,
    to_socket: bpy.types.NodeSocket,
    node_tree: bpy.types.NodeTree,
):
    """
    Finds the link in the node tree between two given sockets.
    """
    for link in node_tree.links:
        if link.from_socket == from_socket and link.to_socket == to_socket:
            return link
    return None


def create_link(
    from_socket: bpy.types.NodeSocket,
    to_socket: bpy.types.NodeSocket,
    node_tree: bpy.types.NodeTree,
):
    """
    Creates a link between two given sockets.
    """
    return node_tree.links.new(from_socket, to_socket)


def turn_off_joint_debugging(blend_node: bpy.types.Node):
    """
    Turn off debugging geometries for joint nodes
    """
    if len(blend_node.inputs["Show Center of Parent"].links) == 1:
        link = blend_node.inputs["Show Center of Parent"].links[0]
        node_tree = link.id_data
        node_tree.links.remove(link)
    if len(blend_node.inputs["Show Center of Child"].links) == 1:
        link = blend_node.inputs["Show Center of Child"].links[0]
        node_tree = link.id_data
        node_tree.links.remove(link)
    if len(blend_node.inputs["Show Joint"].links) == 1:
        link = blend_node.inputs["Show Joint"].links[0]
        node_tree = link.id_data
        node_tree.links.remove(link)

    # set the Value input to 0
    if len(blend_node.inputs["Value"].links) == 1:
        link = blend_node.inputs["Value"].links[0]
        node_tree = link.id_data
        node_tree.links.remove(link)

    blend_node.inputs["Show Center of Parent"].default_value = False
    blend_node.inputs["Show Center of Child"].default_value = False
    blend_node.inputs["Show Joint"].default_value = False
    blend_node.inputs["Value"].default_value = 0.0


def get_functional_geonodes(link, visited):
    """
    Given a link, points to the next functional geo node (i.e., not a group node).
    Skips joint related geo nodes.
    """

    to_nodes = []

    def recurse_forward(node, socket_name, link):
        if link in visited:
            return

        if not is_node_group(node) or "Joint" in node.node_tree.name:
            to_nodes.append((node, link))
            return

        inner_input_group = get_node_by_idname(node.node_tree, "NodeGroupInput")
        for l in inner_input_group.outputs[socket_name].links:
            next_socket = l.to_socket
            next_node = next_socket.node
            recurse_forward(next_node, next_socket.name, l)

    from_socket, to_socket = link.from_socket, link.to_socket
    from_node, to_node = from_socket.node, to_socket.node
    # first find the functional node this is coming from
    while is_node_group(from_node):
        if "Joint" in from_node.node_tree.name:
            break
        inner_output_group = get_node_by_idname(from_node.node_tree, "NodeGroupOutput")
        socket_name = from_socket.name
        from_socket = inner_output_group.inputs[socket_name].links[0].from_socket
        from_node = from_socket.node

    # second find the functional node the link goes to
    recurse_forward(to_node, to_socket.name, link)

    return from_node, to_nodes
