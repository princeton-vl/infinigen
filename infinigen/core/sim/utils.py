# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Abhishek Joshi: primary author
# - Max Gonzalez Saez-Diez: functions required for sim unit tests

import inspect
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
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
    return is_node_group(node) and (
        "hinge_joint" in node.node_tree.name or "Hinge Joint" in node.node_tree.name
    )


def is_sliding(node: bpy.types.Node) -> bool:
    """
    Returns true if the current node is a sliding joint node
    """
    return is_node_group(node) and (
        "sliding_joint" in node.node_tree.name or "Sliding Joint" in node.node_tree.name
    )


def is_joint(node: bpy.types.Node) -> bool:
    """
    Returns true if the current node is a joint node
    """
    return is_hinge(node) or is_sliding(node)


def is_add_metadata(node: bpy.types.Node) -> bool:
    """
    Returns true fi the node is an add metadata node.
    """
    return is_node_group(node) and (
        "add_jointed_geometry_metadata" in node.node_tree.name
        or "Add Jointed Geometry Metadata" in node.node_tree.name
    )


def is_duplicate(node: bpy.types.Node) -> bool:
    """
    Returns true if the node is a duplicate joint node.
    """
    return is_node_group(node) and (
        "duplicate_joints_on_parent" in node.node_tree.name
        or "Duplicate Joints on Parent" in node.node_tree.name
    )


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


def set_default_joint_state(node: bpy.types.Node):
    """
    Sets the joint to the default state (joint value = 0, turn debugging off)
    """
    if len(node.inputs["Show Joint"].links) == 1:
        link = node.inputs["Show Joint"].links[0]
        node_tree = link.id_data
        node_tree.links.remove(link)
    node.inputs["Show Joint"].default_value = False

    if len(node.inputs["Value"].links) == 1:
        link = node.inputs["Value"].links[0]
        node_tree = link.id_data
        node_tree.links.remove(link)
    node.inputs["Value"].default_value = 0.0


def get_functional_geonodes(link, visited):
    """
    Given a link, points to the next functional geo node (i.e., not a group node).
    Skips joint related geo nodes.
    """

    to_nodes = []

    def recurse_forward(node, socket_name, link):
        if link in visited:
            return

        if not is_node_group(node) or "joint" in node.node_tree.name.lower():
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
        if "joint" in from_node.node_tree.name.lower():
            break
        inner_output_group = get_node_by_idname(from_node.node_tree, "NodeGroupOutput")
        socket_name = from_socket.name
        from_socket = inner_output_group.inputs[socket_name].links[0].from_socket
        from_node = from_socket.node

    # second find the functional node the link goes to
    recurse_forward(to_node, to_socket.name, link)

    return from_node, to_nodes


def load_class_from_path(path: str, classname: str):
    p = Path(path)
    if p.is_dir():
        p = p / "__init__.py"
    spec = spec_from_file_location(p.stem, str(p))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = getattr(mod, classname)
    if not inspect.isclass(cls):
        raise TypeError(f"{classname} is not a class in {p}")
    return cls


def find_joints(obj):
    """Helper function to find all joints in a spawned asset."""
    all_joints = list()

    def _find_joints(node, node_group, all_joints, level):
        if (node, node_group) in [(j[0], j[1]) for j in all_joints]:
            return

        for input_socket in node.inputs:
            if input_socket.name == "Joint Label":
                # Try to grab its default value if available
                default_val = getattr(input_socket, "default_value", "")
                all_joints.append([node, node_group, level, default_val])
                return

        # Recurse into node groups
        if node.type == "GROUP" and node.node_tree:
            for child_node in node.node_tree.nodes:
                _find_joints(child_node, node.node_tree, all_joints, level + 1)

    for mod in obj.modifiers:
        if mod and mod.node_group:
            node_group = mod.node_group
            for node in node_group.nodes:
                _find_joints(node, node_group, all_joints, 0)

    return all_joints


def get_metadata_all_joints_input(obj):
    """Helper function to check if each joint input has metadata for both parent and child bodies. This requires a metadata node RIGHT BEFORE each joint node in the node tree even if duplicate comes before."""
    all_joints_info = list()

    def _get_metadata_all_joints_input(node, node_group, all_joints):
        if (node, node_group) in [(j[0], j[1]) for j in all_joints]:
            return

        for input_socket in node.inputs:
            if input_socket.name == "Joint Label":
                group_name = getattr(input_socket, "default_value", "")
                all_joints.append(
                    [
                        node,
                        node_group,
                        group_name,
                    ]
                )
                node_upstream = None
                for item in ["Parent", "Child"]:
                    try:
                        link = node.inputs[item].links[0]
                        node_upstream = link.from_node
                    except Exception:
                        node_upstream = None

                    # Traverse upstream until non-Reroute node
                    while node_upstream and node_upstream.type == "REROUTE":
                        try:
                            link = node_upstream.inputs[0].links[0]
                            node_upstream = link.from_node
                        except Exception:
                            node_upstream = None
                            break

                    node_has_metadata = False
                    if node_upstream and hasattr(node_upstream, "node_tree"):
                        parent_tree_name = getattr(
                            node_upstream.node_tree, "name", ""
                        ).lower()
                        node_has_metadata = (
                            "add_jointed_geometry_metadata" in parent_tree_name
                        )

                    try:
                        if (
                            node.inputs["Parent"].links[0].from_socket.name == "Parent"
                            and node.inputs["Child"].links[0].from_socket.name
                            == "Child"
                        ):
                            node_has_metadata = True
                    except Exception:
                        pass

                    all_joints[-1].append(node_has_metadata)

                return

        # Recurse into node groups
        if node.type == "GROUP" and node.node_tree:
            for child_node in node.node_tree.nodes:
                _get_metadata_all_joints_input(child_node, node.node_tree, all_joints)

    for mod in obj.modifiers:
        if mod and mod.node_group:
            node_group = mod.node_group
            for node in node_group.nodes:
                _get_metadata_all_joints_input(node, node_group, all_joints_info)

    return all_joints_info


def verify_joint_parent_child_output_used_correctly(obj):
    """Helper function to verify that parent and child inputs do not point to duplicate nodes."""
    all_joints_info = list()

    def _verify_joint_parent_child_output_used_correctly(node, node_group, all_joints):
        if (node, node_group) in [(j[0], j[1]) for j in all_joints]:
            return

        for input_socket in node.inputs:
            if input_socket.name == "Joint Label":
                group_name = getattr(input_socket, "default_value", "")
                parent_to_non_duplicate, child_to_non_duplicate = False, False
                parent_to_parent, child_to_parent = False, False

                # check if parent output goes to a duplicate node
                if len(node.outputs["Parent"].links) > 0:
                    link = node.outputs["Parent"].links[0]
                    to_node = link.to_node

                    while to_node.type == "REROUTE":
                        try:
                            link = to_node.outputs[0].links[0]
                            to_node = link.to_node
                        except Exception:
                            to_node = None
                            break

                    if not (
                        hasattr(to_node, "node_tree")
                        and "duplicate" in to_node.node_tree.name.lower()
                    ):
                        parent_to_non_duplicate = True
                        parent_to_parent = (
                            True if link.to_socket.name == "Parent" else False
                        )

                # check if child output goes to a duplicate node
                if len(node.outputs["Child"].links) > 0:
                    link = node.outputs["Child"].links[0]
                    to_node = link.to_node

                    while to_node.type == "REROUTE":
                        try:
                            link = to_node.outputs[0].links[0]
                            to_node = link.to_node
                        except Exception:
                            to_node = None
                            break

                    if not (
                        hasattr(to_node, "node_tree")
                        and "duplicate" in to_node.node_tree.name.lower()
                    ):
                        child_to_non_duplicate = True
                        child_to_parent = (
                            True if link.to_socket.name == "Child" else False
                        )

                # Exception: Both Parent/Child go to Parent/Child inputs on a joint.
                if parent_to_parent and child_to_parent:
                    parent_to_non_duplicate = False
                    child_to_non_duplicate = False

                all_joints.append(
                    [
                        node,
                        node_group,
                        group_name,
                        parent_to_non_duplicate,
                        child_to_non_duplicate,
                    ]
                )
                return

        # Recurse into node groups
        if node.type == "GROUP" and node.node_tree:
            for child_node in node.node_tree.nodes:
                _verify_joint_parent_child_output_used_correctly(
                    child_node, node.node_tree, all_joints
                )

    for mod in obj.modifiers:
        if mod and mod.node_group:
            node_group = mod.node_group
            for node in node_group.nodes:
                _verify_joint_parent_child_output_used_correctly(
                    node, node_group, all_joints_info
                )

    return all_joints_info


def check_if_asset_scaled_after_joint(obj):
    """Helper function to check if any scaling is applied to the asset after a joint node in the node tree."""
    seen = list()

    def _verify_no_scale_after_joint(node, node_group):
        if (node, node_group) in [(j[0], j[1]) for j in seen]:
            return

        for output in node.outputs:
            if output.type.startswith("GEOMETRY"):
                for link in output.links:
                    to_node = link.to_node
                    if to_node.type == "TRANSFORM_GEOMETRY":
                        scale_input = to_node.inputs.get("Scale")
                        default_scale = scale_input.default_value
                        if not all(abs(s - 1.0) < 1e-6 for s in default_scale):
                            raise ValueError("Scaling applied after joint in node")

                        if to_node.inputs["Scale"].links:
                            raise ValueError(
                                "Scaling applied after joint in node. Nothing should be linked to Scale input after joint."
                            )

                    if is_node_group(to_node) and not (
                        is_duplicate(to_node) or is_join(to_node)
                    ):
                        for child_node in to_node.node_tree.nodes:
                            _verify_no_scale_after_joint(child_node, to_node.node_tree)
                    else:
                        _verify_no_scale_after_joint(to_node, node_group)

        seen.append((node, node_group))
        return

    for mod in obj.modifiers:
        if mod and mod.node_group:
            node_group = mod.node_group
            for node in node_group.nodes:
                if not is_joint(node):
                    continue

                _verify_no_scale_after_joint(node, node_group)
