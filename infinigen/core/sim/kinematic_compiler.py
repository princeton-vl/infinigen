# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Abhishek Joshi: primary author

import logging
from collections import defaultdict
from typing import Dict, List

import bpy

from infinigen.core.sim import utils
from infinigen.core.sim.kinematic_node import (
    JointType,
    KinematicNode,
    KinematicType,
    kinematic_node_factory,
)


def get_string_input(node: bpy.types.Node, input_name: str) -> str:
    """
    Gets string input given a node and the input name.
    """
    if input_name not in node.inputs:
        return ""

    if len(node.inputs[input_name].links) == 1:
        str_val = node.inputs[input_name].links[0].from_socket.node.string
    else:
        str_val = node.inputs[input_name].default_value

    return str_val


def get_labels(node_tree: bpy.types.NodeTree):
    """
    Gets and updates all the labels for the parts of the asset
    """
    labels = []
    q = [node_tree]
    while q:
        nt = q.pop(0)
        for node in nt.nodes:
            if utils.is_node_group(node) and "Joint" not in node.node_tree.name:
                q.append(node.node_tree)
            if utils.is_add_metadata(node):
                labels.append(get_string_input(node, "Label"))

    return labels


def set_joint_id(joint_idn: str, joint_node: bpy.types.Node) -> None:
    """
    Sets the joint id.
    """
    node_tree = joint_node.id_data
    joint_id = node_tree.nodes.new("FunctionNodeInputString")
    joint_id.string = joint_idn
    node_tree.links.new(
        joint_id.outputs["String"], joint_node.inputs["Joint ID (do not set)"]
    )


def set_duplicate_id(duplicate_idn: str, duplicate_node: bpy.types.Node) -> None:
    """
    Updates the default name for the duplicate store named attribute
    """
    node_tree = duplicate_node.id_data
    joint_id = node_tree.nodes.new("FunctionNodeInputString")
    joint_id.string = duplicate_idn
    node_tree.links.new(
        joint_id.outputs["String"], duplicate_node.inputs["Duplicate ID (do not set)"]
    )


def get_geometry_graph(
    mods: List[bpy.types.NodesModifier],
) -> Dict[bpy.types.Node, List]:
    """
    Constructs a mapping between nodes and their children connected
    through exclusively GEOMETRY sockets
    """
    geo_graph = defaultdict(list)
    visited_links = set()

    def add_to_geometry_graph(from_node, to_nodes):
        for to_node, link in to_nodes:
            geo_graph[to_node].append((from_node, link))
            assert (
                link not in visited_links
            ), f"({link.id_data}) Link from {link.from_node} to {link.to_node} already added."
            visited_links.add(link)

    output_node = utils.get_node_by_idname(mods[0].node_group, "NodeGroupOutput")
    link = output_node.inputs["Geometry"].links[0]
    from_node, to_nodes = utils.get_functional_geonodes(link, visited_links)
    add_to_geometry_graph(from_node, to_nodes)

    # add nodes from each of the modifers (and any node groups they have) to our
    # geometry graph, children stored as (child node, link)
    queue = [mod.node_group for mod in mods]
    seen_groups = set()
    while len(queue) > 0:
        node_tree = queue.pop(0)

        if node_tree not in seen_groups:
            print(f"Adding node tree {node_tree}")
            seen_groups.add(node_tree)
        else:
            print(f"Already seen node tree {node_tree}")
            continue

        # add all the groups in this node tree
        for node in node_tree.nodes:
            if utils.is_node_group(node) and "Joint" not in node.node_tree.name:
                queue.append(node.node_tree)

        for link in node_tree.links:
            if link.to_socket.type == "GEOMETRY":
                from_node, to_node = link.from_node, link.to_node
                if (
                    from_node.bl_idname == "NodeGroupInput"
                    or to_node.bl_idname == "NodeGroupOutput"
                ):
                    continue

                from_node, to_nodes = utils.get_functional_geonodes(link, visited_links)
                add_to_geometry_graph(from_node, to_nodes)

    return geo_graph


def add_kinematic_node_as_child(node, child, idx):
    """
    Adds a child to a kinematic node
    """

    if child.kinematic_type == KinematicType.NONE:
        if len(child.children.keys()) == 0:
            node.add_child(idx, kinematic_node_factory(KinematicType.ASSET))
        else:
            for child in child.get_all_children():
                if child.kinematic_type != KinematicType.NONE:
                    node.add_child(idx, child)
                else:
                    node.add_child(idx, kinematic_node_factory(KinematicType.ASSET))
    else:
        node.add_child(idx, child)


def compile(obj: bpy.types.Object) -> Dict:
    """
    Compiles the Blender geometry nodes graph into MJCF format
    """

    KinematicNode.reset_counts()

    # Build kinematic connection graph (simplifies blender graph)
    mods = [mod for mod in obj.modifiers if mod.type == "NODES"]
    if len(mods) == 0:
        logging.error("No modifiers defined. Exitting.")

    # create a graph representation using only geometry links
    geo_graph = get_geometry_graph(mods)

    blend_to_kinematic_node = {}

    # dictionaries used for semantic information
    semantic_labels = {}

    visited = set()

    def get_kinematic_node(node: bpy.types.Node) -> KinematicNode:
        """
        Returns an equivalent kinematic node for the given blender node
        """
        # check if there's already a kinematic equivalent for current node
        if node in blend_to_kinematic_node:
            return blend_to_kinematic_node[node]

        if utils.is_join(node):
            kinematic_type = KinematicType.JOINT
            joint_type = JointType.WELD
        elif utils.is_hinge(node):
            kinematic_type = KinematicType.JOINT
            joint_type = JointType.HINGE
        elif utils.is_sliding(node):
            kinematic_type = KinematicType.JOINT
            joint_type = JointType.SLIDING
        elif utils.is_duplicate(node):
            kinematic_type = KinematicType.DUPLICATE
            joint_type = JointType.NONE
        elif utils.is_switch(node):
            kinematic_type = KinematicType.SWITCH
            joint_type = JointType.NONE
        else:
            kinematic_type = KinematicType.NONE
            joint_type = JointType.NONE

        new_kinematic_node = kinematic_node_factory(
            kinematic_type, joint_type=joint_type
        )
        blend_to_kinematic_node[node] = new_kinematic_node
        return new_kinematic_node

    def build_kinematic_graph(blend_node: bpy.types.Node) -> KinematicNode:
        """
        Builds a kinematic graph that we can use to create the sim ready asset.
        """
        if blend_node in visited:
            return blend_to_kinematic_node[blend_node]

        root = get_kinematic_node(blend_node)

        # update joint related nodes
        if utils.is_joint(blend_node):
            set_joint_id(root.idn, blend_node)
            utils.turn_off_joint_debugging(blend_node)
        if utils.is_duplicate(blend_node):
            set_duplicate_id(root.idn, blend_node)

        if utils.is_join(blend_node) or utils.is_joint(blend_node):
            for i, (child, link) in enumerate(geo_graph[blend_node]):
                child_subgraph = build_kinematic_graph(child)
                idx = i
                if utils.is_joint(blend_node):
                    idx = 1 if link.to_socket.name == "Child" else 0

                    # store the part labels for semantic information
                    joint_label = get_string_input(blend_node, "Joint Label")
                    parent_label = get_string_input(blend_node, "Parent Label")
                    child_label = get_string_input(blend_node, "Child Label")
                    semantic_labels[root.idn] = {
                        "joint": joint_label if joint_label != "" else root.idn,
                        "parent": parent_label
                        if parent_label != ""
                        else f"{root.idn}parent",
                        "child": child_label
                        if child_label != ""
                        else f"{root.idn}child",
                    }
                # injects an attribute so that we can track joint path

                utils.inject_store_named_attr(
                    link, default_name=root.idn, default_value=idx
                )
                add_kinematic_node_as_child(root, child_subgraph, idx)

            # sanity check
            if utils.is_joint(blend_node):
                assert len(root.children.keys()) == 2

        elif utils.is_duplicate(blend_node):
            for i, (child, link) in enumerate(geo_graph[blend_node]):
                if link.to_socket.name == "Points":
                    continue
                child_subgraph = build_kinematic_graph(child)
                # duplicates child bodies on the parent and requires the previous
                # node to be a joint
                assert child_subgraph.kinematic_type == KinematicType.JOINT
                idx = 1 if link.to_socket.name == "Child" else 0
                add_kinematic_node_as_child(root, child_subgraph, idx)

        elif utils.is_switch(blend_node):
            # stores the value of the switch node that each geometry corresponds to
            for i, (child, link) in enumerate(geo_graph[blend_node]):
                child_subgraph = build_kinematic_graph(child)
                if "IndexSwitch" in blend_node.bl_idname:
                    idx = int(link.to_socket.name)
                else:
                    idx = 0 if link.to_socket.name == "False" else 1
                utils.inject_store_named_attr(
                    link, default_name=root.idn, default_value=idx
                )
                add_kinematic_node_as_child(root, child_subgraph, idx)

        else:
            for i, (child, link) in enumerate(geo_graph[blend_node]):
                child_subgraph = build_kinematic_graph(child)
                add_kinematic_node_as_child(root, child_subgraph, i)

        visited.add(blend_node)
        return root

    # build the kinematic graph starting from the output of the last modifier
    output_node = utils.get_node_by_idname(mods[-1].node_group, "NodeGroupOutput")
    root = build_kinematic_graph(output_node)
    root.set_idn("root")
    graph = root.get_graph()

    # create a dictionary for metadata
    metadata = {}
    for k, sl in semantic_labels.items():
        metadata[k] = {
            "joint label": sl["joint"],
            "parent body label": sl["parent"],
            "child body label": sl["child"],
        }

    # Assuming there is only one modifier for now
    # TODO: fix this to allow for multiple modifiers
    labels = get_labels(mods[-1].node_group)

    kinematic_info = {"graph": graph, "metadata": metadata, "labels": labels}

    return kinematic_info
