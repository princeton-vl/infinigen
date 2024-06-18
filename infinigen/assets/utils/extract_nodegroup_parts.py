# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import bpy

from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler, geometry_node_group_empty_new
from infinigen.core.util import blender as butil


def extract_nodegroup_geo(target_obj, nodegroup, k, ng_params=None):
    assert k in nodegroup.outputs
    assert target_obj.type == "MESH"

    vert = butil.spawn_vert("extract_nodegroup_geo.temp")

    butil.modify_mesh(vert, type="NODES", apply=False)
    if vert.modifiers[0].node_group == None:
        group = geometry_node_group_empty_new()
        vert.modifiers[0].node_group = group
    ng = vert.modifiers[0].node_group
    nw = NodeWrangler(ng)
    obj_inp = nw.new_node(Nodes.ObjectInfo, [target_obj])

    group_input_kwargs = {**ng_params}
    if "Geometry" in nodegroup.inputs:
        group_input_kwargs["Geometry"] = obj_inp.outputs["Geometry"]
    group = nw.new_node(nodegroup.name, input_kwargs=group_input_kwargs)

    geo = group.outputs[k]

    if k.endswith("Curve"):
        # curves dont export from geonodes well, convert it to a mesh
        geo = nw.new_node(Nodes.CurveToMesh, [geo])

    output = nw.new_node(Nodes.GroupOutput, input_kwargs={"Geometry": geo})

    butil.apply_modifiers(vert)
    bpy.data.node_groups.remove(ng)
    return vert
