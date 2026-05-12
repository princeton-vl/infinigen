# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo


from infinigen.assets.objects.tables.table_utils import (
    nodegroup_generate_radius_curve,
    nodegroup_n_gon_cylinder,
)
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler


@node_utils.to_nodegroup(
    "nodegroup_generate_leg_straight", singleton=False, type="GeometryNodeTree"
)
def nodegroup_generate_leg_straight(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Leg Height", 0.0000),
            ("NodeSocketFloat", "Leg Diameter", 1.0000),
            ("NodeSocketInt", "Resolution", 0),
            ("NodeSocketInt", "N-gon", 32),
            ("NodeSocketFloat", "Fillet Ratio", 0.0100),
        ],
    )

    generateradiuscurve = nw.new_node(
        nodegroup_generate_radius_curve(kwargs["Leg Curve Control Points"]).name,
        input_kwargs={"Resolution": group_input.outputs["Resolution"]},
    )

    ngoncylinder = nw.new_node(
        nodegroup_n_gon_cylinder().name,
        input_kwargs={
            "Radius Curve": generateradiuscurve,
            "Height": group_input.outputs["Leg Height"],
            "N-gon": group_input.outputs["N-gon"],
            "Profile Width": group_input.outputs["Leg Diameter"],
            "Aspect Ratio": 1.0000,
            "Fillet Ratio": group_input.outputs["Fillet Ratio"],
            "Resolution": group_input.outputs["Resolution"],
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": ngoncylinder.outputs["Mesh"]},
        attrs={"is_active_output": True},
    )
