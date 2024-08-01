# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo


from infinigen.assets.objects.tables.table_utils import nodegroup_n_gon_cylinder
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler


@node_utils.to_nodegroup("nodegroup_strecher", singleton=False, type="GeometryNodeTree")
def nodegroup_strecher(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    curve_line = nw.new_node(
        Nodes.CurveLine,
        input_kwargs={
            "Start": (1.0000, 0.0000, 1.0000),
            "End": (1.0000, 0.0000, -1.0000),
        },
    )

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketInt", "N-gon", 32),
            ("NodeSocketFloat", "Profile Width", 0.200),
        ],
    )

    ngoncylinder = nw.new_node(
        nodegroup_n_gon_cylinder().name,
        input_kwargs={
            "Radius Curve": curve_line,
            "Height": 1.0000,
            "N-gon": group_input.outputs["N-gon"],
            "Profile Width": group_input.outputs["Profile Width"],
            "Aspect Ratio": 1.0000,
            "Resolution": 64,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": ngoncylinder.outputs["Mesh"]},
        attrs={"is_active_output": True},
    )
