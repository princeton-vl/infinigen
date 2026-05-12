# Copyright (C) 2024, Princeton University.

# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Yiming Zuo

from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler


@node_utils.to_nodegroup(
    "nodegroup_scalar_positive_modulo", singleton=False, type="ShaderNodeTree"
)
def nodegroup_scalar_positive_modulo(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "input_1", 0.5000),
            ("NodeSocketFloat", "input_2", 0.5000),
        ],
    )

    modulo = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs[0], 1: group_input.outputs[1]},
        attrs={"operation": "MODULO"},
    )

    less_than = nw.new_node(
        Nodes.Math,
        input_kwargs={0: modulo, 1: 0.0000},
        attrs={"operation": "LESS_THAN"},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: less_than, 1: group_input.outputs[1]},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: modulo, 1: multiply})

    group_output = nw.new_node(
        Nodes.GroupOutput, input_kwargs={"Value": add}, attrs={"is_active_output": True}
    )


@node_utils.to_nodegroup(
    "nodegroup_u_v_recenter", singleton=False, type="ShaderNodeTree"
)
def nodegroup_u_v_recenter(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "UV", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketVector", "HalfSize", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketFloat", "Rot Angle", 0.0000),
        ],
    )

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["UV"], 1: group_input.outputs["HalfSize"]},
        attrs={"operation": "SUBTRACT"},
    )

    vector_rotate = nw.new_node(
        Nodes.VectorRotate,
        input_kwargs={
            "Vector": subtract.outputs["Vector"],
            "Angle": group_input.outputs["Rot Angle"],
        },
    )

    absolute = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: vector_rotate},
        attrs={"operation": "ABSOLUTE"},
    )

    subtract_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: group_input.outputs["HalfSize"],
            1: absolute.outputs["Vector"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Vector": subtract_1.outputs["Vector"]},
        attrs={"is_active_output": True},
    )
