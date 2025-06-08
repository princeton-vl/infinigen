import bpy

from infinigen.core.nodes.node_transpiler.transpiler import (
    create_inputs_dict,
    process_single_input,
)
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler


def setup_math_node_test():
    node_tree = bpy.data.node_groups.new(name="test_node_tree", type="GeometryNodeTree")
    nw = NodeWrangler(node_tree)

    math_node = nw.new_node(Nodes.Math)
    value1 = nw.new_node(Nodes.Value)
    value1.outputs[0].default_value = 1.0
    value2 = nw.new_node(Nodes.Value)
    value2.outputs[0].default_value = 2.0

    nw.connect_input(math_node.inputs[0], value1)
    nw.connect_input(math_node.inputs[1], value2)

    return node_tree, math_node, value1, value2


def test_disabled_input_socket():
    node_tree, math_node, value1, value2 = setup_math_node_test()
    memo = {}

    # Both inputs enabled with ADD operation
    math_node.operation = "ADD"
    result = process_single_input(
        node_tree, math_node, math_node.inputs[1], "Value", memo
    )
    assert result is not None, "Second input should be processed when operation is ADD"

    # Second input disabled with ABS operation
    math_node.operation = "ABSOLUTE"
    result = process_single_input(
        node_tree, math_node, math_node.inputs[1], "Value", memo
    )
    assert result is None, "Second input should be skipped when operation is ABSOLUTE"

    inputs_dict, code, targets = create_inputs_dict(node_tree, math_node, memo)
    assert (
        len(inputs_dict) == 1
    ), "Only one input should be in the dict for ABSOLUTE operation"


def test_default_value_disabled_socket():
    node_tree = bpy.data.node_groups.new(name="test_node_tree", type="GeometryNodeTree")
    nw = NodeWrangler(node_tree)

    math_node = nw.new_node(Nodes.Math)
    math_node.operation = "ABSOLUTE"
    math_node.inputs[1].default_value = 5.0

    result = process_single_input(
        node_tree, math_node, math_node.inputs[1], "Value", memo={}
    )
    assert (
        result is None
    ), "Disabled socket should be skipped even if it has a default value"
