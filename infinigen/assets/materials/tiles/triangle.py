# Copyright (C) 2024, Princeton University.

# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Yiming Zuo

from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler

from .utils import nodegroup_scalar_positive_modulo


@node_utils.to_nodegroup(
    "nodegroup_triangle_single", singleton=False, type="ShaderNodeTree"
)
def nodegroup_triangle_single(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    value_2 = nw.new_node(Nodes.Value)
    value_2.outputs[0].default_value = 0.0000

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Coordinate", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketFloat", "Size", 0.5000),
        ],
    )

    separate_xyz_3 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Coordinate"]}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Size"], 1: 0.8660},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Size"], 1: 0.7500},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply, "Y": multiply_1}
    )

    separate_xyz_4 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": combine_xyz_2}
    )

    snap = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_3.outputs["Y"], 1: separate_xyz_4.outputs["Y"]},
        attrs={"operation": "SNAP"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: snap, 1: 0.5774}, attrs={"operation": "MULTIPLY"}
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply_2, 1: separate_xyz_3.outputs["X"]}
    )

    group_3 = nw.new_node(
        nodegroup_scalar_positive_modulo().name,
        input_kwargs={0: add, 1: separate_xyz_4.outputs["X"]},
    )

    group_4 = nw.new_node(
        nodegroup_scalar_positive_modulo().name,
        input_kwargs={0: separate_xyz_3.outputs["Y"], 1: separate_xyz_4.outputs["Y"]},
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": group_3, "Y": group_4}
    )

    multiply_3 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_2, 1: (0.5000, 0.3333, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_3, 1: multiply_3.outputs["Vector"]},
        attrs={"operation": "SUBTRACT"},
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": subtract.outputs["Vector"]}
    )

    dot_product = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute, 1: (0.0000, 1.0000, 0.0000)},
        attrs={"operation": "DOT_PRODUCT"},
    )

    dot_product_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute, 1: (0.8660, -0.5000, 0.0000)},
        attrs={"operation": "DOT_PRODUCT"},
    )

    dot_product_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute, 1: (-0.8660, -0.5000, 0.0000)},
        attrs={"operation": "DOT_PRODUCT"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": dot_product.outputs["Value"],
            "Y": dot_product_1.outputs["Value"],
            "Z": dot_product_2.outputs["Value"],
        },
    )

    subtract_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: value_2, 1: combine_xyz},
        attrs={"operation": "SUBTRACT"},
    )

    snap_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_3.outputs["Y"], 1: separate_xyz_4.outputs["Y"]},
        attrs={"operation": "SNAP"},
    )

    snap_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add, 1: separate_xyz_4.outputs["X"]},
        attrs={"operation": "SNAP"},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": snap_1, "Y": snap_2}
    )

    white_noise_texture = nw.new_node(
        Nodes.WhiteNoiseTexture, input_kwargs={"Vector": combine_xyz_1}
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: snap_2, 1: separate_xyz_4.outputs["X"]},
        attrs={"operation": "DIVIDE"},
    )

    group = nw.new_node(
        nodegroup_scalar_positive_modulo().name, input_kwargs={0: divide, 1: 2.0000}
    )

    greater_than = nw.new_node(
        Nodes.Math, input_kwargs={0: group}, attrs={"operation": "GREATER_THAN"}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "UV": subtract_1.outputs["Vector"],
            "Color": white_noise_texture.outputs["Color"],
            "Value": greater_than,
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_triangle", singleton=False, type="ShaderNodeTree")
def nodegroup_triangle(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Coordinate", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketFloat", "Subtiles Number", 1.0000),
            ("NodeSocketFloat", "Aspect Ratio", 5.0000),
            ("NodeSocketFloat", "border", 0.1000),
            ("NodeSocketFloat", "Flatness", 0.9000),
        ],
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["border"], 1: 0.5774},
        attrs={"operation": "MULTIPLY"},
    )

    reroute = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply})

    value_6 = nw.new_node(Nodes.Value)
    value_6.outputs[0].default_value = 1.0000

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: value_6, 1: 0.2500},
        attrs={"operation": "MULTIPLY"},
    )

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.4330

    mapping_1 = nw.new_node(
        Nodes.Mapping,
        input_kwargs={"Vector": group_input.outputs["Coordinate"], "Scale": value},
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": mapping_1})

    group_8 = nw.new_node(
        nodegroup_triangle_single().name,
        input_kwargs={"Coordinate": reroute_1, "Size": value_6},
    )

    separate_xyz_5 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_8.outputs["UV"]}
    )

    value_4 = nw.new_node(Nodes.Value)
    value_4.outputs[0].default_value = 0.0000

    smooth_max = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: separate_xyz_5.outputs["X"],
            1: separate_xyz_5.outputs["Y"],
            2: value_4,
        },
        attrs={"operation": "SMOOTH_MAX"},
    )

    smooth_max_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: smooth_max, 1: separate_xyz_5.outputs["Z"], 2: value_4},
        attrs={"operation": "SMOOTH_MAX"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_1, 1: smooth_max_1},
        attrs={"operation": "SUBTRACT"},
    )

    mix_1 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: reroute, 6: subtract},
        attrs={"blend_type": "BURN", "data_type": "RGBA"},
    )

    mix_4 = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            0: group_input.outputs["Flatness"],
            6: mix_1.outputs[2],
            7: (1.0000, 1.0000, 1.0000, 1.0000),
        },
        attrs={"blend_type": "DODGE", "data_type": "RGBA"},
    )

    vector_rotate = nw.new_node(
        Nodes.VectorRotate, input_kwargs={"Vector": reroute_1, "Angle": 3.1416}
    )

    group_7 = nw.new_node(
        nodegroup_triangle_single().name,
        input_kwargs={"Coordinate": vector_rotate, "Size": value_6},
    )

    separate_xyz_6 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_7.outputs["UV"]}
    )

    smooth_max_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: separate_xyz_6.outputs["X"],
            1: separate_xyz_6.outputs["Y"],
            2: value_4,
        },
        attrs={"operation": "SMOOTH_MAX"},
    )

    smooth_max_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: smooth_max_2, 1: separate_xyz_6.outputs["Z"], 2: value_4},
        attrs={"operation": "SMOOTH_MAX"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_1, 1: smooth_max_3},
        attrs={"operation": "SUBTRACT"},
    )

    mix_3 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: reroute, 6: subtract_1},
        attrs={"blend_type": "BURN", "data_type": "RGBA"},
    )

    mix_2 = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            0: group_input.outputs["Flatness"],
            6: mix_3.outputs[2],
            7: (1.0000, 1.0000, 1.0000, 1.0000),
        },
        attrs={"blend_type": "DODGE", "data_type": "RGBA"},
    )

    mix_5 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: 1.0000, 6: mix_4.outputs[2], 7: mix_2.outputs[2]},
        attrs={"blend_type": "ADD", "data_type": "RGBA"},
    )

    mix_7 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: 1.0000, 6: mix_2.outputs[2], 7: group_7.outputs["Color"]},
        attrs={"blend_type": "MULTIPLY", "data_type": "RGBA"},
    )

    mix_6 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: 1.0000, 6: mix_4.outputs[2], 7: group_8.outputs["Color"]},
        attrs={"blend_type": "MULTIPLY", "data_type": "RGBA"},
    )

    mix_8 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: 1.0000, 6: mix_7.outputs[2], 7: mix_6.outputs[2]},
        attrs={"blend_type": "ADD", "data_type": "RGBA"},
    )

    greater_than = nw.new_node(
        Nodes.Math,
        input_kwargs={0: mix_4.outputs[2], 1: 0.0000},
        attrs={"operation": "GREATER_THAN"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: mix_4.outputs[2], 1: group_8.outputs["Value"]},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: mix_2.outputs[2], 1: group_7.outputs["Value"]},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: multiply_2, 1: multiply_3})

    greater_than_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add, 1: 0.0000},
        attrs={"operation": "GREATER_THAN"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Result": mix_5.outputs[2],
            "Tile Color": mix_8.outputs[2],
            "Tile Type 1": greater_than,
            "Tile Type 2": greater_than_1,
        },
        attrs={"is_active_output": True},
    )
