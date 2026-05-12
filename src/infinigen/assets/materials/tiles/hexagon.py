# Copyright (C) 2024, Princeton University.

# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Yiming Zuo

from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler

from .utils import nodegroup_scalar_positive_modulo


@node_utils.to_nodegroup(
    "nodegroup_distance_to_axis", singleton=False, type="ShaderNodeTree"
)
def nodegroup_distance_to_axis(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Vector", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketVector", "Axis", (0.0000, 0.0000, 0.0000)),
        ],
    )

    length = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Vector"]},
        attrs={"operation": "LENGTH"},
    )

    power = nw.new_node(
        Nodes.Math,
        input_kwargs={0: length.outputs["Value"], 1: 2.0000},
        attrs={"operation": "POWER"},
    )

    normalize = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Axis"]},
        attrs={"operation": "NORMALIZE"},
    )

    dot_product = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Vector"], 1: normalize.outputs["Vector"]},
        attrs={"operation": "DOT_PRODUCT"},
    )

    power_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: dot_product.outputs["Value"], 1: 2.0000},
        attrs={"operation": "POWER"},
    )

    subtract = nw.new_node(
        Nodes.Math, input_kwargs={0: power, 1: power_1}, attrs={"operation": "SUBTRACT"}
    )

    sqrt = nw.new_node(
        Nodes.Math, input_kwargs={0: subtract}, attrs={"operation": "SQRT"}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Distance": sqrt},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_hex_single", singleton=False, type="ShaderNodeTree")
def nodegroup_hex_single(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

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
        input_kwargs={0: group_input.outputs["Size"], 1: 1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Size"], 1: 1.7321},
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

    group = nw.new_node(
        nodegroup_distance_to_axis().name,
        input_kwargs={"Vector": reroute, "Axis": (0.0000, 1.0000, 0.0000)},
    )

    group_1 = nw.new_node(
        nodegroup_distance_to_axis().name,
        input_kwargs={"Vector": reroute, "Axis": (0.8660, -0.5000, 0.0000)},
    )

    group_2 = nw.new_node(
        nodegroup_distance_to_axis().name,
        input_kwargs={"Vector": reroute, "Axis": (-0.8660, -0.5000, 0.0000)},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": group, "Y": group_1, "Z": group_2}
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

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": snap_1, "Y": snap_2})

    white_noise_texture = nw.new_node(
        Nodes.WhiteNoiseTexture, input_kwargs={"Vector": combine_xyz}
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: snap_2, 1: separate_xyz_4.outputs["X"]},
        attrs={"operation": "DIVIDE"},
    )

    group_5 = nw.new_node(
        nodegroup_scalar_positive_modulo().name, input_kwargs={0: divide, 1: 2.0000}
    )

    greater_than = nw.new_node(
        Nodes.Math, input_kwargs={0: group_5}, attrs={"operation": "GREATER_THAN"}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "UV": combine_xyz_1,
            "Color": white_noise_texture.outputs["Color"],
            "Value": greater_than,
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_hexagon", singleton=False, type="ShaderNodeTree")
def nodegroup_hexagon(nw: NodeWrangler):
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
        input_kwargs={0: group_input.outputs["border"], 1: 0.8660},
        attrs={"operation": "MULTIPLY"},
    )

    reroute = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply})

    size = nw.new_node(Nodes.Value, label="Size")
    size.outputs[0].default_value = 1.0000

    multiply_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: size}, attrs={"operation": "MULTIPLY"}
    )

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.5000

    mapping = nw.new_node(
        Nodes.Mapping,
        input_kwargs={"Vector": group_input.outputs["Coordinate"], "Scale": value},
    )

    group_5 = nw.new_node(
        nodegroup_hex_single().name, input_kwargs={"Coordinate": mapping, "Size": size}
    )

    separate_xyz_5 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_5.outputs["UV"]}
    )

    value_3 = nw.new_node(Nodes.Value)
    value_3.outputs[0].default_value = 0.0000

    smooth_max = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: separate_xyz_5.outputs["X"],
            1: separate_xyz_5.outputs["Y"],
            2: value_3,
        },
        attrs={"operation": "SMOOTH_MAX"},
    )

    smooth_max_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: smooth_max, 1: separate_xyz_5.outputs["Z"], 2: value_3},
        attrs={"operation": "SMOOTH_MAX"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_1, 1: smooth_max_1},
        attrs={"operation": "SUBTRACT"},
    )

    mix = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: reroute, 6: subtract},
        attrs={"blend_type": "BURN", "data_type": "RGBA"},
    )

    mix_1 = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            0: group_input.outputs["Flatness"],
            6: mix.outputs[2],
            7: (1.0000, 1.0000, 1.0000, 1.0000),
        },
        attrs={"blend_type": "DODGE", "data_type": "RGBA"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: size}, attrs={"operation": "MULTIPLY"}
    )

    multiply_3 = nw.new_node(
        Nodes.Math, input_kwargs={0: size, 1: 0.8660}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_2, "Y": multiply_3}
    )

    multiply_4 = nw.new_node(
        Nodes.Math, input_kwargs={0: size, 1: 99.0000}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_4})

    add = nw.new_node(Nodes.VectorMath, input_kwargs={0: combine_xyz, 1: combine_xyz_1})

    add_1 = nw.new_node(
        Nodes.VectorMath, input_kwargs={0: mapping, 1: add.outputs["Vector"]}
    )

    group_6 = nw.new_node(
        nodegroup_hex_single().name,
        input_kwargs={"Coordinate": add_1.outputs["Vector"], "Size": size},
    )

    separate_xyz_6 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_6.outputs["UV"]}
    )

    smooth_max_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: separate_xyz_6.outputs["X"],
            1: separate_xyz_6.outputs["Y"],
            2: value_3,
        },
        attrs={"operation": "SMOOTH_MAX"},
    )

    smooth_max_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: smooth_max_2, 1: separate_xyz_6.outputs["Z"], 2: value_3},
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

    mix_4 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: 1.0000, 6: mix_1.outputs[2], 7: mix_2.outputs[2]},
        attrs={"blend_type": "ADD", "data_type": "RGBA"},
    )

    mix_5 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: 1.0000, 6: mix_1.outputs[2], 7: group_5.outputs["Color"]},
        attrs={"blend_type": "MULTIPLY", "data_type": "RGBA"},
    )

    mix_6 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: 1.0000, 6: mix_2.outputs[2], 7: group_6.outputs["Color"]},
        attrs={"blend_type": "MULTIPLY", "data_type": "RGBA"},
    )

    mix_7 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: 1.0000, 6: mix_5.outputs[2], 7: mix_6.outputs[2]},
        attrs={"blend_type": "ADD", "data_type": "RGBA"},
    )

    greater_than = nw.new_node(
        Nodes.Math,
        input_kwargs={0: mix_1.outputs[2], 1: 0.0000},
        attrs={"operation": "GREATER_THAN"},
    )

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: mix_1.outputs[2], 1: group_5.outputs["Value"]},
        attrs={"operation": "MULTIPLY"},
    )

    greater_than_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_5, 1: 0.0000},
        attrs={"operation": "GREATER_THAN"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Result": mix_4.outputs[2],
            "Tile Color": mix_7.outputs[2],
            "Tile Type 1": greater_than,
            "Tile Type 2": greater_than_1,
        },
        attrs={"is_active_output": True},
    )
