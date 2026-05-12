# Copyright (C) 2024, Princeton University.

# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Yiming Zuo

from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler

from .utils import nodegroup_scalar_positive_modulo


@node_utils.to_nodegroup("nodegroup_half_shell", singleton=False, type="ShaderNodeTree")
def nodegroup_half_shell(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Coordinate", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketFloat", "Radius", 0.0000),
            ("NodeSocketFloat", "Edge", 0.0000),
            ("NodeSocketFloat", "3D-ness", 0.8333),
        ],
    )

    separate_xyz_4 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Coordinate"]}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Radius"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    snap = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_4.outputs["X"], 1: multiply},
        attrs={"operation": "SNAP"},
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: snap, 1: group_input.outputs["Radius"]}
    )

    snap_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_4.outputs["Y"], 1: multiply},
        attrs={"operation": "SNAP"},
    )

    add_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: snap_1, 1: group_input.outputs["Radius"]}
    )

    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": add, "Y": add_1})

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Coordinate"], 1: combine_xyz_5},
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz_5 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract.outputs["Vector"]}
    )

    greater_than = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_5.outputs["Y"], 1: 0.0000},
        attrs={"operation": "GREATER_THAN"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 1.0000, 1: greater_than},
        attrs={"operation": "SUBTRACT"},
    )

    length = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: subtract.outputs["Vector"]},
        attrs={"operation": "LENGTH"},
    )

    absolute = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_5.outputs["X"]},
        attrs={"operation": "ABSOLUTE"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: absolute, 1: 1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_1, 1: length.outputs["Value"]},
        attrs={"operation": "DIVIDE"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_5.outputs["Y"], 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_2, 1: length.outputs["Value"]},
        attrs={"operation": "DIVIDE"},
    )

    add_2 = nw.new_node(Nodes.Math, input_kwargs={0: divide, 1: divide_1})

    multiply_3 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_2, 1: 0.7071}, attrs={"operation": "MULTIPLY"}
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_3, 1: 1.4142},
        attrs={"operation": "MULTIPLY"},
    )

    power = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_4, 1: 2.0000},
        attrs={"operation": "POWER"},
    )

    subtract_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: power, 1: 1.0000}, attrs={"operation": "SUBTRACT"}
    )

    sqrt = nw.new_node(
        Nodes.Math, input_kwargs={0: subtract_2}, attrs={"operation": "SQRT"}
    )

    subtract_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_4, 1: sqrt},
        attrs={"operation": "SUBTRACT"},
    )

    divide_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: length.outputs["Value"], 1: subtract_3},
        attrs={"operation": "DIVIDE"},
    )

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_1, 1: divide_2},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: greater_than, 1: length.outputs["Value"]},
        attrs={"operation": "MULTIPLY"},
    )

    add_3 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_5, 1: multiply_6})

    subtract_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Radius"], 1: add_3},
        attrs={"operation": "SUBTRACT"},
    )

    mix_5 = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            0: group_input.outputs["Edge"],
            6: subtract_4,
            7: (0.2574, 0.2574, 0.2574, 1.0000),
        },
        attrs={"clamp_factor": False, "blend_type": "BURN", "data_type": "RGBA"},
    )

    mix_6 = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            0: group_input.outputs["3D-ness"],
            6: mix_5.outputs[2],
            7: (1.0000, 1.0000, 1.0000, 1.0000),
        },
        attrs={"blend_type": "DODGE", "data_type": "RGBA"},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": snap, "Y": snap_1})

    white_noise_texture = nw.new_node(
        Nodes.WhiteNoiseTexture, input_kwargs={"Vector": combine_xyz}
    )

    multiply_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply, 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    group_1 = nw.new_node(
        nodegroup_scalar_positive_modulo().name, input_kwargs={0: snap_1, 1: multiply_7}
    )

    multiply_8 = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply}, attrs={"operation": "MULTIPLY"}
    )

    greater_than_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_1, 1: multiply_8},
        attrs={"operation": "GREATER_THAN"},
    )

    group = nw.new_node(
        nodegroup_scalar_positive_modulo().name, input_kwargs={0: snap, 1: multiply_7}
    )

    greater_than_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group, 1: multiply_8},
        attrs={"operation": "GREATER_THAN"},
    )

    subtract_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: greater_than_1, 1: greater_than_2},
        attrs={"operation": "SUBTRACT"},
    )

    absolute_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: subtract_5}, attrs={"operation": "ABSOLUTE"}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Result": mix_6.outputs[2],
            "Color": white_noise_texture.outputs["Color"],
            "Even": absolute_1,
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_shell", singleton=False, type="ShaderNodeTree")
def nodegroup_shell(nw: NodeWrangler):
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

    radius = nw.new_node(Nodes.Value, label="Radius")
    radius.outputs[0].default_value = 1.0000

    reroute_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["border"]}
    )

    power = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Flatness"], 1: 1.5000},
        attrs={"operation": "POWER"},
    )

    reroute = nw.new_node(Nodes.Reroute, input_kwargs={"Input": power})

    group = nw.new_node(
        nodegroup_half_shell().name,
        input_kwargs={
            "Coordinate": group_input.outputs["Coordinate"],
            "Radius": radius,
            "Edge": reroute_1,
            "3D-ness": reroute,
        },
    )

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": radius, "Y": radius}
    )

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 99.0000

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_5, 1: value},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: group_input.outputs["Coordinate"],
            1: multiply.outputs["Vector"],
        },
    )

    group_1 = nw.new_node(
        nodegroup_half_shell().name,
        input_kwargs={
            "Coordinate": add.outputs["Vector"],
            "Radius": radius,
            "Edge": reroute_1,
            "3D-ness": reroute,
        },
    )

    mix_5 = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            0: 1.0000,
            6: group.outputs["Result"],
            7: group_1.outputs["Result"],
        },
        attrs={"blend_type": "ADD", "data_type": "RGBA"},
    )

    greater_than = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group.outputs["Result"], 1: 0.0000},
        attrs={"operation": "GREATER_THAN"},
    )

    mix_6 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: 1.0000, 6: greater_than, 7: group.outputs["Color"]},
        attrs={"blend_type": "MULTIPLY", "data_type": "RGBA"},
    )

    greater_than_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_1.outputs["Result"], 1: 0.0000},
        attrs={"operation": "GREATER_THAN"},
    )

    mix_7 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: 1.0000, 6: greater_than_1, 7: group_1.outputs["Color"]},
        attrs={"blend_type": "MULTIPLY", "data_type": "RGBA"},
    )

    mix_8 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: 1.0000, 6: mix_6.outputs[2], 7: mix_7.outputs[2]},
        attrs={"blend_type": "ADD", "data_type": "RGBA"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: greater_than_1, 1: group_1.outputs["Even"]},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group.outputs["Even"], 1: greater_than},
        attrs={"operation": "MULTIPLY"},
    )

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_1, 1: multiply_2})

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Result": mix_5.outputs[2],
            "Tile Color": mix_8.outputs[2],
            "Tile Type 1": greater_than,
            "Tile Type 2": add_1,
        },
        attrs={"is_active_output": True},
    )
