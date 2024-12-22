# Copyright (C) 2024, Princeton University.

# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Yiming Zuo

from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler

from .utils import nodegroup_scalar_positive_modulo


@node_utils.to_nodegroup(
    "nodegroup_diamond_single", singleton=False, type="ShaderNodeTree"
)
def nodegroup_diamond_single(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = 0.7070

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Coordinate", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketFloat", "Width", 0.0000),
        ],
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Coordinate"]}
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Width"]}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute, 1: 1.5000},
        attrs={"operation": "MULTIPLY"},
    )

    snap = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: multiply},
        attrs={"operation": "SNAP"},
    )

    divide = nw.new_node(
        Nodes.Math, input_kwargs={0: snap, 1: multiply}, attrs={"operation": "DIVIDE"}
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Width"], 1: 0.8660},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: divide, 1: multiply_1},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: separate_xyz.outputs["X"], 1: multiply_2}
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add, "Y": separate_xyz.outputs["Y"]}
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute, 1: 1.7321},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_3, "Y": reroute}
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": combine_xyz}
    )

    snap_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add, 1: separate_xyz_1.outputs["X"]},
        attrs={"operation": "SNAP"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": snap_1, "Y": snap})

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_2, 1: combine_xyz_1},
        attrs={"operation": "SUBTRACT"},
    )

    scale = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz, "Scale": 0.5000},
        attrs={"operation": "SCALE"},
    )

    subtract_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: subtract.outputs["Vector"], 1: scale.outputs["Vector"]},
        attrs={"operation": "SUBTRACT"},
    )

    value_3 = nw.new_node(Nodes.Value)
    value_3.outputs[0].default_value = 1.0000

    divide_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: value_3, 1: scale.outputs["Vector"]},
        attrs={"operation": "DIVIDE"},
    )

    multiply_4 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: subtract_1.outputs["Vector"], 1: divide_1.outputs["Vector"]},
        attrs={"operation": "MULTIPLY"},
    )

    vector_rotate = nw.new_node(
        Nodes.VectorRotate,
        input_kwargs={"Vector": multiply_4.outputs["Vector"], "Angle": 0.7854},
    )

    absolute = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: vector_rotate},
        attrs={"operation": "ABSOLUTE"},
    )

    subtract_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: value_1, 1: absolute.outputs["Vector"]},
        attrs={"operation": "SUBTRACT"},
    )

    white_noise_texture = nw.new_node(
        Nodes.WhiteNoiseTexture, input_kwargs={"Vector": combine_xyz_1}
    )

    divide_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: snap, 1: multiply}, attrs={"operation": "DIVIDE"}
    )

    group_1 = nw.new_node(
        nodegroup_scalar_positive_modulo().name, input_kwargs={0: divide_2, 1: 2.0000}
    )

    greater_than = nw.new_node(
        Nodes.Math, input_kwargs={0: group_1}, attrs={"operation": "GREATER_THAN"}
    )

    divide_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: snap_1, 1: separate_xyz_1.outputs["X"]},
        attrs={"operation": "DIVIDE"},
    )

    group = nw.new_node(
        nodegroup_scalar_positive_modulo().name, input_kwargs={0: divide_3, 1: 2.0000}
    )

    greater_than_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: group}, attrs={"operation": "GREATER_THAN"}
    )

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: greater_than, 1: greater_than_1},
        attrs={"operation": "MULTIPLY"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "UV": subtract_2.outputs["Vector"],
            "Color": white_noise_texture.outputs["Color"],
            "Value": multiply_5,
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_diamond", singleton=False, type="ShaderNodeTree")
def nodegroup_diamond(nw: NodeWrangler):
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

    reroute_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Flatness"]}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["border"], 1: 1.5000},
        attrs={"operation": "MULTIPLY"},
    )

    reroute = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply})

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.5000

    mapping = nw.new_node(
        Nodes.Mapping,
        input_kwargs={"Vector": group_input.outputs["Coordinate"], "Scale": value},
    )

    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = 1.0000

    group = nw.new_node(
        nodegroup_diamond_single().name,
        input_kwargs={"Coordinate": mapping, "Width": value_1},
    )

    separate_xyz_3 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group.outputs["UV"]}
    )

    value_3 = nw.new_node(Nodes.Value)
    value_3.outputs[0].default_value = 0.0000

    smooth_min = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: separate_xyz_3.outputs["X"],
            1: separate_xyz_3.outputs["Y"],
            2: value_3,
        },
        attrs={"operation": "SMOOTH_MIN"},
    )

    mix = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: reroute, 6: smooth_min},
        attrs={"blend_type": "BURN", "data_type": "RGBA"},
    )

    mix_1 = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            0: reroute_1,
            6: mix.outputs[2],
            7: (1.0000, 1.0000, 1.0000, 1.0000),
        },
        attrs={"blend_type": "DODGE", "data_type": "RGBA"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: value_1}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply_1})

    vector_rotate = nw.new_node(
        Nodes.VectorRotate,
        input_kwargs={"Vector": mapping, "Center": combine_xyz, "Angle": -2.0944},
    )

    group_1 = nw.new_node(
        nodegroup_diamond_single().name,
        input_kwargs={"Coordinate": vector_rotate, "Width": value_1},
    )

    separate_xyz_4 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_1.outputs["UV"]}
    )

    smooth_min_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: separate_xyz_4.outputs["X"],
            1: separate_xyz_4.outputs["Y"],
            2: value_3,
        },
        attrs={"operation": "SMOOTH_MIN"},
    )

    mix_2 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: reroute, 6: smooth_min_1},
        attrs={"blend_type": "BURN", "data_type": "RGBA"},
    )

    mix_3 = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            0: reroute_1,
            6: mix_2.outputs[2],
            7: (1.0000, 1.0000, 1.0000, 1.0000),
        },
        attrs={"blend_type": "DODGE", "data_type": "RGBA"},
    )

    mix_4 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: 1.0000, 6: mix_1.outputs[2], 7: mix_3.outputs[2]},
        attrs={"blend_type": "ADD", "data_type": "RGBA"},
    )

    vector_rotate_1 = nw.new_node(
        Nodes.VectorRotate,
        input_kwargs={"Vector": mapping, "Center": combine_xyz, "Angle": 2.0944},
    )

    group_2 = nw.new_node(
        nodegroup_diamond_single().name,
        input_kwargs={"Coordinate": vector_rotate_1, "Width": value_1},
    )

    separate_xyz_5 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_2.outputs["UV"]}
    )

    smooth_min_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: separate_xyz_5.outputs["X"],
            1: separate_xyz_5.outputs["Y"],
            2: value_3,
        },
        attrs={"operation": "SMOOTH_MIN"},
    )

    mix_5 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: reroute, 6: smooth_min_2},
        attrs={"blend_type": "BURN", "data_type": "RGBA"},
    )

    mix_6 = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            0: reroute_1,
            6: mix_5.outputs[2],
            7: (1.0000, 1.0000, 1.0000, 1.0000),
        },
        attrs={"blend_type": "DODGE", "data_type": "RGBA"},
    )

    mix_7 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: 1.0000, 6: mix_4.outputs[2], 7: mix_6.outputs[2]},
        attrs={"blend_type": "ADD", "data_type": "RGBA"},
    )

    mix_8 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: 1.0000, 6: mix_1.outputs[2], 7: group.outputs["Color"]},
        attrs={"blend_type": "MULTIPLY", "data_type": "RGBA"},
    )

    mix_9 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: 1.0000, 6: group_1.outputs["Color"], 7: mix_3.outputs[2]},
        attrs={"blend_type": "MULTIPLY", "data_type": "RGBA"},
    )

    mix_11 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: 1.0000, 6: mix_8.outputs[2], 7: mix_9.outputs[2]},
        attrs={"blend_type": "ADD", "data_type": "RGBA"},
    )

    mix_10 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: 1.0000, 6: mix_6.outputs[2], 7: group_2.outputs["Color"]},
        attrs={"blend_type": "MULTIPLY", "data_type": "RGBA"},
    )

    mix_12 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: 1.0000, 6: mix_11.outputs[2], 7: mix_10.outputs[2]},
        attrs={"blend_type": "ADD", "data_type": "RGBA"},
    )

    greater_than = nw.new_node(
        Nodes.Math,
        input_kwargs={0: mix_1.outputs[2], 1: 0.0000},
        attrs={"operation": "GREATER_THAN"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: mix_6.outputs[2], 1: group_2.outputs["Value"]},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group.outputs["Value"], 1: mix_1.outputs[2]},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: mix_3.outputs[2], 1: group_1.outputs["Value"]},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: multiply_3, 1: multiply_4})

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_2, 1: add})

    greater_than_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add_1, 1: 0.0000},
        attrs={"operation": "GREATER_THAN"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Result": mix_7.outputs[2],
            "Tile Color": mix_12.outputs[2],
            "Tile Type 1": greater_than,
            "Tile Type 2": greater_than_1,
        },
        attrs={"is_active_output": True},
    )
