# Copyright (C) 2024, Princeton University.

# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Yiming Zuo

from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler

from .utils import nodegroup_scalar_positive_modulo


@node_utils.to_nodegroup("nodegroup_mix", singleton=False, type="ShaderNodeTree")
def nodegroup_mix(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Mask", 0.0000),
            ("NodeSocketVector", "Vector 1", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketVector", "Vector 2", (0.0000, 0.0000, 0.0000)),
        ],
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 1.0000, 1: group_input.outputs["Mask"]},
        attrs={"operation": "SUBTRACT"},
    )

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Vector 2"], 1: subtract},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: group_input.outputs["Vector 1"],
            1: group_input.outputs["Mask"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: multiply.outputs["Vector"], 1: multiply_1.outputs["Vector"]},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Vector": add.outputs["Vector"]},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_herringbone", singleton=False, type="ShaderNodeTree"
)
def nodegroup_herringbone(nw: NodeWrangler):
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
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Aspect Ratio"]}
    )

    brick_width = nw.new_node(Nodes.Value, label="Brick Width")
    brick_width.outputs[0].default_value = 1.0000

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": reroute_1, "Y": brick_width}
    )

    scale = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz, "Scale": 0.5000},
        attrs={"operation": "SCALE"},
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": 1.0000,
            "Y": group_input.outputs["Subtiles Number"],
            "Z": 1.0000,
        },
    )

    divide = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: scale.outputs["Vector"], 1: combine_xyz_3},
        attrs={"operation": "DIVIDE"},
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": divide.outputs["Vector"]}
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Coordinate"]}
    )

    snap = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: brick_width},
        attrs={"operation": "SNAP"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: snap},
        attrs={"operation": "SUBTRACT"},
    )

    snap_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract, 1: reroute_1},
        attrs={"operation": "SNAP"},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_1, 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    group_1 = nw.new_node(
        nodegroup_scalar_positive_modulo().name, input_kwargs={0: snap_1, 1: multiply}
    )

    greater_than = nw.new_node(
        Nodes.Math, input_kwargs={0: group_1}, attrs={"operation": "GREATER_THAN"}
    )

    group = nw.new_node(
        nodegroup_scalar_positive_modulo().name,
        input_kwargs={0: subtract, 1: reroute_1},
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: brick_width, 1: group_input.outputs["Subtiles Number"]},
        attrs={"operation": "DIVIDE"},
    )

    group_3 = nw.new_node(
        nodegroup_scalar_positive_modulo().name,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: divide_1},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": group, "Y": group_3}
    )

    snap_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: brick_width},
        attrs={"operation": "SNAP"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: snap_2, 1: brick_width},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: multiply_1, 1: brick_width})

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: add},
        attrs={"operation": "SUBTRACT"},
    )

    group_2 = nw.new_node(
        nodegroup_scalar_positive_modulo().name,
        input_kwargs={0: subtract_1, 1: reroute_1},
    )

    divide_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: brick_width, 1: group_input.outputs["Subtiles Number"]},
        attrs={"operation": "DIVIDE"},
    )

    group_4 = nw.new_node(
        nodegroup_scalar_positive_modulo().name,
        input_kwargs={0: separate_xyz.outputs["X"], 1: divide_2},
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": group_2, "Y": group_4}
    )

    group_5 = nw.new_node(
        nodegroup_mix().name,
        input_kwargs={
            "Mask": greater_than,
            "Vector 1": combine_xyz_1,
            "Vector 2": combine_xyz_2,
        },
    )

    subtract_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_5, 1: reroute},
        attrs={"operation": "SUBTRACT"},
    )

    absolute = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: subtract_2.outputs["Vector"]},
        attrs={"operation": "ABSOLUTE"},
    )

    subtract_3 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute, 1: absolute.outputs["Vector"]},
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract_3.outputs["Vector"]}
    )

    smooth_min = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: separate_xyz_1.outputs["X"],
            1: separate_xyz_1.outputs["Y"],
            2: 0.0000,
        },
        attrs={"operation": "SMOOTH_MIN"},
    )

    mix = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: group_input.outputs["border"], 6: smooth_min},
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

    snap_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract, 1: reroute_1},
        attrs={"operation": "SNAP"},
    )

    snap_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: divide_1},
        attrs={"operation": "SNAP"},
    )

    combine_xyz_6 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": snap_3, "Y": snap_4}
    )

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.5000

    add_1 = nw.new_node(Nodes.VectorMath, input_kwargs={0: combine_xyz_6, 1: value})

    snap_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_1, 1: reroute_1},
        attrs={"operation": "SNAP"},
    )

    snap_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: divide_2},
        attrs={"operation": "SNAP"},
    )

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": snap_5, "Y": snap_6}
    )

    group_6 = nw.new_node(
        nodegroup_mix().name,
        input_kwargs={
            "Mask": greater_than,
            "Vector 1": add_1.outputs["Vector"],
            "Vector 2": combine_xyz_5,
        },
    )

    white_noise_texture_1 = nw.new_node(
        Nodes.WhiteNoiseTexture, input_kwargs={"Vector": group_6}
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: divide_1, 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    group_9 = nw.new_node(
        nodegroup_scalar_positive_modulo().name, input_kwargs={0: snap_4, 1: multiply_2}
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: divide_2, 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    group_7 = nw.new_node(
        nodegroup_scalar_positive_modulo().name, input_kwargs={0: snap_6, 1: multiply_3}
    )

    group_8 = nw.new_node(
        nodegroup_mix().name,
        input_kwargs={"Mask": greater_than, "Vector 1": group_9, "Vector 2": group_7},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Result": mix_1.outputs[2],
            "Tile Color": white_noise_texture_1.outputs["Color"],
            "Tile Type 1": greater_than,
            "Tile Type 2": group_8,
        },
        attrs={"is_active_output": True},
    )
