# Copyright (C) 2024, Princeton University.

# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Yiming Zuo

from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler

from .utils import nodegroup_scalar_positive_modulo


@node_utils.to_nodegroup(
    "nodegroup_positive_modulo", singleton=False, type="ShaderNodeTree"
)
def nodegroup_positive_modulo(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Vector_1", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketVector", "Vector_2", (0.0000, 0.0000, 0.0000)),
        ],
    )

    modulo = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs[0], 1: group_input.outputs[1]},
        attrs={"operation": "MODULO"},
    )

    separate_xyz_3 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": modulo.outputs["Vector"]}
    )

    less_than = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_3.outputs["X"], 1: 0.0000},
        attrs={"operation": "LESS_THAN"},
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs[1]}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: less_than, 1: separate_xyz.outputs["X"]},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: separate_xyz_3.outputs["X"], 1: multiply}
    )

    less_than_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_3.outputs["Y"], 1: 0.0000},
        attrs={"operation": "LESS_THAN"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: less_than_1, 1: separate_xyz.outputs["Y"]},
        attrs={"operation": "MULTIPLY"},
    )

    add_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: separate_xyz_3.outputs["Y"], 1: multiply_1}
    )

    less_than_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_3.outputs["Z"], 1: 0.0000},
        attrs={"operation": "LESS_THAN"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: less_than_2, 1: separate_xyz.outputs["Z"]},
        attrs={"operation": "MULTIPLY"},
    )

    add_2 = nw.new_node(Nodes.Math, input_kwargs={1: multiply_2})

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add, "Y": add_1, "Z": add_2}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Vector": combine_xyz_3},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_birck", singleton=False, type="ShaderNodeTree")
def nodegroup_birck(nw: NodeWrangler):
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

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Coordinate"]}
    )

    brick_width = nw.new_node(Nodes.Value, label="Brick Width")
    brick_width.outputs[0].default_value = 1.0000

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Subtiles Number"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    snap = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply, 1: 2.0000}, attrs={"operation": "SNAP"}
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: brick_width, 1: snap},
        attrs={"operation": "MULTIPLY"},
    )

    group_4 = nw.new_node(
        nodegroup_scalar_positive_modulo().name,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: multiply_1},
    )

    less_than = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_4, 1: brick_width},
        attrs={"operation": "LESS_THAN"},
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Aspect Ratio"]}
    )

    header_size = nw.new_node(Nodes.Value, label="header_size")
    header_size.outputs[0].default_value = 0.5000

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute, 1: header_size},
        attrs={"operation": "MULTIPLY"},
    )

    mix_3 = nw.new_node(
        Nodes.Mix, input_kwargs={0: less_than, 2: reroute, 3: multiply_2}
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": mix_3.outputs["Result"], "Y": brick_width}
    )

    value_4 = nw.new_node(Nodes.Value)
    value_4.outputs[0].default_value = 0.5000

    multiply_3 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_1, 1: value_4},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: brick_width, 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    group_1 = nw.new_node(
        nodegroup_scalar_positive_modulo().name,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: multiply_4},
    )

    less_than_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_1, 1: brick_width},
        attrs={"operation": "LESS_THAN"},
    )

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: less_than, 1: 0.2500},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: multiply_5})

    multiply_6 = nw.new_node(
        Nodes.Math, input_kwargs={0: reroute, 1: add}, attrs={"operation": "MULTIPLY"}
    )

    multiply_add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: less_than_1, 1: multiply_6, 2: separate_xyz.outputs["X"]},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": multiply_add, "Y": separate_xyz.outputs["Y"]},
    )

    group = nw.new_node(
        nodegroup_positive_modulo().name,
        input_kwargs={0: combine_xyz, 1: combine_xyz_1},
    )

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: multiply_3.outputs["Vector"], 1: group},
        attrs={"operation": "SUBTRACT"},
    )

    snap_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz, 1: combine_xyz_1},
        attrs={"operation": "SNAP"},
    )

    white_noise_texture = nw.new_node(
        Nodes.WhiteNoiseTexture,
        input_kwargs={"Vector": snap_1.outputs["Vector"]},
        attrs={"noise_dimensions": "4D"},
    )

    separate_color = nw.new_node(
        Nodes.SeparateColor,
        input_kwargs={"Color": white_noise_texture.outputs["Color"]},
    )

    map_range = nw.new_node(
        Nodes.MapRange,
        input_kwargs={"Value": separate_color.outputs["Red"], 3: -1.0000},
    )

    rotate_amount = nw.new_node(Nodes.Value, label="Rotate amount")
    rotate_amount.outputs[0].default_value = 0.0000

    multiply_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: map_range.outputs["Result"], 1: rotate_amount},
        attrs={"operation": "MULTIPLY"},
    )

    vector_rotate = nw.new_node(
        Nodes.VectorRotate,
        input_kwargs={"Vector": subtract.outputs["Vector"], "Angle": multiply_7},
        attrs={"rotation_type": "Z_AXIS"},
    )

    absolute = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: vector_rotate},
        attrs={"operation": "ABSOLUTE"},
    )

    subtract_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: multiply_3.outputs["Vector"], 1: absolute.outputs["Vector"]},
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract_1.outputs["Vector"]}
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

    separate_xyz_2 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": snap_1.outputs["Vector"]}
    )

    separate_xyz_3 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": combine_xyz_1}
    )

    multiply_8 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_3.outputs["X"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    group_3 = nw.new_node(
        nodegroup_scalar_positive_modulo().name,
        input_kwargs={0: separate_xyz_2.outputs["X"], 1: multiply_8},
    )

    multiply_9 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_3.outputs["X"]},
        attrs={"operation": "MULTIPLY"},
    )

    greater_than = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_3, 1: multiply_9},
        attrs={"operation": "GREATER_THAN"},
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": less_than})

    multiply_10 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: greater_than, 1: reroute_1},
        attrs={"operation": "MULTIPLY"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Result": mix_1.outputs[2],
            "Tile Color": white_noise_texture.outputs["Color"],
            "Tile Type 1": less_than_1,
            "Tile Type 2": multiply_10,
        },
        attrs={"is_active_output": True},
    )
