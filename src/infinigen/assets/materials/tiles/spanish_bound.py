# Copyright (C) 2024, Princeton University.

# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Yiming Zuo

from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler

from .utils import nodegroup_scalar_positive_modulo, nodegroup_u_v_recenter


@node_utils.to_nodegroup(
    "nodegroup_spanish_bond", singleton=False, type="ShaderNodeTree"
)
def nodegroup_spanish_bond(nw: NodeWrangler):
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

    reroute_3 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Aspect Ratio"]}
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: reroute_3, 1: brick_width})

    group = nw.new_node(
        nodegroup_scalar_positive_modulo().name, input_kwargs={0: subtract, 1: add}
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Subtiles Number"]}
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: brick_width, 1: reroute_2},
        attrs={"operation": "DIVIDE"},
    )

    group_3 = nw.new_node(
        nodegroup_scalar_positive_modulo().name,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: divide},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": group, "Y": group_3}
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": reroute_3, "Y": brick_width}
    )

    scale = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz, "Scale": 0.5000},
        attrs={"operation": "SCALE"},
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": 1.0000, "Y": reroute_2, "Z": 1.0000}
    )

    divide_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: scale.outputs["Vector"], 1: combine_xyz_3},
        attrs={"operation": "DIVIDE"},
    )

    group_10 = nw.new_node(
        nodegroup_u_v_recenter().name,
        input_kwargs={"UV": combine_xyz_1, "HalfSize": divide_1.outputs["Vector"]},
    )

    group_5 = nw.new_node(
        nodegroup_scalar_positive_modulo().name, input_kwargs={0: snap, 1: add}
    )

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: brick_width})

    subtract_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: add, 1: add_1}, attrs={"operation": "SUBTRACT"}
    )

    greater_than = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_5, 1: subtract_1},
        attrs={"operation": "GREATER_THAN"},
    )

    less_than = nw.new_node(
        Nodes.Math, input_kwargs={0: group_5}, attrs={"operation": "LESS_THAN"}
    )

    add_2 = nw.new_node(Nodes.Math, input_kwargs={0: greater_than, 1: less_than})

    greater_than_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group, 1: reroute_3},
        attrs={"operation": "GREATER_THAN"},
    )

    big_tiles = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 1.0000, 1: greater_than_1},
        label="big tiles",
        attrs={"operation": "SUBTRACT"},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add_2, 1: big_tiles},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_10, 1: multiply},
        attrs={"operation": "MULTIPLY"},
    )

    add_3 = nw.new_node(
        Nodes.Math, input_kwargs={0: separate_xyz.outputs["X"], 1: brick_width}
    )

    snap_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_3, 1: brick_width}, attrs={"operation": "SNAP"}
    )

    add_4 = nw.new_node(
        Nodes.Math, input_kwargs={0: separate_xyz.outputs["Y"], 1: snap_1}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": add})

    group_2 = nw.new_node(
        nodegroup_scalar_positive_modulo().name, input_kwargs={0: add_4, 1: reroute_1}
    )

    divide_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: brick_width, 1: reroute_2},
        attrs={"operation": "DIVIDE"},
    )

    group_4 = nw.new_node(
        nodegroup_scalar_positive_modulo().name,
        input_kwargs={0: separate_xyz.outputs["X"], 1: divide_2},
    )

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": group_2, "Y": group_4}
    )

    group_12 = nw.new_node(
        nodegroup_u_v_recenter().name,
        input_kwargs={"UV": combine_xyz_5, "HalfSize": divide_1.outputs["Vector"]},
    )

    group_7 = nw.new_node(
        nodegroup_scalar_positive_modulo().name, input_kwargs={0: snap_1, 1: add}
    )

    add_5 = nw.new_node(Nodes.Math, input_kwargs={0: brick_width})

    subtract_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: add, 1: add_5}, attrs={"operation": "SUBTRACT"}
    )

    greater_than_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_7, 1: subtract_2},
        attrs={"operation": "GREATER_THAN"},
    )

    less_than_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_7}, attrs={"operation": "LESS_THAN"}
    )

    add_6 = nw.new_node(Nodes.Math, input_kwargs={0: greater_than_2, 1: less_than_1})

    greater_than_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_2, 1: reroute_3},
        attrs={"operation": "GREATER_THAN"},
    )

    big_tiles_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 1.0000, 1: greater_than_3},
        label="big tiles",
        attrs={"operation": "SUBTRACT"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add_6, 1: big_tiles_1},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_3 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_12, 1: multiply_2},
        attrs={"operation": "MULTIPLY"},
    )

    add_7 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: multiply_1.outputs["Vector"], 1: multiply_3.outputs["Vector"]},
    )

    add_8 = nw.new_node(Nodes.Math, input_kwargs={0: multiply, 1: multiply_2})

    subtract_3 = nw.new_node(
        Nodes.Math, input_kwargs={0: 1.0000, 1: add_8}, attrs={"operation": "SUBTRACT"}
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": separate_xyz.outputs["X"], "Y": separate_xyz.outputs["Y"]},
    )

    separate_xyz_2 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": combine_xyz_2}
    )

    subtract_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Y"], 1: brick_width},
        attrs={"operation": "SUBTRACT"},
    )

    reroute = nw.new_node(Nodes.Reroute, input_kwargs={"Input": add})

    group_6 = nw.new_node(
        nodegroup_scalar_positive_modulo().name,
        input_kwargs={0: subtract_4, 1: reroute},
    )

    subtract_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_3, 1: brick_width},
        attrs={"operation": "SUBTRACT"},
    )

    absolute = nw.new_node(
        Nodes.Math, input_kwargs={0: subtract_5}, attrs={"operation": "ABSOLUTE"}
    )

    center_subdivide = nw.new_node(Nodes.Value, label="Center Subdivide")
    center_subdivide.outputs[0].default_value = 1.0000

    divide_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: absolute, 1: center_subdivide},
        attrs={"operation": "DIVIDE"},
    )

    group_8 = nw.new_node(
        nodegroup_scalar_positive_modulo().name, input_kwargs={0: group_6, 1: divide_3}
    )

    group_1 = nw.new_node(
        nodegroup_scalar_positive_modulo().name,
        input_kwargs={0: separate_xyz_2.outputs["X"], 1: reroute},
    )

    group_11 = nw.new_node(
        nodegroup_scalar_positive_modulo().name, input_kwargs={0: group_1, 1: divide_3}
    )

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": group_8, "Y": group_11}
    )

    divide_4 = nw.new_node(
        Nodes.Math, input_kwargs={0: absolute, 1: 2.0000}, attrs={"operation": "DIVIDE"}
    )

    divide_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: divide_4, 1: center_subdivide},
        attrs={"operation": "DIVIDE"},
    )

    group_9 = nw.new_node(
        nodegroup_u_v_recenter().name,
        input_kwargs={"UV": combine_xyz_4, "HalfSize": divide_5},
    )

    multiply_4 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: subtract_3, 1: group_9},
        attrs={"operation": "MULTIPLY"},
    )

    add_9 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: add_7.outputs["Vector"], 1: multiply_4.outputs["Vector"]},
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": add_9.outputs["Vector"]}
    )

    smooth_min = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: separate_xyz_1.outputs["X"],
            1: separate_xyz_1.outputs["Y"],
            2: 0.0500,
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

    snap_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: divide},
        attrs={"operation": "SNAP"},
    )

    snap_3 = nw.new_node(
        Nodes.Math, input_kwargs={0: subtract, 1: add}, attrs={"operation": "SNAP"}
    )

    combine_xyz_7 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": snap_2, "Y": snap_3}
    )

    multiply_5 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: multiply, 1: combine_xyz_7},
        attrs={"operation": "MULTIPLY"},
    )

    snap_4 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_4, 1: reroute_1}, attrs={"operation": "SNAP"}
    )

    snap_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: divide_2},
        attrs={"operation": "SNAP"},
    )

    combine_xyz_8 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": snap_4, "Y": snap_5}
    )

    multiply_6 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: multiply_2, 1: combine_xyz_8},
        attrs={"operation": "MULTIPLY"},
    )

    add_10 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: multiply_5.outputs["Vector"], 1: multiply_6.outputs["Vector"]},
    )

    snap_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_4, 1: reroute},
        attrs={"operation": "SNAP"},
    )

    snap_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["X"], 1: reroute},
        attrs={"operation": "SNAP"},
    )

    combine_xyz_9 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": snap_6, "Y": snap_7}
    )

    multiply_7 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_9, 1: subtract_3},
        attrs={"operation": "MULTIPLY"},
    )

    add_11 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: add_10.outputs["Vector"], 1: multiply_7.outputs["Vector"]},
    )

    white_noise_texture = nw.new_node(
        Nodes.WhiteNoiseTexture, input_kwargs={"Vector": add_11.outputs["Vector"]}
    )

    greater_than_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_3, 1: 0.0000},
        attrs={"operation": "GREATER_THAN"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Result": mix_1.outputs[2],
            "Tile Color": white_noise_texture.outputs["Color"],
            "Tile Type 1": greater_than_4,
            "Tile Type 2": greater_than_4,
        },
        attrs={"is_active_output": True},
    )
