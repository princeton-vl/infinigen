# Copyright (C) 2024, Princeton University.

# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Yiming Zuo

from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler

from .utils import nodegroup_scalar_positive_modulo


@node_utils.to_nodegroup("nodegroup_cheveron", singleton=False, type="ShaderNodeTree")
def nodegroup_cheveron(nw: NodeWrangler):
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

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": scale.outputs["Vector"]}
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Coordinate"]}
    )

    group_1 = nw.new_node(
        nodegroup_scalar_positive_modulo().name,
        input_kwargs={0: separate_xyz.outputs["X"], 1: reroute_1},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_1, 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_1, 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    snap = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: reroute_1},
        attrs={"operation": "SNAP"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_1, 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    group_2 = nw.new_node(
        nodegroup_scalar_positive_modulo().name, input_kwargs={0: snap, 1: multiply_1}
    )

    greater_than = nw.new_node(
        Nodes.Math, input_kwargs={0: group_2}, attrs={"operation": "GREATER_THAN"}
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract, 1: greater_than},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: multiply_2, 1: group_1})

    rot_amount = nw.new_node(Nodes.Value, label="rot amount")
    rot_amount.outputs[0].default_value = 1.0000

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add, 1: rot_amount},
        attrs={"operation": "MULTIPLY"},
    )

    add_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: separate_xyz.outputs["Y"], 1: multiply_3}
    )

    group = nw.new_node(
        nodegroup_scalar_positive_modulo().name, input_kwargs={0: add_1, 1: brick_width}
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": group_1, "Y": group}
    )

    subtract_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_1, 1: reroute},
        attrs={"operation": "SUBTRACT"},
    )

    absolute = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: subtract_1.outputs["Vector"]},
        attrs={"operation": "ABSOLUTE"},
    )

    subtract_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute, 1: absolute.outputs["Vector"]},
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract_2.outputs["Vector"]}
    )

    smooth_min = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: separate_xyz_1.outputs["X"],
            1: separate_xyz_1.outputs["Y"],
            2: 0.1000,
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

    snap_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_1, 1: brick_width}, attrs={"operation": "SNAP"}
    )

    snap_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: reroute_1},
        attrs={"operation": "SNAP"},
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": snap_1, "Y": snap_2}
    )

    white_noise_texture = nw.new_node(
        Nodes.WhiteNoiseTexture, input_kwargs={"Vector": combine_xyz_3}
    )

    snap_3 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_1, 1: brick_width}, attrs={"operation": "SNAP"}
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: brick_width, 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    group_3 = nw.new_node(
        nodegroup_scalar_positive_modulo().name, input_kwargs={0: snap_3, 1: multiply_4}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Result": mix_1.outputs[2],
            "Tile Color": white_noise_texture.outputs["Color"],
            "Tile Type 1": greater_than,
            "Tile Type 2": group_3,
        },
        attrs={"is_active_output": True},
    )
