# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo
# Acknowledgement: This file draws inspiration https://www.youtube.com/watch?v=XDqRa0ExDqs by Ryan King Art


from numpy.random import uniform

from infinigen.assets.materials import common
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.util.color import color_category


@node_utils.to_nodegroup("nodegroup_crack", singleton=False, type="ShaderNodeTree")
def nodegroup_crack(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Seed", 0.0000),
            ("NodeSocketFloat", "Amount", 1.0000),
            ("NodeSocketFloat", "Scale", 0.0000),
            ("NodeSocketFloat", "Snake Crack", 0.3000),
        ],
    )

    texture_coordinate_1 = nw.new_node(Nodes.TextureCoord)

    musgrave_texture_2 = nw.new_node(
        Nodes.MusgraveTexture,
        input_kwargs={
            "Vector": texture_coordinate_1.outputs["Object"],
            "W": group_input.outputs["Seed"],
            "Scale": group_input.outputs["Scale"],
            "Detail": 15.0000,
            "Dimension": 0.2000,
        },
        attrs={"musgrave_dimensions": "4D"},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Scale"]},
        attrs={"operation": "MULTIPLY"},
    )

    noise_texture_2 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": texture_coordinate_1.outputs["Object"],
            "W": group_input.outputs["Seed"],
            "Scale": multiply,
            "Detail": 15.0000,
        },
        attrs={"noise_dimensions": "4D"},
    )

    voronoi_texture = nw.new_node(
        Nodes.VoronoiTexture,
        input_kwargs={"Vector": noise_texture_2.outputs["Fac"], "Scale": 1.2000},
        attrs={"feature": "DISTANCE_TO_EDGE"},
    )

    map_range_4 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": voronoi_texture.outputs["Distance"],
            2: 0.0200,
            3: 2.0000,
            4: 0.0000,
        },
    )

    mix_7 = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            0: group_input.outputs["Snake Crack"],
            6: musgrave_texture_2,
            7: map_range_4.outputs["Result"],
        },
        attrs={"blend_type": "ADD", "data_type": "RGBA"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Scale"], 1: 0.6000},
        attrs={"operation": "MULTIPLY"},
    )

    musgrave_texture_3 = nw.new_node(
        Nodes.MusgraveTexture,
        input_kwargs={
            "Vector": texture_coordinate_1.outputs["Object"],
            "W": group_input.outputs["Seed"],
            "Scale": multiply_1,
            "Detail": 15.0000,
            "Dimension": 1.0000,
        },
        attrs={"musgrave_dimensions": "4D"},
    )

    map_range_2 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={"Value": group_input.outputs["Amount"], 3: 1.0000, 4: -0.5000},
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: map_range_2.outputs["Result"], 1: 0.1000}
    )

    map_range_1 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": musgrave_texture_3,
            1: map_range_2.outputs["Result"],
            2: add,
        },
    )

    mix_4 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: 1.0000, 6: mix_7.outputs[2], 7: map_range_1.outputs["Result"]},
        attrs={"blend_type": "DARKEN", "data_type": "RGBA"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Scale"], 1: 0.3000},
        attrs={"operation": "MULTIPLY"},
    )

    musgrave_texture_4 = nw.new_node(
        Nodes.MusgraveTexture,
        input_kwargs={
            "Vector": texture_coordinate_1.outputs["Object"],
            "W": group_input.outputs["Seed"],
            "Scale": multiply_2,
            "Detail": 15.0000,
            "Dimension": 1.0000,
        },
        attrs={"musgrave_dimensions": "4D"},
    )

    add_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: map_range_2.outputs["Result"], 1: 0.1000}
    )

    map_range = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": musgrave_texture_4,
            1: map_range_2.outputs["Result"],
            2: add_1,
        },
    )

    mix_5 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: 1.0000, 6: mix_4.outputs[2], 7: map_range.outputs["Result"]},
        attrs={"blend_type": "DARKEN", "data_type": "RGBA"},
    )

    color_ramp = nw.new_node(Nodes.ColorRamp, input_kwargs={"Fac": mix_5.outputs[2]})
    color_ramp.color_ramp.elements[0].position = 0.0000
    color_ramp.color_ramp.elements[0].color = [0.0000, 0.0000, 0.0000, 1.0000]
    color_ramp.color_ramp.elements[1].position = 1.0000
    color_ramp.color_ramp.elements[1].color = [1.0000, 1.0000, 1.0000, 1.0000]

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Color": color_ramp.outputs["Color"]},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_concrete", singleton=False, type="ShaderNodeTree")
def nodegroup_concrete(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input_1 = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketColor", "Base Color", (0.8000, 0.8000, 0.8000, 1.0000)),
            ("NodeSocketFloat", "Scale", 0.0000),
            ("NodeSocketFloat", "Seed", 0.0000),
            ("NodeSocketFloat", "Roughness", 0.0000),
            ("NodeSocketFloat", "Crack Amount", 0.0000),
            ("NodeSocketFloat", "Crack Scale", 0.0000),
            ("NodeSocketFloat", "Snake Crack", 0.3000),
        ],
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_1.outputs["Scale"], 1: 10.0000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply, 1: group_input_1.outputs["Crack Scale"]},
        attrs={"operation": "MULTIPLY"},
    )

    group = nw.new_node(
        nodegroup_crack().name,
        input_kwargs={
            "Seed": group_input_1.outputs["Seed"],
            "Amount": group_input_1.outputs["Crack Amount"],
            "Scale": multiply_1,
            "Snake Crack": group_input_1.outputs["Snake Crack"],
        },
    )

    map_range_3 = nw.new_node(Nodes.MapRange, input_kwargs={"Value": group})

    texture_coordinate = nw.new_node(Nodes.TextureCoord)

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_1.outputs["Scale"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    noise_texture_1 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": texture_coordinate.outputs["Object"],
            "W": group_input_1.outputs["Seed"],
            "Scale": multiply_2,
            "Detail": 15.0000,
        },
        attrs={"noise_dimensions": "4D"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_1.outputs["Scale"], 1: 5.0000},
        attrs={"operation": "MULTIPLY"},
    )

    musgrave_texture_1 = nw.new_node(
        Nodes.MusgraveTexture,
        input_kwargs={
            "Vector": texture_coordinate.outputs["Object"],
            "W": group_input_1.outputs["Seed"],
            "Scale": multiply_3,
            "Detail": 15.0000,
            "Dimension": 1.0000,
            "Lacunarity": 3.0000,
        },
        attrs={"musgrave_dimensions": "4D"},
    )

    mix_2 = nw.new_node(
        Nodes.Mix, input_kwargs={6: musgrave_texture_1}, attrs={"data_type": "RGBA"}
    )

    hue_saturation_value_1 = nw.new_node(
        "ShaderNodeHueSaturation",
        input_kwargs={"Value": 0.6000, "Color": group_input_1.outputs["Base Color"]},
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_1.outputs["Scale"], 1: 20.0000},
        attrs={"operation": "MULTIPLY"},
    )

    noise_texture = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": texture_coordinate.outputs["Object"],
            "W": group_input_1.outputs["Seed"],
            "Scale": multiply_4,
            "Detail": 15.0000,
            "Distortion": 0.2000,
        },
        attrs={"noise_dimensions": "4D"},
    )

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_1.outputs["Scale"], 1: 20.0000},
        attrs={"operation": "MULTIPLY"},
    )

    musgrave_texture = nw.new_node(
        Nodes.MusgraveTexture,
        input_kwargs={
            "Vector": texture_coordinate.outputs["Object"],
            "W": group_input_1.outputs["Seed"],
            "Scale": multiply_5,
            "Detail": 15.0000,
            "Dimension": 0.2000,
        },
        attrs={"musgrave_dimensions": "4D"},
    )

    mix = nw.new_node(
        Nodes.Mix,
        input_kwargs={6: noise_texture.outputs["Fac"], 7: musgrave_texture},
        attrs={"data_type": "RGBA"},
    )

    hue_saturation_value = nw.new_node(
        "ShaderNodeHueSaturation",
        input_kwargs={"Value": 1.4000, "Color": group_input_1.outputs["Base Color"]},
    )

    mix_1 = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            0: mix.outputs[2],
            6: group_input_1.outputs["Base Color"],
            7: hue_saturation_value,
        },
        attrs={"data_type": "RGBA"},
    )

    mix_3 = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            0: mix_2.outputs[2],
            6: hue_saturation_value_1,
            7: mix_1.outputs[2],
        },
        attrs={"data_type": "RGBA"},
    )

    hue_saturation_value_2 = nw.new_node(
        "ShaderNodeHueSaturation",
        input_kwargs={
            "Value": noise_texture_1.outputs["Fac"],
            "Fac": 0.2000,
            "Color": mix_3.outputs[2],
        },
    )

    hue_saturation_value_3 = nw.new_node(
        "ShaderNodeHueSaturation",
        input_kwargs={"Value": 0.2000, "Color": group_input_1.outputs["Base Color"]},
    )

    mix_6 = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            0: map_range_3.outputs["Result"],
            6: hue_saturation_value_2,
            7: hue_saturation_value_3,
        },
        attrs={"data_type": "RGBA"},
    )

    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": mix_6.outputs[2],
            "Roughness": group_input_1.outputs["Roughness"],
        },
    )

    multiply_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_1.outputs["Crack Amount"], 1: 0.6000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_1, 1: 5.0000},
        attrs={"operation": "MULTIPLY"},
    )

    group_1 = nw.new_node(
        nodegroup_crack().name,
        input_kwargs={
            "Seed": group_input_1.outputs["Seed"],
            "Amount": multiply_6,
            "Scale": multiply_7,
        },
    )

    multiply_8 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_1.outputs["Roughness"], 1: 1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_9 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_1, 1: multiply_8},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_10 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_8, 1: group},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: multiply_9, 1: multiply_10})

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.3000

    multiply_11 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: value, 1: group_input_1.outputs["Roughness"]},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_12 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_11, 1: mix_1.outputs[2]},
        attrs={"operation": "MULTIPLY"},
    )

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: add, 1: multiply_12})

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"BSDF": principled_bsdf, "Displacement": add_1},
        attrs={"is_active_output": True},
    )


def shader_concrete(
    nw: NodeWrangler,
    scale=1.0,
    base_color=None,
    seed=None,
    roughness=None,
    crack_amount=None,
    crack_scale=None,
    snake_crack=None,
    **kwargs,
):
    # Code generated using version 2.6.4 of the node_transpiler
    if seed is None:
        seed = uniform(-1000.0, 1000.0)
    if roughness is None:
        roughness = uniform(0.5, 1.0)
    if crack_amount is None:
        crack_amount = uniform(0.2, 0.8)
    if crack_scale is None:
        crack_scale = uniform(1.0, 3.0)
    if snake_crack is None:
        snake_crack = uniform(0.0, 1.0)
    if base_color is None:
        base_color = color_category("concrete")

    group = nw.new_node(
        nodegroup_concrete().name,
        input_kwargs={
            "Base Color": base_color,
            "Scale": scale,
            "Seed": seed,
            "Roughness": roughness,
            "Crack Amount": crack_amount,
            "Crack Scale": crack_scale,
            "Snake Crack": snake_crack,
        },
    )

    displacement_1 = nw.new_node(
        "ShaderNodeDisplacement",
        input_kwargs={
            "Height": group.outputs["Displacement"],
            "Midlevel": 0.0000,
            "Scale": 0.0500,
        },
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": group.outputs["BSDF"], "Displacement": displacement_1},
        attrs={"is_active_output": True},
    )


def apply(obj, selection=None, **kwargs):
    common.apply(obj, shader_concrete, selection=selection, **kwargs)
