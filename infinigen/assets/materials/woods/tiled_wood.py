# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo
# Acknowledgement: This file draws inspiration https://www.youtube.com/watch?v=rd2jhGV6tqo by Ryan King Art


from numpy.random import randint, uniform

from infinigen.assets.materials import common
from infinigen.assets.materials.bark_random import hex_to_rgb
from infinigen.assets.materials.woods.wood import get_color
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.util.random import clip_gaussian


@node_utils.to_nodegroup("nodegroup_tiling", singleton=False, type="ShaderNodeTree")
def nodegroup_tiling(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    texture_coordinate = nw.new_node(Nodes.TextureCoord)

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Horizontal Scale", 0.5000),
            ("NodeSocketFloat", "Vertical Scale", 0.5),
            ("NodeSocketFloat", "Seed", 0.5000),
        ],
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 1.0000, 1: group_input.outputs["Horizontal Scale"]},
        attrs={"operation": "DIVIDE"},
    )

    multiply = nw.new_node(
        Nodes.Math, input_kwargs={0: divide}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply})

    vec = texture_coordinate.outputs["Object"]

    vec = nw.new_node(Nodes.Mapping, [vec, uniform(0, 1, 3)])

    add = nw.new_node(Nodes.VectorMath, input_kwargs={0: vec, 1: combine_xyz})

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 1.0000, 1: group_input.outputs["Vertical Scale"]},
        attrs={"operation": "DIVIDE"},
    )

    brick_texture = nw.new_node(
        Nodes.BrickTexture,
        input_kwargs={
            "Vector": add.outputs["Vector"],
            "Color2": (0, 0, 0, 1.0000),
            "Scale": 1.0000,
            "Mortar Size": 0.0050,
            "Mortar Smooth": 1.0000,
            "Bias": -0.5000,
            "Brick Width": divide_1,
            "Row Height": divide,
        },
        attrs={"squash_frequency": 1},
    )

    multiply_add = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: brick_texture.outputs["Color"],
            1: 1000.0000,
            2: group_input.outputs["Seed"],
        },
        attrs={"operation": "MULTIPLY_ADD"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Tile Color": brick_texture.outputs["Color"],
            "Seed": multiply_add,
            "Displacement": brick_texture.outputs["Color"],
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_tiled_wood", singleton=False, type="ShaderNodeTree")
def nodegroup_tiled_wood(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    texture_coordinate = nw.new_node(Nodes.TextureCoord)

    mapping_2 = nw.new_node(
        Nodes.Mapping,
        input_kwargs={
            "Vector": texture_coordinate.outputs["Object"],
            "Scale": (5.0000, 100.0000, 100.0000),
        },
    )

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Tile Horizontal Scale", 0.0000),
            ("NodeSocketFloat", "Tile Vertical Scale", 2.9600),
            ("NodeSocketColor", "Main Color", (0.0000, 0.0000, 0.0000, 1.0000)),
            ("NodeSocketFloat", "Seed", 0.0000),
        ],
    )

    group = nw.new_node(
        nodegroup_tiling().name,
        input_kwargs={
            "Horizontal Scale": group_input.outputs["Tile Horizontal Scale"],
            "Vertical Scale": group_input.outputs["Tile Vertical Scale"],
            "Seed": group_input.outputs["Seed"],
        },
    )

    musgrave_texture_2 = nw.new_node(
        Nodes.MusgraveTexture,
        input_kwargs={
            "Vector": mapping_2,
            "W": group.outputs["Seed"],
            "Scale": 10.0000,
            "Detail": 15.0000,
            "Dimension": 7.0000,
        },
        attrs={"musgrave_dimensions": "4D"},
    )

    map_range_2 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={"Value": musgrave_texture_2, 3: 1.0000, 4: -1.0000},
    )

    mapping_1 = nw.new_node(
        Nodes.Mapping, input_kwargs={"Vector": texture_coordinate.outputs["Object"]}
    )

    noise_texture_1 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": mapping_1,
            "W": group.outputs["Seed"],
            "Scale": 0.5000,
            "Detail": 1.0000,
            "Distortion": 1.1000,
        },
        attrs={"noise_dimensions": "4D"},
    )

    musgrave_texture_1 = nw.new_node(
        Nodes.MusgraveTexture,
        input_kwargs={
            "W": group.outputs["Seed"],
            "Scale": noise_texture_1.outputs["Fac"],
            "Detail": 15.0000,
            "Dimension": 0.2000,
            "Lacunarity": 2.4000,
        },
        attrs={"musgrave_dimensions": "4D"},
    )

    map_range = nw.new_node(
        Nodes.MapRange,
        input_kwargs={"Value": musgrave_texture_1, 3: -1.4000, 4: 1.5000},
    )

    map_range_1 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={"Value": map_range.outputs["Result"], 3: 1.0000, 4: 0.5000},
    )

    mapping = nw.new_node(
        Nodes.Mapping,
        input_kwargs={
            "Vector": texture_coordinate.outputs["Object"],
            "Scale": (0.1500, 1.0000, 0.1500),
        },
    )

    noise_texture = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": mapping,
            "W": group.outputs["Seed"],
            "Detail": 5.0000,
            "Distortion": 1.0000,
        },
        attrs={"noise_dimensions": "4D"},
    )

    musgrave_texture = nw.new_node(
        Nodes.MusgraveTexture,
        input_kwargs={
            "Vector": noise_texture.outputs["Fac"],
            "W": group.outputs["Seed"],
            "Scale": 4.0000,
            "Detail": 10.0000,
            "Dimension": 0.0000,
        },
        attrs={"musgrave_dimensions": "4D"},
    )

    mix = nw.new_node(
        Nodes.Mix,
        input_kwargs={6: noise_texture.outputs["Fac"], 7: musgrave_texture},
        attrs={"data_type": "RGBA"},
    )

    mix_1 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: 0.9000, 6: map_range_1.outputs["Result"], 7: mix.outputs[2]},
        attrs={"blend_type": "MULTIPLY", "data_type": "RGBA"},
    )

    mix_2 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: 0.9500, 6: map_range_2.outputs["Result"], 7: mix_1.outputs[2]},
        attrs={"blend_type": "MULTIPLY", "data_type": "RGBA"},
    )

    hue_saturation_value = nw.new_node(
        "ShaderNodeHueSaturation",
        input_kwargs={
            "Saturation": 0.8000,
            "Value": 0.2000,
            "Fac": 0.0,
            "Color": group_input.outputs["Main Color"],
        },
    )

    mix_3 = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            0: mix_2.outputs[2],
            6: hue_saturation_value,
            7: group_input.outputs["Main Color"],
        },
        attrs={"data_type": "RGBA"},
    )

    mix_4 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: 1.0000, 6: mix_3.outputs[2], 7: group.outputs["Tile Color"]},
        attrs={"blend_type": "MULTIPLY", "data_type": "RGBA"},
    )

    color = mix_4.outputs[2]
    roughness = nw.build_float_curve(
        color, [(0, uniform(0.3, 0.5)), (1, uniform(0.8, 1.0))]
    )
    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF, input_kwargs={"Base Color": color, "Roughness": roughness}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: mix_2.outputs[2], 1: 0.1000},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group.outputs["Displacement"], 1: multiply}
    )

    multiply_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: add, 1: 0.0100}, attrs={"operation": "MULTIPLY"}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"BSDF": principled_bsdf, "Displacement": multiply_1},
        attrs={"is_active_output": True},
    )


def shader_wood_tiled(
    nw: NodeWrangler, hscale=None, vscale=None, base_color=None, seed=None, **kwargs
):
    # Code generated using version 2.6.4 of the node_transpiler

    if hscale is None:
        hscale = clip_gaussian(6, 4, 3, 9)
    if vscale is None:
        vscale = uniform(0.05, 0.2) * hscale
    if seed is None:
        seed = uniform(-1000.0, 1000.0)
    if base_color is None:
        base_color = get_color()

    group = nw.new_node(
        nodegroup_tiled_wood().name,
        input_kwargs={
            "Tile Horizontal Scale": hscale,
            "Tile Vertical Scale": vscale,
            "Seed": seed,
            "Main Color": base_color,
        },
    )

    displacement = nw.new_node(
        "ShaderNodeDisplacement",
        input_kwargs={"Height": group.outputs["Displacement"], "Midlevel": 0.0000},
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": group.outputs["BSDF"], "Displacement": displacement},
        attrs={"is_active_output": True},
    )


def get_random_light_wood_params():
    color_fac = [0xDEB887, 0xCDAA7D, 0xFFF8DC]
    color_factory = [hex_to_rgb(c) for c in color_fac]
    return color_factory[randint(len(color_fac))]


def apply(obj, selection=None, **kwargs):
    common.apply(obj, shader_wood_tiled, selection=selection, **kwargs)


# def make_sphere():
#     return new_plane()
