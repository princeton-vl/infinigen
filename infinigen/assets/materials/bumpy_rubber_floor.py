# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo


from numpy.random import uniform

from infinigen.core import surface
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.util.color import hsv2rgba


@node_utils.to_nodegroup("nodegroup_node_group", singleton=False, type="ShaderNodeTree")
def nodegroup_node_group(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    texture_coordinate = nw.new_node(Nodes.TextureCoord)

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketColor", "Base Color", (0.8000, 0.8000, 0.8000, 1.0000)),
            ("NodeSocketFloat", "Scale", 1.0000),
            ("NodeSocketFloat", "Seed", 0.0000),
            ("NodeSocketFloat", "Roughness", 0.4000),
        ],
    )

    mapping = nw.new_node(
        Nodes.Mapping,
        input_kwargs={
            "Vector": texture_coordinate.outputs["Generated"],
            "Scale": group_input.outputs["Scale"],
        },
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": mapping})

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Seed"]}
    )

    noise_texture_9 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": reroute_1,
            "W": reroute,
            "Scale": 18.0000,
            "Detail": 3.0000,
            "Roughness": 0.4500,
        },
        attrs={"noise_dimensions": "4D"},
    )

    map_range_6 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={"Value": noise_texture_9.outputs["Fac"], 3: 0.6000, 4: 1.4000},
    )

    hue_saturation_value = nw.new_node(
        Nodes.HueSaturationValue,
        input_kwargs={
            "Value": map_range_6.outputs["Result"],
            "Color": group_input.outputs["Base Color"],
        },
    )

    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": hue_saturation_value,
            "Specular IOR Level": 0.9,
            "Roughness": group_input.outputs["Roughness"],
        },
    )

    noise_texture = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": mapping,
            "W": group_input.outputs["Seed"],
            "Scale": 2.0000,
            "Detail": 6.0000,
        },
        attrs={"noise_dimensions": "4D"},
    )

    voronoi_texture = nw.new_node(
        Nodes.VoronoiTexture,
        input_kwargs={"Vector": noise_texture.outputs["Fac"], "Randomness": 0.0000},
        attrs={"feature": "DISTANCE_TO_EDGE", "voronoi_dimensions": "4D"},
    )

    map_range = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": voronoi_texture.outputs["Distance"],
            2: 0.0300,
            3: 1.0000,
            4: 0.0000,
        },
    )

    noise_texture_1 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": mapping,
            "W": group_input.outputs["Seed"],
            "Scale": 2.5000,
            "Detail": 6.0000,
        },
        attrs={"noise_dimensions": "4D"},
    )

    map_range_1 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={"Value": noise_texture_1.outputs["Fac"], 1: 0.5500, 2: 0.5700},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: map_range.outputs["Result"], 1: map_range_1.outputs["Result"]},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply, 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    noise_texture_2 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": mapping,
            "W": group_input.outputs["Seed"],
            "Scale": 10.0000,
            "Detail": 15.0000,
            "Distortion": 0.1000,
        },
        attrs={"noise_dimensions": "4D"},
    )

    map_range_2 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={"Value": noise_texture_2.outputs["Fac"], 1: 0.6300, 2: 0.6800},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: map_range_2.outputs["Result"], 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: multiply_1, 1: multiply_2})

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 200.0000

    noise_texture_3 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": mapping,
            "W": group_input.outputs["Seed"],
            "Scale": value,
        },
        attrs={"noise_dimensions": "4D"},
    )

    voronoi_texture_1 = nw.new_node(
        Nodes.VoronoiTexture,
        input_kwargs={
            "Vector": mapping,
            "W": group_input.outputs["Seed"],
            "Scale": value,
        },
        attrs={"voronoi_dimensions": "4D"},
    )

    mix = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            0: 0.4000,
            2: noise_texture_3.outputs["Fac"],
            3: voronoi_texture_1.outputs["Distance"],
        },
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: mix.outputs["Result"], 1: 0.1000},
        attrs={"operation": "MULTIPLY"},
    )

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: add, 1: multiply_3})

    noise_texture_4 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": mapping,
            "W": group_input.outputs["Seed"],
            "Scale": 4.0000,
            "Detail": 1.0000,
            "Roughness": 0.4500,
        },
        attrs={"noise_dimensions": "4D"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: noise_texture_4.outputs["Fac"]},
        attrs={"operation": "SUBTRACT"},
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract, 1: 3.0000},
        attrs={"operation": "MULTIPLY"},
    )

    add_2 = nw.new_node(Nodes.Math, input_kwargs={0: add_1, 1: multiply_4})

    noise_texture_5 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": mapping,
            "W": group_input.outputs["Seed"],
            "Scale": 40.0000,
            "Detail": 15.0000,
            "Distortion": 0.1000,
        },
        attrs={"noise_dimensions": "4D"},
    )

    map_range_3 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": noise_texture_5.outputs["Fac"],
            1: 0.6500,
            2: 0.6400,
            3: 1.0000,
            4: 0.0000,
        },
    )

    noise_texture_7 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": mapping,
            "W": group_input.outputs["Seed"],
            "Scale": 12.0000,
            "Detail": 6.0000,
        },
        attrs={"noise_dimensions": "4D"},
    )

    map_range_4 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={"Value": noise_texture_7.outputs["Fac"], 1: 0.5500, 2: 0.5700},
    )

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: map_range_3.outputs["Result"],
            1: map_range_4.outputs["Result"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply_5}, attrs={"operation": "SUBTRACT"}
    )

    multiply_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_1, 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    add_3 = nw.new_node(Nodes.Math, input_kwargs={0: add_2, 1: multiply_6})

    noise_texture_6 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": reroute_1,
            "W": reroute,
            "Scale": 30.0000,
            "Detail": 3.0000,
            "Roughness": 0.4500,
        },
        attrs={"noise_dimensions": "4D"},
    )

    subtract_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: noise_texture_6.outputs["Fac"]},
        attrs={"operation": "SUBTRACT"},
    )

    multiply_7 = nw.new_node(
        Nodes.Math, input_kwargs={0: subtract_2}, attrs={"operation": "MULTIPLY"}
    )

    add_4 = nw.new_node(Nodes.Math, input_kwargs={0: add_3, 1: multiply_7})

    noise_texture_8 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": reroute_1,
            "W": reroute,
            "Scale": 20.0000,
            "Detail": 3.0000,
            "Roughness": 0.4500,
        },
        attrs={"noise_dimensions": "4D"},
    )

    map_range_5 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": noise_texture_8.outputs["Fac"],
            1: 0.5500,
            2: 0.5100,
            3: -0.5000,
            4: 0.5000,
        },
    )

    multiply_8 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: map_range_5.outputs["Result"], 1: 0.0500},
        attrs={"operation": "MULTIPLY"},
    )

    add_5 = nw.new_node(Nodes.Math, input_kwargs={0: add_4, 1: multiply_8})

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "BSDF": principled_bsdf,
            "Displacement": add_5,
            "tmp_viewer": principled_bsdf,
        },
        attrs={"is_active_output": True},
    )


def shader_bumpy_rubber(nw: NodeWrangler, scale=2.0, base_color=None, seed=None):
    # Code generated using version 2.6.5 of the node_transpiler
    if base_color is None:
        base_color = hsv2rgba(uniform(0, 1), uniform(0.2, 0.5), uniform(0.4, 0.7))
    if seed is None:
        seed = uniform(-1000.0, 1000.0)

    roughness = uniform(0.1, 0.3)

    group = nw.new_node(
        nodegroup_node_group().name,
        input_kwargs={
            "Base Color": base_color,
            "Scale": scale,
            "Seed": seed,
            "Roughness": roughness,
        },
    )

    displacement = nw.new_node(
        Nodes.Displacement,
        input_kwargs={
            "Height": group.outputs["Displacement"],
            "Midlevel": 0.0000,
            "Scale": 0.0010,
        },
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={
            "Surface": group.outputs["tmp_viewer"],
            "Displacement": displacement,
        },
        attrs={"is_active_output": True},
    )


def apply(obj, selection=None, **kwargs):
    surface.add_material(obj, shader_bumpy_rubber, selection=selection)
