# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo
# Acknowledgement: This file draws inspiration https://www.youtube.com/watch?v=In9V4-ih16o by Ryan King Art


from numpy.random import uniform

from infinigen.assets.color_fits import real_color_distribution
from infinigen.assets.materials import common
from infinigen.assets.utils.uv import unwrap_faces
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler


@node_utils.to_nodegroup("nodegroup_leather", singleton=False, type="ShaderNodeTree")
def nodegroup_leather(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    texture_coordinate = nw.new_node(Nodes.TextureCoord)

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Seed", 0.0000),
            ("NodeSocketFloat", "Scale", 0.0000),
            ("NodeSocketColor", "Base Color", (0.0000, 0.0000, 0.0000, 1.0000)),
        ],
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Scale"], 1: 10.0000},
        attrs={"operation": "MULTIPLY"},
    )

    noise_texture = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": texture_coordinate.outputs["Object"],
            "W": group_input.outputs["Seed"],
            "Scale": multiply,
            "Detail": 15.0000,
            "Distortion": 0.2000,
        },
        attrs={"noise_dimensions": "4D"},
    )

    color_ramp = nw.new_node(
        Nodes.ColorRamp, input_kwargs={"Fac": noise_texture.outputs["Fac"]}
    )
    color_ramp.color_ramp.elements[0].position = 0.2841
    color_ramp.color_ramp.elements[0].color = [0.0000, 0.0000, 0.0000, 1.0000]
    color_ramp.color_ramp.elements[1].position = 0.9455
    color_ramp.color_ramp.elements[1].color = [1.0000, 1.0000, 1.0000, 1.0000]

    mix = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            0: 0.0200,
            6: texture_coordinate.outputs["Object"],
            7: noise_texture.outputs["Color"],
        },
        attrs={"blend_type": "LINEAR_LIGHT", "data_type": "RGBA"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Scale"], 1: 800.0000},
        attrs={"operation": "MULTIPLY"},
    )

    voronoi_texture = nw.new_node(
        Nodes.VoronoiTexture,
        input_kwargs={
            "Vector": mix.outputs[2],
            "W": group_input.outputs["Seed"],
            "Scale": multiply_1,
        },
        attrs={"voronoi_dimensions": "4D", "feature": "DISTANCE_TO_EDGE"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: voronoi_texture.outputs["Distance"],
            1: group_input.outputs["Scale"],
        },
        attrs={"use_clamp": True, "operation": "MULTIPLY"},
    )

    hue_saturation_value = nw.new_node(
        "ShaderNodeHueSaturation",
        input_kwargs={"Value": 0.6000, "Color": group_input.outputs["Base Color"]},
    )

    mix_1 = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            0: multiply_2,
            6: group_input.outputs["Base Color"],
            7: hue_saturation_value,
        },
        attrs={"data_type": "RGBA"},
    )

    hue_saturation_value_1 = nw.new_node(
        "ShaderNodeHueSaturation",
        input_kwargs={"Value": 0.4000, "Color": group_input.outputs["Base Color"]},
    )

    mix_2 = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            0: color_ramp.outputs["Color"],
            6: mix_1.outputs[2],
            7: hue_saturation_value_1,
        },
        attrs={"data_type": "RGBA"},
    )

    map_range = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": mix_2.outputs[2],
            3: uniform(0.3, 0.5),
            4: uniform(0.5, 0.7),
        },
    )

    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": mix_2.outputs[2],
            "Roughness": map_range.outputs["Result"],
        },
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: mix_1.outputs[2], 1: -0.2000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: color_ramp.outputs["Color"], 1: 0.0500},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: multiply_3, 1: multiply_4})

    multiply_5 = nw.new_node(
        Nodes.Math, input_kwargs={0: add, 1: 0.0200}, attrs={"operation": "MULTIPLY"}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"BSDF": principled_bsdf, "Displacement": multiply_5},
        attrs={"is_active_output": True},
    )


def shader_leather(nw: NodeWrangler, scale=1.0, base_color=None, seed=None, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler
    if seed is None:
        seed = uniform(-1000.0, 1000.0)

    # if base_color is None:
    #     base_color = color_category('leather')
    base_color = real_color_distribution("sofa_leather")

    group = nw.new_node(
        nodegroup_leather().name,
        input_kwargs={"Seed": seed, "Scale": scale, "Base Color": base_color},
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


def apply(obj, selection=None, **kwargs):
    unwrap_faces(obj, selection)
    common.apply(obj, shader_leather, selection=selection, **kwargs)
