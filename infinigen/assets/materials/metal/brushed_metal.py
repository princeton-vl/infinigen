# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo
# Acknowledgement: This file draws inspiration https://www.youtube.com/watch?v=QcAMYRgR03k by blenderian


from numpy.random import uniform

from infinigen.core import surface
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler


@node_utils.to_nodegroup(
    "nodegroup_brushed_metal", singleton=False, type="ShaderNodeTree"
)
def nodegroup_brushed_metal(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    texture_coordinate = nw.new_node(Nodes.TextureCoord)

    mapping_1 = nw.new_node(
        Nodes.Mapping,
        input_kwargs={
            "Vector": texture_coordinate.outputs["Object"],
            "Scale": (0.2000, 0.2000, 5.0000),
        },
    )

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketColor", "Base Color", (0.8000, 0.8000, 0.8000, 1.0000)),
            ("NodeSocketFloat", "Scale", 0.0000),
            ("NodeSocketFloat", "Seed", 0.0000),
        ],
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Scale"], 1: 100.0000},
        attrs={"operation": "MULTIPLY"},
    )

    noise_texture_2 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": mapping_1,
            "W": group_input.outputs["Seed"],
            "Scale": multiply,
            "Detail": 15.0000,
            "Roughness": 0.4000,
            "Distortion": 0.1000,
        },
        attrs={"noise_dimensions": "4D"},
    )

    mapping = nw.new_node(
        Nodes.Mapping,
        input_kwargs={
            "Vector": texture_coordinate.outputs["Object"],
            "Scale": (1.0000, 1.0000, 20.0000),
        },
    )

    noise_texture_1 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": texture_coordinate.outputs["Object"],
            "W": group_input.outputs["Seed"],
            "Scale": 0.1000,
            "Detail": 15.0000,
            "Roughness": 0.0000,
        },
        attrs={"noise_dimensions": "4D"},
    )

    mix = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: 0.2000, 6: mapping, 7: noise_texture_1.outputs["Color"]},
        attrs={"data_type": "RGBA"},
    )

    noise_texture = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": mix.outputs[2],
            "W": group_input.outputs["Seed"],
            "Scale": multiply,
            "Detail": 15.0000,
            "Roughness": 0.6000,
            "Distortion": 0.1000,
        },
        attrs={"noise_dimensions": "4D"},
    )

    mix_1 = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            0: 1.0000,
            6: noise_texture_2.outputs["Fac"],
            7: noise_texture.outputs["Fac"],
        },
        attrs={"blend_type": "DARKEN", "data_type": "RGBA"},
    )

    map_range = nw.new_node(
        Nodes.MapRange,
        input_kwargs={"Value": mix_1, 1: 0.4000, 2: 0.6000, 3: 0.8000, 4: 1.2000},
    )

    hue_saturation_value = nw.new_node(
        "ShaderNodeHueSaturation",
        input_kwargs={
            "Value": map_range.outputs["Result"],
            "Color": group_input.outputs["Base Color"],
        },
    )

    map_range_1 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={"Value": mix_1, 1: 0.4000, 2: 0.6000, 3: 0.2000, 4: 0.3000},
    )

    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": hue_saturation_value,
            "Metallic": 1.0000,
            "Specular IOR Level": 0.0000,
            "Roughness": map_range_1.outputs["Result"],
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"BSDF": principled_bsdf, "tmp_viewer": principled_bsdf},
        attrs={"is_active_output": True},
    )


def shader_brushed_metal(
    nw: NodeWrangler, scale=1.0, base_color=None, seed=None, **kwargs
):
    # Code generated using version 2.6.4 of the node_transpiler
    if seed is None:
        seed = uniform(-1000.0, 1000.0)
    if base_color is None:
        from infinigen.assets.materials.metal import sample_metal_color

        base_color = sample_metal_color(**kwargs)

    group = nw.new_node(
        nodegroup_brushed_metal().name,
        input_kwargs={"Base Color": base_color, "Scale": scale, "Seed": seed},
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": group.outputs["BSDF"]},
        attrs={"is_active_output": True},
    )


def apply(obj, selection=None, **kwargs):
    surface.add_material(
        obj, shader_brushed_metal, selection=selection, input_kwargs=kwargs
    )
