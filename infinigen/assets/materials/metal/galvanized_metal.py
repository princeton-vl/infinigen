# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo
# Acknowledgement: This file draws inspiration https://www.youtube.com/watch?v=ECl2pQ1jQm8 by Ryan King Art

from numpy.random import uniform

from infinigen.assets.materials import common
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler


@node_utils.to_nodegroup(
    "nodegroup_galvanized_metal", singleton=False, type="ShaderNodeTree"
)
def nodegroup_galvanized_metal(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketColor", "Base Color", (0.8000, 0.8000, 0.8000, 1.0000)),
            ("NodeSocketFloat", "Scale", 0.0000),
            ("NodeSocketFloat", "Seed", 0.0000),
        ],
    )

    texture_coordinate = nw.new_node(Nodes.TextureCoord)

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Scale"], 1: 5.0000},
        attrs={"operation": "MULTIPLY"},
    )

    noise_texture = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": texture_coordinate.outputs["Object"],
            "W": group_input.outputs["Seed"],
            "Scale": multiply,
            "Detail": 15.0000,
            "Roughness": 0.4000,
            "Distortion": 0.2000,
        },
        attrs={"noise_dimensions": "4D"},
    )

    mix = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            0: 0.0500,
            6: texture_coordinate.outputs["Object"],
            7: noise_texture.outputs["Color"],
        },
        attrs={"clamp_factor": False, "data_type": "RGBA"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Scale"], 1: 500.0000},
        attrs={"operation": "MULTIPLY"},
    )

    voronoi_texture = nw.new_node(
        Nodes.VoronoiTexture,
        input_kwargs={
            "Vector": mix.outputs[2],
            "W": group_input.outputs["Seed"],
            "Scale": multiply_1,
        },
        attrs={"distance": "MINKOWSKI", "voronoi_dimensions": "4D"},
    )

    map_range = nw.new_node(
        Nodes.MapRange,
        input_kwargs={"Value": voronoi_texture.outputs["Color"], 3: 0.1000, 4: 0.5000},
    )

    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": group_input.outputs["Base Color"],
            "Metallic": 1.0000,
            "Specular IOR Level": 0.0000,
            "Roughness": map_range.outputs["Result"],
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"BSDF": principled_bsdf},
        attrs={"is_active_output": True},
    )


def shader_galvanized_metal(
    nw: NodeWrangler, scale=1.0, base_color=None, seed=None, **kwargs
):
    # Code generated using version 2.6.4 of the node_transpiler
    if seed is None:
        seed = uniform(-1000.0, 1000.0)
    if base_color is None:
        from infinigen.assets.materials.metal import sample_metal_color

        base_color = sample_metal_color(**kwargs)

    group = nw.new_node(
        nodegroup_galvanized_metal().name,
        input_kwargs={"Base Color": base_color, "Scale": scale, "Seed": seed},
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": group},
        attrs={"is_active_output": True},
    )


def apply(obj, selection=None, **kwargs):
    common.apply(obj, shader_galvanized_metal, selection=selection, **kwargs)
