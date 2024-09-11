# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Hongyu Wen

from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler


def shader_glass_001(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    glass_bsdf = nw.new_node(Nodes.GlassBSDF, input_kwargs={"IOR": 1.5000})

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": glass_bsdf},
        attrs={"is_active_output": True},
    )


def shader_black_medal_001(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    anisotropic_bsdf = nw.new_node(
        "ShaderNodeBsdfAnisotropic",
        input_kwargs={"Color": (0.0167, 0.0167, 0.0167, 1.0000)},
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": anisotropic_bsdf},
        attrs={"is_active_output": True},
    )


def shader_white_metal_001(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    texture_coordinate = nw.new_node(Nodes.TextureCoord)

    mapping = nw.new_node(
        Nodes.Mapping,
        input_kwargs={
            "Vector": texture_coordinate.outputs["Object"],
            "Scale": (1.0000, 1.0000, 50.0000),
        },
    )

    noise_texture_1 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": mapping,
            "Scale": 20.0000,
            "Detail": 20.0000,
            "Distortion": 1.0000,
        },
    )

    colorramp = nw.new_node(
        Nodes.ColorRamp, input_kwargs={"Fac": noise_texture_1.outputs["Color"]}
    )
    colorramp.color_ramp.elements[0].position = 0.2500
    colorramp.color_ramp.elements[0].color = [0.5244, 0.5244, 0.5244, 1.0000]
    colorramp.color_ramp.elements[1].position = 1.0000
    colorramp.color_ramp.elements[1].color = [0.9698, 0.9698, 0.9698, 1.0000]

    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": colorramp.outputs["Color"],
            "Subsurface Color": (1.0000, 1.0000, 1.0000, 1.0000),
            "Metallic": 1.0000,
            "Specular IOR Level": 1.0000,
            "Roughness": 0.1000,
            "Anisotropic": 0.9182,
            "Sheen Weight": 0.0455,
            "Sheen Tint": (0, 0, 0, 0.4948),
        },
        attrs={"subsurface_method": "BURLEY"},
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": principled_bsdf},
        attrs={"is_active_output": True},
    )
