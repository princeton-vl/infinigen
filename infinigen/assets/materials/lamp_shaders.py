# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Hongyu Wen

from numpy.random import uniform as U

from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler


def shader_metal(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    anisotropic_bsdf = nw.new_node(
        "ShaderNodeBsdfAnisotropic",
        input_kwargs={"Color": (0.3224, 0.3224, 0.3224, 1.0000), "Roughness": 0.1000},
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": anisotropic_bsdf},
        attrs={"is_active_output": True},
    )


def shader_lampshade(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    light_path = nw.new_node(Nodes.LightPath)

    object_info = nw.new_node(Nodes.ObjectInfo_Shader)

    white_noise_texture = nw.new_node(
        Nodes.WhiteNoiseTexture,
        input_kwargs={"Vector": object_info.outputs["Random"]},
        attrs={"noise_dimensions": "4D"},
    )

    mix = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            0: 0.9000,
            6: white_noise_texture.outputs["Color"],
            7: (0.5000, 0.4444, 0.3669, 1.0000),
        },
        attrs={"data_type": "RGBA"},
    )

    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": mix.outputs[2],
            "Subsurface Weight": U(0.03, 0.08),
            "Subsurface Radius": (0.1000, 0.1000, 0.1000),
            "Roughness": U(0.5, 0.8),
            "IOR": 4.0000,
            "Transmission Weight": U(0.05, 0.2),
        },
    )

    translucent_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": mix.outputs[2],
            "Roughness": 0.7,
        },
    )

    mix_shader = nw.new_node(
        Nodes.MixShader,
        input_kwargs={
            "Fac": light_path.outputs["Is Camera Ray"],
            1: principled_bsdf,
            2: translucent_bsdf,
        },
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": mix_shader},
        attrs={"is_active_output": True},
    )


def shader_black(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={"Base Color": (0.0039, 0.0039, 0.0039, 1.0000)},
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": principled_bsdf},
        attrs={"is_active_output": True},
    )
