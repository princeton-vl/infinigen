# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Hongyu Wen

from numpy.random import uniform as U

from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.util.color import hsv2rgba


def shader_lamp_bulb_nonemissive(nw: NodeWrangler):
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

    transparent_bsdf = nw.new_node(
        Nodes.TransparentBSDF, input_kwargs={"Color": mix.outputs[2]}
    )

    translucent_bsdf = nw.new_node(
        Nodes.TranslucentBSDF, input_kwargs={"Color": mix.outputs[2]}
    )

    mix_shader = nw.new_node(
        Nodes.MixShader,
        input_kwargs={
            "Fac": light_path.outputs["Is Camera Ray"],
            1: transparent_bsdf,
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

    color = hsv2rgba(U(0.45, 0.55), U(0, 0.1), U(0, 1))
    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF, input_kwargs={"Base Color": color}
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": principled_bsdf},
        attrs={"is_active_output": True},
    )
