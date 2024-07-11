# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo
# Acknowledgement: This file draws inspiration https://www.youtube.com/watch?v=wTzk9T06gdw by Ryan King Arts

from infinigen.core import surface
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler


def shader_marble(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler

    texture_coordinate = nw.new_node(Nodes.TextureCoord)

    scale = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: texture_coordinate.outputs["Object"]},
        attrs={"operation": "SCALE"},
    )

    vector_rotate = nw.new_node(
        Nodes.VectorRotate,
        input_kwargs={"Vector": scale.outputs["Vector"]},
        attrs={"rotation_type": "EULER_XYZ"},
    )

    seed = nw.new_node(Nodes.Value, label="seed")
    seed.outputs[0].default_value = 0.0000

    scale_1 = nw.new_node(Nodes.Value, label="scale")
    scale_1.outputs[0].default_value = 3.0000

    add = nw.new_node(Nodes.Math, input_kwargs={0: scale_1, 1: 1.0000})

    noise_texture_2 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": vector_rotate,
            "W": seed,
            "Scale": add,
            "Detail": 15.0000,
        },
        attrs={"noise_dimensions": "4D"},
    )

    map_range = nw.new_node(
        Nodes.MapRange,
        input_kwargs={"Value": noise_texture_2.outputs["Fac"], 1: 0.4800, 2: 0.6000},
    )

    noise_texture = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": vector_rotate,
            "W": seed,
            "Scale": scale_1,
            "Detail": 15.0000,
        },
        attrs={"noise_dimensions": "4D"},
    )

    noise_texture_3 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": noise_texture.outputs["Fac"],
            "Scale": 8.0000,
            "Detail": 15.0000,
        },
    )

    voronoi_texture = nw.new_node(
        Nodes.VoronoiTexture,
        input_kwargs={
            "Vector": noise_texture_3.outputs["Fac"],
            "W": 1.6400,
            "Scale": 3.0000,
        },
        attrs={"feature": "DISTANCE_TO_EDGE", "voronoi_dimensions": "4D"},
    )

    colorramp_1 = nw.new_node(
        Nodes.ColorRamp, input_kwargs={"Fac": voronoi_texture.outputs["Distance"]}
    )
    colorramp_1.color_ramp.elements[0].position = 0.0000
    colorramp_1.color_ramp.elements[0].color = [1.0000, 1.0000, 1.0000, 1.0000]
    colorramp_1.color_ramp.elements[1].position = 0.0300
    colorramp_1.color_ramp.elements[1].color = [0.0000, 0.0000, 0.0000, 1.0000]

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: map_range.outputs["Result"], 1: colorramp_1.outputs["Color"]},
        attrs={"operation": "MULTIPLY"},
    )

    noise_texture_1 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": noise_texture.outputs["Fac"],
            "W": seed,
            "Scale": 8.0000,
            "Detail": 15.0000,
        },
        attrs={"noise_dimensions": "4D"},
    )

    mix_1 = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            0: 0.8000,
            6: noise_texture.outputs["Fac"],
            7: noise_texture_1.outputs["Fac"],
        },
        attrs={"data_type": "RGBA"},
    )

    colorramp = nw.new_node(Nodes.ColorRamp, input_kwargs={"Fac": mix_1.outputs[2]})
    colorramp.color_ramp.elements[0].position = 0.3000
    colorramp.color_ramp.elements[0].color = [1.0000, 1.0000, 1.0000, 1.0000]
    colorramp.color_ramp.elements[1].position = 0.9000
    colorramp.color_ramp.elements[1].color = [0.0000, 0.0000, 0.0000, 1.0000]

    mix = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            0: multiply,
            6: colorramp.outputs["Color"],
            7: (0.0376, 0.0179, 0.0033, 1.0000),
        },
        attrs={"data_type": "RGBA"},
    )

    bump = nw.new_node(
        "ShaderNodeBump", input_kwargs={"Strength": 0.0200, "Height": multiply}
    )

    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": mix_1.outputs[2],
            "Specular": 0.6000,
            "Roughness": 0.1000,
            "Normal": bump,
        },
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": principled_bsdf},
        attrs={"is_active_output": True},
    )


class Marble:
    shader = shader_marble

    def generate(self):
        return surface.shaderfunc_to_material(shader_marble)

    __call__ = generate
