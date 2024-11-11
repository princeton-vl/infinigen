# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo
# Acknowledgement: This file draws inspiration https://www.youtube.com/watch?v=jDEijCwz6to by Lachlan Sarv

import numpy as np
from numpy.random import uniform

from infinigen.assets.materials import common
from infinigen.assets.utils.object import new_cube
from infinigen.core.nodes import Nodes, NodeWrangler
from infinigen.core.util.color import hsv2rgba, rgb2hsv
from infinigen.core.util.random import log_uniform


def get_color():
    from infinigen.assets.materials.bark_random import get_random_bark_params

    _, color_params = get_random_bark_params(np.random.randint(1e7))
    h, s, v = rgb2hsv(color_params["Color"][:-1])
    return hsv2rgba(
        h + uniform(-0.0, 0.05), s + uniform(-0.3, 0.2), v * log_uniform(0.2, 20)
    )


def shader_wood(nw: NodeWrangler, color=None, w=None, vertical=False, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler

    vec = nw.new_node(Nodes.TextureCoord).outputs["Object"]
    if vertical:
        vec = nw.new_node(
            Nodes.Mapping,
            [vec],
            input_kwargs={"Rotation": (np.pi / 2, 0, np.pi / 2 * np.random.randint(2))},
        )

    mapping_2 = nw.new_node(
        Nodes.Mapping,
        input_kwargs={"Vector": vec, "Scale": (5.0000, 100.0000, 100.0000)},
    )

    if color is None:
        color = get_color()
    if w is None:
        w = uniform(0, 1)
    musgrave_texture_2 = nw.new_node(
        Nodes.MusgraveTexture,
        input_kwargs={
            "Vector": mapping_2,
            "W": w,
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

    mapping_1 = nw.new_node(Nodes.Mapping, input_kwargs={"Vector": vec})

    noise_texture_1 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": mapping_1,
            "W": w,
            "Scale": 0.5000,
            "Detail": 1.0000,
            "Distortion": 1.1000,
        },
        attrs={"noise_dimensions": "4D"},
    )

    musgrave_texture_1 = nw.new_node(
        Nodes.MusgraveTexture,
        input_kwargs={
            "W": w,
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
        Nodes.Mapping, input_kwargs={"Vector": vec, "Scale": (0.1500, 1.0000, 0.1500)}
    )

    noise_texture = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": mapping,
            "W": w,
            "Detail": 5.0000,
            "Distortion": 1.0000,
        },
        attrs={"noise_dimensions": "4D"},
    )

    musgrave_texture = nw.new_node(
        Nodes.MusgraveTexture,
        input_kwargs={
            "Vector": noise_texture.outputs["Fac"],
            "W": w,
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
            "Color": color,
        },
    )

    mix_3 = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            0: mix_2.outputs[2],
            6: hue_saturation_value,
            7: color,
        },
        attrs={"data_type": "RGBA"},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: mix_2.outputs[2], 1: log_uniform(0.0002, 0.01)},
        attrs={"operation": "MULTIPLY"},
    )

    displacement = nw.new_node(
        "ShaderNodeDisplacement", input_kwargs={"Height": multiply, "Midlevel": 0.0000}
    )

    color = mix_3.outputs[2]
    roughness = uniform(0.0, 0.4)
    roughness = nw.build_float_curve(
        nw.new_node(Nodes.NoiseTexture, input_kwargs={"Scale": log_uniform(40, 50)}),
        [(0, roughness), (1, roughness + uniform(0.0, 0.8))],
    )
    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": color,
            "Roughness": roughness,
            "Coat Weight": np.clip(uniform(0, 1.4), 0, 1),
        },
    )
    nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": principled_bsdf, "Displacement": displacement},
    )


def apply(obj, selection=None, **kwargs):
    # TODO HACK - avoiding circular imports for now
    from infinigen.assets.materials.shelf_shaders import (
        shader_shelves_black_wood,
        shader_shelves_white,
        shader_shelves_wood,
    )

    r = uniform()
    if r < 1 / 12:
        shader = shader_shelves_white
    elif r < 2 / 12:
        shader = shader_shelves_wood
    elif r < 3 / 12:
        shader = shader_shelves_black_wood
    else:
        shader = shader_wood
    common.apply(obj, shader, selection, **kwargs)


def make_sphere():
    return new_cube()
