# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Meenal Parakh


from numpy.random import uniform

from infinigen.assets.materials import common
from infinigen.assets.utils.uv import unwrap_faces
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler


def get_texture_params():
    return {
        "_hue": uniform(0, 1.0),
        "_saturation": uniform(0.5, 1.0),
        "_line_density": uniform(5, 20.0),
    }


def shader_lined_fur_base(
    nw: NodeWrangler, _hue=0.3, _saturation=0.7, _line_density=10
):
    # Code generated using version 2.6.5 of the node_transpiler

    hue = nw.new_node(Nodes.Value)
    hue.outputs[0].default_value = _hue

    saturation = nw.new_node(Nodes.Value)
    saturation.outputs[0].default_value = _saturation

    line_density = nw.new_node(Nodes.Value)
    line_density.outputs[0].default_value = _line_density

    combine_color = nw.new_node(
        Nodes.CombineColor,
        input_kwargs={"Red": hue, "Green": saturation, "Blue": 0.6000},
        attrs={"mode": "HSV"},
    )

    texture_coordinate = nw.new_node(Nodes.TextureCoord)

    mapping = nw.new_node(
        Nodes.Mapping, input_kwargs={"Vector": texture_coordinate.outputs["Object"]}
    )

    wave_texture = nw.new_node(
        Nodes.WaveTexture,
        input_kwargs={
            "Vector": mapping,
            "Scale": line_density,
            "Distortion": 1.0000,
            "Detail": 1.0000,
        },
    )

    color_ramp_1 = nw.new_node(
        Nodes.ColorRamp, input_kwargs={"Fac": wave_texture.outputs["Color"]}
    )
    color_ramp_1.color_ramp.elements[0].position = 0.0073
    color_ramp_1.color_ramp.elements[0].color = [0.0000, 0.0000, 0.0000, 1.0000]
    color_ramp_1.color_ramp.elements[1].position = 0.2255
    color_ramp_1.color_ramp.elements[1].color = [1.0000, 1.0000, 1.0000, 1.0000]

    mapping_1 = nw.new_node(
        Nodes.Mapping,
        input_kwargs={
            "Vector": texture_coordinate.outputs["Object"],
            "Scale": (1.0000, 1.0000, 87.4000),
        },
    )

    noise_texture = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": mapping_1,
            "Scale": 2.7000,
            "Detail": 7.3000,
            "Distortion": 7.0000,
        },
    )

    color_ramp = nw.new_node(
        Nodes.ColorRamp, input_kwargs={"Fac": noise_texture.outputs["Fac"]}
    )
    color_ramp.color_ramp.elements[0].position = 0.3018
    color_ramp.color_ramp.elements[0].color = [1.0000, 1.0000, 1.0000, 1.0000]
    color_ramp.color_ramp.elements[1].position = 0.4691
    color_ramp.color_ramp.elements[1].color = [0.0000, 0.0000, 0.0000, 1.0000]

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: color_ramp_1.outputs["Color"], 1: color_ramp.outputs["Color"]},
        attrs={"operation": "MULTIPLY", "use_clamp": True},
    )

    scale = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_color, "Scale": multiply},
        attrs={"operation": "SCALE"},
    )

    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={"Base Color": scale.outputs["Vector"], "Roughness": 1.0000},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 1.0000, 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract, 1: -0.3000},
        attrs={"operation": "MULTIPLY"},
    )

    mapping_2 = nw.new_node(
        Nodes.Mapping, input_kwargs={"Vector": texture_coordinate.outputs["Object"]}
    )

    noise_texture_1 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": mapping_2,
            "Scale": 42.3000,
            "Detail": 7.3000,
            "Distortion": 16.6000,
        },
    )

    color_ramp_2 = nw.new_node(
        Nodes.ColorRamp, input_kwargs={"Fac": noise_texture_1.outputs["Fac"]}
    )
    color_ramp_2.color_ramp.elements[0].position = 0.3018
    color_ramp_2.color_ramp.elements[0].color = [1.0000, 1.0000, 1.0000, 1.0000]
    color_ramp_2.color_ramp.elements[1].position = 0.4691
    color_ramp_2.color_ramp.elements[1].color = [0.0000, 0.0000, 0.0000, 1.0000]

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: color_ramp_2.outputs["Color"], 1: multiply},
        attrs={"operation": "MULTIPLY", "use_clamp": True},
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: multiply_1, 1: multiply_2})

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": principled_bsdf, "Displacement": add},
        attrs={"is_active_output": True},
    )


def shader_fabric_random(nw: NodeWrangler, **kwargs):
    fabric_params = get_texture_params()
    return shader_lined_fur_base(nw, **fabric_params)


def apply(obj, selection=None, **kwargs):
    unwrap_faces(obj, selection)
    common.apply(obj, shader_fabric_random, selection, **kwargs)
