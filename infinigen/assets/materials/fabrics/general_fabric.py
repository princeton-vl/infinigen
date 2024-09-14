# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo
# Acknowledgement: This file draws inspiration https://www.youtube.com/watch?v=umrARvXC_MI by Ryan King Art


import numpy as np
from numpy.random import uniform

from infinigen.assets.materials import common
from infinigen.assets.utils.uv import unwrap_faces
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.util.color import color_category


def func_fabric(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler

    texture_coordinate = nw.new_node(Nodes.TextureCoord)

    group_input = {
        "Weave Scale": 0.0,
        "Color Pattern Scale": 0.0,
        "Color1": (0.7991, 0.1046, 0.1195, 1.0000),
        "Color2": (1.0000, 0.5271, 0.5711, 1.0000),
    }
    group_input.update(kwargs)

    wave_texture_1 = nw.new_node(
        Nodes.WaveTexture,
        input_kwargs={
            "Vector": texture_coordinate.outputs["UV"],
            "Scale": group_input["Weave Scale"],
            "Distortion": 7.0000,
            "Detail": 15.0000,
        },
        attrs={"bands_direction": "Y"},
    )

    value_2 = nw.new_node(Nodes.Value)
    value_2.outputs[0].default_value = 0.1000

    map_range = nw.new_node(
        Nodes.MapRange,
        input_kwargs={"Value": wave_texture_1.outputs["Color"], 1: value_2},
    )

    wave_texture = nw.new_node(
        Nodes.WaveTexture,
        input_kwargs={
            "Vector": texture_coordinate.outputs["UV"],
            "Scale": group_input["Weave Scale"],
            "Distortion": 7.0000,
            "Detail": 15.0000,
        },
    )

    map_range_1 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={"Value": wave_texture.outputs["Color"], 1: value_2},
    )

    mix = nw.new_node(
        Nodes.Mix,
        input_kwargs={6: map_range.outputs["Result"], 7: map_range_1.outputs["Result"]},
        attrs={"data_type": "RGBA"},
    )

    greater_than = nw.new_node(
        Nodes.Math,
        input_kwargs={0: mix.outputs[2], 1: 0.1000},
        attrs={"operation": "GREATER_THAN"},
    )

    transparent_bsdf = nw.new_node(Nodes.TransparentBSDF)

    less_than = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input["Color Pattern Scale"], 1: 0.0001},
        attrs={"operation": "LESS_THAN"},
    )

    brick_texture_2 = nw.new_node(
        Nodes.BrickTexture,
        input_kwargs={
            "Vector": texture_coordinate.outputs["UV"],
            "Color1": group_input["Color1"],
            "Mortar": group_input["Color2"],
            "Scale": group_input["Color Pattern Scale"],
            "Mortar Size": 0.0000,
            "Bias": -1.0000,
            "Row Height": 0.5000,
        },
        attrs={"offset_frequency": 1, "squash": 0.0000},
    )

    vector_rotate = nw.new_node(
        Nodes.VectorRotate,
        input_kwargs={
            "Vector": texture_coordinate.outputs["UV"],
            "Rotation": (0.0000, 0.0000, 1.5708),
        },
        attrs={"rotation_type": "EULER_XYZ"},
    )

    brick_texture = nw.new_node(
        Nodes.BrickTexture,
        input_kwargs={
            "Vector": vector_rotate,
            "Color1": group_input["Color1"],
            "Mortar": group_input["Color2"],
            "Scale": group_input["Color Pattern Scale"],
            "Mortar Size": 0.0000,
            "Bias": -1.0000,
            "Row Height": 0.5000,
        },
        attrs={"offset_frequency": 1, "squash": 0.0000},
    )

    mix_2 = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            0: 1.0000,
            6: brick_texture_2.outputs["Color"],
            7: brick_texture.outputs["Color"],
        },
        attrs={"data_type": "RGBA", "blend_type": "ADD"},
    )

    mix_4 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: less_than, 6: mix_2.outputs[2], 7: group_input["Color1"]},
        attrs={"data_type": "RGBA"},
    )

    mix_3 = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            0: mix.outputs[2],
            6: (0.0000, 0.0000, 0.0000, 1.0000),
            7: mix_4.outputs[2],
        },
        attrs={"data_type": "RGBA"},
    )

    map_range_2 = nw.new_node(
        Nodes.MapRange, input_kwargs={"Value": mix.outputs[2], 3: 1.0000, 4: 0.9000}
    )

    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": mix_3.outputs[2],
            "Roughness": map_range_2.outputs["Result"],
            "Sheen Weight": 1.0000,
            "Sheen Tint": 1.0000,
        },
    )

    mix_shader = nw.new_node(
        Nodes.MixShader,
        input_kwargs={"Fac": greater_than, 1: transparent_bsdf, 2: principled_bsdf},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input["Weave Scale"], 1: 5.0000},
        attrs={"operation": "MULTIPLY"},
    )

    musgrave_texture = nw.new_node(
        Nodes.MusgraveTexture, input_kwargs={"Scale": multiply}
    )

    mix_1 = nw.new_node(
        Nodes.Mix,
        input_kwargs={6: musgrave_texture, 7: mix.outputs[2]},
        attrs={"data_type": "RGBA"},
    )

    subtract = nw.new_node(
        Nodes.Math, input_kwargs={0: mix_1.outputs[2]}, attrs={"operation": "SUBTRACT"}
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract, 1: 0.0010},
        attrs={"operation": "MULTIPLY"},
    )

    displacement = nw.new_node(
        "ShaderNodeDisplacement",
        input_kwargs={"Height": multiply_1, "Midlevel": 0.0000},
    )

    return {"Shader": mix_shader, "Displacement": displacement}


def shader_fabric(
    nw: NodeWrangler,
    weave_scale=500.0,
    color_scale=None,
    color_1=None,
    color_2=None,
    **kwargs,
):
    # Code generated using version 2.6.4 of the node_transpiler

    if color_scale is None:
        color_scale = np.random.choice([0.0, uniform(5.0, 20.0)])
    if color_1 is None:
        color_1 = color_category("fabric")
    if color_2 is None:
        color_2 = color_category("white")

    group = func_fabric(
        nw,
        **{
            "Weave Scale": weave_scale,
            "Color Pattern Scale": color_scale,
            "Color1": color_1,
            "Color2": color_2,
        },
    )

    displacement = nw.new_node(
        "ShaderNodeDisplacement",
        input_kwargs={"Height": group["Displacement"], "Midlevel": 0.0000},
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": group["Shader"], "Displacement": displacement},
        attrs={"is_active_output": True},
    )


def apply(obj, selection=None, **kwargs):
    if not isinstance(obj, list):
        obj = [obj]
    for o in obj:
        unwrap_faces(o, selection)
    common.apply(obj, shader_fabric, selection, **kwargs)
