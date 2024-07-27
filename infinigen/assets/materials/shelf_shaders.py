# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Beining Han
# Acknowledgement: This file draws inspiration from https://www.youtube.com/watch?v=jDEijCwz6to by Lachlan Sarv


import numpy as np
from numpy.random import normal, uniform

from infinigen.assets.materials import (
    metal_shader_list,
    shader_glass,
    shader_rough_plastic,
    wood,
)
from infinigen.core import surface
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.util.color import hsv2rgba


def shader_shelves_white(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler
    rgb = kwargs.get("rgb", [0.9, 0.9, 0.9])
    base_color = (*rgb, 1.0)
    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": base_color,
            "Roughness": kwargs.get("roughness", 0.9),
        },
    )
    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": principled_bsdf},
        attrs={"is_active_output": True},
    )


def shader_shelves_white_sampler():
    params = dict()
    v = uniform(0.7, 1.0)
    base_color = [
        v * (1.0 + normal(0, 0.005)),
        v * (1.0 + normal(0, 0.005)),
        v * (1.0 + normal(0, 0.005)),
    ]
    params["rgb"] = base_color
    params["roughness"] = uniform(0.7, 1.0)
    return params


def shader_shelves_black_metallic(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler

    color = (*kwargs.get("rgb", [0.0, 0.0, 0.0]), 1.0)
    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={"Base Color": color, "Metallic": kwargs.get("metallic", 0.65)},
    )
    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": principled_bsdf},
        attrs={"is_active_output": True},
    )


def shader_shelves_black_metallic_sampler():
    params = dict()
    base_color = [uniform(0, 0.01), uniform(0, 0.01), uniform(0, 0.01)]
    params["rgb"] = base_color
    params["metallic"] = uniform(0.45, 0.75)
    return params


def shader_shelves_white_metallic(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler

    rgb = kwargs.get("rgb", [0.9, 0.9, 0.9])
    base_color = (*rgb, 1.0)
    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": base_color,
            "Metallic": kwargs.get("metallic", 0.65),
        },
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": principled_bsdf},
        attrs={"is_active_output": True},
    )


def shader_shelves_white_metallic_sampler():
    params = dict()
    v = uniform(0.7, 1.0)
    base_color = [
        v * (1.0 + normal(0, 0.005)),
        v * (1.0 + normal(0, 0.005)),
        v * (1.0 + normal(0, 0.005)),
    ]
    params["rgb"] = base_color
    params["metallic"] = uniform(0.45, 0.75)
    return params


def shader_shelves_black_wood(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.5 of the node_transpiler

    texture_coordinate_1 = nw.new_node(Nodes.TextureCoord)
    wave_scale = kwargs.get("wave_scale", 2.0)
    if kwargs.get("z_axis_texture", False):
        wave_scale = (wave_scale, wave_scale, 0.1)
    else:
        wave_scale = (wave_scale, 0.1, 0.1)

    mapping_1 = nw.new_node(
        Nodes.Mapping,
        input_kwargs={
            "Vector": texture_coordinate_1.outputs["Object"],
            "Scale": (0.1, 0.1, 2.0)
            if kwargs.get("z_axis_texture", False)
            else (0.1, 2.0, 2.0),
        },
    )

    noise_texture_1 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": mapping_1,
            "Scale": 100.0000,
            "Detail": 10.0000,
            "Distortion": 2.0000,
        },
    )

    voronoi_texture = nw.new_node(
        Nodes.VoronoiTexture,
        input_kwargs={"Vector": noise_texture_1.outputs["Fac"], "Scale": 40.0000},
    )

    colorramp_1 = nw.new_node(
        Nodes.ColorRamp, input_kwargs={"Fac": voronoi_texture.outputs["Color"]}
    )
    colorramp_1.color_ramp.elements[0].position = 0.0864
    colorramp_1.color_ramp.elements[0].color = [0.0000, 0.0000, 0.0000, 1.0000]
    colorramp_1.color_ramp.elements[1].position = 0.1091
    colorramp_1.color_ramp.elements[1].color = [1.0000, 1.0000, 1.0000, 1.0000]

    texture_coordinate = nw.new_node(Nodes.TextureCoord)

    mapping = nw.new_node(
        Nodes.Mapping,
        input_kwargs={
            "Vector": texture_coordinate.outputs["Object"],
            "Scale": wave_scale,
        },
    )

    noise_texture = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": mapping,
            "Scale": 3.0000,
            "Detail": 15.0000,
            "Distortion": 2.0000,
        },
    )

    musgrave_texture = nw.new_node(
        Nodes.MusgraveTexture,
        input_kwargs={
            "Vector": noise_texture.outputs["Fac"],
            "Scale": 20.0000,
            "Detail": 3.0000,
        },
    )

    mix_1 = nw.new_node(
        Nodes.Mix,
        input_kwargs={6: musgrave_texture, 7: noise_texture.outputs["Color"]},
        attrs={"data_type": "RGBA"},
    )

    colorramp_2 = nw.new_node(Nodes.ColorRamp, input_kwargs={"Fac": mix_1.outputs[2]})
    colorramp_2.color_ramp.elements[0].position = 0.0818
    colorramp_2.color_ramp.elements[0].color = [0.0000, 0.0000, 0.0000, 1.0000]
    colorramp_2.color_ramp.elements[1].position = 0.8500
    colorramp_2.color_ramp.elements[1].color = [1.0000, 1.0000, 1.0000, 1.0000]

    mix_2 = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            0: 0.6000,
            6: colorramp_1.outputs["Color"],
            7: colorramp_2.outputs["Color"],
        },
        attrs={"data_type": "RGBA"},
    )

    dark_scale = kwargs.get("dark_scale", 0.005)
    gray_scale = kwargs.get("gray_scale", 0.02)
    color_scale = [*kwargs.get("rgb", [0.02, 0.002, 0.002]), 1.0]
    colorramp = nw.new_node(Nodes.ColorRamp, input_kwargs={"Fac": mix_2})
    colorramp.color_ramp.elements.new(0)
    colorramp.color_ramp.elements[0].position = 0.15
    colorramp.color_ramp.elements[0].color = [
        dark_scale,
        dark_scale,
        dark_scale,
        1.0000,
    ]
    colorramp.color_ramp.elements[1].position = 0.5
    colorramp.color_ramp.elements[1].color = [
        gray_scale,
        gray_scale,
        gray_scale,
        1.0000,
    ]
    colorramp.color_ramp.elements[2].position = 1.0000
    colorramp.color_ramp.elements[2].color = color_scale

    mix_3 = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            0: 0.0040,
            6: colorramp_1.outputs["Color"],
            7: colorramp_2.outputs["Color"],
        },
        attrs={"data_type": "RGBA"},
    )

    bump = nw.new_node(
        Nodes.Bump, input_kwargs={"Strength": 0.5000, "Height": mix_3.outputs[2]}
    )

    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": colorramp.outputs["Color"],
            "Roughness": kwargs.get("roughness", 0.9),
            "Normal": bump,
        },
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": principled_bsdf},
        attrs={"is_active_output": True},
    )


def shader_shelves_black_wood_sampler():
    params = dict()
    params["wave_scale"] = uniform(1.0, 3.0)
    params["dark_scale"] = uniform(0.0, 0.01)
    params["gray_scale"] = uniform(0.01, 0.03)
    params["rgb"] = [uniform(0.015, 0.035), uniform(0.0, 0.01), uniform(0.0, 0.01)]
    params["roughness"] = uniform(0.75, 1.0)
    return params


def shader_shelves_wood(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.5 of the node_transpiler

    texture_coordinate_1 = nw.new_node(Nodes.TextureCoord)
    wave_scale = kwargs.get("wave_scale", 2.0)
    if kwargs.get("z_axis_texture", False):
        wave_scale = (wave_scale, wave_scale, 0.1)
    else:
        wave_scale = (wave_scale, 0.1, 0.1)

    mapping_1 = nw.new_node(
        Nodes.Mapping,
        input_kwargs={
            "Vector": texture_coordinate_1.outputs["Object"],
            "Scale": (0.1, 0.1, 2.0)
            if kwargs.get("z_axis_texture", False)
            else (0.1, 2.0, 2.0),
        },
    )

    noise_texture_1 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": mapping_1,
            "Scale": 100.0000,
            "Detail": 10.0000,
            "Distortion": 2.0000,
        },
    )

    voronoi_texture = nw.new_node(
        Nodes.VoronoiTexture,
        input_kwargs={"Vector": noise_texture_1.outputs["Fac"], "Scale": 40.0000},
    )

    colorramp_1 = nw.new_node(
        Nodes.ColorRamp, input_kwargs={"Fac": voronoi_texture.outputs["Color"]}
    )
    colorramp_1.color_ramp.elements[0].position = 0.0864
    colorramp_1.color_ramp.elements[0].color = [0.0000, 0.0000, 0.0000, 1.0000]
    colorramp_1.color_ramp.elements[1].position = 0.1091
    colorramp_1.color_ramp.elements[1].color = [1.0000, 1.0000, 1.0000, 1.0000]

    texture_coordinate = nw.new_node(Nodes.TextureCoord)

    mapping = nw.new_node(
        Nodes.Mapping,
        input_kwargs={
            "Vector": texture_coordinate.outputs["Object"],
            "Scale": wave_scale,
        },
    )

    noise_texture = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": mapping,
            "Scale": 3.0000,
            "Detail": 15.0000,
            "Distortion": 2.0000,
        },
    )

    musgrave_texture = nw.new_node(
        Nodes.MusgraveTexture,
        input_kwargs={
            "Vector": noise_texture.outputs["Fac"],
            "Scale": 20.0000,
            "Detail": 3.0000,
        },
    )

    mix_1 = nw.new_node(
        Nodes.Mix,
        input_kwargs={6: musgrave_texture, 7: noise_texture.outputs["Color"]},
        attrs={"data_type": "RGBA"},
    )

    colorramp_2 = nw.new_node(Nodes.ColorRamp, input_kwargs={"Fac": mix_1.outputs[2]})
    colorramp_2.color_ramp.elements[0].position = 0.0818
    colorramp_2.color_ramp.elements[0].color = [0.0000, 0.0000, 0.0000, 1.0000]
    colorramp_2.color_ramp.elements[1].position = 0.8500
    colorramp_2.color_ramp.elements[1].color = [1.0000, 1.0000, 1.0000, 1.0000]

    mix_2 = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            0: 0.6000,
            6: colorramp_1.outputs["Color"],
            7: colorramp_2.outputs["Color"],
        },
        attrs={"data_type": "RGBA"},
    )

    bright_hsv = kwargs.get("bright_hsv", [0.068, 0.665, 0.805])
    mid_hsv = kwargs.get("mid_hsv", [0.042, 0.853, 0.447])
    dark_hsv = kwargs.get("dark_hsv", [0.043, 0.882, 0.183])

    colorramp = nw.new_node(Nodes.ColorRamp, input_kwargs={"Fac": mix_2})
    colorramp.color_ramp.elements.new(0)
    colorramp.color_ramp.elements[0].position = 0.02
    colorramp.color_ramp.elements[0].color = hsv2rgba(dark_hsv)
    colorramp.color_ramp.elements[1].position = 0.11
    colorramp.color_ramp.elements[1].color = hsv2rgba(mid_hsv)
    colorramp.color_ramp.elements[2].position = 0.8
    colorramp.color_ramp.elements[2].color = hsv2rgba(bright_hsv)

    mix_3 = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            0: 0.0040,
            6: colorramp_1.outputs["Color"],
            7: colorramp_2.outputs["Color"],
        },
        attrs={"data_type": "RGBA"},
    )

    bump = nw.new_node(
        Nodes.Bump, input_kwargs={"Strength": 0.5000, "Height": mix_3.outputs[2]}
    )

    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": colorramp.outputs["Color"],
            "Roughness": kwargs.get("roughness", 0.9),
            "Normal": bump,
        },
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": principled_bsdf},
        attrs={"is_active_output": True},
    )


def shader_shelves_wood_sampler():
    params = dict()
    params["bright_hsv"] = [uniform(0.03, 0.09), uniform(0.5, 0.7), uniform(0.7, 1.0)]
    params["mid_hsv"] = [uniform(0.02, 0.06), uniform(0.6, 1.0), uniform(0.3, 0.6)]
    params["dark_hsv"] = [uniform(0.03, 0.05), uniform(0.6, 1.0), uniform(0.1, 0.3)]
    params["wave_scale"] = uniform(1.0, 3.0)
    params["roughness"] = uniform(0.75, 1.0)
    return params


def get_shelf_material(name, **kwargs):
    match name:
        case "white":
            shader_func = np.random.choice(
                [shader_shelves_white, shader_rough_plastic], p=[0.6, 0.4]
            )
        case "black_wood":
            shader_func = np.random.choice(
                [shader_shelves_black_wood, wood.shader_wood], p=[0.6, 0.4]
            )
        case "wood":
            shader_func = np.random.choice(
                [shader_shelves_wood, wood.shader_wood], p=[0.6, 0.4]
            )

        case "glass":
            shader_func = shader_glass
        case _:
            shader_func = np.random.choice(
                [
                    shader_shelves_white,
                    shader_rough_plastic,
                    shader_shelves_black_wood,
                    wood.shader_wood,
                    shader_shelves_wood,
                ],
                p=[0.3, 0.2, 0.3, 0.1, 0.1],
            )
    r = uniform()
    if name == "metal":
        shader_func = np.random.choice(metal_shader_list)
    else:
        shader_func = np.random.choice(
            [
                shader_shelves_white,
                shader_rough_plastic,
                shader_shelves_black_wood,
                wood.shader_wood,
                shader_shelves_wood,
            ],
            p=[0.3, 0.2, 0.3, 0.1, 0.1],
        )
    # elif r < .3:
    #     shader_func = rg(fabric_shader_list)
    return surface.shaderfunc_to_material(shader_func, **kwargs)
