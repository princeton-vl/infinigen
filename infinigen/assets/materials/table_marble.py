# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo
# Acknowledgement: This file draws inspiration https://www.youtube.com/watch?v=wTzk9T06gdw by Ryan King Arts


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
            "Vector": noise_texture.outputs["Color"],
            "Scale": 8.0000,
            "Detail": 15.0000,
        },
    )

    voronoi_texture = nw.new_node(
        Nodes.VoronoiTexture,
        input_kwargs={
            "Vector": noise_texture_3.outputs["Color"],
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
            "Vector": noise_texture.outputs["Color"],
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
            "Specular IOR Level": 0.6000,
            "Roughness": 0.1000,
            "Normal": bump,
        },
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": principled_bsdf},
        attrs={"is_active_output": True},
    )


def shader_wood(nw: NodeWrangler, **kwargs):
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

    mapping_2 = nw.new_node(
        Nodes.Mapping,
        input_kwargs={"Vector": vector_rotate, "Scale": (5.0000, 100.0000, 100.0000)},
    )

    seed = nw.new_node(Nodes.Value, label="seed")
    seed.outputs[0].default_value = 0.0000

    musgrave_texture_2 = nw.new_node(
        Nodes.MusgraveTexture,
        input_kwargs={
            "Vector": mapping_2,
            "W": seed,
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

    mapping_1 = nw.new_node(Nodes.Mapping, input_kwargs={"Vector": vector_rotate})

    noise_texture_1 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": mapping_1,
            "W": seed,
            "Scale": 0.5000,
            "Detail": 1.0000,
            "Distortion": 1.1000,
        },
        attrs={"noise_dimensions": "4D"},
    )

    musgrave_texture_1 = nw.new_node(
        Nodes.MusgraveTexture,
        input_kwargs={
            "W": seed,
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
        Nodes.Mapping,
        input_kwargs={"Vector": vector_rotate, "Scale": (0.1500, 1.0000, 0.1500)},
    )

    noise_texture = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": mapping,
            "W": seed,
            "Detail": 5.0000,
            "Distortion": 1.0000,
        },
        attrs={"noise_dimensions": "4D"},
    )

    musgrave_texture = nw.new_node(
        Nodes.MusgraveTexture,
        input_kwargs={
            "Vector": noise_texture.outputs["Fac"],
            "W": seed,
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
        attrs={"data_type": "RGBA", "blend_type": "MULTIPLY"},
    )

    mix_2 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: 0.9500, 6: map_range_2.outputs["Result"], 7: mix_1.outputs[2]},
        attrs={"data_type": "RGBA", "blend_type": "MULTIPLY"},
    )

    rgb = nw.new_node(Nodes.RGB)
    rgb.outputs[0].default_value = (0.0242, 0.0056, 0.0027, 1.0000)

    rgb_1 = nw.new_node(Nodes.RGB)
    rgb_1.outputs[0].default_value = (0.5089, 0.2122, 0.0685, 1.0000)

    mix_3 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: mix_2.outputs[2], 6: rgb, 7: rgb_1},
        attrs={"data_type": "RGBA"},
    )

    bump = nw.new_node(
        "ShaderNodeBump", input_kwargs={"Strength": 0.2000, "Height": mix_2.outputs[2]}
    )

    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={"Base Color": mix_3.outputs[2], "Normal": bump},
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": principled_bsdf},
        attrs={"is_active_output": True},
    )
