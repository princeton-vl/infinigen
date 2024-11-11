# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Stamatis Alexandropoulos
# Acknowledgement: This file draws inspiration from https://www.youtube.com/watch?v=55MMAnTYhWI by Dikko

from numpy.random import uniform

from infinigen.assets.materials import common
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.util.color import color_category


def shader_velvet(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.5 of the node_transpiler

    texture_coordinate = nw.new_node(Nodes.TextureCoord)

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": texture_coordinate.outputs["Object"]}
    )

    mapping = nw.new_node(Nodes.Mapping, input_kwargs={"Vector": reroute})

    voronoi_texture = nw.new_node(
        Nodes.VoronoiTexture, input_kwargs={"Vector": mapping, "Scale": 1.0000}
    )

    mix_6 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: 0.1125, 6: voronoi_texture.outputs["Color"]},
        attrs={"data_type": "RGBA"},
    )

    musgrave_texture = nw.new_node(
        Nodes.MusgraveTexture,
        input_kwargs={
            "Vector": mapping,
            "Scale": 9.6000,
            "Detail": 11.4000,
            "Dimension": 0.1000,
            "Lacunarity": 1.9000,
        },
        attrs={"musgrave_type": "MULTIFRACTAL"},
    )

    mix = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            0: uniform(0, 0.8),
            6: musgrave_texture,
            7: (0.6044, 0.6044, 0.6044, 1.0000),
        },
        attrs={"data_type": "RGBA", "blend_type": "MULTIPLY"},
    )

    mix_1 = nw.new_node(
        Nodes.Mix,
        input_kwargs={6: mix_6.outputs[2], 7: mix.outputs[2]},
        attrs={"data_type": "RGBA"},
    )

    color_ramp = nw.new_node(Nodes.ColorRamp, input_kwargs={"Fac": mix_1.outputs[2]})
    color_ramp.color_ramp.elements[0].position = 0.0000
    color_ramp.color_ramp.elements[0].color = [1.0000, 1.0000, 1.0000, 1.0000]
    color_ramp.color_ramp.elements[1].position = 0.8455
    color_ramp.color_ramp.elements[1].color = [0.0000, 0.0000, 0.0000, 1.0000]

    rgb = nw.new_node(Nodes.RGB)
    rgb.outputs[0].default_value = color_category("textile")
    # (0.3547, 0.3018, 0.3087, 1.0000)

    brightness_contrast = nw.new_node(
        "ShaderNodeBrightContrast", input_kwargs={"Color": rgb, "Bright": 0.0500}
    )

    mix_2 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: color_ramp.outputs["Color"], 6: brightness_contrast, 7: rgb},
        attrs={"data_type": "RGBA"},
    )

    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": mix_2.outputs[2],
            "Specular IOR Level": 0.0000,
            "Roughness": uniform(0.4, 0.9),
            "Anisotropic": 0.7614,
            "Anisotropic Rotation": 1.0000,
            "Sheen Weight": 16.2273,
        },
    )

    mapping_1 = nw.new_node(
        Nodes.Mapping,
        input_kwargs={
            "Vector": reroute,
            "Rotation": (0.0000, 0.0000, 1.0157),
            "Scale": (2.2000, 2.2000, 2.2000),
        },
    )

    wave_texture_1 = nw.new_node(
        Nodes.WaveTexture,
        input_kwargs={
            "Vector": mapping_1,
            "Scale": 500.0000,
            "Distortion": 4.0000,
            "Detail": 6.7000,
            "Detail Scale": 1.5000,
            "Detail Roughness": 0.4308,
        },
        attrs={"bands_direction": "DIAGONAL"},
    )

    mix_3 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: 1.0000, 6: mapping_1, 7: wave_texture_1.outputs["Color"]},
        attrs={"data_type": "RGBA", "blend_type": "MULTIPLY"},
    )

    mix_4 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: 1.0000, 6: color_ramp.outputs["Color"], 7: mix_3.outputs[2]},
        attrs={"data_type": "RGBA", "blend_type": "MULTIPLY"},
    )

    displacement = nw.new_node(
        Nodes.Displacement,
        input_kwargs={"Height": mix_4.outputs[2], "Midlevel": 0.0000, "Scale": 0.0150},
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": principled_bsdf, "Displacement": displacement},
        attrs={"is_active_output": True},
    )


def apply(obj, selection=None, **kwargs):
    common.apply(obj, shader_velvet, selection, **kwargs)
    # surface.add_material(obj, shader_velvet, selection=selection)


# apply(bpy.context.active_object)
