# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Meenal Parakh
# Acknowledgement: This file draws inspiration from following sources:

# https://www.youtube.com/watch?v=DfoMWLQ-BkM by 5 Minutes Blender
# https://www.youtube.com/watch?v=tS_U3twxKKg by PIXXO 3D
# https://www.youtube.com/watch?v=OCay8AsVD84 by Antonio Palladino
# https://www.youtube.com/watch?v=5dS3N90wPkc by Dr Blender
# https://www.youtube.com/watch?v=12c1J6LhK4Y by blenderian
# https://www.youtube.com/watch?v=kVvOk_7PoUE by Blender Box
# https://www.youtube.com/watch?v=WTK7E443l1E by blenderbitesize
# https://www.youtube.com/watch?v=umrARvXC_MI by Ryan King Art


from numpy.random import uniform

from infinigen.assets.materials import common
from infinigen.assets.utils.uv import unwrap_faces
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler


def get_texture_params():
    return {
        "_color": uniform(0.2, 1.0, 3),
        "_roughness": uniform(0, 1.0),
        "_thread_density_x": uniform(100, 300),
        "_relative_density_y": uniform(0.75, 1.33),
    }


def shader_material(
    nw: NodeWrangler,
    _color=[0.2, 0.2, 0.2],
    _roughness=0.4,
    _thread_density_x=200,
    _relative_density_y=1.0,
    _map="UV",
):
    red = nw.new_node(Nodes.Value, label="red")
    red.outputs[0].default_value = _color[0]

    green = nw.new_node(Nodes.Value, label="green")
    green.outputs[0].default_value = _color[1]

    blue = nw.new_node(Nodes.Value, label="blue")
    blue.outputs[0].default_value = _color[2]

    roughness = nw.new_node(Nodes.Value, label="roughness")
    roughness.outputs[0].default_value = _roughness

    thread_density_x = nw.new_node(Nodes.Value, label="thread_density_x")
    thread_density_x.outputs[0].default_value = _thread_density_x

    relative_density_y = nw.new_node(Nodes.Value, label="relative_density_y")
    relative_density_y.outputs[0].default_value = _relative_density_y

    combine_color = nw.new_node(
        Nodes.CombineColor, input_kwargs={"Red": red, "Green": green, "Blue": blue}
    )
    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={"Base Color": combine_color, "Roughness": roughness},
    )

    texture_coordinate = nw.new_node(Nodes.TextureCoord)

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": texture_coordinate.outputs[_map]}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: thread_density_x, 1: relative_density_y},
        attrs={"operation": "MULTIPLY"},
    )

    wave_texture_1 = nw.new_node(
        Nodes.WaveTexture,
        input_kwargs={
            "Vector": reroute,
            "Scale": multiply,
            "Distortion": 5.0000,
            "Detail": 6.1000,
        },
        attrs={"bands_direction": "Y"},
    )

    principled_bsdf_1 = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": wave_texture_1.outputs["Color"],
            "Subsurface Color": (0.0000, 0.0000, 0.0000, 1.0000),
            "Roughness": roughness,
        },
    )

    mix_shader = nw.new_node(
        Nodes.MixShader,
        input_kwargs={"Fac": 0.1333, 1: principled_bsdf, 2: principled_bsdf_1},
    )

    wave_texture = nw.new_node(
        Nodes.WaveTexture,
        input_kwargs={
            "Vector": reroute,
            "Scale": thread_density_x,
            "Distortion": 3.8000,
            "Detail": 6.1000,
        },
    )

    color_ramp = nw.new_node(
        Nodes.ColorRamp, input_kwargs={"Fac": wave_texture.outputs["Color"]}
    )
    color_ramp.color_ramp.elements[0].position = 0.8109
    color_ramp.color_ramp.elements[0].color = [0.0000, 0.0000, 0.0000, 1.0000]
    color_ramp.color_ramp.elements[1].position = 1.0000
    color_ramp.color_ramp.elements[1].color = [1.0000, 1.0000, 1.0000, 1.0000]

    invert_color = nw.new_node(
        Nodes.Invert, input_kwargs={"Fac": 0.8400, "Color": color_ramp.outputs["Color"]}
    )

    color_ramp_1 = nw.new_node(
        Nodes.ColorRamp, input_kwargs={"Fac": wave_texture_1.outputs["Color"]}
    )
    color_ramp_1.color_ramp.elements[0].position = 0.0727
    color_ramp_1.color_ramp.elements[0].color = [0.0000, 0.0000, 0.0000, 1.0000]
    color_ramp_1.color_ramp.elements[1].position = 0.8655
    color_ramp_1.color_ramp.elements[1].color = [1.0000, 1.0000, 1.0000, 1.0000]

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: invert_color, 1: color_ramp_1.outputs["Color"]}
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": mix_shader, "Displacement": add},
        attrs={"is_active_output": True},
    )


def shader_fine_knit_fabric(nw: NodeWrangler, **kwargs):
    fabric_params = get_texture_params()
    fabric_params["_map"] = "Object"
    return shader_material(nw, **fabric_params)


def apply(obj, selection=None, **kwargs):
    unwrap_faces(obj, selection)
    common.apply(obj, shader_fine_knit_fabric, selection, **kwargs)
