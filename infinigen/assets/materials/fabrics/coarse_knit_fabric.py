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


from numpy.random import choice, uniform

from infinigen.assets.materials import common
from infinigen.assets.utils.uv import unwrap_faces
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler


def get_texture_params():
    return {
        "_pattern_mixer": choice([uniform(0.0, 0.75), uniform(0.75, 1.0)]),
        "_pattern_density": choice([uniform(0.1, 1.0), uniform(1.0, 10.0)]),
        "_color": uniform(0.0, 1.0, 3),
        "_brick_knit": choice(
            [uniform(0.0, 0.05), uniform(0.05, 0.95), uniform(0.95, 1.0)]
        ),
        "_knit_resolution": uniform(0.5, 3.0),
        "_brick_resolution": uniform(10.0, 30.0),
        "_crease_resolution": uniform(10.0, 80.0),
        "_smoothness": choice([uniform(0.0, 0.2), uniform(0.2, 1.0)]),
        "_color_shader_frac": uniform(0.1, 0.9),
    }


def shader_fabric_base(
    nw: NodeWrangler,
    _pattern_mixer=1.0,
    _pattern_density=0.15,
    _color=[5, 10.0, 10.0],
    _brick_knit=0.0,
    _knit_resolution=3.0,
    _brick_resolution=40,
    _crease_resolution=200,
    _smoothness=0.7,
    _color_shader_frac=0.01,
):
    # Code generated using version 2.6.5 of the node_transpiler

    pattern_mixer = nw.new_node(Nodes.Value)
    pattern_mixer.outputs[0].default_value = _pattern_mixer

    pattern_density = nw.new_node(Nodes.Value)
    pattern_density.outputs[0].default_value = _pattern_density

    color_r = nw.new_node(Nodes.Value)
    color_r.outputs[0].default_value = _color[0]

    color_g = nw.new_node(Nodes.Value)
    color_g.outputs[0].default_value = _color[1]

    color_b = nw.new_node(Nodes.Value)
    color_b.outputs[0].default_value = _color[2]

    brick_knit = nw.new_node(Nodes.Value)
    brick_knit.outputs[0].default_value = _brick_knit

    knit_resolution = nw.new_node(Nodes.Value)
    knit_resolution.outputs[0].default_value = _knit_resolution

    brick_resolution = nw.new_node(Nodes.Value)
    brick_resolution.outputs[0].default_value = _brick_resolution

    crease_resolution = nw.new_node(Nodes.Value)
    crease_resolution.outputs[0].default_value = _crease_resolution

    smoothness = nw.new_node(Nodes.Value)
    smoothness.outputs[0].default_value = _smoothness

    color_shader_frac = nw.new_node(Nodes.Value)
    color_shader_frac.outputs[0].default_value = _color_shader_frac

    texture_coordinate = nw.new_node(Nodes.TextureCoord)

    mapping_1 = nw.new_node(
        Nodes.Mapping,
        input_kwargs={
            "Vector": texture_coordinate.outputs["Object"],
            "Scale": (3.2000, 1.0000, 1.0000),
        },
    )

    brick_texture = nw.new_node(
        Nodes.BrickTexture,
        input_kwargs={"Vector": mapping_1, "Scale": brick_resolution},
    )

    color_ramp_1 = nw.new_node(
        Nodes.ColorRamp, input_kwargs={"Fac": brick_texture.outputs["Color"]}
    )
    color_ramp_1.color_ramp.elements[0].position = 0.0000
    color_ramp_1.color_ramp.elements[0].color = [1.0000, 1.0000, 1.0000, 1.0000]
    color_ramp_1.color_ramp.elements[1].position = 1.0000
    color_ramp_1.color_ramp.elements[1].color = [0.0000, 0.0000, 0.0000, 1.0000]

    mapping = nw.new_node(
        Nodes.Mapping,
        input_kwargs={
            "Vector": texture_coordinate.outputs["Object"],
            "Rotation": (0.0000, 0.0000, 0.7854),
            "Scale": (238.8000, 1.0000, 35.6000),
        },
    )

    voronoi_texture = nw.new_node(
        Nodes.VoronoiTexture,
        input_kwargs={
            "Vector": mapping,
            "Scale": pattern_density,
            "Randomness": 0.0000,
        },
        attrs={"feature": "F2"},
    )

    color_ramp = nw.new_node(
        Nodes.ColorRamp, input_kwargs={"Fac": voronoi_texture.outputs["Distance"]}
    )
    color_ramp.color_ramp.elements[0].position = 0.1018
    color_ramp.color_ramp.elements[0].color = [1.0000, 1.0000, 1.0000, 1.0000]
    color_ramp.color_ramp.elements[1].position = 1.0000
    color_ramp.color_ramp.elements[1].color = [0.0000, 0.0000, 0.0000, 1.0000]

    mix = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            0: pattern_mixer,
            6: color_ramp_1.outputs["Color"],
            7: color_ramp.outputs["Color"],
        },
        attrs={"clamp_result": True, "data_type": "RGBA"},
    )

    reroute = nw.new_node(Nodes.Reroute, input_kwargs={"Input": mix.outputs[2]})

    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": reroute,
            "Specular IOR Level": 0.6309,
            "Roughness": 0.9945,
        },
    )

    combine_color = nw.new_node(
        Nodes.CombineColor,
        input_kwargs={"Red": color_r, "Green": color_g, "Blue": color_b},
    )

    principled_bsdf_1 = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": combine_color,
            "Specular IOR Level": 0.6309,
            "Roughness": 0.9945,
        },
    )

    mix_shader = nw.new_node(
        Nodes.MixShader,
        input_kwargs={
            "Fac": color_shader_frac,
            1: principled_bsdf,
            2: principled_bsdf_1,
        },
    )

    # bump_1 = nw.new_node(Nodes.Bump, input_kwargs={'Height': color_ramp_1.outputs["Color"]})

    scale = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: color_ramp_1.outputs["Color"], "Scale": brick_knit},
        attrs={"operation": "SCALE"},
    )

    mapping_2 = nw.new_node(
        Nodes.Mapping,
        input_kwargs={
            "Vector": texture_coordinate.outputs["Object"],
            "Rotation": (0.0000, 0.0000, 0.6196),
            "Scale": (217.5000, 176.2000, 42.0000),
        },
    )

    voronoi_texture_1 = nw.new_node(
        Nodes.VoronoiTexture,
        input_kwargs={
            "Vector": mapping_2,
            "Scale": knit_resolution,
            "Randomness": 0.0000,
        },
        attrs={"feature": "F2"},
    )

    mapping_3 = nw.new_node(
        Nodes.Mapping, input_kwargs={"Vector": texture_coordinate.outputs["Object"]}
    )

    noise_texture = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={"Vector": mapping_3, "Scale": crease_resolution},
    )

    add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: noise_texture.outputs["Fac"], 1: smoothness},
        attrs={"use_clamp": True},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: voronoi_texture_1.outputs["Distance"], 1: add},
        attrs={"use_clamp": True, "operation": "MULTIPLY"},
    )

    # bump = nw.new_node(Nodes.Bump, input_kwargs={'Height': multiply})

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 1.0000, 1: brick_knit},
        attrs={"operation": "SUBTRACT"},
    )

    scale_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: multiply, "Scale": subtract},
        attrs={"operation": "SCALE"},
    )

    # scale_1 = nw.new_node(Nodes.VectorMath, input_kwargs={0: bump, 'Scale': subtract}, attrs={'operation': 'SCALE'})

    add_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: scale.outputs["Vector"], 1: scale_1.outputs["Vector"]},
    )

    vector_displacement = nw.new_node(
        "ShaderNodeVectorDisplacement", input_kwargs={"Vector": add_1.outputs["Vector"]}
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": mix_shader, "Displacement": vector_displacement},
        attrs={"is_active_output": True},
    )


def shader_coarse_knit_fabric(nw: NodeWrangler, **kwargs):
    fabric_params = get_texture_params()
    return shader_fabric_base(nw, **fabric_params)


def apply(obj, selection=None, **kwargs):
    unwrap_faces(obj, selection)
    common.apply(obj, shader_coarse_knit_fabric, selection, **kwargs)
