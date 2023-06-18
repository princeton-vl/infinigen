# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Ankit Goyal, Mingzhe Wang, Zeyu Ma
# Date Signed: June 5 2023

import gin

from numpy.random import uniform

from nodes import node_utils
from nodes.node_wrangler import Nodes
from surfaces import surface
from terrain.utils import SurfaceTypes
from util.random import random_color_neighbour
from util.math import FixedSeed

type = SurfaceTypes.SDFPerturb
mod_name = "geo_mud"
name = "mud"

def shader_mud(nw):
    nw.force_input_consistency()
    position = nw.new_node('ShaderNodeNewGeometry')
    mud_frac, mud_noise = geo_mud(nw, geometry=False)
    # colors for mud
    mud_col_1 = random_color_neighbour((0.0772, 0.0446, 0.0219, 1.0), 0.05,
                                       0.05, 0.05)
    mud_col_2 = random_color_neighbour((0.1136, 0.0722, 0.043, 1.0), 0.05,
                                       0.05, 0.05)
    mud_col_3 = random_color_neighbour((0.15, 0.0997, 0.0642, 1.0), 0.05, 0.05,
                                       0.05)
    # colors for water
    wat_col_1 = random_color_neighbour(mud_col_1, 0.02, 0.02, 0.1, only_more_val=True)
    wat_col_2 = random_color_neighbour(mud_col_2, 0.02, 0.02, 0.1, only_more_val=True)
    # mud wetness
    mud_wet = uniform(0.1, 0.8)

    colorramp_7 = nw.new_node(
        Nodes.ColorRamp, input_kwargs={"Fac": mud_frac}
    )
    colorramp_7.color_ramp.elements[0].position = 0.0
    colorramp_7.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_7.color_ramp.elements[1].position = 0.1
    colorramp_7.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    value_7 = nw.new_node(Nodes.Value)
    value_7.outputs[0].default_value = 0.25

    value_9 = nw.new_node(Nodes.Value)
    value_9.outputs[0].default_value = 13.0

    math_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: value_7, 1: value_9},
        attrs={"operation": "MULTIPLY"},
    )

    noise_texture_5 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": position,
            "Scale": math_7,
            "Detail": 16.0,
            "Roughness": 0.7,
            "Distortion": 1.0,
            "W": uniform(0, 10),
        },
        attrs={"noise_dimensions": "4D"},
    )

    colorramp_8 = nw.new_node(
        Nodes.ColorRamp, input_kwargs={"Fac": noise_texture_5.outputs["Fac"]}
    )
    colorramp_8.color_ramp.elements[0].position = 0.0
    colorramp_8.color_ramp.elements[0].color = wat_col_1
    colorramp_8.color_ramp.elements[1].position = 1.0
    colorramp_8.color_ramp.elements[1].color = wat_col_2

    water_roughness = 0.1
    principled_bsdf_3 = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": colorramp_8.outputs["Color"],
            "Specular": 0.8,
            "Roughness": water_roughness,
        },
    )

    colorramp_6 = nw.new_node(
        Nodes.ColorRamp, input_kwargs={"Fac": mud_noise}
    )
    colorramp_6.color_ramp.elements.new(1)
    colorramp_6.color_ramp.elements[0].position = 0.0
    colorramp_6.color_ramp.elements[0].color = mud_col_1
    colorramp_6.color_ramp.elements[1].position = 0.5
    colorramp_6.color_ramp.elements[1].color = mud_col_2
    colorramp_6.color_ramp.elements[2].position = 1.0
    colorramp_6.color_ramp.elements[2].color = mud_col_3

    colorramp_5 = nw.new_node(
        Nodes.ColorRamp, input_kwargs={"Fac": mud_noise}
    )
    colorramp_5.color_ramp.elements[0].position = 0.0
    colorramp_5.color_ramp.elements[0].color = (
        water_roughness,
        water_roughness,
        water_roughness,
        1.0,
    )
    mud_roughness = water_roughness + mud_wet
    colorramp_5.color_ramp.elements[1].position = 1.0
    colorramp_5.color_ramp.elements[1].color = (
        mud_roughness,
        mud_roughness,
        mud_roughness,
        1.0,
    )

    principled_bsdf_4 = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": colorramp_6.outputs["Color"],
            "Specular": 0.2,
            "Roughness": colorramp_5.outputs["Color"],
        },
    )

    mix_shader_1 = nw.new_node(
        Nodes.MixShader,
        input_kwargs={
            "Fac": colorramp_7.outputs["Color"],
            1: principled_bsdf_3,
            2: principled_bsdf_4,
        },
    )

    return mix_shader_1

@gin.configurable
def geo_mud(nw, selection=None, flat_only=False, random_seed=0, geometry=True):
    nw.force_input_consistency()
    if nw.node_group.type == "SHADER":
        position = nw.new_node('ShaderNodeNewGeometry')
        normal = (nw.new_node('ShaderNodeNewGeometry'), 1)
    else:
        position = nw.new_node(Nodes.InputPosition)
        normal = nw.new_node(Nodes.InputNormal)
    with FixedSeed(random_seed):

        # large bump scale
        lar_bum_sca = nw.new_value(uniform(0.5, 2), "lar_bum_sca")
        # small bump scale
        sma_bum_sca = nw.new_value(uniform(2, 3), "sma_bum_sca")
        # depth of puddles and bumpiness
        dep_pud = nw.new_value(0.1 * (uniform(0.75, 1) if uniform() < 0.5 else uniform(1, 1.75)), "dep_pud")
        # per water
        per_water = uniform(-0.1, 0.1)

        if flat_only:
            vector_math_4 = nw.new_node(
                Nodes.VectorMath,
                input_kwargs={0: normal, 1: (0.0, 0.0, 1.0)},
                attrs={"operation": "DOT_PRODUCT"},
            )
            colorramp_4 = nw.new_node(
                Nodes.ColorRamp, input_kwargs={"Fac": vector_math_4.outputs["Value"]}
            )
        else:
            colorramp_4 = nw.new_node(
                Nodes.ColorRamp, input_kwargs={"Fac": 1}
            )
        
        colorramp_4.color_ramp.elements.new(1)
        colorramp_4.color_ramp.elements[0].position = 0.95
        colorramp_4.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
        colorramp_4.color_ramp.elements[1].position = 0.99 #
        colorramp_4.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
        colorramp_4.color_ramp.elements[2].position = 1.0
        colorramp_4.color_ramp.elements[2].color = (1.0, 1.0, 1.0, 1.0)

        noise_texture = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={
                "Vector": position,
                "Scale": 1.0,
                "W": nw.new_value(uniform(0, 10), "noise_texture_w"),
                "Detail": 9,
                "Roughness": 0.4,
            },
            attrs={"noise_dimensions": "4D"},
        )

        colorramp_5 = nw.new_node(
            Nodes.ColorRamp, input_kwargs={"Fac": noise_texture.outputs["Fac"]}, label="color_ramp_VAR"
        )
        colorramp_5.color_ramp.elements[0].position = 0.4 + per_water
        colorramp_5.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
        colorramp_5.color_ramp.elements[1].position = 0.6 + per_water
        colorramp_5.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

        mix_1 = nw.new_node(
            Nodes.MixRGB,
            input_kwargs={
                "Fac": colorramp_4.outputs["Color"],
                "Color1": (1.0, 1.0, 1.0, 1.0),
                "Color2": colorramp_5.outputs["Color"],
            },
        )

        value_1 = nw.new_node(Nodes.Value)
        value_1.outputs[0].default_value = 0.0

        vector_math_11 = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={0: value_1, 1: normal},
            attrs={"operation": "MULTIPLY"},
        )

        voronoi_texture = nw.new_node(
            Nodes.VoronoiTexture,
            input_kwargs={"Scale": sma_bum_sca, "Randomness": 10.0, "W": nw.new_value(uniform(0, 10), "voronoi_texture_w")},
            attrs={"voronoi_dimensions": "4D", "feature": "SMOOTH_F1"},
        )

        voronoi_texture_1 = nw.new_node(
            Nodes.VoronoiTexture,
            input_kwargs={"Scale": lar_bum_sca, "W": nw.new_value(uniform(0, 10), "voronoi_texture_1_w")},
            attrs={"voronoi_dimensions": "4D", "feature": "SMOOTH_F1"},
        )

        value_3 = nw.new_node(Nodes.Value)
        value_3.outputs[0].default_value = 2.0

        vector_math_9 = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={0: voronoi_texture_1.outputs["Distance"], 1: value_3},
            attrs={"operation": "MULTIPLY"},
        )

        vector_math_7 = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={
                0: voronoi_texture.outputs["Distance"],
                1: vector_math_9.outputs["Vector"],
            },
        )

        noise_texture_1 = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={"W": nw.new_value(uniform(0, 10), "noise_texture_1_w")},
            attrs={"noise_dimensions": "4D"},
        )

        value_5 = nw.new_node(Nodes.Value)
        value_5.outputs[0].default_value = 0.5

        vector_math_10 = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={0: noise_texture_1.outputs["Fac"], 1: value_5},
            attrs={"operation": "MULTIPLY"},
        )

        noise_texture_3 = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={"Scale": 10.0, "Detail": 16.0, "W": nw.new_value(uniform(0, 10), "noise_texture_3_w")},
            attrs={"noise_dimensions": "4D"},
        )

        value_6 = nw.new_node(Nodes.Value)
        value_6.outputs[0].default_value = 0.4

        vector_math_12 = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={0: noise_texture_3.outputs["Fac"], 1: value_6},
            attrs={"operation": "MULTIPLY"},
        )

        vector_math_6 = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={
                0: vector_math_10.outputs["Vector"],
                1: vector_math_12.outputs["Vector"],
            },
        )

        vector_math_3 = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={
                0: vector_math_7.outputs["Vector"],
                1: vector_math_6.outputs["Vector"],
            },
        )

        value = nw.new_node(Nodes.Value)
        value.outputs[0].default_value = 3.9

        vector_math = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={0: vector_math_3.outputs["Vector"], 1: value},
            attrs={"operation": "DIVIDE"},
        )

        float_curve = nw.new_node(
            Nodes.FloatCurve, input_kwargs={"Value": vector_math.outputs["Vector"]}
        )
        node_utils.assign_curve(
            float_curve.mapping.curves[0],
            [(0.0, 0.0), (0.2955, 0.175), (0.5045, 0.5063), (0.6818, 0.8313), (1.0, 1.0)],
        )

        vector_math_2 = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={0: float_curve, 1: dep_pud},
            attrs={"operation": "MULTIPLY"},
        )

        vector_math_1 = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={0: vector_math_2.outputs["Vector"], 1: normal},
            attrs={"operation": "MULTIPLY"},
        )

        offset = nw.new_node(
            Nodes.MixRGB,
            input_kwargs={
                "Fac": mix_1,
                "Color1": vector_math_11.outputs["Vector"],
                "Color2": vector_math_1.outputs["Vector"],
            },
        )
        mud_frac = mix_1
        mud_noise = noise_texture_1

    if geometry:  
        groupinput = nw.new_node(Nodes.GroupInput)
        if selection is not None:
            offset = nw.multiply(offset, surface.eval_argument(nw, selection))
        set_position = nw.new_node(Nodes.SetPosition, input_kwargs={"Geometry": groupinput,  "Offset": offset})
        nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': set_position})
    else:
        return mud_frac, mud_noise


def apply(obj, selection=None, **kwargs):
    surface.add_geomod(obj, geo_mud, selection=selection)
    surface.add_material(obj, shader_mud, selection=selection)