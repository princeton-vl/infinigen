# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Ankit Goyal, Zeyu Ma
# Date Signed: June 15 2023

import gin
from nodes.node_wrangler import Nodes
from numpy.random import uniform
from surfaces import surface
from terrain.utils import SurfaceTypes
from util.math import FixedSeed
from util.random import random_color_neighbour

type = SurfaceTypes.SDFPerturb
mod_name = "geo_cracked_ground"
name = "cracked_ground"

def shader_cracked_ground(nw):
    nw.force_input_consistency()
    crackedground_base_color, crackedground_roughness = geo_cracked_ground(nw, geometry=False)

    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": crackedground_base_color,
            "Roughness": crackedground_roughness,
        },
    )

    return principled_bsdf

@gin.configurable
def geo_cracked_ground(nw, selection=None, random_seed=0, geometry=True):
    nw.force_input_consistency()
    if nw.node_group.type == "SHADER":
        position = nw.new_node('ShaderNodeNewGeometry')
        normal = (nw.new_node('ShaderNodeNewGeometry'), 1)
    else:
        position = nw.new_node(Nodes.InputPosition)
        normal = nw.new_node(Nodes.InputNormal)

    with FixedSeed(random_seed):
        # scale of cracks; smaller means larger cracks
        sca_crac = nw.new_value(uniform(1, 3), "sca_crac")
        # percentage of area with crac, 0 means in half of area
        per_crac = uniform(-0.25, 0.1)
        # color crack
        col_crac = random_color_neighbour((0.2016, 0.107, 0.0685, 1.0), 0.1, 0.1, 0.1)
        col_crac = tuple([0.5 * x for x in col_crac[0:3]] + [1])
        col_crac_node = nw.new_node(Nodes.ColorRamp, label="col_crac")
        col_crac_node.color_ramp.elements.remove(col_crac_node.color_ramp.elements[1])
        col_crac_node.color_ramp.elements[0].color = col_crac

        # width of the crack
        wid_crac = uniform(0.01, 0.04)
        # colors for ground
        col_1 = random_color_neighbour((0.3005, 0.1119, 0.0284, 1.0), 0.1, 0.1, 0.1)
        col_2 = random_color_neighbour((0.6038, 0.4397, 0.2159, 1.0), 0.1, 0.1, 0.1)
        # scale of the grains, smaller means larger grains
        sca_gra = nw.new_value(uniform(20, 300), "sca_gra")

        noise_texture = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={
                "Vector": position,
                "Scale": 6.0,
                "Detail": 16.0,
                "W": nw.new_value(uniform(0, 10), "noise_texture_w"),
            },
            attrs={"noise_dimensions": "4D"},
        )

        colorramp = nw.new_node(
            Nodes.ColorRamp, input_kwargs={"Fac": noise_texture.outputs["Fac"]}, label="colorramp_VAR"
        )
        colorramp.color_ramp.elements.new(1)
        colorramp.color_ramp.elements[0].position = 0.0
        colorramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
        colorramp.color_ramp.elements[1].position = 0.459
        colorramp.color_ramp.elements[1].color = col_1
        colorramp.color_ramp.elements[2].position = 1.0
        colorramp.color_ramp.elements[2].color = col_2

        vector_math = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={0: colorramp.outputs["Color"], 1: normal},
            attrs={"operation": "MULTIPLY"},
        )

        value = nw.new_node(Nodes.Value)
        value.outputs["Value"].default_value = 0.51

        vector_math_1 = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={0: vector_math.outputs["Vector"], 1: value},
            attrs={"operation": "MULTIPLY"},
        )

        voronoi_texture = nw.new_node(
            Nodes.VoronoiTexture,
            input_kwargs={"Vector": position, "Scale": 50.0, "W": nw.new_value(uniform(0, 10), "voronoi_texture_w")},
            attrs={"feature": "SMOOTH_F1", "voronoi_dimensions": "4D"},
        )

        colorramp_2 = nw.new_node(
            Nodes.ColorRamp, input_kwargs={"Fac": voronoi_texture.outputs["Distance"]}
        )
        colorramp_2.color_ramp.elements[0].position = 0.114
        colorramp_2.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
        colorramp_2.color_ramp.elements[1].position = 0.336
        colorramp_2.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

        vector_math_2 = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={0: colorramp_2.outputs["Color"], 1: normal},
            attrs={"operation": "MULTIPLY"},
        )

        value_1 = nw.new_node(Nodes.Value)
        value_1.outputs["Value"].default_value = -0.31

        vector_math_3 = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={0: vector_math_2.outputs["Vector"], 1: value_1},
            attrs={"operation": "MULTIPLY"},
        )

        vector_math_10 = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={
                0: vector_math_1.outputs["Vector"],
                1: vector_math_3.outputs["Vector"],
            },
        )

        noise_texture_1 = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={
                "Vector": position,
                "Scale": 3.6,
                "Detail": 16.0,
                "Roughness": 0.48,
                "W": nw.new_value(uniform(0, 10), "noise_texture_1_w"),
            },
            attrs={"noise_dimensions": "4D"},
        )

        voronoi_texture_2 = nw.new_node(
            Nodes.VoronoiTexture,
            input_kwargs={"Vector": noise_texture_1.outputs["Color"], "W": nw.new_value(uniform(0, 10), "voronoi_texture_2_w")},
            attrs={"feature": "DISTANCE_TO_EDGE", "voronoi_dimensions": "4D"},
        )

        colorramp_6 = nw.new_node(
            Nodes.ColorRamp, input_kwargs={"Fac": voronoi_texture_2.outputs["Distance"]}
        )
        colorramp_6.color_ramp.elements[0].position = 0.232
        colorramp_6.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
        colorramp_6.color_ramp.elements[1].position = 1.0
        colorramp_6.color_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)

        vector_math_4 = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={0: colorramp_6.outputs["Color"], 1: normal},
            attrs={"operation": "MULTIPLY"},
        )

        value_2 = nw.new_node(Nodes.Value)
        value_2.outputs["Value"].default_value = 0.68

        vector_math_5 = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={0: vector_math_4.outputs["Vector"], 1: value_2},
            attrs={"operation": "MULTIPLY"},
        )

        vector_math_11 = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={
                0: vector_math_10.outputs["Vector"],
                1: vector_math_5.outputs["Vector"],
            },
        )

        voronoi_texture_1 = nw.new_node(
            Nodes.VoronoiTexture,
            input_kwargs={"Vector": position, "Scale": sca_gra},
            attrs={"feature": "SMOOTH_F1"},
        )

        colorramp_3 = nw.new_node(
            Nodes.ColorRamp, input_kwargs={"Fac": voronoi_texture_1.outputs["Color"]}
        )
        colorramp_3.color_ramp.elements[0].position = 0.323
        colorramp_3.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
        colorramp_3.color_ramp.elements[1].position = 0.436
        colorramp_3.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

        vector_math_6 = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={0: colorramp_3.outputs["Color"], 1: normal},
            attrs={"operation": "MULTIPLY"},
        )

        value_3 = nw.new_node(Nodes.Value)
        value_3.outputs["Value"].default_value = 0.28

        vector_math_7 = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={0: vector_math_6.outputs["Vector"], 1: value_3},
            attrs={"operation": "MULTIPLY"},
        )

        vector_math_12 = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={
                0: vector_math_11.outputs["Vector"],
                1: vector_math_7.outputs["Vector"],
            },
        )

        noise_texture_3 = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={"Vector": position, "Roughness": 0.52, "W": nw.new_value(uniform(0, 10), "noise_texture_3_w")},
            attrs={"noise_dimensions": "4D"},
        )

        colorramp_4 = nw.new_node(
            Nodes.ColorRamp, input_kwargs={"Fac": noise_texture_3.outputs["Fac"]}, label="colorramp_4_VAR"
        )
        colorramp_4.color_ramp.elements[0].position = 0.49 + per_crac
        colorramp_4.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
        colorramp_4.color_ramp.elements[1].position = 0.5 + per_crac
        colorramp_4.color_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)

        noise_texture_2 = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={
                "Vector": position,
                "Scale": 2.3,
                "Detail": 16.0,
                "W": nw.new_value(uniform(0, 10), "noise_texture_2_w"),
            },
            attrs={"noise_dimensions": "4D"},
        )

        voronoi_texture_3 = nw.new_node(
            Nodes.VoronoiTexture,
            input_kwargs={
                "Vector": noise_texture_2.outputs["Color"],
                "Scale": sca_crac,
                "W": nw.new_value(uniform(0, 10), "voronoi_texture_3_w"),
            },
            attrs={"feature": "DISTANCE_TO_EDGE", "voronoi_dimensions": "4D"},
        )

        colorramp_5 = nw.new_node(
            Nodes.ColorRamp, input_kwargs={"Fac": voronoi_texture_3.outputs["Distance"]}, label="colorramp_5_VAR"
        )
        colorramp_5.color_ramp.elements[0].position = 0.0
        colorramp_5.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
        colorramp_5.color_ramp.elements[1].position = wid_crac
        colorramp_5.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

        mix_2 = nw.new_node(
            Nodes.MixRGB,
            input_kwargs={
                "Fac": colorramp_4.outputs["Color"],
                "Color1": colorramp_5.outputs["Color"],
                "Color2": (1.0, 1.0, 1.0, 1.0),
            },
        )

        vector_math_8 = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={0: mix_2, 1: normal},
            attrs={"operation": "MULTIPLY"},
        )

        value_4 = nw.new_node(Nodes.Value)
        value_4.outputs["Value"].default_value = 1

        vector_math_9 = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={0: vector_math_8.outputs["Vector"], 1: value_4},
            attrs={"operation": "MULTIPLY"},
        )

        vector_math_13 = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={
                0: vector_math_12.outputs["Vector"],
                1: vector_math_9.outputs["Vector"],
            },
        )

        value_5 = nw.new_node(Nodes.Value)
        value_5.outputs["Value"].default_value = 0.02

        offset = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={0: vector_math_13.outputs["Vector"], 1: value_5},
            attrs={"operation": "MULTIPLY"},
        )

        mix = nw.new_node(
            Nodes.MixRGB,
            input_kwargs={
                "Fac": colorramp_2.outputs["Color"],
                "Color1": (0.2346, 0.0823, 0.0194, 1.0),
                "Color2": colorramp.outputs["Color"],
            },
        )

        mix_1 = nw.new_node(
            Nodes.MixRGB,
            input_kwargs={
                "Fac": mix_2,
                "Color1": col_crac_node,
                "Color2": mix,
            },
        )

        colorramp_1 = nw.new_node(Nodes.ColorRamp, input_kwargs={"Fac": mix_1})
        colorramp_1.color_ramp.elements[0].position = 0.0
        colorramp_1.color_ramp.elements[0].color = (0.6308, 0.6308, 0.6308, 1.0)
        colorramp_1.color_ramp.elements[1].position = 1.0
        colorramp_1.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

        crackedground_roughness = colorramp_1
        crackedground_base_color = mix_1

    if geometry:  
        groupinput = nw.new_node(Nodes.GroupInput)
        if selection is not None:
            offset = nw.multiply(offset, surface.eval_argument(nw, selection))
        set_position = nw.new_node(Nodes.SetPosition, input_kwargs={"Geometry": groupinput,  "Offset": offset})
        nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': set_position})
    else:
        return crackedground_base_color, crackedground_roughness


def apply(obj, selection=None, **kwargs):
    surface.add_geomod(obj, geo_cracked_ground, selection=selection)
    surface.add_material(obj, shader_cracked_ground, selection=selection)
