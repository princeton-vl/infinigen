# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Ankit Goyal, Mingzhe Wang, Zeyu Ma
# Acknowledgement: This file draws inspiration from https://www.youtube.com/watch?v=YKRK82JeBo8 by Ryan King Art


import os

import bpy
import gin
from infinigen.core.nodes.node_wrangler import Nodes
from numpy.random import uniform, normal as N
from infinigen.core import surface
from infinigen.assets.materials.utils.surface_utils import sample_color, sample_ratio
from infinigen.core.util.organization import SurfaceTypes
from infinigen.core.util.math import FixedSeed

from .mountain import geo_MOUNTAIN_general


type = SurfaceTypes.SDFPerturb
mod_name = "geo_stone"
name = "stone"

def shader_stone(nw):
    nw.force_input_consistency()
    stone_base_color, stone_roughness = geo_stone(nw, geometry=False)

    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": stone_base_color,
            "Roughness": stone_roughness,
        },
    )

    return principled_bsdf

@gin.configurable
def geo_stone(nw, selection=None, random_seed=0, geometry=True):
    nw.force_input_consistency()
    if nw.node_group.type == "SHADER":
        position = nw.new_node('ShaderNodeNewGeometry')
        normal = (nw.new_node('ShaderNodeNewGeometry'), 1)
    else:
        position = nw.new_node(Nodes.InputPosition)
        normal = nw.new_node(Nodes.InputNormal)

    with FixedSeed(random_seed):
        # size of low frequency bumps, higher means smaller bumps
        size_bumps_lf = uniform(0, 30)
        # height of low frequency bumps
        heig_bumps_lf = nw.new_value(uniform(.08, .15), "heig_bumps_lf")
        # density of cracks, lower means cracks are present in smaller area
        dens_crack = uniform(0, 0.1)
        # scale cracks
        scal_crack = uniform(5, 10)/2
        # width of the crack
        widt_crack = uniform(0.08, 0.12)
        scale = 0.5

        musgrave_texture = nw.new_node(
            Nodes.MusgraveTexture,
            input_kwargs={"Vector": position, "Scale": nw.new_value(size_bumps_lf * scale, "size_bumps_lf"), "W": nw.new_value(uniform(0, 10), "musgrave_texture_w")},
            attrs={"musgrave_dimensions": "4D"},
        )

        vector_math = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={0: musgrave_texture, 1: normal},
            attrs={"operation": "MULTIPLY"},
        )

        vector_math_3 = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={0: vector_math.outputs["Vector"], 1: heig_bumps_lf},
            attrs={"operation": "MULTIPLY"},
        )

        noise_texture = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={
                "Vector": position,
                "Scale": 12.0 * scale,
                "Detail": 16.0,
                "W": nw.new_value(uniform(0, 10), "noise_texture_w"),
            },
            attrs={"noise_dimensions": "4D"},
        )

        vector_math_1 = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={0: noise_texture.outputs["Fac"], 1: normal},
            attrs={"operation": "MULTIPLY"},
        )

        value_1 = nw.new_node(Nodes.Value)
        value_1.outputs["Value"].default_value = 0.1

        vector_math_4 = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={0: vector_math_1.outputs["Vector"], 1: value_1},
            attrs={"operation": "MULTIPLY"},
        )

        vector_math_6 = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={
                0: vector_math_3.outputs["Vector"],
                1: vector_math_4.outputs["Vector"],
            },
        )

        noise_texture_2 = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={"Vector": position, "Scale": 5.0 * scale, "W": nw.new_value(uniform(0, 10), "noise_texture_2_w")},
            attrs={"noise_dimensions": "4D"},
        )

        colorramp_2 = nw.new_node(
            Nodes.ColorRamp, input_kwargs={"Fac": noise_texture_2.outputs["Fac"]}, label="colorramp_2_VAR"
        )
        colorramp_2.color_ramp.elements[0].position = 0.445 + (2 * dens_crack) - 0.1
        colorramp_2.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
        colorramp_2.color_ramp.elements[1].position = 0.505 + (2 * dens_crack) - 0.1
        colorramp_2.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

        noise_texture_1 = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={
                "Vector": position,
                "Scale": 1.0 * scale,
                "Detail": 16.0,
                "W": nw.new_value(uniform(0, 10), "noise_texture_1_w"),
            },
            attrs={"noise_dimensions": "4D"},
        )

        wave_texture = nw.new_node(
            Nodes.WaveTexture,
            input_kwargs={"Vector": noise_texture_1.outputs["Color"], "Scale": nw.new_value(N(2, 0.5), "wave_texture_scale"), "Distortion": nw.new_value(N(6, 2), "wave_texture_distortion"), "Detail": nw.new_value(N(15, 5), "wave_texture_detail")},
        )

        colorramp_1 = nw.new_node(
            Nodes.ColorRamp, input_kwargs={"Fac": wave_texture.outputs["Fac"]}, label="colorramp_1_VAR"
        )
        colorramp_1.color_ramp.elements[0].position = 0.0
        colorramp_1.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
        colorramp_1.color_ramp.elements[1].position = widt_crack
        colorramp_1.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

        mix = nw.new_node(
            Nodes.MixRGB,
            input_kwargs={
                "Fac": colorramp_2.outputs["Color"],
                "Color1": colorramp_1.outputs["Color"],
            },
        )

        vector_math_2 = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={0: mix, 1: normal},
            attrs={"operation": "MULTIPLY"},
        )

        value_2 = nw.new_node(Nodes.Value)
        value_2.outputs["Value"].default_value = 0.05

        vector_math_5 = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={0: vector_math_2.outputs["Vector"], 1: value_2},
            attrs={"operation": "MULTIPLY"},
        )

        vector_math_7 = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={
                0: vector_math_6.outputs["Vector"],
                1: vector_math_5.outputs["Vector"],
            },
        )

        value_3 = nw.new_node(Nodes.Value)
        value_3.outputs["Value"].default_value = 0.08

        vector_math_8 = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={0: vector_math_7.outputs["Vector"], 1: value_3},
            attrs={"operation": "MULTIPLY"},
        )

        noise_texture_3 = nw.new_node(Nodes.NoiseTexture,
            input_kwargs={'Vector': position, "W": nw.new_value(uniform(0, 10), "noise_texture_3_w"), 'Scale': nw.new_value(sample_ratio(5, 3/4, 4/3), "noise_texture_3_scale")},
            attrs={"noise_dimensions": "4D"})
        
        subtract = nw.new_node(Nodes.Math,
            input_kwargs={0: noise_texture_3.outputs["Fac"]},
            attrs={'operation': 'SUBTRACT'})
        
        multiply_8 = nw.new_node(Nodes.VectorMath,
            input_kwargs={0: subtract, 1: normal},
            attrs={'operation': 'MULTIPLY'})
        
        value_5 = nw.new_node(Nodes.Value)
        value_5.outputs[0].default_value = 0.05
        
        multiply_9 = nw.new_node(Nodes.VectorMath,
            input_kwargs={0: multiply_8.outputs["Vector"], 1: value_5},
            attrs={'operation': 'MULTIPLY'})
        
        noise_texture_4 = nw.new_node(Nodes.NoiseTexture,
            input_kwargs={'Vector': position, 'Scale': nw.new_value(sample_ratio(20, 3/4, 4/3), "noise_texture_4_scale"), "W": nw.new_value(uniform(0, 10), "noise_texture_4_w")},
            attrs={'noise_dimensions': '4D'})
        
        colorramp_5 = nw.new_node(Nodes.ColorRamp,
            input_kwargs={'Fac': noise_texture_4.outputs["Fac"]})
        colorramp_5.color_ramp.elements.new(0)
        colorramp_5.color_ramp.elements.new(0)
        colorramp_5.color_ramp.elements[0].position = 0.0
        colorramp_5.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
        colorramp_5.color_ramp.elements[1].position = 0.3
        colorramp_5.color_ramp.elements[1].color = (0.5, 0.5, 0.5, 1.0)
        colorramp_5.color_ramp.elements[2].position = 0.7
        colorramp_5.color_ramp.elements[2].color = (0.5, 0.5, 0.5, 1.0)
        colorramp_5.color_ramp.elements[3].position = 1.0
        colorramp_5.color_ramp.elements[3].color = (1.0, 1.0, 1.0, 1.0)
        
        subtract_1 = nw.new_node(Nodes.Math,
            input_kwargs={0: colorramp_5.outputs["Color"]},
            attrs={'operation': 'SUBTRACT'})
        
        multiply_10 = nw.new_node(Nodes.VectorMath,
            input_kwargs={0: subtract_1, 1: normal},
            attrs={'operation': 'MULTIPLY'})
        
        value_6 = nw.new_node(Nodes.Value)
        value_6.outputs[0].default_value = 0.1
        
        multiply_11 = nw.new_node(Nodes.VectorMath,
            input_kwargs={0: multiply_10.outputs["Vector"], 1: value_6},
            attrs={'operation': 'MULTIPLY'})

        offset = nw.add(multiply_9, vector_math_8, multiply_11)

        colorramp = nw.new_node(
            Nodes.ColorRamp, input_kwargs={"Fac": noise_texture.outputs["Fac"]}, label="colorramp_1_VAR"
        )
        color1 = uniform(0, 0.05)
        color2 = uniform(0.05, 0.1)
        color3 = uniform(0, 0.05)
        colorramp.color_ramp.elements.new(1)
        colorramp.color_ramp.elements[0].position = 0.223
        colorramp.color_ramp.elements[0].color = (color1, color1, color1, 1.0)
        colorramp.color_ramp.elements[1].position = 0.509
        colorramp.color_ramp.elements[1].color = (color2, color2, color2, 1.0)
        colorramp.color_ramp.elements[2].position = 1.0
        colorramp.color_ramp.elements[2].color = (color3, color3, color3, 1.0)
        sample_color(colorramp.color_ramp.elements[1].color, offset=0.01)
        sample_color(colorramp.color_ramp.elements[2].color, offset=0.01)
        
        stone_base_color = nw.new_node(
            Nodes.MixRGB,
            input_kwargs={
                "Fac": mix,
                "Color1": (0.0, 0.0, 0.0, 1.0),
                "Color2": colorramp.outputs["Color"],
            },
        )

        rough_min = uniform(0.6, 0.7)
        rough_max = uniform(0.7, 0.8)
        colorramp_3 = nw.new_node(
            Nodes.ColorRamp, input_kwargs={"Fac": noise_texture.outputs["Fac"]}, label="colorramp_3_VAR"
        )
        colorramp_3.color_ramp.elements[0].position = 0.082
        colorramp_3.color_ramp.elements[0].color = (rough_min, rough_min, rough_min, 1.0)
        colorramp_3.color_ramp.elements[1].position = 0.768
        colorramp_3.color_ramp.elements[1].color = (rough_max, rough_max, rough_max, 1.0)

        stone_roughness = colorramp_3

    if geometry:  
        groupinput = nw.new_node(Nodes.GroupInput)
        noise_params = {"scale": ("uniform", 10, 20), "detail": 9, "roughness": 0.6, "zscale": ("log_uniform", 0.007, 0.013)}
        offset = nw.add(offset, geo_MOUNTAIN_general(nw, 3, noise_params, 0, {}, {}))
        if selection is not None:
            offset = nw.multiply(offset, surface.eval_argument(nw, selection))
        set_position = nw.new_node(Nodes.SetPosition, input_kwargs={"Geometry": groupinput,  "Offset": offset})
        nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': set_position})
    else:
        return stone_base_color, stone_roughness



def apply(obj, selection=None, **kwargs):
    surface.add_geomod(
        obj,
        geo_stone,
        selection=selection,
    )
    surface.add_material(obj, shader_stone, selection=selection)

