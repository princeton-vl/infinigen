# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Ankit Goyal, Mingzhe Wang, Zeyu Ma
# Acknowledgement: This file draws inspiration from https://www.youtube.com/watch?v=9Tq-6HReNEk by Ryan King Art


from numpy.random import uniform as U, normal as N
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core import surface
from infinigen.core.util.random import random_color_neighbour
from infinigen.core.util.organization import SurfaceTypes
from infinigen.core.util.math import FixedSeed
import gin

type = SurfaceTypes.SDFPerturb
mod_name = "geo_cobblestone"
name = "cobble_stone"


def shader_cobblestone(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler, and modified
    nw.force_input_consistency()
    stone_color = geo_cobblestone(nw, geometry=False)
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': nw.new_node('ShaderNodeNewGeometry'), 'Scale': N(10, 1.5) / 25, 'W': U(-5, 5)},
        attrs={'noise_dimensions': '4D'})

    colorramp_1 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': noise_texture.outputs["Fac"]})
    colorramp_1.color_ramp.elements[0].position = 0.0
    colorramp_1.color_ramp.elements[0].color = random_color_neighbour((0.014, 0.013, 0.014, 1.0), 0.2, 0.1, 0.1)
    colorramp_1.color_ramp.elements[1].position = 1.0
    colorramp_1.color_ramp.elements[1].color = random_color_neighbour((0.047, 0.068, 0.069, 1.0), 0.2, 0.1, 0.1)


    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': stone_color.outputs["Color"], 'Color1': (0.0, 0.0, 0.0, 1.0), 'Color2': colorramp_1.outputs["Color"]})

    roughness_low = N(0.25, 0.05)
    roughness_high = N(0.75, 0.05)
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': stone_color.outputs["Color"]})
    colorramp.color_ramp.elements[0].position = 0.0
    colorramp.color_ramp.elements[0].color = (roughness_high, roughness_high, roughness_high, 1.0)
    colorramp.color_ramp.elements[1].position = 1.0
    colorramp.color_ramp.elements[1].color = (roughness_low, roughness_low, roughness_low, 1.0)

    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': mix, 'Roughness': colorramp.outputs["Color"]})

    return principled_bsdf

@gin.configurable
def geo_cobblestone(nw: NodeWrangler, selection=None, random_seed=0, geometry=True):
    # Code generated using version 2.4.3 of the node_transpiler, and modified
    nw.force_input_consistency()
    if nw.node_group.type == "SHADER":
        position = nw.new_node('ShaderNodeNewGeometry')
        normal = (nw.new_node('ShaderNodeNewGeometry'), 1)
    else:
        position = nw.new_node(Nodes.InputPosition)
        normal = nw.new_node(Nodes.InputNormal)

    with FixedSeed(random_seed):
        # scale of the stone, inversely proportional
        sca_sto = nw.new_value(U(9, 15)/2, "sca_sto")
        # uniformity of the stone, inversely proportional
        uni_sto = nw.new_value(U(0.5, 0.9), "uni_sto")
        # depth of stone
        dep_sto = nw.new_value(U(0.02, 0.04), "dep_sto")


        group_input = nw.new_node(Nodes.GroupInput,
            expose_input=[('NodeSocketGeometry', 'Geometry', None)])

        noise_texture = nw.new_node(Nodes.NoiseTexture,
            input_kwargs={'Vector': position, 'W': nw.new_value(U(-5, 5), "W1"), 'Scale': nw.new_value(N(6.0, 0.5), "Scale1")},
            attrs={'noise_dimensions': '4D'})

        voronoi_texture_2 = nw.new_node(Nodes.VoronoiTexture,
            input_kwargs={'W': nw.new_value(U(-5, 5), "W2"), 'Scale': sca_sto, 'Randomness': uni_sto},
            attrs={'voronoi_dimensions': '4D'})

        noise_texture_3 = nw.new_node(Nodes.NoiseTexture,
            input_kwargs={'Vector': voronoi_texture_2.outputs["Position"], 'Scale': nw.new_value(N(20, 2), "Scale2")})

        colorramp_4 = nw.new_node(Nodes.ColorRamp,
            input_kwargs={'Fac': noise_texture_3.outputs["Fac"]})
        colorramp_4.color_ramp.interpolation = "CONSTANT"
        colorramp_4.color_ramp.elements[0].position = 0.1159
        colorramp_4.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
        colorramp_4.color_ramp.elements[1].position = 0.475
        colorramp_4.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

        voronoi_texture_1 = nw.new_node(Nodes.VoronoiTexture,
            input_kwargs={'Vector': position, 'W': nw.new_value(U(-5, 5), "W3"), 'Scale': sca_sto, 'Randomness': uni_sto},
            attrs={'voronoi_dimensions': '4D', 'feature': 'DISTANCE_TO_EDGE'})

        multiply = nw.new_node(Nodes.Math,
            input_kwargs={0: 1.5, 1: sca_sto},
            attrs={'operation': 'MULTIPLY'})

        voronoi_texture_3 = nw.new_node(Nodes.VoronoiTexture,
            input_kwargs={'Vector': position, 'W': nw.new_value(U(-5, 5), "W4"), 'Scale': multiply, 'Randomness': uni_sto},
            attrs={'voronoi_dimensions': '4D', 'feature': 'DISTANCE_TO_EDGE'})

        mix_3 = nw.new_node(Nodes.MixRGB,
            input_kwargs={'Fac': colorramp_4.outputs["Color"], 'Color1': voronoi_texture_1.outputs["Distance"], 'Color2': voronoi_texture_3.outputs["Distance"]})

        mix = nw.new_node(Nodes.MixRGB,
            input_kwargs={'Fac': noise_texture.outputs["Fac"], 'Color1': mix_3})

        colorramp = nw.new_node(Nodes.ColorRamp,
            input_kwargs={'Fac': mix}, label="colorramp_VAR")
        colorramp.color_ramp.elements[0].position = U(0.26, 0.29)
        colorramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
        colorramp.color_ramp.elements[1].position = 0.377
        colorramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

        if not geometry: return colorramp

        multiply_1 = nw.new_node(Nodes.VectorMath,
            input_kwargs={0: colorramp.outputs["Color"], 1: normal},
            attrs={'operation': 'MULTIPLY'})

        multiply_2 = nw.new_node(Nodes.VectorMath,
            input_kwargs={0: multiply_1.outputs["Vector"], 1: dep_sto},
            attrs={'operation': 'MULTIPLY'})

        noise_texture_4 = nw.new_node(Nodes.NoiseTexture,
            input_kwargs={'Scale': nw.new_value(N(20, 2), "Scale3"), 'Detail': 10.0, 'Distortion': 2.0})

        subtract = nw.new_node(Nodes.VectorMath,
            input_kwargs={0: noise_texture_4.outputs["Fac"], 1: (0.5, 0.5, 0.5)},
            attrs={'operation': 'SUBTRACT'})

        multiply_5 = nw.new_node(Nodes.VectorMath,
            input_kwargs={0: subtract.outputs["Vector"], 1: normal},
            attrs={'operation': 'MULTIPLY'})

        value_8 = nw.new_value(U(0.01, 0.02), "value_8")

        multiply_6 = nw.new_node(Nodes.VectorMath,
            input_kwargs={0: multiply_5.outputs["Vector"], 1: value_8},
            attrs={'operation': 'MULTIPLY'})

        set_position_1 = nw.new_node(Nodes.SetPosition,
            input_kwargs={'Geometry': group_input, 'Offset': nw.add(multiply_6, multiply_2)})


        group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': set_position_1})


def apply(obj, selection=None, **kwargs):
    surface.add_geomod(obj,geo_cobblestone, selection=selection)
    surface.add_material(obj, shader_cobblestone, selection=selection, reuse=False)