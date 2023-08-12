# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Ankit Goyal, Zeyu Ma


import gin
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes
from numpy.random import uniform
from infinigen.core import surface
from infinigen.core.util.organization import SurfaceTypes
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import random_color_neighbour

type = SurfaceTypes.SDFPerturb
mod_name = "geometry_soil"
name = "soil"

@node_utils.to_nodegroup(
    "nodegroup_displacement_to_offset", singleton=False, type="GeometryNodeTree"
)
def nodegroup_displacement_to_offset(nw):
    # Code generated using version 2.3.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Vector", (0.0, 0.0, 0.0)),
            ("NodeSocketFloat", "Magnitude", 1.0),
        ],
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Vector"],
            1: group_input.outputs["Magnitude"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    normal = nw.new_node(Nodes.InputNormal)

    multiply_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: multiply, 1: normal},
        attrs={"operation": "MULTIPLY"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput, input_kwargs={"Vector": multiply_1.outputs["Vector"]}
    )


def nodegroup_pebble(nw):
    if nw.node_group.type == "SHADER":
        position = nw.new_node('ShaderNodeNewGeometry')
    else:
        position = nw.new_node(Nodes.InputPosition)
    
    # Code generated using version 2.3.1 of the node_transpiler

    noise1_w = nw.new_node(Nodes.Value, label="noise1_w ~ U(0, 10)")
    noise1_w.outputs[0].default_value = uniform(0.0, 10.0)

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "PebbleScale", 5.0),
            ("NodeSocketFloat", "NoiseMag", 0.2),
        ],
    )

    noise_texture = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={"W": noise1_w, "Scale": group_input.outputs["PebbleScale"]},
        attrs={"noise_dimensions": "4D"},
    )

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: noise_texture.outputs["Color"],
            1: group_input.outputs["NoiseMag"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(
        Nodes.VectorMath, input_kwargs={0: multiply.outputs["Vector"], 1: position}
    )

    vornoi1_w = nw.new_node(Nodes.Value, label="vornoi1_w ~ U(0, 10)")
    vornoi1_w.outputs[0].default_value = uniform(0.0, 10.0)

    voronoi_texture_2 = nw.new_node(
        Nodes.VoronoiTexture,
        input_kwargs={
            "Vector": add.outputs["Vector"],
            "W": vornoi1_w,
            "Scale": group_input.outputs["PebbleScale"],
        },
        attrs={"voronoi_dimensions": "4D"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Distance": voronoi_texture_2.outputs["Distance"]},
    )



@node_utils.to_nodegroup("nodegroup_pebble", singleton=False)
def nodegroup_pebble_geo(nw):
    nw.force_input_consistency()
    nodegroup_pebble(nw)

@node_utils.to_nodegroup("nodegroup_pebble", singleton=False, type='ShaderNodeTree')
def nodegroup_pebble_shader(nw):
    nw.force_input_consistency()
    nodegroup_pebble(nw)


def shader_soil(nw):
    nw.force_input_consistency()
    big_stone = geometry_soil(nw, geometry=False)
    # Code generated using version 2.3.1 of the node_transpiler
    darkness = 1.5
    soil_col_1 = random_color_neighbour((0.28 / darkness, 0.11 / darkness, 0.042 / darkness, 1.0), 0.05, 0.1, 0.1)
    soil_col_2 = random_color_neighbour((0.22 / darkness , 0.0906 / darkness , 0.035 / darkness, 1.0), 0.05, 0.1, 0.1)
    peb_col_1 = random_color_neighbour((0.3813, 0.1714, 0.0782, 1.0), 0.1, 0.1, 0.1)
    peb_col_2 = random_color_neighbour((0.314, 0.1274, 0.0578, 1.0), 0.1, 0.1, 0.1)

    ambient_occlusion = nw.new_node(Nodes.AmbientOcclusion)

    colorramp_1 = nw.new_node(
        Nodes.ColorRamp, input_kwargs={"Fac": ambient_occlusion.outputs["Color"]}
    )
    colorramp_1.color_ramp.elements[0].position = 0.8
    colorramp_1.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_1.color_ramp.elements[1].position = 1.0
    colorramp_1.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    noise1_w = nw.new_node(Nodes.Value, label="noise1_w ~ U(0, 10)")
    noise1_w.outputs[0].default_value = uniform(0.0, 10.0)

    noise_texture = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={"W": noise1_w, "Scale": 10.0},
        attrs={"noise_dimensions": "4D"},
    )

    mix_1 = nw.new_node(
        Nodes.MixRGB,
        input_kwargs={
            "Color1": colorramp_1.outputs["Color"],
            "Color2": noise_texture.outputs["Fac"],
        },
    )

    colorramp = nw.new_node(Nodes.ColorRamp, input_kwargs={"Fac": mix_1})
    colorramp.color_ramp.elements[0].position = 0.0
    colorramp.color_ramp.elements[0].color = soil_col_1
    colorramp.color_ramp.elements[1].position = 1.0
    colorramp.color_ramp.elements[1].color = soil_col_2

    colorramp_3 = nw.new_node(
        Nodes.ColorRamp, input_kwargs={"Fac": big_stone}
    )
    colorramp_3.color_ramp.elements[0].position = 0.0
    colorramp_3.color_ramp.elements[0].color = peb_col_1
    colorramp_3.color_ramp.elements[1].position = 1.0
    colorramp_3.color_ramp.elements[1].color = peb_col_2

    mix = nw.new_node(
        Nodes.MixRGB,
        input_kwargs={
            "Fac": big_stone,
            "Color1": colorramp.outputs["Color"],
            "Color2": colorramp_3.outputs["Color"],
        },
    )

    multiply_add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: big_stone, 1: -1.0, 2: 1.0},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    colorramp_2 = nw.new_node(Nodes.ColorRamp, input_kwargs={"Fac": multiply_add})
    colorramp_2.color_ramp.elements.new(0.0)
    colorramp_2.color_ramp.elements.new(0.0)
    colorramp_2.color_ramp.elements[0].position = 0.0
    colorramp_2.color_ramp.elements[0].color = (0.5, 0.5, 0.5, 1.0)
    colorramp_2.color_ramp.elements[1].position = 0.8636
    colorramp_2.color_ramp.elements[1].color = (0.7, 0.7, 0.7, 1.0)
    colorramp_2.color_ramp.elements[2].position = 0.9427
    colorramp_2.color_ramp.elements[2].color = (0.95, 0.95, 0.95, 1.0)
    colorramp_2.color_ramp.elements[3].position = 1.0
    colorramp_2.color_ramp.elements[3].color = (0.98, 0.98, 0.98, 1.0)

    principled_bsdf_4 = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": mix,
            "Specular": 0.2,
            "Roughness": colorramp_2.outputs["Color"],
        },
    )

    return principled_bsdf_4

@gin.configurable
def geometry_soil(nw, selection=None, random_seed=0, geometry=True):
    nw.force_input_consistency()
    if nw.node_group.type == "SHADER":
        position = nw.new_node('ShaderNodeNewGeometry')
        normal = (nw.new_node('ShaderNodeNewGeometry'), 1)
    else:
        position = nw.new_node(Nodes.InputPosition)
        normal = nw.new_node(Nodes.InputNormal)
    
    with FixedSeed(random_seed):
        # Code generated using version 2.3.1 of the node_transpiler

        peb1_size = nw.new_value(uniform(2.0, 5.0), "peb1_size ~ U(2, 5)")
        peb1_noise_mag = nw.new_value((1 / peb1_size.outputs[0].default_value) * uniform(1.5, 2), "peb1_noise_mag ~ U(0.1, 0.5)")

        group = nw.new_node(
            nodegroup_pebble_geo().name if nw.node_group.type != "SHADER" else nodegroup_pebble_shader().name,
            input_kwargs={"PebbleScale": peb1_size, "NoiseMag": peb1_noise_mag},
        )

        peb1_roundness = uniform(0.5, 1.0)
        peb1_amount = uniform(0.2, 0.5)

        colorramp = nw.new_node(Nodes.ColorRamp, input_kwargs={"Fac": group}, label="colorramp_VAR")
        colorramp.color_ramp.elements[0].position = 0.0
        colorramp.color_ramp.elements[0].color = (
            peb1_roundness,
            peb1_roundness,
            peb1_roundness,
            1.0,
        )
        colorramp.color_ramp.elements.new(1)
        colorramp.color_ramp.elements[1].color = (
            peb1_roundness / 2,
            peb1_roundness / 2,
            peb1_roundness / 2,
            1.0,
        )
        colorramp.color_ramp.elements[1].position = peb1_amount / 8
        colorramp.color_ramp.elements[2].position = peb1_amount
        colorramp.color_ramp.elements[2].color = (0.0, 0.0, 0.0, 1.0)

        peb2_size = nw.new_value(uniform(5, 9), "peb2_size ~ U(5, 9)")
        peb2_noise_scale = nw.new_value((1 / peb2_size.outputs[0].default_value) * uniform(1.5, 2), "peb2_noise_scale ~ U(0.05, 0.2)")

        group_3 = nw.new_node(
            nodegroup_pebble_geo().name if nw.node_group.type != "SHADER" else nodegroup_pebble_shader().name,
            input_kwargs={"PebbleScale": peb2_size, "NoiseMag": peb2_noise_scale},
        )

        peb2_roundness = uniform(0.3, 0.8)
        peb2_amount = uniform(0.2, 0.5)
        colorramp_2 = nw.new_node(Nodes.ColorRamp, input_kwargs={"Fac": group_3}, label="colorramp_2_VAR")
        colorramp_2.color_ramp.elements[0].position = 0.0
        colorramp_2.color_ramp.elements[0].color = (
            peb2_roundness,
            peb2_roundness,
            peb2_roundness,
            1.0,
        )
        colorramp_2.color_ramp.elements.new(1)
        colorramp_2.color_ramp.elements[1].color = (
            peb2_roundness / 2,
            peb2_roundness / 2,
            peb2_roundness / 2,
            1.0,
        )
        colorramp_2.color_ramp.elements[1].position = peb2_amount / 8
        colorramp_2.color_ramp.elements[2].position = peb2_amount
        colorramp_2.color_ramp.elements[2].color = (0.0, 0.0, 0.0, 1.0)

        add = nw.new_node(
            Nodes.Math,
            input_kwargs={0: colorramp.outputs["Color"], 1: colorramp_2.outputs["Color"]},
        )

        big_stone = colorramp

        peb3_size = nw.new_value(uniform(12.0, 18.0), "peb3_size ~ U(12, 18)")
        peb3_noise_scale = nw.new_value(uniform(0.05, 0.35), "peb3_noise_scale ~ U(0.05, 0.35)")

        group_2 = nw.new_node(
            nodegroup_pebble_geo().name if nw.node_group.type != "SHADER" else nodegroup_pebble_shader().name,
            input_kwargs={"PebbleScale": peb3_size, "NoiseMag": peb3_noise_scale},
        )

        colorramp_1 = nw.new_node(Nodes.ColorRamp, input_kwargs={"Fac": group_2})
        colorramp_1.color_ramp.elements[0].position = 0.0
        colorramp_1.color_ramp.elements[0].color = (0.15, 0.15, 0.15, 1.0)
        colorramp_1.color_ramp.elements[1].position = 0.9
        colorramp_1.color_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)

        add_1 = nw.new_node(
            Nodes.Math, input_kwargs={0: add, 1: colorramp_1.outputs["Color"]}
        )

    if geometry:
        offset = nw.new_node(
            nodegroup_displacement_to_offset().name,
            input_kwargs={"Vector": add_1, "Magnitude": 0.1},
        )
        groupinput = nw.new_node(Nodes.GroupInput)
        if selection is not None:
            offset = nw.multiply(offset, surface.eval_argument(nw, selection))
        set_position = nw.new_node(Nodes.SetPosition, input_kwargs={"Geometry": groupinput,  "Offset": offset})
        nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': set_position})
    else:
        return big_stone


def apply(obj, selection=None, **kwargs):
    surface.add_geomod(
        obj, geometry_soil, selection=selection
    )
    surface.add_material(obj, shader_soil, selection=selection)
