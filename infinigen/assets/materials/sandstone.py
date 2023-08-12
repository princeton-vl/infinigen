# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Ankit Goyal, Mingzhe Wang, Zeyu Ma


 
# Code generated using version v2.0.0 of the node_transpiler
import gin
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes
from numpy.random import uniform
from infinigen.core import surface
from infinigen.core.util.organization import SurfaceTypes
from infinigen.core.util.random import random_color_neighbour
from infinigen.core.util.random import random_general as rg
from .mountain import geo_MOUNTAIN_general
from mathutils import Vector

type = SurfaceTypes.SDFPerturb
mod_name = "geometry_sandstone"
name = "sandstone"

@node_utils.to_nodegroup("nodegroup_roughness", singleton=False)
def nodegroup_roughness(nw):
    nw.force_input_consistency()
    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Noise 1 Scale", 15.0),
            ("NodeSocketFloat", "Noise 2 Scale", 8.0),
            ("NodeSocketFloat", "Noise 1 Magnitude", 0.3),
            ("NodeSocketFloat", "Noise 2 Magnitude", 0.15),
            ("NodeSocketVector", "Normal", (0.0, 0.0, 0.0)),
        ],
    )

    noise_texture_1 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Scale": group_input.outputs["Noise 1 Scale"],
            "Roughness": 0.5,
            "W": nw.new_value(uniform(0, 10), "noise_texture_1_w"),
        },
        attrs={"noise_dimensions": "4D"},
    )


    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: noise_texture_1.outputs["Color"], 1: group_input.outputs["Normal"]},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: multiply.outputs["Vector"],
            1: group_input.outputs["Noise 1 Magnitude"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    noise_texture_2 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Scale": group_input.outputs["Noise 2 Scale"],
            "Detail": 0.0,
            "W": nw.new_value(uniform(0, 10), "noise_texture_2_w"),
        },
        attrs={"noise_dimensions": "4D"},
    )


    multiply_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: noise_texture_2.outputs["Color"], 1: group_input.outputs["Normal"]},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_3 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: multiply_2.outputs["Vector"],
            1: group_input.outputs["Noise 2 Magnitude"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: multiply_1.outputs["Vector"], 1: multiply_3.outputs["Vector"]},
    )

    value_3 = nw.new_node(Nodes.Value)
    value_3.outputs[0].default_value = 0.05

    multiply_4 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: add.outputs["Vector"], 1: value_3},
        attrs={"operation": "MULTIPLY"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput, input_kwargs={"Vector": multiply_4.outputs["Vector"]}
    )


@node_utils.to_nodegroup("nodegroup_cracked_with_mask", singleton=False, type="GeometryNodeTree")
def nodegroup_cracked_with_mask(nw):
    nw.force_input_consistency()
    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "NoiseScale", 2.0),
            ("NodeSocketFloat", "CrackMagnitude", 0.005),
            ("NodeSocketFloat", "VornoiScale", 5.0),
        ],
    )

    noise_texture = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Scale": group_input.outputs["NoiseScale"],
            "Distortion": 1.0,
            "W": nw.new_value(uniform(0, 10), "noise_texture_w"),
        },
        attrs={"noise_dimensions": "4D"},
    )

    voronoi_texture = nw.new_node(
        Nodes.VoronoiTexture,
        input_kwargs={
            "Vector": noise_texture.outputs["Color"],
            "Scale": group_input.outputs["VornoiScale"],
            "W": nw.new_value(uniform(0, 10), "voronoi_texture_w"),
        },
        attrs={"feature": "DISTANCE_TO_EDGE", "voronoi_dimensions": "4D"},
    )

    colorramp = nw.new_node(
        Nodes.ColorRamp, input_kwargs={"Fac": voronoi_texture.outputs["Distance"]}
    )
    colorramp.color_ramp.elements[0].position = 0.0
    colorramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp.color_ramp.elements[1].position = 0.06
    colorramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: colorramp.outputs["Color"], 1: 1.0},
        attrs={"operation": "SUBTRACT"},
    )

    group = nw.new_node(
        nodegroup_displacement_to_offset().name,
        input_kwargs={
            "Vector": subtract,
            "Magnitude": group_input.outputs["CrackMagnitude"],
        },
    )

    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={"Vector": group})


@node_utils.to_nodegroup("nodegroup_polynomial", singleton=False)
def nodegroup_polynomial(nw):
    nw.force_input_consistency()
    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "X", 0.5),
            ("NodeSocketFloat", "Y", 0.5),
            ("NodeSocketFloat", "Z", 0.5),
            ("NodeSocketFloat", "alpha_x", 0.0),
            ("NodeSocketFloat", "alpha_y", 0.0),
            ("NodeSocketFloat", "alpha_z", 0.0),
            ("NodeSocketFloat", "pow_x", 1.0),
            ("NodeSocketFloat", "pow_y", 1.0),
            ("NodeSocketFloat", "pow_z", 1.0),
        ],
    )

    power = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["X"], 1: group_input.outputs["pow_x"]},
        attrs={"operation": "POWER"},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["alpha_x"], 1: power},
        attrs={"operation": "MULTIPLY"},
    )

    power_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Y"], 1: group_input.outputs["pow_y"]},
        attrs={"operation": "POWER"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["alpha_y"], 1: power_1},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: multiply, 1: multiply_1})

    power_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Z"], 1: group_input.outputs["pow_z"]},
        attrs={"operation": "POWER"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["alpha_z"], 1: power_2},
        attrs={"operation": "MULTIPLY"},
    )

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: add, 1: multiply_2})

    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={"Value": add_1})


@node_utils.to_nodegroup("nodegroup_add_noise", singleton=False)
def nodegroup_add_noise(nw):
    nw.force_input_consistency()
    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Value", 0.5),
            ("NodeSocketFloat", "NoiseMag", 1.0),
            ("NodeSocketFloat", "NoiseScale", 5.0),
            ("NodeSocketFloat", "NoiseDetail", 2.0),
            ("NodeSocketFloat", "NoiseRoughness", 0.5),
            ("NodeSocketFloat", "NoiseDistortion", 0.0),
        ],
    )

    noise_texture_1 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Scale": group_input.outputs["NoiseScale"],
            "Detail": group_input.outputs["NoiseDetail"],
            "Roughness": group_input.outputs["NoiseRoughness"],
            "Distortion": group_input.outputs["NoiseDistortion"],
            "W": nw.new_value(uniform(0, 10), "noise_texture_1_w"),
        },
        attrs={"noise_dimensions": "4D"},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: noise_texture_1.outputs["Fac"],
            1: group_input.outputs["NoiseMag"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["Value"], 1: multiply}
    )

    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={"Value": add})



@node_utils.to_nodegroup("nodegroup_displacement_to_offset", singleton=False)
def nodegroup_displacement_to_offset(nw):
    nw.force_input_consistency()
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

@gin.configurable
def shader(nw, color=None):
    nw.force_input_consistency()
    per_dark_1 = uniform(-0.1, 0.1)
    per_dark_2 = uniform(-0.1, 0.1)
    per_dark_3 = uniform(-0.1, 0.1)
    if color is None:
        col_1 = random_color_neighbour((0.262, 0.057, 0.035, 1.0), 0.1, 0.1, 0.1)
        col_2 = random_color_neighbour((0.8, 0.225, 0.135, 1.0), 0.1, 0.1, 0.1)
    else:
        col_1 = rg(color)
        col_2 = rg(color)

    ambient_occlusion = nw.new_node("ShaderNodeAmbientOcclusion")

    colorramp_0 = nw.new_node(Nodes.ColorRamp, input_kwargs={"Fac": ambient_occlusion})
    colorramp_0.color_ramp.elements[0].position = 0.8
    colorramp_0.color_ramp.elements[0].color = (0, 0, 0, 1)
    colorramp_0.color_ramp.elements[1].position = 1.0
    colorramp_0.color_ramp.elements[1].color = (1, 1, 1, 1)

    vector = nw.new_node('ShaderNodeNewGeometry', [])

    noise_texture = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": vector,
            "Scale": 2.0,
            "W": uniform(0, 10),
        },
        attrs={"noise_dimensions": "4D"},
    )

    noise_texture_1 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": vector,
            "Scale": 10.0,
            "W": uniform(0, 10),
        },
        attrs={"noise_dimensions": "4D"},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: noise_texture.outputs["Fac"],
            1: noise_texture_1.outputs["Fac"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    divide = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply}, attrs={"operation": "DIVIDE"}
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: colorramp_0, 1: divide},
        attrs={"operation": "MULTIPLY"},
    )

    colorramp = nw.new_node(Nodes.ColorRamp, input_kwargs={"Fac": multiply_1})
    colorramp.color_ramp.elements.new(1)
    colorramp.color_ramp.elements[0].position = 0.28 + per_dark_1
    colorramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp.color_ramp.elements[1].position = 0.42 + per_dark_2
    colorramp.color_ramp.elements[1].color = (0.296, 0.296, 0.296, 1.0)
    colorramp.color_ramp.elements[2].position = 0.58 + per_dark_3
    colorramp.color_ramp.elements[2].color = (1.0, 1.0, 1.0, 1.0)

    colorramp_1 = nw.new_node(
        Nodes.ColorRamp, input_kwargs={"Fac": colorramp.outputs["Color"]}
    )
    colorramp_1.color_ramp.elements[0].position = 0.0
    colorramp_1.color_ramp.elements[0].color = col_1
    colorramp_1.color_ramp.elements[1].position = 1.0
    colorramp_1.color_ramp.elements[1].color = col_2

    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": colorramp_1.outputs["Color"],
            "Roughness": 0.9,
            "Specular": 0.1,
        },
    )

    return principled_bsdf

@gin.configurable
def geometry_sandstone(nw, selection=None, is_rock=False, **kwargs):
    nw.force_input_consistency()
    if is_rock:
        roug_mag = nw.new_value(uniform(0.1, 0.5), "roug_mag")
        side_step_displacement_to_offset_magnitude = 0.0

    else:
        roug_mag = nw.new_value(uniform(0.3, 0.5), "roug_mag")
        side_step_displacement_to_offset_magnitude = nw.new_value(uniform(0.5, 1.5), "side_step_displacement_to_offset_magnitude")

    side_step_poly_aplha_x = nw.new_value(uniform(0, 2), "side_step_poly_aplha_x")
    side_step_poly_aplha_y = nw.new_value(uniform(0, 2), "side_step_poly_aplha_y")
    crack_magnitude_1 = nw.new_value(uniform(0.0, 0.012), "crack_magnitude_1")
    crack_magnitude_2 = nw.new_value(uniform(0.0, 0.012), "crack_magnitude_2")
    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
    )

    normal = nw.new_node("GeometryNodeInputNormal", [])
    
    group_3 = nw.new_node(
        nodegroup_roughness().name,
        input_kwargs={"Noise 1 Scale": 200.0, "Noise 1 Magnitude": 0.5, 'Normal': normal},
    )

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_3, 1: roug_mag},
        attrs={"operation": "MULTIPLY"},
    )

    offset = multiply.outputs["Vector"]

    position_1 = nw.new_node(Nodes.InputPosition)

    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position_1})

    group_8 = nw.new_node(
        nodegroup_add_noise().name,
        input_kwargs={
            "Value": separate_xyz_1.outputs["X"],
            "NoiseMag": 0.5,
            "NoiseScale": 2.0,
        },
    )

    group_9 = nw.new_node(
        nodegroup_add_noise().name,
        input_kwargs={
            "Value": separate_xyz_1.outputs["Y"],
            "NoiseMag": 0.5,
            "NoiseScale": 2.0,
        },
    )

    group_7 = nw.new_node(
        nodegroup_polynomial().name,
        input_kwargs={
            "X": group_8,
            "Y": group_9,
            "alpha_x": side_step_poly_aplha_x,
            "alpha_y": side_step_poly_aplha_y,
        },
    )

    noise_texture = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": group_7,
            "Scale": 10.0,
            "W": nw.new_value(uniform(0, 10), "noise_texture_w"),
        },
        attrs={"noise_dimensions": "4D"},
    )

    noise_texture_2 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Scale": 2.0,
            "W": nw.new_value(uniform(0, 10), "noise_texture_2_w"),
        },
        attrs={"noise_dimensions": "4D"},
    )

    colorramp = nw.new_node(
        Nodes.ColorRamp, input_kwargs={"Fac": noise_texture_2.outputs["Fac"]}
    )
    colorramp.color_ramp.elements[0].position = 0.4
    colorramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp.color_ramp.elements[1].position = 0.6
    colorramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: noise_texture.outputs["Fac"], 1: colorramp.outputs["Color"]},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_1, 1: 0.02},
        attrs={"operation": "MULTIPLY"},
    )

    group_4 = nw.new_node(
        nodegroup_displacement_to_offset().name,
        input_kwargs={
            "Vector": multiply_2,
            "Magnitude": side_step_displacement_to_offset_magnitude,
        },
    )

    offset1 = group_4

    noise_texture_1 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "W": nw.new_value(uniform(0, 10), "noise_texture_1_w"),
        },
        attrs={"noise_dimensions": "4D"},
    )

    colorramp_1 = nw.new_node(
        Nodes.ColorRamp, input_kwargs={"Fac": noise_texture_1.outputs["Fac"]}
    )
    colorramp_1.color_ramp.elements[0].position = 0.4
    colorramp_1.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_1.color_ramp.elements[1].position = 0.6
    colorramp_1.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    group_11 = nw.new_node(
        nodegroup_cracked_with_mask().name,
        input_kwargs={"CrackMagnitude": crack_magnitude_1, "VornoiScale": 2.0},
    )

    group_12 = nw.new_node(
        nodegroup_cracked_with_mask().name,
        input_kwargs={
            "NoiseScale": 3.0,
            "CrackMagnitude": crack_magnitude_2,
            "VornoiScale": 3.0,
        },
    )

    add_1 = nw.new_node(Nodes.VectorMath, input_kwargs={0: group_11, 1: group_12})

    multiply_3 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: colorramp_1.outputs["Color"], 1: add_1.outputs["Vector"]},
        attrs={"operation": "MULTIPLY"},
    )

    warped_position = nw.add(
        position_1,
        nw.multiply(
            nw.new_node(
                Nodes.NoiseTexture,
                input_kwargs={
                    "Vector": position_1,
                    "Scale": nw.new_value(1, "stripe_warp_scale"),
                }
            ),
            nw.new_value(0.2, "stripe_warp_mag"),
        )
    )
    
    offset2 = nw.add(
        multiply_3,
        nw.multiply(
            nw.new_node(Nodes.WaveTexture, input_kwargs={
                "Vector": warped_position,
                "Scale": nw.new_value(20, "stripe_scale"),
            }, attrs={
                "bands_direction": "Z",
                "wave_profile": "SAW",
            }),
            nw.new_value(0.005, "stripe_mag"),
            normal,
        )
    )
    
    noise_params = {"scale": ("uniform", 10, 20), "detail": 9, "roughness": 0.6, "zscale": ("log_uniform", 0.05, 0.1)}

    offset = nw.add(
        geo_MOUNTAIN_general(nw, 3, noise_params, 0, {}, {}),
        offset,
        offset1,
        offset2,
    )

    if selection is not None:
        offset = nw.new_node(
            Nodes.VectorMath,
            [offset, surface.eval_argument(nw, selection)],
            attrs={"operation": "MULTIPLY"},
        )
    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": group_input.outputs["Geometry"],
            "Offset": offset,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput, input_kwargs={"Geometry": set_position}
    )


def apply(obj, selection=None, **kwargs):
    if not isinstance(obj, list):
        obj = [obj]
    if not len(obj):
        return
    geomod_args = {"is_rock": max(obj[0].dimensions) < 5}
    geomod_args.update(kwargs)
    surface.add_geomod(
        obj,
        geometry_sandstone,
        selection=selection,
        attributes=[],
        input_kwargs=geomod_args,
    )
    surface.add_material(obj, shader, selection=selection)
