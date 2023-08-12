# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Ankit Goyal, Zeyu Ma


# Code generated using version v2.0.0 of the node_transpiler
import math

import gin
from mathutils import Vector
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes
from numpy.random import uniform
from infinigen.core import surface
from infinigen.core.util.organization import SurfaceTypes
from infinigen.terrain.utils import drive_param
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import random_color_neighbour

type = SurfaceTypes.BlenderDisplacement
mod_name = "lava_geo"
name = "lava"

def nodegroup_polynomial_base(nw):
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

@node_utils.to_nodegroup("nodegroup_polynomial", singleton=False)
def nodegroup_polynomial_geo(nw):
    nodegroup_polynomial_base(nw)

@node_utils.to_nodegroup("nodegroup_polynomial", singleton=False, type='ShaderNodeTree')
def nodegroup_polynomial_shader(nw):
    nodegroup_polynomial_base(nw)


def lava_shader(nw):
    nw.force_input_consistency()
    lava_dir = lava_geo(nw, geometry=False)

    # rock brightness
    rock_col = random_color_neighbour((0.02, 0.02, 0.02, 1), 1, 0.02, 0.03)
    # rock roughness
    rock_rou = uniform(0.6, 0.9)
    # vornoi noises col ramps
    vor_0_cr_0 = uniform(0.01, 0.03)
    vor_0_cr_1 = uniform(0.1, 0.15)
    vor_1_cr_0 = uniform(0.0, 0.01)
    vor_1_cr_1 = uniform(0.25, 0.45)
    # amount of rock, inversely proportional
    amo_roc = uniform(0.01, 0.15)
    # lava emission
    lava_emi = uniform(20, 60)
    # min lava temp
    min_lava_temp = uniform(1000, 1500)
    # max lava temp
    max_lava_temp = min_lava_temp + uniform(0, 1000)

    # print(f"{amo_roc=}")
    # print(f"{vor_0_cr_0=} {vor_0_cr_1=} {vor_1_cr_0=} {vor_1_cr_1=}")
    # print(f"{lava_emi=} {min_lava_temp=} {max_lava_temp=}")

    noise_texture_2 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={"Detail": 16.0, "Distortion": 2.0, "W": uniform(0, 10)},
        attrs={"noise_dimensions": "4D"},
    )

    voronoi_texture = nw.new_node(
        Nodes.VoronoiTexture,
        input_kwargs={"Vector": noise_texture_2.outputs["Fac"], "Scale": 10.0},
        attrs={"voronoi_dimensions": "4D", "feature": "DISTANCE_TO_EDGE"},
    )
    drive_param(voronoi_texture.inputs["W"], scale=0.003, offset=uniform(0, 10))


    colorramp_1 = nw.new_node(
        Nodes.ColorRamp, input_kwargs={"Fac": voronoi_texture.outputs["Distance"]}
    )
    colorramp_1.color_ramp.elements[0].position = vor_0_cr_0
    colorramp_1.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_1.color_ramp.elements[1].position = vor_0_cr_1
    colorramp_1.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    noise_texture_3 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={"W": uniform(0, 10), "Distortion": 2.0},
        attrs={"noise_dimensions": "4D"},
    )

    voronoi_texture_1 = nw.new_node(
        Nodes.VoronoiTexture,
        input_kwargs={"Vector": noise_texture_3.outputs["Fac"], "Scale": 10.0},
        attrs={"voronoi_dimensions": "4D", "feature": "DISTANCE_TO_EDGE"},
    )
    drive_param(voronoi_texture_1.inputs["W"], scale=0.003, offset=uniform(0, 10))


    colorramp_2 = nw.new_node(
        Nodes.ColorRamp, input_kwargs={"Fac": voronoi_texture_1.outputs["Distance"]}
    )
    colorramp_2.color_ramp.elements[0].position = vor_1_cr_0
    colorramp_2.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_2.color_ramp.elements[1].position = vor_1_cr_1
    colorramp_2.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    mix_1 = nw.new_node(
        Nodes.MixRGB,
        input_kwargs={
            "Color1": colorramp_1.outputs["Color"],
            "Color2": colorramp_2.outputs["Color"],
        },
    )

    ambient_occlusion_1 = nw.new_node("ShaderNodeAmbientOcclusion")

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: ambient_occlusion_1.outputs["Color"],
            # determines how strong the small scale noise are
            # this makes the lava look turbulent
            1: 0 if uniform() < 0.2 else uniform(0.0, 0.5)},
        attrs={"operation": "SUBTRACT"},
    )

    mix = nw.new_node(
        Nodes.MixRGB,
        input_kwargs={
            "Fac": mix_1,
            "Color1": subtract,
            "Color2": ambient_occlusion_1.outputs["Color"],
        },
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: mix, 1: lava_dir})

    divide = nw.new_node(
        Nodes.Math, input_kwargs={0: add, 1: 2.0}, attrs={"operation": "DIVIDE"}
    )

    colorramp = nw.new_node(Nodes.ColorRamp, input_kwargs={"Fac": divide})
    colorramp.color_ramp.elements[0].position = 0.0
    colorramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp.color_ramp.elements[1].position = 0.85 + amo_roc
    colorramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    invert = nw.new_node(
        "ShaderNodeInvert", input_kwargs={"Color": lava_dir}
    )

    multiply = nw.new_node(
        Nodes.Math, input_kwargs={0: invert, 1: max_lava_temp - min_lava_temp}, attrs={"operation": "MULTIPLY"}
    )

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: min_lava_temp, 1: multiply})

    blackbody_1 = nw.new_node(
        "ShaderNodeBlackbody", input_kwargs={"Temperature": add_1}
    )

    noise_emission = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={"W": uniform(0, 10), "Scale": 0.5}
    )
    
    strength_emission = nw.new_node(Nodes.Math, input_kwargs={0: noise_emission.outputs["Fac"], 1: lava_emi})

    emission_1 = nw.new_node(
        "ShaderNodeEmission", input_kwargs={"Color": blackbody_1, "Strength":
                                            strength_emission}
    )

    noise_bsdf = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={"W": uniform(0, 10), "Scale": 0.5,
        "Detail": 10.0},
        attrs={"noise_dimensions": "4D"},
    )

    color_bsdf = nw.new_node(Nodes.ColorRamp, input_kwargs={"Fac": noise_bsdf.outputs["Fac"]})

    color_bsdf.color_ramp.elements[0].position = 0.0
    color_bsdf.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    color_bsdf.color_ramp.elements[1].position = 1.0
    color_bsdf.color_ramp.elements[1].color = rock_col

    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={"Base Color": color_bsdf, "Roughness": rock_rou},
    )

    mix_shader = nw.new_node(
        Nodes.MixShader,
        input_kwargs={
            "Fac": colorramp.outputs["Color"],
            1: emission_1,
            2: principled_bsdf,
        },
    )


    return mix_shader

@gin.configurable
def lava_geo(nw, selection=None, random_seed=0, geometry=True):
    nw.force_input_consistency()
    if nw.node_group.type == "SHADER":
        position = nw.new_node('ShaderNodeNewGeometry')
        # normal = (nw.new_node('ShaderNodeNewGeometry'), 1)
    else:
        position = nw.new_node(Nodes.InputPosition)
        # normal = nw.new_node(Nodes.InputNormal)
        normal = Vector([0, 0, 1])

    with FixedSeed(random_seed):
        # scale wave
        wave_sca = nw.new_value(uniform(3.5, 4.5), "wave_sca")
        # direction of wave
        dir_x = uniform(-2, 2)
        dir_y = nw.new_value(math.sqrt(5 - (dir_x ** 2)), "dir_y")
        dir_x = nw.new_value(dir_x, "dir_x")
        # print(f"{wave_sca=} {dir_x=} {dir_y=}")

        group_input = nw.new_node(
            Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
        )

        noise_texture_1 = nw.new_node(
            Nodes.NoiseTexture,
            attrs={"noise_dimensions": "4D"},
        )
        drive_param(noise_texture_1.inputs["W"], 0.01)

        separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position})

        group_3 =  nw.scalar_divide(
            nw.scalar_add(
                separate_xyz.outputs["X"],
                200
            ),
            400
        )

        group_4 =  nw.scalar_divide(
            nw.scalar_add(
                separate_xyz.outputs["Y"],
                200
            ),
            400
        )

        group =  nw.scalar_divide(
            nw.scalar_add(
                separate_xyz.outputs["Z"],
                0
            ),
            20
        )

        group_2 = nw.new_node(
            nodegroup_polynomial_geo().name if nw.node_group.type != "SHADER" else nodegroup_polynomial_shader().name,
            input_kwargs={
                "X": group_3,
                "Y": group_4,
                "Z": group,
                "alpha_x": dir_x,
                "alpha_y": dir_y,
                "alpha_z": 1.0,
                "pow_x": 2.0,
                "pow_y": 2.0,
                "pow_z": 2.0,
            },
        )

        multiply_add = nw.new_node(
            Nodes.Math,
            input_kwargs={0: noise_texture_1.outputs["Fac"], 1: 0.2, 2: group_2},
            attrs={"operation": "MULTIPLY_ADD"},
        )

        group_1 =  nw.scalar_divide(
            nw.scalar_add(
                multiply_add,
                0
            ),
            3
        )

        noise_texture = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={
                "W": nw.new_value(uniform(0, 10), "noise_texture_w"),
                "Scale": 0.35,
                "Detail": 1.0,
                "Distortion": 5.0,
            },
            attrs={"noise_dimensions": "4D"},
        )

        value_3 = nw.new_node(Nodes.Value)
        value_3.outputs[0].default_value = 0.2

        multiply = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={0: noise_texture.outputs["Fac"], 1: value_3},
            attrs={"operation": "MULTIPLY"},
        )

        add = nw.new_node(
            Nodes.VectorMath, input_kwargs={0: group_1, 1: multiply.outputs["Vector"]}
        )

        wave_texture = nw.new_node(
            Nodes.WaveTexture,
            input_kwargs={
                "Vector": add.outputs["Vector"],
                "Scale": wave_sca,
                "Distortion": 1.0,
                "Detail": 0.0,
            },
        )

        float_curve = nw.new_node(Nodes.FloatCurve, input_kwargs={"Value": group_1})
        node_utils.assign_curve(
            float_curve.mapping.curves[0],
            [(0.0, 0.0), (0.25, 0.4937), (0.5818, 0.8625), (1.0, 1.0)],
        )

        value = nw.new_node(Nodes.Value)
        value.outputs[0].default_value = 0.05

        multiply_1 = nw.new_node(
            Nodes.Math,
            input_kwargs={0: float_curve, 1: value},
            attrs={"operation": "MULTIPLY"},
        )

        multiply_2 = nw.new_node(
            Nodes.Math,
            input_kwargs={0: wave_texture.outputs["Color"], 1: multiply_1},
            attrs={"operation": "MULTIPLY"},
        )

        voronoi_texture = nw.new_node(
            Nodes.VoronoiTexture,
            input_kwargs={"W": nw.new_value(uniform(0, 10), "voronoi_texture_w"), "Vector": position, "Scale": 1.0},
            attrs={"voronoi_dimensions": "4D", "feature": "SMOOTH_F1"},
        )

        value_1 = nw.new_node(Nodes.Value)
        value_1.outputs[0].default_value = 0.05

        multiply_3 = nw.new_node(
            Nodes.Math,
            input_kwargs={0: voronoi_texture.outputs["Distance"], 1: value_1},
            attrs={"operation": "MULTIPLY"},
        )

        add_1 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_2, 1: multiply_3})

        lava_dir = float_curve

    if geometry:
        offset = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={0: add_1, 1: normal},
            attrs={"operation": "MULTIPLY"},
        )
        groupinput = nw.new_node(Nodes.GroupInput)
        if selection is not None:
            offset = nw.multiply(offset, surface.eval_argument(nw, selection))
        set_position = nw.new_node(Nodes.SetPosition, input_kwargs={"Geometry": groupinput,  "Offset": offset})
        nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': set_position})
    else:
        return lava_dir


def apply(obj, selection=None, **kwargs):
    surface.add_geomod(
        obj, lava_geo, selection=selection,
    )
    surface.add_material(obj, lava_shader, selection=selection)
