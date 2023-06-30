import math

import gin
from mathutils import Vector
from nodes.node_wrangler import Nodes
from numpy.random import uniform
from terrain.utils import SurfaceTypes, drive_param
from util.math import FixedSeed
from util.random import random_color_neighbour

type = SurfaceTypes.BlenderDisplacement
mod_name = "lava_geo"
name = "lava"

def nodegroup_polynomial_base(nw):
@node_utils.to_nodegroup("nodegroup_polynomial", singleton=False)
def nodegroup_polynomial_geo(nw):
    nodegroup_polynomial_base(nw)
@node_utils.to_nodegroup("nodegroup_polynomial", singleton=False, type='ShaderNodeTree')
def nodegroup_polynomial_shader(nw):
    nodegroup_polynomial_base(nw)


def lava_shader(nw):
    nw.force_input_consistency()
    lava_dir = lava_geo(nw, geometry=False)
        input_kwargs={"Vector": noise_texture_2.outputs["Fac"], "Scale": 10.0},
    drive_param(voronoi_texture.inputs["W"], scale=0.003, offset=uniform(0, 10))

        input_kwargs={"Vector": noise_texture_3.outputs["Fac"], "Scale": 10.0},
    drive_param(voronoi_texture_1.inputs["W"], scale=0.003, offset=uniform(0, 10))

    add = nw.new_node(Nodes.Math, input_kwargs={0: mix, 1: lava_dir})
        "ShaderNodeInvert", input_kwargs={"Color": lava_dir}

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
        # group_3 = nw.new_node(
        #     nodegroup_normalize_0_1().name,
        #     input_kwargs={
        #         "Attribute": separate_xyz.outputs["X"],
        #         "Geometry": group_input.outputs["Geometry"],
        #     },
        # )
        group_3 =  nw.scalar_divide(
            nw.scalar_add(
                separate_xyz.outputs["X"],
                200
            ),
            400
        )
        # group_4 = nw.new_node(
        #     nodegroup_normalize_0_1().name,
        #     input_kwargs={
        #         "Attribute": separate_xyz.outputs["Y"],
        #         "Geometry": group_input.outputs["Geometry"],
        #     },
        # )
        group_4 =  nw.scalar_divide(
            nw.scalar_add(
                separate_xyz.outputs["Y"],
                200
            ),
            400
        )
        # group = nw.new_node(
        #     nodegroup_normalize_0_1().name,
        #     input_kwargs={
        #         "Attribute": separate_xyz.outputs["Z"],
        #         "Geometry": group_input.outputs["Geometry"],
        #     },
        # )
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
        # group_1 = nw.new_node(
        #     nodegroup_normalize_0_1().name,
        #     input_kwargs={
        #         "Attribute": multiply_add,
        #         "Geometry": group_input.outputs["Geometry"],
        #     },
        # )
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
        obj, lava_geo, selection=selection,
    surface.add_material(obj, lava_shader, selection=selection)
