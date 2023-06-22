# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Ankit Goyal, Zeyu Ma
# Date Signed: June 5 2023

import gin
from numpy.random import uniform

from nodes.node_wrangler import Nodes
from surfaces import surface
from terrain.utils import SurfaceTypes
from util.math import FixedSeed
from util.random import random_color_neighbour

type = SurfaceTypes.SDFPerturb
mod_name = "geo_ice"
name = "ice"

def shader_ice(nw, **kwargs):
    nw.force_input_consistency()
    roughness = geo_ice(nw, geometry=False)
    # color
    col_ice = random_color_neighbour((0.6795, 0.8148, 1.0, 1.0), 0.05, 0.1, 0.1)
    # tranmission
    tra = 0 #uniform(0.75, 0.95) if uniform() > 0.5 else uniform(0.95, 1)

    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            # "Base Color": tuple(col_ice),
            "Subsurface Radius": (0.1, 0.1, 0.2),
            "Roughness": roughness,
            "IOR": 1.31,
            "Transmission": tra,
            "Subsurface": 1,
            "Subsurface Color": tuple(col_ice),
        },
    )
    
    return principled_bsdf

@gin.configurable
def geo_ice(nw, selection=None, random_seed=0,  geometry=True, **kwargs):
    nw.force_input_consistency()
    if nw.node_group.type == "SHADER":
        position = nw.new_node('ShaderNodeNewGeometry')
        normal = (nw.new_node('ShaderNodeNewGeometry'), 1)
    else:
        position = nw.new_node(Nodes.InputPosition)
        normal = nw.new_node(Nodes.InputNormal)

    with FixedSeed(random_seed):
        # how rough the geometry of the surface is
        sur_rou = nw.new_value(uniform(0.02, 0.1), "sur_rou")
        # distortion on the surface of the ice
        sur_dis = nw.new_value(uniform(0, 1) if (uniform() < 0.5) else uniform(1, 5), "sur_dis")
        # percentage of the surface that is uneven
        sur_une = uniform(-0.3, 0) if (uniform() < 0.5) else uniform(0, 0.05)

        noise_texture = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={
                "W": nw.new_value(uniform(0, 10), "noise_texture_w"),
                "Vector": position,
                "Scale": 4.0,
                "Detail": 15.0,
            },
            attrs={"noise_dimensions": "4D"},
        )

        colorramp_1 = nw.new_node(
            Nodes.ColorRamp, input_kwargs={"Fac": noise_texture.outputs["Fac"]}, label="color_ramp_1_VAR"
        )
        colorramp_1.color_ramp.elements[0].position = 0.5 + sur_une
        colorramp_1.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
        colorramp_1.color_ramp.elements[1].position = 0.766
        colorramp_1.color_ramp.elements[1].color = (0.1329, 0.1329, 0.1329, 1.0)

        vector_math = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={0: colorramp_1.outputs["Color"], 1: normal},
            attrs={"operation": "MULTIPLY"},
        )

        value = nw.new_node(Nodes.Value)
        value.outputs["Value"].default_value = 0.2

        vector_math_1 = nw.new_node(
            Nodes.VectorMath, input_kwargs={0: vector_math.outputs["Vector"], 1: value}
        )

        noise_texture_1 = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={
                "W": nw.new_value(uniform(0, 10), "noise_texture_1_w"),
                "Vector": position,
                "Scale": 1.0,
                "Detail": 15.0,
                "Roughness": 0.8,
                "Distortion": sur_dis,
            },
            attrs={"noise_dimensions": "4D"},
        )

        vector_math_3 = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={0: noise_texture_1.outputs["Fac"], 1: normal},
            attrs={"operation": "MULTIPLY"},
        )

        value_1 = nw.new_node(Nodes.Value)
        value_1.outputs["Value"].default_value = 0.06

        vector_math_2 = nw.new_node(
            Nodes.VectorMath, input_kwargs={0: vector_math_3.outputs["Vector"], 1: value_1}
        )

        vector_math_4 = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={
                0: vector_math_1.outputs["Vector"],
                1: vector_math_2.outputs["Vector"],
            },
        )

        offset = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={0: vector_math_4.outputs["Vector"], 1: sur_rou},
            attrs={"operation": "MULTIPLY"},
        )

        colorramp = nw.new_node(
            Nodes.ColorRamp, input_kwargs={"Fac": noise_texture.outputs["Fac"]}, label="color_ramp_VAR"
        )
        colorramp.color_ramp.elements[0].position = 0.5 + sur_une
        colorramp.color_ramp.elements[0].color = (0.0844, 0.0844, 0.0844, 1.0)
        colorramp.color_ramp.elements[1].position = 0.766
        colorramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

        roughness = colorramp

    if geometry:  
        groupinput = nw.new_node(Nodes.GroupInput)
        if selection is not None:
            offset = nw.multiply(offset, surface.eval_argument(nw, selection))
        set_position = nw.new_node(Nodes.SetPosition, input_kwargs={"Geometry": groupinput,  "Offset": offset})
        nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': set_position})
    else:
        return roughness


def apply(obj, selection=None, **kwargs):
    surface.add_geomod(obj, geo_ice, selection=selection)
    surface.add_material(obj, shader_ice, selection=selection)

