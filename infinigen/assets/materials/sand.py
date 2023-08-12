# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma
# Acknowledgement: This file draws inspiration from https://www.youtube.com/watch?v=y02x-p_0wP0 by Sam Bowman


import gin
from mathutils import Vector

from infinigen.core.nodes.node_wrangler import Nodes
from infinigen.core import surface
from infinigen.core.util.organization import SurfaceTypes
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import random_general as rg

type = SurfaceTypes.SDFPerturb
mod_name = "geo_SAND"
name = "sand"

@gin.configurable('shader')
def shader_SAND(
        nw,
        color=("palette", "desert"),
        random_seed=0,
        wet=False,
        wet_part=("uniform", 0.2, 0.25),
        *args,
        **kwargs
    ):
    nw.force_input_consistency()

    with FixedSeed(random_seed):
        position = (nw.new_node("ShaderNodeTexCoord", []), 3)
        assert(color is not None)
        if wet:
            position = nw.new_node('ShaderNodeNewGeometry')
            factor = nw.scalar_divide(nw.separate(position)[2], 3) # this needs to be consistent with value in coast.gin
            factor = nw.scalar_add(factor, -0.5, nw.new_node(Nodes.NoiseTexture, input_kwargs={"Scale": 0.1}))
            sand_color = nw.new_node(Nodes.ColorRamp, [factor])
            sand_color.color_ramp.elements[0].position = rg(wet_part)
            sand_color.color_ramp.elements[0].color = rg(("color_category", "wet_sand"))
            sand_color.color_ramp.elements[1].position = sand_color.color_ramp.elements[0].position + 0.11
            sand_color.color_ramp.elements[1].color = rg(("color_category", "dry_sand"))
            roughness = nw.new_node(Nodes.ColorRamp, [factor])
            roughness.color_ramp.elements[0].position = sand_color.color_ramp.elements[0].position / 2
            roughness.color_ramp.elements[0].color = (0.1, 0.1, 0.1, 0.1)
            roughness.color_ramp.elements[1].position = sand_color.color_ramp.elements[1].position
            roughness.color_ramp.elements[1].color = (1, 1, 1, 1)
        else:
            sand_color = tuple(rg(color))
            roughness = 1.0
        bsdf_sand = nw.new_node("ShaderNodeBsdfPrincipled", input_kwargs={
            "Base Color": sand_color,
            "Roughness": roughness,
        })
    return bsdf_sand

@gin.configurable('geo')
def geo_SAND(nw,
    n_waves=3,
    wave_scale=("log_uniform", 0.2, 4),
    wave_distortion=4,
    noise_scale=125,
    noise_detail=9, # tune down if there are numerical spikes
    noise_roughness=0.9,
    selection=None,
):
    nw.force_input_consistency()
    normal = nw.new_node("GeometryNodeInputNormal", [])
    position = nw.new_node("GeometryNodeInputPosition", [])

    offsets = []
    for i in range(n_waves):
        wave_scale_node = nw.new_value(rg(wave_scale), f"wave_scale_{i}")
        
        
        position_shift0 = nw.new_node(Nodes.Vector, label=f"position_shift_0_{i}")
        position_shift0.vector = nw.get_position_translation_seed(f"position_shift_0_{i}")
        position_shift1 = nw.new_node(Nodes.Vector, label=f"position_shift_1_{i}")
        position_shift1.vector = nw.get_position_translation_seed(f"position_shift_1_{i}")
        position_shift2 = nw.new_node(Nodes.Vector, label=f"position_shift_2_{i}")
        position_shift2.vector = nw.get_position_translation_seed(f"position_shift_2_{i}")
        position_shift3 = nw.new_node(Nodes.Vector, label=f"position_shift_3_{i}")
        position_shift3.vector = nw.get_position_translation_seed(f"position_shift_3_{i}")

        mag = nw.power(1e5, nw.scalar_sub(nw.new_node(Nodes.NoiseTexture, input_kwargs={
            "Vector": nw.add(position, position_shift3),
            "Scale": 0.1,
        }), 0.6))
        mag.use_clamp = 1
        offsets.append(nw.multiply(
            nw.add(
                nw.new_node(Nodes.WaveTexture, [
                    nw.add(
                        position,
                        position_shift0,
                        (nw.new_node(Nodes.NoiseTexture, input_kwargs={
                            "Scale": nw.new_value(1, "warp_scale"),
                            "Detail": nw.new_value(9, "warp_detail"),
                        }), 1),
                    ),
                    wave_scale_node,
                    wave_distortion
                ]),
                nw.new_node(Nodes.WaveTexture, [
                    nw.add(position, position_shift1),
                    nw.scalar_multiply(wave_scale_node, 0.98),
                    wave_distortion
                ]),
                nw.multiply(
                    nw.new_node(Nodes.NoiseTexture, [
                        nw.add(position, position_shift2),
                        None,
                        noise_scale,
                        noise_detail,
                        noise_roughness
                    ]),
                    Vector([1] * 3),
                )
            ),
            normal,
            mag,
            Vector([0.01] * 3)
        ))
    offset = nw.add(*offsets)
    groupinput = nw.new_node(Nodes.GroupInput)
    if selection is not None:
        offset = nw.multiply(offset, surface.eval_argument(nw, selection))
    set_position = nw.new_node(Nodes.SetPosition, input_kwargs={"Geometry": groupinput,  "Offset": offset})
    groupoutput = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': set_position})




def apply(objs, selection=None, **kwargs):
    surface.add_geomod(objs, geo_SAND, selection=selection)
    surface.add_material(objs, shader_SAND, selection=selection, 
        input_kwargs={"obj": objs[0] if isinstance(objs, list) else objs})