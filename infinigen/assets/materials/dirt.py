# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Mingzhe Wang
# Acknowledgement: This file draws inspiration from https://www.youtube.com/watch?v=pEXHCsrTsco

import os

import bpy
import gin
from infinigen.core.nodes.node_wrangler import Nodes
from numpy.random import uniform
from infinigen.core import surface
from infinigen.assets.materials.utils.surface_utils import sample_color, sample_ratio
from infinigen.core.util.organization import SurfaceTypes
from infinigen.core.util.math import FixedSeed

from .mountain import geo_MOUNTAIN_general

type = SurfaceTypes.SDFPerturb
mod_name = "geo_dirt"
name = "dirt"


def shader_dirt(nw):
    nw.force_input_consistency()
    dirt_base_color, dirt_roughness = geo_dirt(nw, selection=None, geometry=False)
    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": dirt_base_color,
            "Roughness": dirt_roughness,
        },
    )

    return principled_bsdf

@gin.configurable
def geo_dirt(nw, selection=None, random_seed=0, geometry=True):
    nw.force_input_consistency()
    if nw.node_group.type == "SHADER":
        position = nw.new_node('ShaderNodeNewGeometry')
        normal = (nw.new_node('ShaderNodeNewGeometry'), 1)
    else:
        position = nw.new_node(Nodes.InputPosition)
        normal = nw.new_node(Nodes.InputNormal)

    with FixedSeed(random_seed):
        # density of cracks, lower means cracks are present in smaller area
        dens_crack = uniform(0, 0.1)
        # scale cracks
        scal_crack = uniform(5, 15)
        # width of the crack
        widt_crack = uniform(0.01, 0.05)

        scale = 0.5

        noise_texture = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={
                "Vector": position,
                "Scale": 8.0 * scale,
                "Detail": 16.0,
                "W": nw.new_value(uniform(0, 10), "noise_texture_w"),
            },
            attrs={"noise_dimensions": "4D"},
        )

        noise_texture_2 = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={"Vector": position, "Scale": 5.0 * scale, "W": nw.new_value(uniform(0, 10), "noise_texture_2_w")},
            attrs={"noise_dimensions": "4D"},
        )

        colorramp_2 = nw.new_node(Nodes.MapRange,
            input_kwargs={"Value": noise_texture_2.outputs["Fac"], 1: nw.new_value(0.445 + (2 * dens_crack) - 0.1, "colorramp_2_a"), 2: nw.new_value(0.505 + (2 * dens_crack) - 0.1, "colorramp_2_b"), 3: 0.0, 4: 1.0},
            attrs={'clamp': True}
        )

        #nw.new_node(
        #    Nodes.ColorRamp, input_kwargs={"Fac": noise_texture_2.outputs["Fac"]},
        #    label = "color_ramp_2_VAR"
        #)
        #colorramp_2.color_ramp.elements[0].position = 0.445 + (2 * dens_crack) - 0.1
        #colorramp_2.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
        #colorramp_2.color_ramp.elements[1].position = 0.505 + (2 * dens_crack) - 0.1
        #colorramp_2.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

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

        voronoi_texture = nw.new_node(
            Nodes.VoronoiTexture,
            input_kwargs={"Vector": noise_texture_1.outputs["Color"], "Scale": nw.new_value(scal_crack * scale, "scal_crack")},
            attrs={"feature": "DISTANCE_TO_EDGE"},
        )

        colorramp_1 = nw.new_node(Nodes.MapRange,
            input_kwargs={"Value":voronoi_texture.outputs["Distance"], 1: 0.0, 2: nw.new_value(widt_crack, "colorramp_1"), 3: 0.0, 4: 1.0},
            # label = "color_ramp_1_VAR",
            attrs={'clamp': True}
        )
        
        #nw.new_node(
        #    Nodes.ColorRamp, input_kwargs={"Fac": voronoi_texture.outputs["Distance"]},
        #    label="color_ramp_1_VAR"
        #)
        #colorramp_1.color_ramp.elements[0].position = 0.0
        #colorramp_1.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
        #colorramp_1.color_ramp.elements[1].position = widt_crack
        #colorramp_1.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

        mix_sub = nw.new_node(Nodes.VectorMath,
            input_kwargs={0: (1.0, 1.0, 1.0), 1: colorramp_2.outputs["Result"]},
            attrs={"operation": "SUBTRACT"},
        )

        mix_mul1 = nw.new_node(Nodes.VectorMath,
            input_kwargs={0: mix_sub.outputs["Vector"], 1: colorramp_1.outputs["Result"]},
            attrs={"operation": "MULTIPLY"},
        )

        mix_mul2 = nw.new_node(Nodes.VectorMath,
            input_kwargs={0: colorramp_2.outputs["Result"], 1: (0.5, 0.5, 0.5)},
            attrs={"operation": "MULTIPLY"},
        )

        mix = nw.new_node(Nodes.VectorMath,
            input_kwargs={0: mix_mul1.outputs["Vector"], 1: mix_mul2.outputs["Vector"]},
        )

        #nw.new_node(
        #    Nodes.MixRGB,
        #    input_kwargs={
        #        "Fac": colorramp_2.outputs["Color"],
        #        "Color1": colorramp_1.outputs["Color"],
        #    },
        #)

        vector_math_2 = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={0: mix, 1: normal},
            attrs={"operation": "MULTIPLY"},
        )

        value_2 = nw.new_node(Nodes.Value)
        value_2.outputs["Value"].default_value = 0.5

        vector_math_5 = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={0: vector_math_2.outputs["Vector"], 1: value_2},
            attrs={"operation": "MULTIPLY"},
        )

        value_3 = nw.new_node(Nodes.Value)
        value_3.outputs["Value"].default_value = 0.08

        vector_math_8 = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={0: vector_math_5.outputs["Vector"], 1: value_3},
            attrs={"operation": "MULTIPLY"},
        )
        
        noise_texture_3 = nw.new_node(Nodes.NoiseTexture,
            input_kwargs={'Vector': position, "W": nw.new_value(uniform(0, 10), "noise_texture_3_w"), 'Scale': sample_ratio(5, 3/4, 4/3)},
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
            input_kwargs={'Vector': position, 'Scale': sample_ratio(20, 3/4, 4/3), "W": nw.new_value(uniform(0, 10), "noise_texture_4_w")},
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
        
        colorramp = nw.new_node(
            Nodes.ColorRamp, input_kwargs={"Fac": noise_texture.outputs["Fac"]}
        )
        colorramp.color_ramp.elements.new(1)
        colorramp.color_ramp.elements[0].position = 0.223
        colorramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
        colorramp.color_ramp.elements[1].position = 0.71
        colorramp.color_ramp.elements[1].color = (0.18, 0.075, 0.05, 1.0)
        colorramp.color_ramp.elements[2].position = 1.0
        colorramp.color_ramp.elements[2].color = (0.19, 0.03, 0.02, 1.0)
        sample_color(colorramp.color_ramp.elements[1].color, offset=0.05)
        sample_color(colorramp.color_ramp.elements[2].color, offset=0.05)
        
        dirt_base_color = nw.new_node(
            Nodes.MixRGB,
            input_kwargs={
                "Fac": mix,
                "Color1": (0.0, 0.0, 0.0, 1.0),
                "Color2": colorramp.outputs["Color"],
            },
        )

        colorramp_3 = nw.new_node(
            Nodes.ColorRamp, input_kwargs={"Fac": noise_texture.outputs["Fac"]}
        )
        colorramp_3.color_ramp.elements[0].position = 0.08
        colorramp_3.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
        colorramp_3.color_ramp.elements[1].position = 0.768
        colorramp_3.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
        
        dirt_roughness = colorramp_3

        offset = nw.add(multiply_11, multiply_9, vector_math_8)

    if geometry:  
        noise_params = {"scale": ("uniform", 1, 5), "detail": 7, "roughness": 0.7, "zscale": ("power_uniform", -1, -0.5)}
        offset = nw.add(
            geo_MOUNTAIN_general(nw, 3, noise_params, 0, {}, {}),
            offset
        )
        groupinput = nw.new_node(Nodes.GroupInput)
        if selection is not None:
            offset = nw.multiply(offset, surface.eval_argument(nw, selection))
        set_position = nw.new_node(Nodes.SetPosition, input_kwargs={"Geometry": groupinput,  "Offset": offset})
        nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': set_position})
    else:
        return dirt_base_color, dirt_roughness



def apply(obj, selection=None, **kwargs):
    surface.add_geomod(
        obj,
        geo_dirt,
        selection=selection,
    )
    surface.add_material(obj, shader_dirt, selection=selection)


if __name__ == "__main__":
    mat = 'dirt'
    if not os.path.isdir(os.path.join('outputs', mat)):
        os.mkdir(os.path.join('outputs', mat))
    for i in range(10):
        bpy.ops.wm.open_mainfile(filepath='landscape_surface_dev.blend')
        apply(bpy.data.objects['Plane.002'])
        bpy.context.scene.render.filepath = os.path.join('outputs', mat, '%s_%d.jpg'%(mat, i))
        bpy.context.scene.render.image_settings.file_format='JPEG'
        bpy.ops.render.render(write_still=True)
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join('outputs', mat, 'landscape_surface_dev_dirt.blend'))
        