# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Mingzhe Wang
# Acknowledgment: This file draws inspiration from https://www.youtube.com/watch?v=xWT_7jUTW4Q by Ryan King Art


import os

import bpy
import mathutils
from numpy.random import uniform, normal, randint
import gin
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface
from infinigen.assets.materials.utils.surface_utils import sample_color, sample_range, sample_ratio
from infinigen.core.util.organization import SurfaceTypes
from infinigen.core.util.math import FixedSeed
from .mountain import geo_MOUNTAIN_general


type = SurfaceTypes.SDFPerturb
mod_name = "geo_rocks"
name = "chunkyrock"

def shader_rocks(nw, rand=True, **input_kwargs):
    nw.force_input_consistency()
    position = nw.new_node('ShaderNodeNewGeometry')
    depth = geo_rocks(nw, geometry=False)
    
    colorramp_3 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': depth})
    colorramp_3.color_ramp.elements[0].position = 0.0285
    colorramp_3.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_3.color_ramp.elements[1].position = 0.1347
    colorramp_3.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    mapping = nw.new_node(Nodes.Mapping,
        input_kwargs={'Vector': position, 'Scale': (0.2, 0.2, 0.2)})
    
    noise_texture_1 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': mapping, 'Detail': 15.0})
    
    rock_color1 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': noise_texture_1.outputs["Fac"], 'Color1': (0.0, 0.0, 0.0, 1.0), 'Color2': (0.01, 0.024, 0.0283, 1.0)})

    if rand:
        sample_color(rock_color1.inputs[6].default_value)
        sample_color(rock_color1.inputs[7].default_value)

    noise_texture_2 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': mapping, 'Detail': 15.0})

    rock_color2 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': noise_texture_2.outputs["Fac"], 'Color1': (0.0, 0.0, 0.0, 1.0), 'Color2': (0.0694, 0.1221, 0.0693, 1.0)})

    if rand:
        sample_color(rock_color2.inputs[6].default_value)
        sample_color(rock_color2.inputs[7].default_value)

    mix_1 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp_3.outputs["Color"], 'Color1': rock_color1, 'Color2': rock_color2})
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': mix_1})

    return principled_bsdf

@gin.configurable
def geo_rocks(nw: NodeWrangler, rand=True, selection=None, random_seed=0, geometry=True, **input_kwargs):
    nw.force_input_consistency()
    if nw.node_group.type == "SHADER":
        position = nw.new_node('ShaderNodeNewGeometry')
        normal = (nw.new_node('ShaderNodeNewGeometry'), 1)
    else:
        position = nw.new_node(Nodes.InputPosition)
        normal = nw.new_node(Nodes.InputNormal)
    
    with FixedSeed(random_seed):
        # Code generated using version 2.4.3 of the node_transpiler
        
        noise_texture = nw.new_node(Nodes.NoiseTexture,
            input_kwargs={'Vector': position})
        
        mix = nw.new_node(Nodes.MixRGB,
            input_kwargs={'Fac': 0.8, 'Color1': noise_texture.outputs["Color"], 'Color2': position})
        
        if rand:
            sample_max = 2
            sample_min = 1/2
            voronoi_texture_scale = nw.new_value(sample_ratio(1, sample_min, sample_max), "voronoi_texture_scale")
            voronoi_texture_w = nw.new_value(sample_range(0, 5), "voronoi_texture_w")
        else:
            voronoi_texture_scale = 1.0
            voronoi_texture_w = 0
        voronoi_texture = nw.new_node(Nodes.VoronoiTexture,
            input_kwargs={'Vector': mix, 'Scale': voronoi_texture_scale, 'W': voronoi_texture_w},
            attrs={'feature': 'DISTANCE_TO_EDGE', 'voronoi_dimensions': '4D'})

        colorramp = nw.new_node(Nodes.ColorRamp,
            input_kwargs={'Fac': voronoi_texture.outputs["Distance"]},
            label="colorramp_VAR",
        )
        colorramp.color_ramp.elements[0].position = 0.0432
        colorramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
        colorramp.color_ramp.elements[1].position = 0.3
        colorramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
        if rand:
            colorramp.color_ramp.elements[0].position = sample_ratio(colorramp.color_ramp.elements[0].position, 0.5, 2)
            colorramp.color_ramp.elements[1].position = sample_ratio(colorramp.color_ramp.elements[1].position, 0.5, 2)

        depth = colorramp
        
        multiply = nw.new_node(Nodes.VectorMath,
            input_kwargs={0: colorramp.outputs["Color"], 1: normal},
            attrs={'operation': 'MULTIPLY'})
        
        value = nw.new_node(Nodes.Value)
        value.outputs[0].default_value = 0.4
        
        offset = nw.new_node(Nodes.VectorMath,
            input_kwargs={0: multiply.outputs["Vector"], 1: value},
            attrs={'operation': 'MULTIPLY'})
        
    
    if geometry:
        groupinput = nw.new_node(Nodes.GroupInput)
        noise_params = {"scale": ("uniform", 10, 20), "detail": 9, "roughness": 0.6, "zscale": ("log_uniform", 0.08, 0.12)}
        offset = nw.add(offset, geo_MOUNTAIN_general(nw, 3, noise_params, 0, {}, {}))
        if selection is not None:
            offset = nw.multiply(offset, surface.eval_argument(nw, selection))
        set_position = nw.new_node(Nodes.SetPosition, input_kwargs={"Geometry": groupinput,  "Offset": offset})
        nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': set_position})
    else:
        return depth


def apply(obj, selection=None, geo_kwargs=None, shader_kwargs=None, **kwargs):
    surface.add_geomod(obj, geo_rocks, selection=selection, input_kwargs=geo_kwargs)
    surface.add_material(obj, shader_rocks, selection=selection, input_kwargs=shader_kwargs)

if __name__ == "__main__":
    mat = 'rock'
    if not os.path.isdir(os.path.join('outputs', mat)):
        os.mkdir(os.path.join('outputs', mat))
    for i in range(10):
        bpy.ops.wm.open_mainfile(filepath='test.blend')
        apply(bpy.data.objects['SolidModel'], geo_kwargs={'rand':True}, shader_kwargs={'rand': True})
        #fn = os.path.join(os.path.abspath(os.curdir), 'giraffe_geo_test.blend')
        #bpy.ops.wm.save_as_mainfile(filepath=fn)
        bpy.context.scene.render.filepath = os.path.join('outputs', mat, '%s_%d.jpg'%(mat, i))
        bpy.context.scene.render.image_settings.file_format='JPEG'
        bpy.ops.render.render(write_still=True)