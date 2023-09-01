# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Mingzhe Wang


import os, sys
import numpy as np
import math as ma
from infinigen.assets.materials.utils.surface_utils import clip, sample_range, sample_ratio, sample_color, geo_voronoi_noise
import bpy
import mathutils
from numpy.random import uniform, normal, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface

def shader_wood(nw: NodeWrangler, rand=False, **input_kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    texture_coordinate_1 = nw.new_node(Nodes.TextureCoord)
    
    mapping_2 = nw.new_node(Nodes.Mapping,
        input_kwargs={'Vector': texture_coordinate_1.outputs["Generated"], 'Rotation': uniform(0,ma.pi*2, 3)})
    
    mapping_1 = nw.new_node(Nodes.Mapping,
        input_kwargs={'Vector': mapping_2, 'Scale': (0.5, sample_range(2, 4) if rand else 3, 0.5)})
    
    musgrave_texture_2 = nw.new_node(Nodes.MusgraveTexture,
        input_kwargs={'Vector': mapping_1, 'Scale': 2.0},
        attrs={'musgrave_dimensions': '4D'})
    if rand:
        musgrave_texture_2.inputs['W'].default_value = sample_range(0, 5)
        musgrave_texture_2.inputs['Scale'].default_value = sample_ratio(2.0, 3/4, 4/3)
    
    noise_texture_1 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': musgrave_texture_2, 'W': 0.7, 'Scale': 10.0},
        attrs={'noise_dimensions': '4D'})
    if rand:
        noise_texture_1.inputs['W'].default_value = sample_range(0, 5)
        noise_texture_1.inputs['Scale'].default_value = sample_ratio(5, 0.5, 2)
    
    colorramp_2 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': noise_texture_1.outputs["Fac"]})
    colorramp_2.color_ramp.elements.new(0)
    colorramp_2.color_ramp.elements[0].position = 0.1727
    colorramp_2.color_ramp.elements[0].color = (0.1567, 0.0162, 0.0017, 1.0)
    colorramp_2.color_ramp.elements[1].position = 0.4364
    colorramp_2.color_ramp.elements[1].color = (0.2908, 0.1007, 0.0148, 1.0)
    colorramp_2.color_ramp.elements[2].position = 0.5864
    colorramp_2.color_ramp.elements[2].color = (0.0814, 0.0344, 0.0125, 1.0)
    if rand:
        colorramp_2.color_ramp.elements[0].position += sample_range(-0.05, 0.05)
        colorramp_2.color_ramp.elements[1].position += sample_range(-0.1, 0.1)
        colorramp_2.color_ramp.elements[2].position += sample_range(-0.05, 0.05)
        for e in colorramp_2.color_ramp.elements:
            sample_color(e.color, offset=0.03)

    colorramp_4 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': noise_texture_1.outputs["Fac"]})
    colorramp_4.color_ramp.elements[0].position = 0.0
    colorramp_4.color_ramp.elements[0].color = (0.4855, 0.4855, 0.4855, 1.0)
    colorramp_4.color_ramp.elements[1].position = 1.0
    colorramp_4.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    principled_bsdf_1 = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': colorramp_2.outputs["Color"], 'Roughness': colorramp_4.outputs["Color"]},
        attrs={'subsurface_method': 'BURLEY'})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf_1})

def apply(obj, geo_kwargs=None, shader_kwargs=None, **kwargs):
    surface.add_material(obj, shader_wood, reuse=False, input_kwargs=shader_kwargs)


if __name__ == "__main__":
    mat = 'wood'
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