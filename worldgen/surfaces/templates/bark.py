# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Mingzhe Wang
# Date Signed: June 15 2023 

import os, sys
import numpy as np
import math as ma
from surfaces.surface_utils import clip, sample_range, sample_ratio, sample_color, geo_voronoi_noise
import bpy
import mathutils
from numpy.random import uniform, normal
from nodes.node_wrangler import Nodes, NodeWrangler
from surfaces import surface
import random

def shader_bark(nw, rand=False, **input_kwargs):

    texture_coordinate = nw.new_node(Nodes.TextureCoord)
    
    mapping = nw.new_node(Nodes.Mapping,
        input_kwargs={'Vector': texture_coordinate.outputs["Object"]})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': mapping, 'Detail': 16.0, 'Roughness': 0.62})
    if rand:
        sample_max = input_kwargs['noise_scale_max'] if 'noise_scale_max' in input_kwargs else 3
        sample_min = input_kwargs['noise_scale_min'] if 'noise_scale_min' in input_kwargs else 1/sample_max
        noise_texture.inputs["Scale"].default_value = sample_ratio(noise_texture.inputs["Scale"].default_value, sample_min, sample_max)

    colorramp_1 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': noise_texture.outputs["Fac"]})
    colorramp_1.color_ramp.elements[0].position = 0.627
    colorramp_1.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_1.color_ramp.elements[1].position = 0.63
    colorramp_1.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    attribute = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'offset'})
    
    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Color1': noise_texture.outputs["Fac"], 'Color2': attribute.outputs["Color"]},
        attrs={'blend_type': 'MULTIPLY'})
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': mix})
    colorramp.color_ramp.elements[0].position = 0.0
    colorramp.color_ramp.elements[0].color = (0.02, 0.0091, 0.0016, 1.0)
    colorramp.color_ramp.elements[1].position = 1.0
    colorramp.color_ramp.elements[1].color = (0.2243, 0.1341, 0.1001, 1.0)
    for e in colorramp.color_ramp.elements:
        sample_color(e.color)
        #print(e.color[0], e.color[1], e.color[2])

    mix_1 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp_1.outputs["Color"], 'Color1': colorramp.outputs["Color"], 'Color2': (0.0897, 0.052, 0.0149, 1.0)})
    if rand:
        for i in range(3):
            mix_1.inputs["Color2"].default_value[i] = (colorramp.color_ramp.elements[0].color[i] + colorramp.color_ramp.elements[1].color[i]) / 2

    colorramp_2 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': noise_texture.outputs["Fac"]})
    colorramp_2.color_ramp.elements[0].position = 0.0
    colorramp_2.color_ramp.elements[0].color = (0.5173, 0.5173, 0.5173, 1.0)
    colorramp_2.color_ramp.elements[1].position = 1.0
    colorramp_2.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': mix_1, 'Roughness': colorramp_2.outputs["Color"]},
        attrs={'subsurface_method': 'BURLEY'})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

def geo_bark(nw, rand=False, **input_kwargs):

    group_input = nw.new_node(Nodes.GroupInput)
    
    position = nw.new_node(Nodes.InputPosition)
    
    value = nw.new_node(Nodes.Value)
    value.outputs["Value"].default_value = input_kwargs.get('scale', 5)
    
    vector_math = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: position, 1: value},
        attrs={'operation': 'MULTIPLY'})
    
    wave_texture = nw.new_node(Nodes.WaveTexture,
        input_kwargs={'Vector': vector_math.outputs["Vector"], 'Scale': 0.5, 'Distortion': 7.0, 'Detail': 8.0, 'Detail Scale': 3.5})
    if rand:
        wave_texture.inputs['Scale'].default_value = sample_ratio(wave_texture.inputs['Scale'].default_value, 1/2, 2)
        wave_texture.inputs["Distortion"].default_value = sample_ratio(wave_texture.inputs['Distortion'].default_value, 0.7, 3)
        wave_texture.inputs["Detail Scale"].default_value = sample_ratio(wave_texture.inputs['Detail Scale'].default_value, 1/2, 2)
        wave_texture.inputs["Detail Roughness"].default_value = sample_range(0.3, 0.7)
        if random.random() < 0.5:
            wave_texture.wave_type = 'BANDS'
            wave_texture.bands_direction = random.choice(['X', 'Y', 'Z', 'DIAGONAL'])
        else:
            wave_texture.wave_type = 'RINGS'
            wave_texture.rings_direction = random.choice(['X', 'Y', 'Z', 'SPHERICAL'])    
        wave_texture.wave_profile = random.choice(['SIN', 'TRI', 'SAW'])
        wave_texture.inputs['Phase Offset'].default_value = sample_range(-2, 2)

    normal = nw.new_node(Nodes.InputNormal)
    
    vector_math_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: wave_texture.outputs["Color"], 1: normal},
        attrs={'operation': 'MULTIPLY'})
    
    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs["Value"].default_value = input_kwargs.get('offsetscale', 0.01)

    vector_math_2 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: vector_math_1.outputs["Vector"], 1: value_1},
        attrs={'operation': 'MULTIPLY'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input, 'Offset': vector_math_2.outputs["Vector"]})
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': set_position, 1: wave_texture.outputs["Color"]},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': capture_attribute.outputs["Geometry"], 'Attribute': capture_attribute.outputs["Attribute"]})

def apply(obj, geo_kwargs=None, shader_kwargs=None, **kwargs):
    surface.add_geomod(obj, geo_bark, apply=False, input_kwargs=geo_kwargs, attributes=['offset'])
    surface.add_material(obj, shader_bark, reuse=False, input_kwargs=shader_kwargs)

if __name__ == "__main__":
    for i in range(10):
        bpy.ops.wm.open_mainfile(filepath='test.blend')
        apply(bpy.data.objects['creature_16_aquatic_0_root_mesh'], geo_kwargs={'rand': True}, shader_kwargs={'rand': True})
        fn = os.path.join(os.path.abspath(os.curdir), 'bark.blend')
        #bpy.ops.wm.save_as_mainfile(filepath=fn)
        bpy.context.scene.render.filepath = os.path.join('surfaces/surface_thumbnails', 'bark_%d.jpg'%(i))
        bpy.context.scene.render.image_settings.file_format='JPEG'
        bpy.ops.render.render(write_still=True)