# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Mingzhe Wang
# Date Signed: June 14 2023 

import os, sys
import numpy as np
import math as ma
from surfaces.surface_utils import clip, sample_range, sample_ratio, sample_color, geo_voronoi_noise
import bpy
import mathutils
from numpy.random import uniform, normal, randint
from nodes.node_wrangler import Nodes, NodeWrangler
from nodes import node_utils
from nodes.color import color_category
from surfaces import surface

def shader_rough_plastic(nw: NodeWrangler, rand=False, **input_kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 1.46
    
    fresnel = nw.new_node(Nodes.Fresnel,
        input_kwargs={'IOR': value})
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': (0.0123, 0.5029, 0.008, 1.0), 'Roughness': 0.1091})
    if rand:
        sample_color(principled_bsdf.inputs['Base Color'].default_value)
    
    glossy_bsdf = nw.new_node('ShaderNodeBsdfGlossy',
        input_kwargs={'Color': (1.0, 1.0, 1.0, 1.0), 'Roughness': 0.1782})
    if rand:
        glossy_bsdf.inputs['Roughness'].default_value = sample_range(0.05, 0.3)

    mix_shader = nw.new_node(Nodes.MixShader,
        input_kwargs={'Fac': fresnel, 1: principled_bsdf, 2: glossy_bsdf})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': mix_shader})

def shader_translucent_plastic(nw: NodeWrangler, rand=False, **input_kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    layer_weight = nw.new_node('ShaderNodeLayerWeight',
        input_kwargs={'Blend': 0.3})
    if rand:
        layer_weight.inputs['Blend'].default_value = sample_range(0.2, 0.4)

    rgb = nw.new_node(Nodes.RGB)
    rgb.outputs[0].default_value = (0.5, 0.0, 0.036, 1.0)
    if rand:
        sample_color(rgb.outputs[0].default_value)

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 1.2
    if rand:
        value.outputs[0].default_value = sample_range(1.2, 1.6)

    glass_bsdf = nw.new_node('ShaderNodeBsdfGlass',
        input_kwargs={'Color': rgb, 'Roughness': 0.2, 'IOR': value})
    
    glossy_bsdf = nw.new_node('ShaderNodeBsdfGlossy',
        input_kwargs={'Roughness': 0.2})
    
    mix_shader = nw.new_node(Nodes.MixShader,
        input_kwargs={'Fac': layer_weight.outputs["Fresnel"], 1: glass_bsdf, 2: glossy_bsdf})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': mix_shader})

def apply(obj, geo_kwargs=None, shader_kwargs=None, **kwargs):
    if 'rough' in shader_kwargs:
        if shader_kwargs['rough']:
            surface.add_material(obj, shader_rough_plastic, reuse=False, input_kwargs=shader_kwargs)
        else:
            surface.add_material(obj, shader_translucent_plastic, reuse=False, input_kwargs=shader_kwargs)
    elif 'translucent' in shader_kwargs:
        if shader_kwargs['translucent']:
            surface.add_material(obj, shader_translucent_plastic, reuse=False, input_kwargs=shader_kwargs)
        else:
            surface.add_material(obj, shader_rough_plastic, reuse=False, input_kwargs=shader_kwargs)
    else:
        if uniform() < 0.5:
            surface.add_material(obj, shader_rough_plastic, reuse=False, input_kwargs=shader_kwargs)
        else:
            surface.add_material(obj, shader_translucent_plastic, reuse=False, input_kwargs=shader_kwargs)
        

if __name__ == "__main__":
    mat = 'plastic'
    if not os.path.isdir(os.path.join('outputs', mat)):
        os.mkdir(os.path.join('outputs', mat))
    for i in range(10):
        bpy.ops.wm.open_mainfile(filepath='test.blend')
        apply(bpy.data.objects['SolidModel'], geo_kwargs={'rand':True, 'subdivide_mesh_level':3}, shader_kwargs={'rand': True})
        #fn = os.path.join(os.path.abspath(os.curdir), 'giraffe_geo_test.blend')
        #bpy.ops.wm.save_as_mainfile(filepath=fn)
        bpy.context.scene.render.filepath = os.path.join('outputs', mat, '%s_%d.jpg'%(mat, i))
        bpy.context.scene.render.image_settings.file_format='JPEG'
        bpy.ops.render.render(write_still=True)