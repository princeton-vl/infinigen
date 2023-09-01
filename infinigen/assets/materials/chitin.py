# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Mingzhe Wang
# Acknowledgement: This file draws inspiration from https://www.youtube.com/watch?v=K45LuDJv_hk by yojigraphics

import os, sys
import numpy as np
import math as ma
from infinigen.assets.materials.utils.surface_utils import clip, sample_range, sample_ratio, sample_color, geo_voronoi_noise
import bpy
import mathutils
from numpy.random import uniform, normal, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category, hsv2rgba
from infinigen.core import surface

def shader_chitin(nw: NodeWrangler, rand=True, **input_kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    geometry = nw.new_node('ShaderNodeNewGeometry')
    
    colorramp_1 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': geometry.outputs["Pointiness"]})
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements[0].position = 0.4091
    colorramp_1.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_1.color_ramp.elements[1].position = 0.4455
    colorramp_1.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    colorramp_1.color_ramp.elements[2].position = 0.5127
    colorramp_1.color_ramp.elements[2].color = (0.0, 0.0, 0.0, 1.0)
    
    colorramp_10 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': colorramp_1.outputs["Color"]})
    colorramp_10.color_ramp.elements[0].position = 0.0
    colorramp_10.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_10.color_ramp.elements[1].position = 0.2273
    colorramp_10.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    colorramp_4 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': geometry.outputs["Pointiness"]})
    colorramp_4.color_ramp.elements[0].position = 0.4909
    colorramp_4.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_4.color_ramp.elements[1].position = 0.6773
    colorramp_4.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: colorramp_10.outputs["Color"], 1: colorramp_4.outputs["Color"]},
        attrs={'operation': 'MULTIPLY'})
    
    colorramp_3 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': multiply})
    colorramp_3.color_ramp.interpolation = "EASE"
    colorramp_3.color_ramp.elements[0].position = 0.0
    colorramp_3.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_3.color_ramp.elements[1].position = 0.0864
    colorramp_3.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    texture_coordinate_1 = nw.new_node(Nodes.TextureCoord)
    
    separate_xyz_3 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': texture_coordinate_1.outputs["Generated"]})
    
    colorramp_6 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': separate_xyz_3.outputs["X"]})
    colorramp_6.color_ramp.elements[0].position = 0.5332
    colorramp_6.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_6.color_ramp.elements[1].position = 0.5427
    colorramp_6.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    attribute_2 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'tag_body'})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: colorramp_6.outputs["Color"], 1: attribute_2.outputs["Color"]},
        attrs={'operation': 'MULTIPLY'})
    
    attribute_1 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'tag_head'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_1, 1: attribute_1.outputs["Color"]},
        attrs={'use_clamp': True})
    
    colorramp_5 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': add})
    colorramp_5.color_ramp.elements[0].position = 0.0
    colorramp_5.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    colorramp_5.color_ramp.elements[1].position = 1.0
    colorramp_5.color_ramp.elements[1].color = (0.0168, 0.0168, 0.0168, 1.0)
    
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: colorramp_3.outputs["Color"], 1: colorramp_5.outputs["Color"]},
        attrs={'operation': 'MULTIPLY'})
    
    attribute_3 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'tag_leg'})
    
    invert_1 = nw.new_node('ShaderNodeInvert',
        input_kwargs={'Color': attribute_3.outputs["Color"]})
    
    multiply_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_2, 1: invert_1},
        attrs={'operation': 'MULTIPLY'})
    
    texture_coordinate = nw.new_node(Nodes.TextureCoord)
    
    separate_xyz_2 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': texture_coordinate.outputs["Object"]})
    
    sign = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["X"]},
        attrs={'operation': 'SIGN'})
    
    mapping_1 = nw.new_node(Nodes.Mapping,
        input_kwargs={'Vector': texture_coordinate.outputs["Object"], 'Rotation': (0.0, 0.0, -0.7854), 'Scale': (1.0, 1.0, 0.0)})
    
    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': mapping_1})
    
    mapping = nw.new_node(Nodes.Mapping,
        input_kwargs={'Vector': texture_coordinate.outputs["Object"], 'Rotation': (0.0, 0.0, 0.7854), 'Scale': (1.0, 10.0, 0.0)})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': mapping})
    
    mix_3 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': sign, 'Color1': separate_xyz_1.outputs["X"], 'Color2': separate_xyz.outputs["X"]})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Scale': 10.0, 'Detail': 10.0})
    
    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.2, 'Color1': mix_3, 'Color2': noise_texture.outputs["Fac"]})
    
    noise_texture_1 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': mix, 'W': 1.4, 'Scale': 100.0, 'Detail': 10.0, 'Roughness': 0.0},
        attrs={'noise_dimensions': '4D'})
    if rand:
        noise_texture_1.inputs['W'].default_value = sample_range(-2,2)
    
    mix_2 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': multiply_3, 'Color1': (0.0, 0.0, 0.0, 1.0), 'Color2': noise_texture_1.outputs["Fac"]})
    
    colorramp_2 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': mix_2})
    colorramp_2.color_ramp.elements[0].position = 0.0
    colorramp_2.color_ramp.elements[0].color = (0.0068, 0.0, 0.0005, 1.0)
    colorramp_2.color_ramp.elements[1].position = 0.9955
    colorramp_2.color_ramp.elements[1].color = (0.1347, 0.0156, 0.0115, 1.0)
    if rand:
        colorramp_2.color_ramp.elements[1].color = hsv2rgba((np.mod(normal(0.2, 0.4), 1), uniform(0, 1), uniform(0.05, 0.5)))
        #for i in range(3):
        #    colorramp_2.color_ramp.elements[1].color[i] /= 7

    invert = nw.new_node('ShaderNodeInvert',
        input_kwargs={'Color': multiply_2})
    
    colorramp_11 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': invert})
    colorramp_11.color_ramp.elements[0].position = 0.3932
    colorramp_11.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_11.color_ramp.elements[1].position = 1.0
    colorramp_11.color_ramp.elements[1].color = (0.5103, 0.5103, 0.5103, 1.0)
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': colorramp_2.outputs["Color"], 'Metallic': 0.7, 'Roughness': colorramp_11.outputs["Color"]},
        attrs={'subsurface_method': 'BURLEY'})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

def geometry_chitin(nw: NodeWrangler, rand=True, **input_kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None)])
    
    normal = nw.new_node(Nodes.InputNormal)
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Scale': 100.0},
        attrs={'noise_dimensions': '4D'})
    if rand:
        noise_texture.inputs['W'].default_value = sample_range(-2,2)

    add = nw.new_node(Nodes.Math,
        input_kwargs={0: noise_texture.outputs["Fac"], 1: -0.5})
    
    multiply = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: normal, 1: add},
        attrs={'operation': 'MULTIPLY'})
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.001
    
    multiply_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: multiply.outputs["Vector"], 1: value},
        attrs={'operation': 'MULTIPLY'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Offset': multiply_1.outputs["Vector"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_position})

def apply(obj, geo_kwargs=None, shader_kwargs=None, **kwargs):
    surface.add_geomod(obj, geometry_chitin, input_kwargs=geo_kwargs)
    surface.add_material(obj, shader_chitin, reuse=False, input_kwargs=shader_kwargs)


if __name__ == "__main__":
    for i in range(1):
        bpy.ops.wm.open_mainfile(filepath='dev_scene_1019.blend')
        #creature(73349, 0).parts(0, factory=QuadrupedBody)
        obj = "creature(36230, 0).parts(0, factory=BeetleBody)"
        #obj = "creature(73349, 0).parts(0, factory=QuadrupedBody)"
        apply(bpy.data.objects[obj], geo_kwargs={'rand': True}, shader_kwargs={'rand': True})
        fn = os.path.join(os.path.abspath(os.curdir), 'dev_scene_test_beetle_attr.blend')
        bpy.ops.wm.save_as_mainfile(filepath=fn)
        #bpy.context.scene.render.filepath = os.path.join('surfaces/surface_thumbnails', 'bone%d.jpg'%(i))
        #bpy.context.scene.render.image_settings.file_format='JPEG'
        #bpy.ops.render.render(write_still=True)
