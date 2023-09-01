# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Mingzhe Wang
# Acknowledgment: This file draws inspiration from https://www.youtube.com/watch?v=FY0lR96Mwas by Sam Bowman

import os, sys
import numpy as np
import math as ma
import bpy
import mathutils
from numpy.random import uniform, normal, randint

from infinigen.assets.materials.utils.surface_utils import clip, sample_range, sample_ratio, sample_color, geo_voronoi_noise
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category

from infinigen.core import surface

def shader_aluminumdisp2tut(nw: NodeWrangler, rand=False, **input_kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    texture_coordinate = nw.new_node(Nodes.TextureCoord)
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': texture_coordinate.outputs["Generated"]})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: 0.1},
        attrs={'operation': 'MULTIPLY'})
    if rand:
        multiply.inputs[1].default_value = sample_range(-1, 1)
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"]},
        attrs={'operation': 'MULTIPLY'})
    if rand:
        multiply_1.inputs[1].default_value = sample_range(-1, 1)
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply, 1: multiply_1})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Z': add})
    
    mapping_1 = nw.new_node(Nodes.Mapping,
        input_kwargs={'Vector': texture_coordinate.outputs["Object"], 'Rotation': combine_xyz})
    
    mapping = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: mapping_1.outputs["Vector"], 1: (1, 75, 1)},
        attrs={'operation': 'MULTIPLY'})

    #mapping = nw.new_node(Nodes.Mapping,
    #    input_kwargs={'Vector': mapping_1, 'Scale': (1.0, sample_range(50, 100) if rand else 75.0, 1.0)})
    
    musgrave_texture = nw.new_node(Nodes.MusgraveTexture,
        input_kwargs={'Vector': mapping, 'W': 0.7, 'Scale': 2.0, 'Detail': 10.0, 'Dimension': 1.0},
        attrs={'musgrave_dimensions': '4D'})
    if rand:
        musgrave_texture.inputs['W'].default_value = sample_range(0, 5)
        musgrave_texture.inputs['Scale'].default_value = sample_ratio(2, 0.5, 2)

    colorramp_4 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': musgrave_texture})
    colorramp_4.color_ramp.elements[0].position = sample_range(0.1, 0.3) if rand else 0.1455
    colorramp_4.color_ramp.elements[0].color = (0.466, 0.466, 0.466, 1.0)
    colorramp_4.color_ramp.elements[1].position = 1.0
    colorramp_4.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    colorramp_1 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': colorramp_4.outputs["Color"]})
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements[0].position = .28
    colorramp_1.color_ramp.elements[0].color = (0.56, 0.61, 0.61, 1.0)
    colorramp_1.color_ramp.elements[1].position = 0.46
    colorramp_1.color_ramp.elements[1].color = (0.206, 0.24, 0.27, 1.0)
    colorramp_1.color_ramp.elements[2].position = 0.71
    colorramp_1.color_ramp.elements[2].color = (0.92, 0.97, 0.95, 1.0)
    
    if rand:
        for e in colorramp_1.color_ramp.elements:
            sample_color(e.color, offset=0.02)


    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': colorramp_4.outputs["Color"]})
    colorramp.color_ramp.elements[0].position = 0.74
    colorramp.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    colorramp.color_ramp.elements[1].position = 1.0
    colorramp.color_ramp.elements[1].color = (0.5162, 0.5162, 0.5162, 1.0)
    
    colorramp_3 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': musgrave_texture})
    colorramp_3.color_ramp.elements[0].position = 0.77
    colorramp_3.color_ramp.elements[0].color = (0.26, 0.26, 0.26, 1.0)
    colorramp_3.color_ramp.elements[1].position = 1.0
    colorramp_3.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': colorramp_1.outputs["Color"], 'Metallic': colorramp.outputs["Color"], 'Roughness': colorramp_3.outputs["Color"]},
        attrs={'subsurface_method': 'BURLEY'})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

def geo_aluminumdisp2tut(nw: NodeWrangler, rand=False, **input_kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None)])
    
    #subdivide_level = nw.new_node(Nodes.Value,
    #    label='SubdivideLevel')
    #subdivide_level.outputs[0].default_value = 0
    
    #subdivide_mesh = nw.new_node(Nodes.SubdivideMesh,
    #    input_kwargs={'Mesh': group_input.outputs["Geometry"], 'Level': subdivide_level})
    
    position = nw.new_node(Nodes.InputPosition)
    
    scale = nw.new_node(Nodes.Value,
        label='Scale')
    scale.outputs[0].default_value = 1.0
    
    multiply = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: position, 1: scale},
        attrs={'operation': 'MULTIPLY'})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': multiply.outputs["Vector"], 'Scale': 4.0},
        attrs={'noise_dimensions': '4D'})
    if rand:
        noise_texture.inputs['W'].default_value = sample_range(0, 5)
        noise_texture.inputs['Scale'].default_value = sample_ratio(6.0, 0.75, 1.5)

    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': noise_texture.outputs["Fac"]})
    colorramp.color_ramp.elements.new(0)
    colorramp.color_ramp.elements[0].position = 0.68
    colorramp.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    colorramp.color_ramp.elements[1].position = 0.79
    colorramp.color_ramp.elements[1].color = (0.093, 0.093, 0.093, 1.0)
    colorramp.color_ramp.elements[2].position = 0.9
    colorramp.color_ramp.elements[2].color = (0.0, 0.0, 0.0, 1.0)
    
    noise_texture_1 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': multiply.outputs["Vector"], 'Scale': 2.0},
        attrs={'noise_dimensions': '4D'})
    
    colorramp_1 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': noise_texture_1.outputs["Fac"]})
    colorramp_1.color_ramp.elements[0].position = 0.46
    colorramp_1.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_1.color_ramp.elements[1].position = 1.0
    colorramp_1.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp.outputs["Color"], 'Color1': colorramp_1.outputs["Color"], 'Color2': (0.521, 0.521, 0.521, 1.0)})
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.5
    
    subtract = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: mix, 1: value},
        attrs={'operation': 'SUBTRACT'})
    
    normal = nw.new_node(Nodes.InputNormal)
    
    multiply_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: subtract.outputs["Vector"], 1: normal},
        attrs={'operation': 'MULTIPLY'})
    
    offset_scale = nw.new_node(Nodes.Value,
        label='OffsetScale')
    offset_scale.outputs[0].default_value = sample_range(0.03, 0.05) if rand else 0.04
    
    multiply_2 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: multiply_1.outputs["Vector"], 1: offset_scale},
        attrs={'operation': 'MULTIPLY'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Offset': multiply_2.outputs["Vector"]})
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': set_position, 1: mix},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': capture_attribute.outputs["Geometry"], 'Attribute': capture_attribute.outputs["Attribute"]})

def apply(obj, geo_kwargs=None, shader_kwargs=None, **kwargs):
    surface.add_geomod(obj, geo_aluminumdisp2tut, apply=False, input_kwargs=geo_kwargs, attributes=['offset'])
    surface.add_material(obj, shader_aluminumdisp2tut, reuse=False, input_kwargs=shader_kwargs)


if __name__ == "__main__":
    mat = 'aluminumdisp2tut'
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