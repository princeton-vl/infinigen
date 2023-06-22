# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Mingzhe Wang
# Date Signed: April 25 2023 

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
    
    mapping = nw.new_node(Nodes.Mapping,
        input_kwargs={'Vector': mapping_1, 'Scale': (1.0, sample_range(50, 100) if rand else 75.0, 1.0)})
    
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
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements[0].position = 0.0
    colorramp_1.color_ramp.elements[0].color = (0.7379, 0.8308, 0.9473, 1.0)
    colorramp_1.color_ramp.elements[1].position = 0.0119
    colorramp_1.color_ramp.elements[1].color = (0.4549, 0.5612, 0.6442, 1.0)
    colorramp_1.color_ramp.elements[2].position = 0.0476
    colorramp_1.color_ramp.elements[2].color = (0.5669, 0.6644, 0.7474, 1.0)
    colorramp_1.color_ramp.elements[3].position = 0.0714
    colorramp_1.color_ramp.elements[3].color = (0.81, 0.9152, 0.9662, 1.0)
    colorramp_1.color_ramp.elements[4].position = 0.0952
    colorramp_1.color_ramp.elements[4].color = (0.7241, 0.7924, 0.8407, 1.0)
    colorramp_1.color_ramp.elements[5].position = 0.131
    colorramp_1.color_ramp.elements[5].color = (0.804, 0.8477, 0.8742, 1.0)
    colorramp_1.color_ramp.elements[6].position = 0.1548
    colorramp_1.color_ramp.elements[6].color = (0.9803, 0.9882, 0.9908, 1.0)
    colorramp_1.color_ramp.elements[7].position = 0.2262
    colorramp_1.color_ramp.elements[7].color = (0.9902, 0.9975, 0.9962, 1.0)
    colorramp_1.color_ramp.elements[8].position = 0.2619
    colorramp_1.color_ramp.elements[8].color = (0.841, 0.9097, 0.951, 1.0)
    colorramp_1.color_ramp.elements[9].position = 0.2976
    colorramp_1.color_ramp.elements[9].color = (0.5182, 0.6006, 0.656, 1.0)
    colorramp_1.color_ramp.elements[10].position = 0.3452
    colorramp_1.color_ramp.elements[10].color = (0.6566, 0.7047, 0.818, 1.0)
    colorramp_1.color_ramp.elements[11].position = 0.369
    colorramp_1.color_ramp.elements[11].color = (0.4919, 0.5651, 0.647, 1.0)
    colorramp_1.color_ramp.elements[12].position = 0.3929
    colorramp_1.color_ramp.elements[12].color = (0.4198, 0.5099, 0.5918, 1.0)
    colorramp_1.color_ramp.elements[13].position = 0.4048
    colorramp_1.color_ramp.elements[13].color = (0.5907, 0.6852, 0.7502, 1.0)
    colorramp_1.color_ramp.elements[14].position = 0.4405
    colorramp_1.color_ramp.elements[14].color = (0.2116, 0.2536, 0.2802, 1.0)
    colorramp_1.color_ramp.elements[15].position = 0.4643
    colorramp_1.color_ramp.elements[15].color = (0.2057, 0.239, 0.2664, 1.0)
    colorramp_1.color_ramp.elements[16].position = 0.4881
    colorramp_1.color_ramp.elements[16].color = (0.366, 0.4329, 0.4656, 1.0)
    colorramp_1.color_ramp.elements[17].position = 0.5119
    colorramp_1.color_ramp.elements[17].color = (0.374, 0.4432, 0.4409, 1.0)
    colorramp_1.color_ramp.elements[18].position = 0.5714
    colorramp_1.color_ramp.elements[18].color = (0.261, 0.2957, 0.2785, 1.0)
    colorramp_1.color_ramp.elements[19].position = 0.5952
    colorramp_1.color_ramp.elements[19].color = (0.9224, 0.9713, 0.9525, 1.0)
    colorramp_1.color_ramp.elements[20].position = 0.6429
    colorramp_1.color_ramp.elements[20].color = (0.7948, 0.8156, 0.7792, 1.0)
    colorramp_1.color_ramp.elements[21].position = 0.6905
    colorramp_1.color_ramp.elements[21].color = (0.8353, 0.812, 0.7957, 1.0)
    colorramp_1.color_ramp.elements[22].position = 0.7024
    colorramp_1.color_ramp.elements[22].color = (0.7605, 0.6939, 0.7011, 1.0)
    colorramp_1.color_ramp.elements[23].position = 0.7143
    colorramp_1.color_ramp.elements[23].color = (0.985, 0.8822, 0.9157, 1.0)
    colorramp_1.color_ramp.elements[24].position = 0.7381
    colorramp_1.color_ramp.elements[24].color = (0.6005, 0.5318, 0.5487, 1.0)
    colorramp_1.color_ramp.elements[25].position = 0.7738
    colorramp_1.color_ramp.elements[25].color = (0.7215, 0.7044, 0.6773, 1.0)
    colorramp_1.color_ramp.elements[26].position = 0.8095
    colorramp_1.color_ramp.elements[26].color = (0.5691, 0.6386, 0.6341, 1.0)
    colorramp_1.color_ramp.elements[27].position = 0.8333
    colorramp_1.color_ramp.elements[27].color = (0.7857, 0.8342, 0.8103, 1.0)
    colorramp_1.color_ramp.elements[28].position = 0.8929
    colorramp_1.color_ramp.elements[28].color = (0.6764, 0.7281, 0.6645, 1.0)
    colorramp_1.color_ramp.elements[29].position = 0.9762
    colorramp_1.color_ramp.elements[29].color = (0.6059, 0.6061, 0.5242, 1.0)
    colorramp_1.color_ramp.elements[30].position = 1.0
    colorramp_1.color_ramp.elements[30].color = (0.6018, 0.5866, 0.5112, 1.0)
    if rand:
        for e in colorramp_1.color_ramp.elements:
            sample_color(e.color, offset=0.03)


    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': colorramp_4.outputs["Color"]})
    colorramp.color_ramp.elements[0].position = 0.7386
    colorramp.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    colorramp.color_ramp.elements[1].position = 1.0
    colorramp.color_ramp.elements[1].color = (0.5162, 0.5162, 0.5162, 1.0)
    
    colorramp_3 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': musgrave_texture})
    colorramp_3.color_ramp.elements[0].position = 0.7682
    colorramp_3.color_ramp.elements[0].color = (0.2633, 0.2633, 0.2633, 1.0)
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