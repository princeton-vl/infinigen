# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Mingzhe Wang
# Acknowledgment: This file draws inspiration from https://www.youtube.com/watch?v=mJVuodaPHTQ and https://www.youtube.com/watch?v=v7a4ouBLIow by Lance Phan


import os, sys
import numpy as np
import math as ma
from infinigen.assets.materials.utils.surface_utils import clip, sample_range, sample_ratio, sample_color, geo_voronoi_noise
import bpy
import mathutils
from numpy.random import normal as normal_func
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core import surface
import random

def shader_scale(nw, rand=True, **input_kwargs):
    math_4 = nw.new_node(Nodes.Math,
        input_kwargs={0: 1.0},
        attrs={'operation': 'SUBTRACT'})
    
    attribute_2 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'Color variations'})
    
    noise_texture_3 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': attribute_2.outputs["Color"], 'W': 1.0},
        attrs={'noise_dimensions': '4D'})
    if rand:
        noise_texture_3.inputs["W"].default_value = sample_range(-5, 5)
    
    colorramp_2 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': noise_texture_3.outputs["Fac"]})
    for i in range(3):
        colorramp_2.color_ramp.elements.new(0.0)
    colorramp_2.color_ramp.elements[0].position = 0.125
    colorramp_2.color_ramp.elements[0].color = (1.0, 0.3168, 0.8521, 1.0)
    colorramp_2.color_ramp.elements[1].position = 0.3295
    colorramp_2.color_ramp.elements[1].color = (0.1647, 0.6793, 0.6392, 1.0)
    colorramp_2.color_ramp.elements[2].position = 0.5
    colorramp_2.color_ramp.elements[2].color = (0.1132, 0.1067, 0.1249, 1.0)
    colorramp_2.color_ramp.elements[3].position = 0.6705
    colorramp_2.color_ramp.elements[3].color = (0.1509, 0.0, 0.0097, 1.0)
    colorramp_2.color_ramp.elements[4].position = 0.9295
    colorramp_2.color_ramp.elements[4].color = (0.0, 0.0, 0.0, 1.0)
    if rand:
        for e in colorramp_2.color_ramp.elements:
            sample_color(e.color)

    vector_math = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: math_4, 1: colorramp_2.outputs["Color"]},
        attrs={'operation': 'MULTIPLY'})
    
    attribute_3 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'offset2'})
    
    math_5 = nw.new_node(Nodes.Math,
        input_kwargs={0: attribute_3.outputs["Vector"], 1: 0.01},
        attrs={'operation': 'GREATER_THAN'})
    
    colorramp_7 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': math_5})
    colorramp_7.color_ramp.elements.new(1)
    colorramp_7.color_ramp.elements[0].position = 0.0
    colorramp_7.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_7.color_ramp.elements[1].position = 0.5
    colorramp_7.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    colorramp_7.color_ramp.elements[2].position = 1.0
    colorramp_7.color_ramp.elements[2].color = (0.302, 0.2765, 0.063, 1.0)
    if rand:
        sample_color(colorramp_7.color_ramp.elements[2].color)

    vector_math_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: vector_math.outputs["Vector"], 1: colorramp_7.outputs["Color"]})
    
    attribute_4 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'Color variations'})
    
    uv_map_1 = nw.new_node('ShaderNodeUVMap')
    uv_map_1.uv_map = 'UVMap'

    noise_texture_5 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': uv_map_1, 'Scale': 50.0},
        attrs={'noise_dimensions': '4D'})
    if rand:
        noise_texture_5.inputs["W"].default_value = sample_range(-5, 5)
    
    mix_3 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 1.0, 'Color1': attribute_4.outputs["Color"], 'Color2': noise_texture_5.outputs["Color"]},
        attrs={'blend_type': 'ADD'})
    
    noise_texture_4 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': mix_3})
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': noise_texture_4.outputs["Fac"]})
    colorramp.color_ramp.elements[0].position = 0.0
    colorramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp.color_ramp.elements[1].position = 1.0
    colorramp.color_ramp.elements[1].color = (0.5078, 0.5078, 0.5078, 1.0)
    
    colormap = random.choice([vector_math.outputs["Vector"], vector_math_1.outputs["Vector"]]) if rand else vector_math.outputs["Vector"]

    principled_bsdf_1 = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': colormap, 'Subsurface': 0.2, 'Subsurface Radius': (0.36, 0.46, 0.6), 'Subsurface Color': (1.0, 0.9405, 0.7747, 1.0), 'Metallic': 0.8, 'Roughness': colorramp.outputs["Color"], 'IOR': 1.69},
        attrs={'subsurface_method': 'BURLEY'})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf_1})

@node_utils.to_nodegroup('nodegroup_node_grid', singleton=False, type='GeometryNodeTree')
def nodegroup_node_grid(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'Value', 0.5)])
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Value"], 1: 2.0},
        attrs={'operation': 'MULTIPLY'})
    
    floor = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply, 1: 2.0},
        attrs={'operation': 'FLOOR'})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: floor},
        attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_1})
    
    trunc = nw.new_node(Nodes.Math,
        input_kwargs={0: add},
        attrs={'operation': 'TRUNC'})
    
    trunc_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_1},
        attrs={'operation': 'TRUNC'})
    
    add_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: trunc_1})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'floor1': trunc, 'floor2': add_1})

def geo_scale(nw, rand=True, **input_kwargs):
    # Code generated using version 2.4.3 of the node_transpiler
 
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None)])
    
    separate_xyz_2 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': nw.expose_input('UVMap', attribute='UVMap', dtype='NodeSocketVector')})

    angle = nw.new_node(Nodes.Value, label='Angle')
    angle.outputs[0].default_value = 0.0000
    
    cosine = nw.new_node(Nodes.Math, input_kwargs={0: angle}, attrs={'operation': 'COSINE'})
    
    multiply = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz_2.outputs["X"], 1: cosine}, attrs={'operation': 'MULTIPLY'})
    
    sine = nw.new_node(Nodes.Math, input_kwargs={0: angle}, attrs={'operation': 'SINE'})
    
    multiply_1 = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz_2.outputs["Y"], 1: sine}, attrs={'operation': 'MULTIPLY'})
    
    subtract = nw.new_node(Nodes.Math, input_kwargs={0: multiply, 1: multiply_1}, attrs={'operation': 'SUBTRACT'})
    
    xscale = nw.new_node(Nodes.Value, label='Xscale')
    xscale.outputs[0].default_value = sample_range(0.7, 1.3)
    
    multiply_2 = nw.new_node(Nodes.Math, input_kwargs={0: subtract, 1: xscale}, attrs={'operation': 'MULTIPLY'})
    
    noise_texture_2 = nw.new_node(Nodes.NoiseTexture, input_kwargs={'W': sample_range(-10, 10), 'Scale': 10.0000}, attrs={'noise_dimensions': '4D'})
    
    xnoise = nw.new_node(Nodes.Value, label='Xnoise')
    xnoise.outputs[0].default_value = sample_range(0.01, 0.03)
    
    multiply_3 = nw.new_node(Nodes.Math, input_kwargs={0: noise_texture_2.outputs["Fac"], 1: xnoise}, attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.Math, input_kwargs={0: multiply_2, 1: multiply_3})
    
    multiply_4 = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz_2.outputs["X"], 1: sine}, attrs={'operation': 'MULTIPLY'})
    
    multiply_5 = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz_2.outputs["Y"], 1: cosine}, attrs={'operation': 'MULTIPLY'})
    
    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_4, 1: multiply_5})
    
    yscale = nw.new_node(Nodes.Value, label='Yscale')
    yscale.outputs[0].default_value = sample_range(0.7, 1.3)
    
    multiply_6 = nw.new_node(Nodes.Math, input_kwargs={0: add_1, 1: yscale}, attrs={'operation': 'MULTIPLY'})
    
    noise_texture_1 = nw.new_node(Nodes.NoiseTexture, input_kwargs={'W': sample_range(-10, 10), 'Scale': 10.0000}, attrs={'noise_dimensions': '4D'})
    
    ynoise = nw.new_node(Nodes.Value, label='Ynoise')
    ynoise.outputs[0].default_value = sample_range(0.01, 0.03)
    
    multiply_7 = nw.new_node(Nodes.Math, input_kwargs={0: noise_texture_1.outputs["Fac"], 1: ynoise}, attrs={'operation': 'MULTIPLY'})
    
    add_2 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_6, 1: multiply_7})
    
    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': add, 'Y': add_2})
    
    scale = nw.new_node(Nodes.Value, label='Scale')
    scale.outputs[0].default_value = sample_ratio(25, 2/3, 3/2)
    
    multiply_8 = nw.new_node(Nodes.VectorMath, input_kwargs={0: combine_xyz_2, 1: scale}, attrs={'operation': 'MULTIPLY'})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': multiply_8})
    
    nodegrid = nw.new_node(nodegroup_node_grid().name, input_kwargs={'Value': separate_xyz.outputs["Y"]})
    
    greater_than = nw.new_node(Nodes.Compare,
        input_kwargs={0: nodegrid.outputs["floor1"], 1: separate_xyz.outputs["Y"]},
        attrs={'operation': 'LESS_THAN'})

    less_than = nw.new_node(Nodes.Compare, input_kwargs={0: nodegrid.outputs["floor1"], 1: separate_xyz.outputs["Y"]})

    nodegrid_1 = nw.new_node(nodegroup_node_grid().name, input_kwargs={'Value': separate_xyz.outputs["X"]})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': nodegrid_1.outputs["floor2"], 'Y': nodegrid.outputs["floor1"]})
    
    multiply_9 = nw.new_node(Nodes.VectorMath, input_kwargs={0: less_than, 1: combine_xyz}, attrs={'operation': 'MULTIPLY'})
    
    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': nodegrid_1.outputs["floor1"], 'Y': nodegrid.outputs["floor2"]})
    
    multiply_10 = nw.new_node(Nodes.VectorMath, input_kwargs={0: greater_than, 1: combine_xyz_1}, attrs={'operation': 'MULTIPLY'})
    
    add_3 = nw.new_node(Nodes.VectorMath, input_kwargs={0: multiply_9.outputs["Vector"], 1: multiply_10.outputs["Vector"]})
    
    subtract_1 = nw.new_node(Nodes.VectorMath, input_kwargs={0: multiply_8, 1: add_3}, attrs={'operation': 'SUBTRACT'})
    
    distance = nw.new_node(Nodes.VectorMath, input_kwargs={0: multiply_8, 1: add_3}, attrs={'operation': 'DISTANCE'})
    
    add_4 = nw.new_node(Nodes.Math, input_kwargs={0: distance.outputs["Value"], 1: 0.0100})
    
    less_than_1 = nw.new_node(Nodes.Compare, input_kwargs={0: add_4, 1: 0.5000}, attrs={'operation': 'LESS_THAN'})
    
    greater_than_1 = nw.new_node(Nodes.Compare, input_kwargs={0: add_4, 1: 0.5000})
    
    multiply_11 = nw.new_node(Nodes.VectorMath, input_kwargs={0: less_than, 1: combine_xyz_1}, attrs={'operation': 'MULTIPLY'})
    
    multiply_12 = nw.new_node(Nodes.VectorMath, input_kwargs={0: greater_than, 1: combine_xyz}, attrs={'operation': 'MULTIPLY'})
    
    add_5 = nw.new_node(Nodes.VectorMath, input_kwargs={0: multiply_11.outputs["Vector"], 1: multiply_12.outputs["Vector"]})
    
    subtract_2 = nw.new_node(Nodes.VectorMath, input_kwargs={0: multiply_8, 1: add_5}, attrs={'operation': 'SUBTRACT'})
    
    multiply_13 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: greater_than_1, 1: subtract_2.outputs["Vector"]},
        attrs={'operation': 'MULTIPLY'})
    
    _multiply_add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: subtract_1.outputs["Vector"], 1: less_than_1, 2: multiply_13.outputs["Vector"]},
        attrs={'operation': 'MULTIPLY_ADD'})
    
    multiply_add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: _multiply_add, 1: (1, -1, 1)},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_14 = nw.new_node(Nodes.VectorMath, input_kwargs={0: greater_than_1, 1: add_5}, attrs={'operation': 'MULTIPLY'})
    
    multiply_add_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: add_3, 1: less_than_1, 2: multiply_14.outputs["Vector"]},
        attrs={'operation': 'MULTIPLY_ADD'})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': multiply_add_1, 'W': sample_range(-10, 10), 'Scale': 33.0000},
        attrs={'noise_dimensions': '4D'})
    
    subtract_3 = nw.new_node(Nodes.MapRange,
        input_kwargs={0: noise_texture.outputs["Fac"], 1: 0.26, 2: 0.74, 3: -0.5, 4: 0.5},
        attrs={'clamp': True}
        )
    
    sine_1 = nw.new_node(Nodes.Math, input_kwargs={0: subtract_3}, attrs={'operation': 'SINE'})
    
    cosine_1 = nw.new_node(Nodes.Math, input_kwargs={0: subtract_3}, attrs={'operation': 'COSINE'})
    
    combine_xyz_color = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': sine_1, 'Y': cosine_1, 'Z': 0.0000})

    add_6 = nw.new_node(Nodes.VectorMath, input_kwargs={0: combine_xyz_color.outputs["Vector"], 1: multiply_add}, attrs={'operation': 'DOT_PRODUCT'})

    distance_1 = nw.new_node(Nodes.VectorMath, input_kwargs={0: multiply_8, 1: add_5}, attrs={'operation': 'DISTANCE'})
    
    add_7 = nw.new_node(Nodes.Math, input_kwargs={0: distance_1.outputs["Value"], 1: 0.0100})
    
    multiply_17 = nw.new_node(Nodes.Math, input_kwargs={0: greater_than_1, 1: add_7}, attrs={'operation': 'MULTIPLY'})
    
    multiply_add_2 = nw.new_node(Nodes.Math, input_kwargs={0: add_4, 1: less_than_1, 2: multiply_17}, attrs={'operation': 'MULTIPLY_ADD'})
    
    multiply_18 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_add_2, 1: 2.0000}, attrs={'operation': 'MULTIPLY'})
    
    multiply_19 = nw.new_node(Nodes.MapRange, 
        input_kwargs={0: multiply_18, 1: 0.9156, 2: 1.0000, 3: 0.0000, 4: 0.5},
        attrs={'clamp': True}
    )

    subtract_4 = nw.new_node(Nodes.Math, input_kwargs={0: add_6, 1: multiply_19}, attrs={'operation': 'SUBTRACT'})

    subtract_5 = nw.new_node(Nodes.Math, input_kwargs={0: subtract_4, 1: 0.0000}, attrs={'operation': 'SUBTRACT'})
    
    normal = nw.new_node(Nodes.InputNormal)
    
    multiply_20 = nw.new_node(Nodes.VectorMath, input_kwargs={0: subtract_5, 1: normal}, attrs={'operation': 'MULTIPLY'})
    
    offset_scale = nw.new_node(Nodes.Value, label='OffsetScale')
    offset_scale.outputs[0].default_value = 0.0020
    
    multiply_21 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: multiply_20.outputs["Vector"], 1: offset_scale},
        attrs={'operation': 'MULTIPLY'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Offset': multiply_21.outputs["Vector"]})
    
    capture_attribute_1 = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': set_position.outputs["Geometry"], 1: multiply_add_1},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    capture_attribute_4 = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': capture_attribute_1.outputs["Geometry"], 1: multiply_19},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={
            'Geometry': capture_attribute_4.outputs["Geometry"], 
            'attr2': capture_attribute_1.outputs["Attribute"], 
            'attr5': capture_attribute_4.outputs["Attribute"]
            },
        attrs={'is_active_output': True})

def apply(obj, geo_kwargs=None, shader_kwargs=None, **kwargs):
    attributes = [
        'Color variations',
        'offset2'
    ]
    surface.add_geomod(obj, geo_scale, apply=False, input_kwargs=geo_kwargs, attributes=attributes)
    surface.add_material(obj, shader_scale, reuse=False, input_kwargs=shader_kwargs)

if __name__ == "__main__":
    template = "scale_new2"
    #outpath = os.path.join("outputs", template)
    #if not os.path.isdir(outpath):
    #    os.mkdir(outpath)
    for i in range(1):
        bpy.ops.wm.open_mainfile(filepath='scale_new2.blend')
        apply(bpy.data.objects['creature_16_aquatic_0_root_mesh.001'], geo_kwargs={'rand': False}, shader_kwargs={'rand': True})
        fn = os.path.join(os.path.abspath(os.curdir), 'dev_test_scale_new2.blend')
        bpy.ops.wm.save_as_mainfile(filepath=fn)
        #bpy.context.scene.render.filepath = os.path.join(outpath, 'scale_%d.jpg'%(i))
        #bpy.context.scene.render.image_settings.file_format='JPEG'
        #bpy.ops.render.render(write_still=True)