# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Mingzhe Wang


import os, sys
import numpy as np
import math as ma
from infinigen.assets.materials.utils.surface_utils import clip, sample_range, sample_ratio, sample_color, geo_voronoi_noise
import bpy
import mathutils
from numpy.random import uniform as U, normal as N, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category, hsv2rgba
from infinigen.core import surface

from infinigen.assets.creatures.util.nodegroups.shader import nodegroup_color_mask

def shader_spots_sparse_attr(nw: NodeWrangler, rand=True, **input_kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    attribute = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'offset'})
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': attribute.outputs["Fac"]})
    colorramp.color_ramp.elements.new(0)
    colorramp.color_ramp.elements[0].position = 0.0
    colorramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp.color_ramp.elements[1].position = 0.4341
    colorramp.color_ramp.elements[1].color = (0.0942, 0.0942, 0.0942, 1.0)
    colorramp.color_ramp.elements[2].position = 0.5
    colorramp.color_ramp.elements[2].color = (1.0, 1.0, 1.0, 1.0)
    
    group = nw.new_node(nodegroup_color_mask().name)
    
    getcolor = lambda: hsv2rgba((U(0.02, 0.06), U(0.05, 0.9), np.abs(N(0.05, 0.1))))

    colorramp_3 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': group})
    colorramp_3.color_ramp.elements[0].position = 0.0
    colorramp_3.color_ramp.elements[0].color = getcolor()
    colorramp_3.color_ramp.elements[1].position = 1.0
    colorramp_3.color_ramp.elements[1].color = hsv2rgba((U(0.02, 0.06), U(0.4, 0.8), U(0.15, 0.7)))
    if rand:
        colorramp_3.color_ramp.elements[0].position = sample_range(0, 0.5)
        colorramp_3.color_ramp.elements[0].color = getcolor()
        #sample_color(colorramp_3.color_ramp.elements[1].color)

    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={
            'Fac': colorramp.outputs["Color"] if U() < 0.6 else 1, 
            'Color1': (0.024, 0.0499, 0.0168, 1.0), 
            'Color2': colorramp_3.outputs["Color"]
        })
    if rand:
        mix.inputs[6].default_value = getcolor()

    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': mix, 'Specular': 0.0, 'Roughness': colorramp.outputs["Color"]})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

def geometry_spots_sparse(nw: NodeWrangler, rand=True, **input_kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None)])
    
    position = nw.new_node(Nodes.InputPosition)
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = U(0.1, 1)
    
    add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: position, 1: value}, attrs={'operation': 'MULTIPLY'})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': add.outputs["Vector"]})
    
    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.1, 'Color1': add.outputs["Vector"], 'Color2': noise_texture.outputs["Fac"]})
    
    voronoi_texture = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': mix, 'Scale': sample_ratio(8, 0.5, 2) if rand else 8.0})

    noise_texture_1 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': add.outputs["Vector"], 'Scale': 15.0, 'Roughness': 1.0})
    
    mix_1 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Color1': voronoi_texture.outputs["Distance"], 'Color2': noise_texture_1.outputs["Fac"]})
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': mix_1})
    colorramp.color_ramp.elements.new(0)
    colorramp.color_ramp.elements[0].position = 0.0
    colorramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp.color_ramp.elements[1].position = 0.4045
    colorramp.color_ramp.elements[1].color = (0.0953, 0.0953, 0.0953, 1.0)
    colorramp.color_ramp.elements[2].position = 0.8091
    colorramp.color_ramp.elements[2].color = (1.0, 1.0, 1.0, 1.0)
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: 1.0, 1: colorramp.outputs["Color"]},
        attrs={'operation': 'SUBTRACT'})
    
    normal = nw.new_node(Nodes.InputNormal)
    
    multiply = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: subtract, 1: normal},
        attrs={'operation': 'MULTIPLY'})
    
    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = 0
    
    multiply_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: multiply.outputs["Vector"], 1: value_1},
        attrs={'operation': 'MULTIPLY'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Offset': multiply_1.outputs["Vector"]})
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': set_position, 1: mix_1},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': capture_attribute.outputs["Geometry"], 'Attribute': capture_attribute.outputs["Attribute"]})

def apply(obj, geo_kwargs=None, shader_kwargs=None, **kwargs):
    surface.add_geomod(obj, geometry_spots_sparse, input_kwargs=geo_kwargs, attributes=['offset'])
    surface.add_material(obj, shader_spots_sparse_attr, reuse=False, input_kwargs=shader_kwargs)

if __name__ == "__main__":
    for i in range(1):
        bpy.ops.wm.open_mainfile(filepath='dev_scene_1019.blend')
        #creature(73349, 0).parts(0, factory=QuadrupedBody)
        apply(bpy.data.objects['creature(73349, 0).parts(0, factory=QuadrupedBody)'], geo_kwargs={'rand': True}, shader_kwargs={'rand': True})
        fn = os.path.join(os.path.abspath(os.curdir), 'dev_scene_test_spot_sparse.blend')
        bpy.ops.wm.save_as_mainfile(filepath=fn)
        #bpy.context.scene.render.filepath = os.path.join('surfaces/surface_thumbnails', 'bone%d.jpg'%(i))
        #bpy.context.scene.render.image_settings.file_format='JPEG'
        #bpy.ops.render.render(write_still=True)