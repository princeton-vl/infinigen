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

from infinigen.assets.creatures.util.nodegroups.shader import nodegroup_color_mask

def shader_brown_circle(nw: NodeWrangler, rand=True, **input_kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    attribute_2 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'local_pos'})
    
    reroute = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': attribute_2.outputs["Color"]})
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 10.0
    
    voronoi_texture = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': reroute, 'Scale': value},
        attrs={'voronoi_dimensions': '2D'})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': reroute})
    
    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.0, 'Color1': reroute, 'Color2': noise_texture.outputs["Color"]})
    
    voronoi_texture_1 = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': mix, 'Scale': value},
        attrs={'voronoi_dimensions': '2D', 'feature': 'SMOOTH_F1'})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: voronoi_texture.outputs["Distance"], 1: voronoi_texture_1.outputs["Distance"]},
        attrs={'operation': 'SUBTRACT'})
    
    power = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract, 1: 2.0},
        attrs={'operation': 'POWER'})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: power, 1: 10000.0},
        attrs={'operation': 'MULTIPLY'})
    
    less_than = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply, 1: 1.0},
        attrs={'operation': 'LESS_THAN'})
    
    divide = nw.new_node(Nodes.Math,
        input_kwargs={0: less_than, 1: 2.0},
        attrs={'operation': 'DIVIDE'})
    
    colorramp_4 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': divide})
    colorramp_4.color_ramp.elements[0].position = 0.0591
    colorramp_4.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_4.color_ramp.elements[1].position = 0.1136
    colorramp_4.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    group = nw.new_node(nodegroup_color_mask().name)
    
    attribute = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'value'})
    
    less_than_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: attribute.outputs["Color"], 1: 0.85},
        attrs={'operation': 'LESS_THAN'})
    
    attribute_1 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'index'})
    
    noise_texture_1 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': attribute_1.outputs["Color"], 'Scale': 100.0})
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': noise_texture_1.outputs["Fac"]})
    colorramp.color_ramp.elements.new(0)
    colorramp.color_ramp.elements.new(0)
    colorramp.color_ramp.elements.new(0)
    colorramp.color_ramp.elements[0].position = 0.0591
    colorramp.color_ramp.elements[0].color = (0.0862, 0.0422, 0.0185, 1.0)
    colorramp.color_ramp.elements[1].position = 0.25
    colorramp.color_ramp.elements[1].color = (0.6512, 0.4016, 0.219, 1.0)
    colorramp.color_ramp.elements[2].position = 0.5
    colorramp.color_ramp.elements[2].color = (0.2281, 0.0947, 0.0245, 1.0)
    colorramp.color_ramp.elements[3].position = 0.8636
    colorramp.color_ramp.elements[3].color = (0.7346, 0.456, 0.2857, 1.0)
    colorramp.color_ramp.elements[4].position = 0.9455
    colorramp.color_ramp.elements[4].color = (0.2134, 0.0921, 0.0372, 1.0)
    
    noise_texture_2 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': reroute, 'Scale': 500.0})
    
    colorramp_1 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': noise_texture_2.outputs["Fac"]})
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements[0].position = 0.0
    colorramp_1.color_ramp.elements[0].color = (0.1356, 0.0648, 0.0273, 1.0)
    colorramp_1.color_ramp.elements[1].position = 0.25
    colorramp_1.color_ramp.elements[1].color = (0.4851, 0.3005, 0.1651, 1.0)
    colorramp_1.color_ramp.elements[2].position = 0.5
    colorramp_1.color_ramp.elements[2].color = (0.0911, 0.0398, 0.0117, 1.0)
    colorramp_1.color_ramp.elements[3].position = 0.75
    colorramp_1.color_ramp.elements[3].color = (0.6724, 0.4179, 0.2623, 1.0)
    colorramp_1.color_ramp.elements[4].position = 1.0
    colorramp_1.color_ramp.elements[4].color = (0.1946, 0.0844, 0.0343, 1.0)
    
    mix_1 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Color1': colorramp.outputs["Color"], 'Color2': colorramp_1.outputs["Color"]},
        attrs={'blend_type': 'MULTIPLY'})
    
    mix_2 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': less_than_1, 'Color1': (0.4969, 0.305, 0.1746, 1.0), 'Color2': mix_1})
    
    colorramp_3 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': group})
    colorramp_3.color_ramp.elements[0].position = 0.0
    colorramp_3.color_ramp.elements[0].color = (0.4969, 0.305, 0.1746, 1.0)
    colorramp_3.color_ramp.elements[1].position = 1.0
    colorramp_3.color_ramp.elements[1].color = (0.9684, 1.0, 0.6723, 1.0)
    
    mix_6 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': group, 'Color1': mix_2, 'Color2': colorramp_3.outputs["Color"]})
    
    power_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract, 1: 2.0},
        attrs={'operation': 'POWER'})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: power_1, 1: 1000000.0},
        attrs={'operation': 'MULTIPLY'})
    
    less_than_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_1, 1: 0.001},
        attrs={'operation': 'LESS_THAN'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: less_than_2, 1: divide},
        attrs={'use_clamp': True})
    
    multiply_2 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: less_than_2, 1: attribute_1.outputs["Color"]},
        attrs={'operation': 'MULTIPLY'})
    
    noise_texture_3 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': multiply_2.outputs["Vector"], 'Scale': 100.0, 'Roughness': 0.49})
    
    mix_3 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': less_than_2, 'Color1': (0.0, 0.0, 0.0, 1.0), 'Color2': noise_texture_3.outputs["Fac"]})
    
    mix_4 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': less_than_2, 'Color1': add, 'Color2': mix_3})
    
    colorramp_2 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': mix_4})
    colorramp_2.color_ramp.elements.new(0)
    colorramp_2.color_ramp.elements.new(0)
    colorramp_2.color_ramp.elements[0].position = 0.0045
    colorramp_2.color_ramp.elements[0].color = (0.0638, 0.0429, 0.0278, 1.0)
    colorramp_2.color_ramp.elements[1].position = 0.4886
    colorramp_2.color_ramp.elements[1].color = (0.1734, 0.1096, 0.0655, 1.0)
    colorramp_2.color_ramp.elements[2].position = 0.5
    colorramp_2.color_ramp.elements[2].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_2.color_ramp.elements[3].position = 0.5591
    colorramp_2.color_ramp.elements[3].color = (0.4524, 0.3119, 0.1992, 1.0)
    
    mix_5 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp_4.outputs["Color"], 'Color1': mix_6, 'Color2': colorramp_2.outputs["Color"]})
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': mix_5})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

def geometry_reptile_vor(nw: NodeWrangler, rand=True, **input_kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketVector', 'value', (0.0, 0.0, 0.0))])
    
    normal = nw.new_node(Nodes.InputNormal)
    
    multiply = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["value"], 1: normal},
        attrs={'operation': 'MULTIPLY'})
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.003
    
    multiply_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: multiply.outputs["Vector"], 1: value},
        attrs={'operation': 'MULTIPLY'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Offset': multiply_1.outputs["Vector"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_position})

def geometry_reptile_vor_attr(nw: NodeWrangler, rand=True, **input_kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None)])
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["Geometry"]})
    
    position = nw.new_node(Nodes.InputPosition)
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 1.0
    
    multiply = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: position, 1: value},
        attrs={'operation': 'MULTIPLY'})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': multiply.outputs["Vector"], 'Scale': 6.0, 'Detail': 15.0})
    
    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.1, 'Color1': multiply.outputs["Vector"], 'Color2': noise_texture.outputs["Fac"]},
        attrs={'blend_type': 'ADD'})
    
    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = 0.0
    
    value_2 = nw.new_node(Nodes.Value)
    value_2.outputs[0].default_value = 80.0
    
    voronoi_texture = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': mix, 'W': value_1, 'Scale': value_2},
        attrs={'voronoi_dimensions': '4D', 'feature': 'DISTANCE_TO_EDGE'})
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': voronoi_texture.outputs["Distance"]})
    colorramp.color_ramp.elements[0].position = 0.02
    colorramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp.color_ramp.elements[1].position = 0.2
    colorramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    noise_texture_1 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': multiply.outputs["Vector"], 'Scale': 100.0})
    
    colorramp_1 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': noise_texture_1.outputs["Fac"]})
    colorramp_1.color_ramp.elements[0].position = 0.1
    colorramp_1.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_1.color_ramp.elements[1].position = 0.4
    colorramp_1.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    mix_1 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp.outputs["Color"], 'Color1': colorramp.outputs["Color"], 'Color2': colorramp_1.outputs["Color"]})
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': set_position, 1: mix_1},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    voronoi_texture_1 = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': mix, 'W': value_1, 'Scale': value_2},
        attrs={'voronoi_dimensions': '4D'})
    
    capture_attribute_1 = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': capture_attribute.outputs["Geometry"], 1: voronoi_texture_1.outputs["Position"]},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': capture_attribute_1.outputs["Geometry"], 'attr1': capture_attribute.outputs["Attribute"], 'attr2': capture_attribute_1.outputs["Attribute"]})

def apply(obj, geo_kwargs=None, shader_kwargs=None, **kwargs):
    surface.add_geomod(obj, geometry_reptile_vor_attr, input_kwargs=geo_kwargs, attributes=['value', 'index'])
    surface.add_geomod(obj, geometry_reptile_vor, input_kwargs=geo_kwargs, attributes=[])
    surface.add_material(obj, shader_brown_circle, input_kwargs=shader_kwargs)

if __name__ == "__main__":
    for i in range(1):
        bpy.ops.wm.open_mainfile(filepath='dev_scene_1019.blend')
        #creature(73349, 0).parts(0, factory=QuadrupedBody)
        import generated_surface_script_replile_gray as gray
        import generated_surface_script_replile_two_color as two_color
        apply(bpy.data.objects['creature(73349, 0).parts(0, factory=QuadrupedBody)'], geo_kwargs={'rand': True}, shader_kwargs={'rand': True, 'mat_name':'brown_circle'})
        gray.apply(bpy.data.objects['creature(19946, 0).parts(0, factory=QuadrupedBody)'], geo_kwargs={'rand': True}, shader_kwargs={'rand': True, 'mat_name':'two_color'})
        two_color.apply(bpy.data.objects['creature(51668, 0).parts(0, factory=QuadrupedBody)'], geo_kwargs={'rand': True}, shader_kwargs={'rand': True, 'mat_name':'gray'})
        fn = os.path.join(os.path.abspath(os.curdir), 'dev_scene_test_brown_circle_attr.blend')
        bpy.ops.wm.save_as_mainfile(filepath=fn)
        #bpy.context.scene.render.filepath = os.path.join('surfaces/surface_thumbnails', 'bone%d.jpg'%(i))
        #bpy.context.scene.render.image_settings.file_format='JPEG'
        #bpy.ops.render.render(write_still=True)