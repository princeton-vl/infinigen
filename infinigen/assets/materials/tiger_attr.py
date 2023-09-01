# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Mingzhe Wang
# Acknowledgment: This file draws inspiration from https://www.youtube.com/watch?v=b9lukB7cWag by Sam Bowman


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
import bpy
import mathutils
from numpy.random import uniform, normal, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface

from infinigen.assets.creatures.util.nodegroups.shader import nodegroup_color_mask

@node_utils.to_nodegroup('nodegroup_tiger_fac', singleton=False, type='ShaderNodeTree')
def nodegroup_tiger_fac(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketVector', 'Vector', (0.0000, 0.0000, 0.0000))])
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': group_input.outputs["Vector"]},
        attrs={'noise_dimensions': '4D'})
    noise_texture.inputs["W"].default_value = uniform(-10, 10)
    
    mix_3 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.8500, 'Color1': noise_texture.outputs["Color"], 'Color2': group_input.outputs["Vector"]})
    mix_3.inputs["Factor"].default_value = uniform(0.8, 0.9)
    
    musgrave_texture_1 = nw.new_node(Nodes.MusgraveTexture,
        input_kwargs={'Vector': mix_3, 'W': 1.0000, 'Scale': 1.0000},
        attrs={'musgrave_dimensions': '4D'})
    musgrave_texture_1.inputs["W"].default_value = uniform(-10, 10)
    
    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = normal(0.1180, 0.0100)
    
    value_2 = nw.new_node(Nodes.Value)
    value_2.outputs[0].default_value = normal(0.0600, 0.0100)
    
    add = nw.new_node(Nodes.Math, input_kwargs={0: value_1, 1: value_2})
    
    greater_than = nw.new_node(Nodes.Math, input_kwargs={0: musgrave_texture_1, 1: add}, attrs={'operation': 'GREATER_THAN'})
    
    subtract = nw.new_node(Nodes.Math, input_kwargs={0: value_1, 1: value_2}, attrs={'operation': 'SUBTRACT'})
    
    greater_than_1 = nw.new_node(Nodes.Math, input_kwargs={0: musgrave_texture_1, 1: subtract}, attrs={'operation': 'GREATER_THAN'})
    
    less_than = nw.new_node(Nodes.Math, input_kwargs={0: greater_than, 1: greater_than_1}, attrs={'operation': 'LESS_THAN'})
    
    colorramp_3 = nw.new_node(Nodes.ColorRamp, input_kwargs={'Fac': musgrave_texture_1})
    colorramp_3.color_ramp.interpolation = "CONSTANT"
    colorramp_3.color_ramp.elements[0].position = 0.0000
    colorramp_3.color_ramp.elements[0].color = [0.0000, 0.0000, 0.0000, 1.0000]
    colorramp_3.color_ramp.elements[1].position = 0.1182
    colorramp_3.color_ramp.elements[1].color = [1.0000, 1.0000, 1.0000, 1.0000]
    
    mapping_2 = nw.new_node(Nodes.Mapping, input_kwargs={'Vector': mix_3, 'Location': (3.0000, 0.0000, 1.0000)})
    mapping_2.inputs["Location"].default_value = (uniform(0, 10), uniform(0, 10), uniform(0, 10))
    
    mapping_1 = nw.new_node(Nodes.Mapping,
        input_kwargs={'Vector': mix_3, 'Location': (1.0000, 5.0000, -10.0000), 'Rotation': (0.7854, 0.0000, 0.0000)})
    mapping_1.inputs["Location"].default_value = (uniform(0, 10), uniform(0, 10), uniform(0, 10))
    
    mix_5 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp_3.outputs["Color"], 'Color1': mapping_2, 'Color2': mapping_1})
    
    wave_texture = nw.new_node(Nodes.WaveTexture,
        input_kwargs={'Vector': mix_5, 'Scale': 1.0000, 'Distortion': 10.0000, 'Detail': 0.0000, 'Phase Offset': 4.0000})
    wave_texture.inputs["Scale"].default_value = normal(1.0000, 0.1000)
    wave_texture.inputs["Distortion"].default_value = normal(10.0000, 0.5000)
    wave_texture.inputs["Phase Offset"].default_value = uniform(-20, 20)

    map_range = nw.new_node(Nodes.MapRange, input_kwargs={'Value': wave_texture.outputs["Fac"], 1: 0.2000, 2: 0.4000})
    
    mix_2 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': less_than, 'Color1': map_range.outputs["Result"], 'Color2': (1.0000, 1.0000, 1.0000, 1.0000)})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Color': mix_2}, attrs={'is_active_output': True})


def shader_tiger_attr(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    local_pos = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'local_pos'})
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = U(2, 4)
    
    attribute = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'tag_head'})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: attribute.outputs["Fac"], 1: N(3, 1)},
        attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: value, 1: multiply})
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: local_pos.outputs["Vector"], 'Scale': add},
        attrs={'operation': 'SCALE'})
    
    group_1 = nw.new_node(nodegroup_tiger_fac().name,
        input_kwargs={'Vector': scale.outputs["Vector"]})
    
    group = nw.new_node(nodegroup_color_mask().name)
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': group})
    colorramp.color_ramp.elements[0].position = 0.0
    colorramp.color_ramp.elements[0].color = hsv2rgba((U(0.015, 0.06), U(0.85, 0.95), U(0.15, 0.7)))
    colorramp.color_ramp.elements[1].position = 1.0
    colorramp.color_ramp.elements[1].color = hsv2rgba((U(0.02, 0.05), U(0.3, 0.7), U(0.15, 0.7)))
    
    mix_5 = nw.new_node(Nodes.MixRGB,
        input_kwargs={
            'Fac': group_1, 
            'Color1': (0.01, 0.01, 0.01, 1.0), 
            'Color2': colorramp.outputs["Color"]})
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': mix_5, 'Specular': 0.0},
        attrs={'subsurface_method': 'BURLEY'})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

def apply(obj, selection=None, **kwargs):
    surface.add_material(obj, shader_tiger_attr, selection=selection)

if __name__ == "__main__":
    for i in range(1):
        bpy.ops.wm.open_mainfile(filepath='dev_scene_1019.blend')
        #creature(73349, 0).parts(0, factory=QuadrupedBody)
        apply(bpy.data.objects['creature(73349, 0).parts(0, factory=QuadrupedBody)'], geo_kwargs={'rand': True}, shader_kwargs={'rand': True})
        fn = os.path.join(os.path.abspath(os.curdir), 'dev_scene_test_tiger_attr.blend')
        bpy.ops.wm.save_as_mainfile(filepath=fn)
        #bpy.context.scene.render.filepath = os.path.join('surfaces/surface_thumbnails', 'bone%d.jpg'%(i))
        #bpy.context.scene.render.image_settings.file_format='JPEG'
        #bpy.ops.render.render(write_still=True)