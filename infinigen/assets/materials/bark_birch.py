# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo
# Acknowledgement: This file draws inspiration from https://www.youtube.com/watch?v=SAbWUs1Rnxw by Sam Bowman

# Code generated using version 2.1.0 of the node_transpiler
import bpy
import mathutils
from numpy.random import uniform, normal
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core import surface

from .bark_random import nodegroup_apply_geo_matv2, nodegroup_shader_canonical_coord, nodegroup_canonical_coord

@node_utils.to_nodegroup('nodegroup_birch_mat_helper', singleton=False, type='ShaderNodeTree')
def nodegroup_birch_mat_helper(nw):
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'Vector', (0.0, 0.0, 0.0))])
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': group_input.outputs["Vector"], 'Scale': 50.0, 'Detail': 10.0})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': noise_texture.outputs["Fac"], 1: 0.3, 2: 0.4, 3: 1.0, 4: 0.0})
    
    musgrave_texture = nw.new_node(Nodes.MusgraveTexture,
        input_kwargs={'Vector': group_input.outputs["Vector"], 'Scale': 2.0, 'Detail': 10.0, 'Dimension': 0.6, 'Lacunarity': 3.0})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': musgrave_texture, 1: 0.3, 2: 0.5})
    
    power = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_1.outputs["Result"], 1: 0.2},
        attrs={'operation': 'POWER', 'use_clamp': True})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group_input.outputs["Vector"]})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: 10.0},
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': separate_xyz.outputs["X"], 'Y': separate_xyz.outputs["Y"], 'Z': multiply})
    
    noise_texture_1 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': combine_xyz, 'Scale': 2.0})
    
    separate_rgb = nw.new_node(Nodes.SeparateRGB,
        input_kwargs={'Image': noise_texture_1.outputs["Color"]})
    
    map_range_2 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_rgb.outputs["B"], 1: 0.341, 2: 0.377})
    
    map_range_3 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_rgb.outputs["R"], 1: 0.341, 2: 0.377})
    
    map_range_4 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_rgb.outputs["G"], 1: 0.341, 2: 0.377})
    
    multiply_add = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_3.outputs["Result"], 1: map_range_4.outputs["Result"]},
        attrs={'operation': 'MULTIPLY_ADD'})
    
    map_range_5 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': multiply_add, 2: 2.0})
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': map_range_5.outputs["Result"]})
    colorramp.color_ramp.elements[0].position = 0.5052
    colorramp.color_ramp.elements[0].color = (0.0252, 0.0395, 0.0176, 1.0)
    colorramp.color_ramp.elements[1].position = 0.8015
    colorramp.color_ramp.elements[1].color = (0.3095, 0.4072, 0.3515, 1.0)
    
    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': map_range_2.outputs["Result"], 'Color1': (0.0823, 0.1095, 0.0595, 1.0), 'Color2': colorramp.outputs["Color"]})
    
    mix_1 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': power, 'Color1': mix, 'Color2': (0.0232, 0.0144, 0.0021, 1.0)})
    
    mix_2 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': map_range.outputs["Result"], 'Color1': mix_1, 'Color2': (0.0437, 0.0482, 0.0222, 1.0)})
    
    map_range_6 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': power, 3: 0.4})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Color': mix_2, 'Result': map_range_6.outputs["Result"]})

@node_utils.to_nodegroup('nodegroup_birch_geo', singleton=False, type='GeometryNodeTree')
def nodegroup_birch_geo(nw):
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'Position', (0.0, 0.0, 0.0))])
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group_input.outputs["Position"]})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: 10.0},
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': separate_xyz.outputs["X"], 'Y': separate_xyz.outputs["Y"], 'Z': multiply})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': combine_xyz, 'Scale': 2.0})
    
    separate_rgb = nw.new_node(Nodes.SeparateRGB,
        input_kwargs={'Image': noise_texture.outputs["Color"]})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_rgb.outputs["G"], 1: 0.341, 2: 0.377, 3: 1.0, 4: 0.0})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_rgb.outputs["R"], 1: 0.341, 2: 0.377, 3: 1.0, 4: 0.0})
    
    multiply_add = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range.outputs["Result"], 1: 0.25, 2: map_range_1.outputs["Result"]},
        attrs={'operation': 'MULTIPLY_ADD'})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_add, 1: 0.01},
        attrs={'operation': 'MULTIPLY'})
    
    noise_texture_1 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': group_input.outputs["Position"], 'Scale': 10.0, 'Detail': 15.0})
    
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: noise_texture_1.outputs["Fac"], 1: 0.08},
        attrs={'operation': 'MULTIPLY'})
    
    musgrave_texture = nw.new_node(Nodes.MusgraveTexture,
        input_kwargs={'Vector': group_input.outputs["Position"], 'Scale': 2.0, 'Detail': 10.0, 'Dimension': 0.6, 'Lacunarity': 3.0})
    
    map_range_2 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': musgrave_texture, 1: 0.3, 2: 0.5})
    
    power = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_2.outputs["Result"], 1: 0.2},
        attrs={'operation': 'POWER', 'use_clamp': True})
    
    multiply_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: power, 1: 0.03},
        attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_2, 1: multiply_3})
    
    add_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_1, 1: add})
    
    multiply_4 = nw.new_node(Nodes.Math,
        input_kwargs={0: add_1, 1: 5.0},
        attrs={'operation': 'MULTIPLY'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Factor': multiply_4})

def shader_birch_mat(nw, selection=None):
    attribute = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'initial_position'})
    
    group = nw.new_node(nodegroup_shader_canonical_coord().name,
        input_kwargs={'Vector': attribute.outputs["Vector"]})
    
    group_1 = nw.new_node(nodegroup_birch_mat_helper().name,
        input_kwargs={'Vector': group})
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': group_1.outputs["Color"], 'Roughness': group_1.outputs["Result"]})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

def geo_bark_birch(nw, selection=None):
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),])

    parent_loc = nw.new_node(Nodes.NamedAttribute, ['parent_skeleton_loc'], attrs={'data_type': 'FLOAT_VECTOR'})
    skeleton_loc = nw.new_node(Nodes.NamedAttribute, ['skeleton_loc'], attrs={'data_type': 'FLOAT_VECTOR'})
        
    position = nw.new_node(Nodes.InputPosition)
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 1: position},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    canonicalcoord = nw.new_node(nodegroup_canonical_coord().name,
        input_kwargs={'Self Location': skeleton_loc, 'Parent Location': parent_loc})
    
    birchgeo = nw.new_node(nodegroup_birch_geo().name,
        input_kwargs={'Position': canonicalcoord})
    
    group = nw.new_node(nodegroup_apply_geo_matv2().name,
        input_kwargs={
            'Geometry': capture_attribute.outputs["Geometry"], 
            'Displacement Amount': nw.multiply(birchgeo, surface.eval_argument(nw, selection)), 
            'Displacement Scale': 0.05, 
            'Material': surface.shaderfunc_to_material(shader_birch_mat)
        })
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': group, 'initial_position': capture_attribute.outputs["Attribute"]})



def apply(obj, selection=None, **kwargs):
    surface.add_geomod(obj, geo_bark_birch, selection=selection, attributes=['initial_position'])
    surface.add_material(obj, shader_birch_mat)
