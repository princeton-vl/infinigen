# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo


# Code generated using version 2.1.0 of the node_transpiler
from typing import Tuple
import bpy
import mathutils
from numpy.random import uniform, normal
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core import surface
import numpy as np

from infinigen.core.util.math import FixedSeed

@node_utils.to_nodegroup('nodegroup_calc_radius', singleton=True, type='GeometryNodeTree')
def nodegroup_calc_radius(nw):
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'Self Location', (0.0, 0.0, 0.0)),
            ('NodeSocketVector', 'Parent Location', (0.0, 0.0, 0.0))])
    
    subtract = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Self Location"], 1: group_input.outputs["Parent Location"]},
        attrs={'operation': 'SUBTRACT'})
    
    normalize = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: subtract.outputs["Vector"]},
        attrs={'operation': 'NORMALIZE'})
    
    position = nw.new_node(Nodes.InputPosition)
    
    subtract_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: position, 1: group_input.outputs["Parent Location"]},
        attrs={'operation': 'SUBTRACT'})
    
    dot_product = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: normalize.outputs["Vector"], 1: subtract_1.outputs["Vector"]},
        attrs={'operation': 'DOT_PRODUCT'})
    
    multiply = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: normalize.outputs["Vector"], 1: dot_product.outputs["Value"]},
        attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Parent Location"], 1: multiply.outputs["Vector"]})
    
    subtract_2 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: add.outputs["Vector"], 1: position},
        attrs={'operation': 'SUBTRACT'})
    
    length = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: subtract_2.outputs["Vector"]},
        attrs={'operation': 'LENGTH'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Radius': length.outputs["Value"]})

@node_utils.to_nodegroup('nodegroup_shader_canonical_coord', singleton=True, type='ShaderNodeTree')
def nodegroup_shader_canonical_coord(nw):
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'scale', 2.0),
            ('NodeSocketVector', 'Vector', (0.0, 0.0, 0.0))])
    
    attribute = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'parent_skeleton_loc'})
    
    attribute_1 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'skeleton_loc'})
    
    subtract = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: attribute_1.outputs["Vector"], 1: attribute.outputs["Vector"]},
        attrs={'operation': 'SUBTRACT'})
    
    cross_product = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: subtract.outputs["Vector"], 1: (0.0, 0.0, 1.0)},
        attrs={'operation': 'CROSS_PRODUCT'})
    
    dot_product = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: subtract.outputs["Vector"], 1: (0.0, 0.0, 1.0)},
        attrs={'operation': 'DOT_PRODUCT'})
    
    length = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: subtract.outputs["Vector"]},
        attrs={'operation': 'LENGTH'})
    
    divide = nw.new_node(Nodes.Math,
        input_kwargs={0: dot_product.outputs["Value"], 1: length.outputs["Value"]},
        attrs={'operation': 'DIVIDE'})
    
    arccosine = nw.new_node(Nodes.Math,
        input_kwargs={0: divide},
        attrs={'operation': 'ARCCOSINE'})
    
    vector_rotate = nw.new_node(Nodes.VectorRotate,
        input_kwargs={'Vector': group_input.outputs["Vector"], 'Center': attribute.outputs["Vector"], 'Axis': cross_product.outputs["Vector"], 'Angle': arccosine})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': attribute.outputs["Vector"]})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': separate_xyz.outputs["X"], 'Y': separate_xyz.outputs["Y"]})
    
    subtract_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: vector_rotate, 1: combine_xyz},
        attrs={'operation': 'SUBTRACT'})
    
    multiply = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: subtract_1.outputs["Vector"], 1: group_input.outputs["scale"]},
        attrs={'operation': 'MULTIPLY'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Coordinate': multiply.outputs["Vector"]})

@node_utils.to_nodegroup('nodegroup_inject_z_noise_and_scale_001', singleton=True, type='GeometryNodeTree')
def nodegroup_inject_z_noise_and_scale_001(nw):
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'Coordinate', (0.0, 0.0, 0.0)),
            ('NodeSocketFloat', 'Noise Scale', 2.0),
            ('NodeSocketFloat', 'Noise Amount', 0.5),
            ('NodeSocketFloat', 'Z Multiplier', 0.5)])
    
    separate_xyz_6 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group_input.outputs["Coordinate"]})
    
    noise_texture_2 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': group_input.outputs["Coordinate"], 'Scale': group_input.outputs["Noise Scale"]})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: noise_texture_2.outputs["Fac"], 1: group_input.outputs["Noise Amount"]},
        attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_6.outputs["Z"], 1: multiply})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: add, 1: group_input.outputs["Z Multiplier"]},
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz_6 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': separate_xyz_6.outputs["X"], 'Y': separate_xyz_6.outputs["Y"], 'Z': multiply_1})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Coordinate': combine_xyz_6})

@node_utils.to_nodegroup('nodegroup_primary_voronoi', singleton=True, type='GeometryNodeTree')
def nodegroup_primary_voronoi(nw):
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'Coordinate', (0.0, 0.0, 0.0)),
            ('NodeSocketFloat', 'Texture Scale', 20.0),
            ('NodeSocketFloatFactor', 'Randomness', 1.0)])
    
    voronoi_texture_3 = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': group_input.outputs["Coordinate"], 'Scale': group_input.outputs["Texture Scale"], 'Randomness': group_input.outputs["Randomness"]},
        attrs={'feature': 'DISTANCE_TO_EDGE'})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': voronoi_texture_3.outputs["Distance"], 2: 0.1})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Displacement': map_range.outputs["Result"]})

@node_utils.to_nodegroup('nodegroup_mix', singleton=True, type='GeometryNodeTree')
def nodegroup_mix(nw):
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'Input 1', 0.5),
            ('NodeSocketFloat', 'Input 2', 0.5),
            ('NodeSocketFloat', 'Mix Weight', 0.5)])
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Input 1"], 1: group_input.outputs["Mix Weight"]},
        attrs={'operation': 'MULTIPLY'})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: 1.0, 1: group_input.outputs["Mix Weight"]},
        attrs={'operation': 'SUBTRACT'})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Input 2"], 1: subtract},
        attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply, 1: multiply_1})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Value': add})

@node_utils.to_nodegroup('nodegroup_adjust_v', singleton=True, type='ShaderNodeTree')
def nodegroup_adjust_v(nw):
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketColor', 'Color', (0.8, 0.8, 0.8, 1.0)),
            ('NodeSocketFloat', 'V Shift', 0.5)])
    
    separate_hsv = nw.new_node('ShaderNodeSeparateHSV',
        input_kwargs={'Color': group_input.outputs["Color"]})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_hsv.outputs["V"], 1: group_input.outputs["V Shift"]})
    
    combine_hsv = nw.new_node(Nodes.CombineHSV,
        input_kwargs={'H': separate_hsv.outputs["H"], 'S': separate_hsv.outputs["S"], 'V': add})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Color': combine_hsv})

@node_utils.to_nodegroup('nodegroup_inject_z_noise_and_scale', singleton=True, type='ShaderNodeTree')
def nodegroup_inject_z_noise_and_scale(nw):
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'Coordinate', (0.0, 0.0, 0.0)),
            ('NodeSocketFloat', 'Noise Scale', 5.0),
            ('NodeSocketFloat', 'Noise Amount', 0.0),
            ('NodeSocketFloat', 'Z Multiplier', 0.5)])
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group_input.outputs["Coordinate"]})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': group_input.outputs["Coordinate"], 'Scale': group_input.outputs["Noise Scale"]})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: noise_texture.outputs["Fac"], 1: group_input.outputs["Noise Amount"]},
        attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: multiply})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: add, 1: group_input.outputs["Z Multiplier"]},
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': separate_xyz.outputs["X"], 'Y': separate_xyz.outputs["Y"], 'Z': multiply_1})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Coordiante': combine_xyz})

@node_utils.to_nodegroup('nodegroup_voronoi', singleton=True, type='ShaderNodeTree')
def nodegroup_voronoi(nw):
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'Coordinate', (0.0, 0.0, 0.0)),
            ('NodeSocketFloat', 'Texture Scale', 5.0),
            ('NodeSocketFloatFactor', 'Randomness', 1.0)])
    
    voronoi_texture = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': group_input.outputs["Coordinate"], 'Scale': group_input.outputs["Texture Scale"], 'Randomness': group_input.outputs["Randomness"]},
        attrs={'feature': 'DISTANCE_TO_EDGE'})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': voronoi_texture.outputs["Distance"], 2: 0.5})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Displacement': map_range.outputs["Result"]})

@node_utils.to_nodegroup('nodegroup_canonical_coord', singleton=True, type='GeometryNodeTree')
def nodegroup_canonical_coord(nw):
    position = nw.new_node(Nodes.InputPosition)
    
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'Self Location', (0.0, 0.0, 0.0)),
            ('NodeSocketVector', 'Parent Location', (0.0, 0.0, 0.0)),
            ('NodeSocketFloat', 'scale', 2.0)])
    
    subtract = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Self Location"], 1: group_input.outputs["Parent Location"]},
        attrs={'operation': 'SUBTRACT'})
    
    cross_product = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: subtract.outputs["Vector"], 1: (0.0, 0.0, 1.0)},
        attrs={'operation': 'CROSS_PRODUCT'})
    
    dot_product = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: subtract.outputs["Vector"], 1: (0.0, 0.0, 1.0)},
        attrs={'operation': 'DOT_PRODUCT'})
    
    length = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: subtract.outputs["Vector"]},
        attrs={'operation': 'LENGTH'})
    
    divide = nw.new_node(Nodes.Math,
        input_kwargs={0: dot_product.outputs["Value"], 1: length.outputs["Value"]},
        attrs={'operation': 'DIVIDE'})
    
    arccosine = nw.new_node(Nodes.Math,
        input_kwargs={0: divide},
        attrs={'operation': 'ARCCOSINE'})
    
    vector_rotate = nw.new_node(Nodes.VectorRotate,
        input_kwargs={'Vector': position, 'Center': group_input.outputs["Parent Location"], 'Axis': cross_product.outputs["Vector"], 'Angle': arccosine})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group_input.outputs["Parent Location"]})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': separate_xyz.outputs["X"], 'Y': separate_xyz.outputs["Y"]})
    
    subtract_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: vector_rotate, 1: combine_xyz},
        attrs={'operation': 'SUBTRACT'})
    
    multiply = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: subtract_1.outputs["Vector"], 1: group_input.outputs["scale"]},
        attrs={'operation': 'MULTIPLY'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Coordinate': multiply.outputs["Vector"]})

@node_utils.to_nodegroup('nodegroup_random_bark_geo', singleton=True, type='GeometryNodeTree')
def nodegroup_random_bark_geo(nw):
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'Position', (0.0, 0.0, 0.0)),
            ('NodeSocketFloat', 'Noise Scale', 2.0),
            ('NodeSocketFloat', 'Noise Amount', 1.0),
            ('NodeSocketFloat', 'Texture Scale', 30.0),
            ('NodeSocketFloatFactor', 'Randomness', 1.0),
            ('NodeSocketFloat', 'Value', 0.05),
            ('NodeSocketFloat', 'Mix Weight', 0.1),
            ('NodeSocketFloat', 'Scale', 15.0),
            ('NodeSocketFloat', 'Detail', 16.0),
            ('NodeSocketFloat', 'Value_1', 2.0),
            ('NodeSocketFloat', 'Z Multiplier', 0.5),
            ('NodeSocketFloat', 'Texture Scale S', 30.0)])

    group_3 = nw.new_node(nodegroup_primary_voronoi().name,
        input_kwargs={'Coordinate': group_input.outputs["Position"], 'Texture Scale': group_input.outputs['Texture Scale S']})
    
    group = nw.new_node(nodegroup_inject_z_noise_and_scale_001().name,
        input_kwargs={'Coordinate': group_input.outputs["Position"], 'Noise Scale': group_input.outputs["Noise Scale"], 'Noise Amount': group_input.outputs["Noise Amount"], 'Z Multiplier': group_input.outputs["Z Multiplier"]})
    
    group_2 = nw.new_node(nodegroup_primary_voronoi().name,
        input_kwargs={'Coordinate': group, 'Texture Scale': group_input.outputs["Texture Scale"], 'Randomness': group_input.outputs["Randomness"]})
    
    group_5 = nw.new_node(nodegroup_mix().name,
        input_kwargs={'Input 1': group_3, 'Input 2': group_2, 'Mix Weight': group_input.outputs["Mix Weight"]})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': group_input.outputs["Position"], 'Scale': group_input.outputs["Value_1"], 'Detail': group_input.outputs["Detail"]})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: noise_texture.outputs["Fac"], 1: group_input.outputs['Noise Scale']},
        attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: group_5, 1: multiply})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: add, 1: group_input.outputs["Value"]},
        attrs={'operation': 'MULTIPLY'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Value': multiply_1})

@node_utils.to_nodegroup('nodegroup_apply_geo_matv2', singleton=True, type='GeometryNodeTree')
def nodegroup_apply_geo_matv2(nw):
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketFloat', 'Displacement Amount', 0.0),
            ('NodeSocketFloat', 'Displacement Scale', 0.0),
            ('NodeSocketMaterial', 'Material', None)])
    
    normal = nw.new_node(Nodes.InputNormal)
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Displacement Amount"], 1: group_input.outputs["Displacement Scale"]},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: normal, 1: multiply},
        attrs={'operation': 'MULTIPLY'})
    
    set_position_1 = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Offset': multiply_1.outputs["Vector"]})
    
    set_material = nw.new_node(Nodes.SetMaterial,
        input_kwargs={'Geometry': set_position_1, 'Material': group_input.outputs["Material"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_material})

def shader_random_bark_mat(nw, base_color:Tuple, geo_params, selection=None):
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: geo_params['Noise Texture Scale'], 1: 4.0},
        attrs={'operation': 'MULTIPLY'})
    
    noise_texture_1 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Scale': multiply})
    
    map_range_2 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': noise_texture_1.outputs["Fac"], 1: 0.35, 2: 0.4, 3: 0.5, 4: 0.0})
    
    attribute_5 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'initial_position'})

    group_canonical = nw.new_node(nodegroup_shader_canonical_coord().name,
        input_kwargs={'Vector': attribute_5.outputs["Vector"]})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: geo_params['Noise Texture Scale'], 1: 2.0},
        attrs={'operation': 'MULTIPLY'})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': group_canonical, 'Scale': multiply_1})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': noise_texture.outputs["Fac"], 1: 0.35, 2: 0.4, 3: 0.5, 4: 0.0})
    
    group_2 = nw.new_node(nodegroup_voronoi().name,
        input_kwargs={'Coordinate': group_canonical, 'Texture Scale': geo_params['Secondary Voronoi Scale']})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': group_2, 3: 0.7, 4: 0.0})
    
    group = nw.new_node(nodegroup_inject_z_noise_and_scale().name,
        input_kwargs={'Coordinate': group_canonical, 'Noise Scale': geo_params['Z Noise Scale'], 'Noise Amount': geo_params['Z Noise Amount'], 'Z Multiplier': geo_params['Z Multiplier']})
    
    group_1 = nw.new_node(nodegroup_voronoi().name,
        input_kwargs={'Coordinate': group, 'Texture Scale': geo_params['Primary Voronoi Scale'], 'Randomness': geo_params['Primary Voronoi Randomness']})
    
    rgb_1 = nw.new_node(Nodes.RGB)
    rgb_1.outputs[0].default_value = base_color

    # todo: this value needs to be assigned
    
    group_3 = nw.new_node(nodegroup_adjust_v().name,
        input_kwargs={'Color': rgb_1, 'V Shift': 0.1})
    
    mix_4 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.1, 'Color1': (0.0, 0.0, 0.0, 1.0), 'Color2': group_3})
    
    mix_3 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': group_1, 'Color1': mix_4, 'Color2': group_3})
    
    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': map_range.outputs["Result"], 'Color1': mix_3, 'Color2': mix_4})
    
    mix_6 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.9, 'Color1': (1.0, 1.0, 1.0, 1.0), 'Color2': group_3})
    
    mix_1 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': map_range_1.outputs["Result"], 'Color1': mix, 'Color2': mix_6})
    
    mix_5 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.9, 'Color1': (0.0, 0.0, 0.0, 1.0), 'Color2': group_3})
    
    mix_2 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': map_range_2.outputs["Result"], 'Color1': mix_1, 'Color2': mix_5})
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': mix_2, 'Roughness': 0.7})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

def geo_bark_random(nw, base_color, geo_params, selection=None):
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketFloat', 'Displacement Scale', geo_params['Displacement Scale']),
            ('NodeSocketFloat', 'Z Noise Scale', geo_params['Z Noise Scale']),
            ('NodeSocketFloat', 'Z Noise Amount', geo_params['Z Noise Amount']),
            ('NodeSocketFloat', 'Z Multiplier', geo_params['Z Multiplier']),
            ('NodeSocketFloat', 'Primary Voronoi Scale', geo_params['Primary Voronoi Scale']),
            ('NodeSocketFloatFactor', 'Primary Voronoi Randomness', geo_params['Primary Voronoi Randomness']),
            ('NodeSocketFloat', 'Secondary Voronoi Mix Weight', geo_params['Secondary Voronoi Mix Weight']),
            ('NodeSocketFloat', 'Secondary Voronoi Scale', geo_params['Secondary Voronoi Scale']),
            ('NodeSocketFloat', 'Noise Texture Scale', geo_params['Noise Texture Scale']),
            ('NodeSocketFloat', 'Noise Texture Detail', geo_params['Noise Texture Detail']),
            ('NodeSocketFloat', 'Noise Texture Weight', geo_params['Noise Texture Weight'])])

    parent_loc = nw.new_node(Nodes.NamedAttribute, ['parent_skeleton_loc'], attrs={'data_type': 'FLOAT_VECTOR'})
    skeleton_loc = nw.new_node(Nodes.NamedAttribute, ['skeleton_loc'], attrs={'data_type': 'FLOAT_VECTOR'})
    
    position = nw.new_node(Nodes.InputPosition)
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 1: position},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    canonicalcoord = nw.new_node(nodegroup_canonical_coord().name,
        input_kwargs={'Self Location': skeleton_loc, 'Parent Location': parent_loc})

    group_1 = nw.new_node(nodegroup_random_bark_geo().name,
        input_kwargs={'Position': canonicalcoord, 'Noise Scale': group_input.outputs["Z Noise Scale"], 'Noise Amount': group_input.outputs["Z Noise Amount"], 3: group_input.outputs["Primary Voronoi Scale"], 'Randomness': group_input.outputs["Primary Voronoi Randomness"], 5: group_input.outputs["Displacement Scale"], 'Mix Weight': group_input.outputs["Secondary Voronoi Mix Weight"], 'Scale': group_input.outputs["Noise Texture Scale"], 'Detail': group_input.outputs["Noise Texture Detail"], 9: group_input.outputs["Noise Texture Weight"], 'Z Multiplier': group_input.outputs["Z Multiplier"], 11: group_input.outputs["Secondary Voronoi Scale"]})

    calc_radius = nw.new_node(nodegroup_calc_radius().name,
        input_kwargs={'Self Location': skeleton_loc, 'Parent Location': parent_loc})

    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: calc_radius, 1: 3.0},
        attrs={'operation': 'MULTIPLY'})

    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply, 1: group_1},
        attrs={'operation': 'MULTIPLY'})
    group = nw.new_node(nodegroup_apply_geo_matv2().name,
        input_kwargs={'Geometry': capture_attribute.outputs["Geometry"], 
                      'Displacement Amount': nw.multiply(multiply_1, surface.eval_argument(nw, selection)), 
                      'Displacement Scale': 0.5, 'Material': surface.shaderfunc_to_material(shader_random_bark_mat, geo_params=geo_params, base_color=base_color)})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': group, 'initial_position': capture_attribute.outputs["Attribute"]})

# https://blender.stackexchange.com/questions/158896/how-set-hex-in-rgb-node-python
def srgb_to_linearrgb(c):
    if   c < 0:       return 0
    elif c < 0.04045: return c/12.92
    else:             return ((c+0.055)/1.055)**2.4

def hex_to_rgb(h, alpha=1):
    r = (h & 0xff0000) >> 16
    g = (h & 0x00ff00) >> 8
    b = (h & 0x0000ff)
    return tuple([srgb_to_linearrgb(c/0xff) for c in (r,g,b)] + [alpha])

def get_random_bark_params(seed):
    with FixedSeed(seed):
        color_factory = [0x4c2f27, 0x69432d, 0x371803, 0x7f4040, 0xcc9576, 0x9e8170, 0x3d2b1f,
        0x8d6a58, 0x8b3325, 0x79443c, 0x88540b, 0x9b5f43, 0x4e3828, 0x4e3828,
        0xc09a6b, 0x944536, 0x3f0110, 0x773c12, 0x6e4e37, 0x5c4033, 0x5c4033,
        0x3c3034, 0x96704c, 0x371b1a, 0x483b32, 0x43141a, 0x471713, 0xc3b090,
        0x6b4423, 0x674d46, 0x5d2e1a, 0x331c1f, 0x7a5640, 0xb99984, 0x71543d,
        0x8f4b28, 0x491a00, 0x836446, 0x7f461b, 0x6a3208, 0x724115, 0xa0522b, 
        0x832a0c, 0x371b1a, 0xc7a373, 0x483b32, 0x635147, 0x664228, 0x5c5248]

        color_factory = [hex_to_rgb(c) for c in color_factory]

        geo_params = {
            'Displacement Scale': np.random.uniform(0.03, 0.07),
            'Z Noise Scale': np.random.uniform(1.0, 3.0),
            'Z Noise Amount': np.random.uniform(0.5, 1.5),
            'Z Multiplier': np.random.uniform(0.1, 0.3),
            'Primary Voronoi Scale': np.random.uniform(20, 40),
            'Primary Voronoi Randomness': np.random.uniform(0.6, 1.0),
            'Secondary Voronoi Mix Weight': np.random.uniform(0.05, 0.2),
            'Secondary Voronoi Scale': np.random.uniform(30, 50),
            'Noise Texture Scale': 15.0,
            'Noise Texture Detail': 16.0,
            'Noise Texture Weight': 2.0}
        color_params = {'Color': color_factory[np.random.randint(len(color_factory))]}

    return geo_params, color_params

def apply(obj, selection=None, **kwargs):

    geo_params, color_params = get_random_bark_params(seed=np.random.randint(1e5))

    surface.add_geomod(obj, geo_bark_random, selection=selection, 
        input_kwargs={'base_color': color_params['Color'], 'geo_params': geo_params}, attributes=['initial_position'])
