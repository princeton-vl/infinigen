# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo
# Acknowledgement: This file draws inspiration https://www.youtube.com/watch?v=X9YmJ0zGWHw by Creative Shrimp


import numpy as np
import bpy
import mathutils
from numpy.random import uniform, normal, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category, hsv2rgba
from infinigen.core import surface
from infinigen.assets.leaves.leaf_v2 import nodegroup_apply_wave

from infinigen.core.util.math import FixedSeed
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

def deg2rad(deg):
    return deg / 180.0 * np.pi

@node_utils.to_nodegroup('nodegroup_vein', singleton=False, type='GeometryNodeTree')
def nodegroup_vein(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'Vector', (0.0, 0.0, 0.0)),
            ('NodeSocketFloatAngle', 'Angle', 0.0),
            ('NodeSocketFloat', 'Length', 0.0),
            ('NodeSocketFloat', 'Start', 0.0),
            ('NodeSocketFloat', 'X Modulated', 0.0),
            ('NodeSocketFloat', 'Anneal', 0.4),
            ('NodeSocketFloat', 'Phase Offset', 0.0)])
    
    absolute = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["X Modulated"]},
        attrs={'operation': 'ABSOLUTE'})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group_input.outputs["Vector"]})
    
    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': absolute, 'Y': separate_xyz.outputs["Y"], 'Z': separate_xyz.outputs["Z"]})
    
    vector_rotate = nw.new_node(Nodes.VectorRotate,
        input_kwargs={'Vector': combine_xyz_1, 'Angle': group_input.outputs["Angle"]},
        attrs={'rotation_type': 'Z_AXIS'})
    
    separate_xyz_3 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': vector_rotate})
    
    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': combine_xyz_1})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_xyz_1.outputs["X"], 2: 0.3})
    
    float_curve = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': map_range_1.outputs["Result"]})
    node_utils.assign_curve(float_curve.mapping.curves[0], [(0.0, 0.0), (0.5932, 0.1969), (1.0, 1.0)])
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: float_curve, 1: 0.2},
        attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_3.outputs["X"], 1: multiply})
    
    sign = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["X Modulated"]},
        attrs={'operation': 'SIGN'})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: sign, 1: 0.1},
        attrs={'operation': 'MULTIPLY'})
    
    add_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: add, 1: multiply_1})
    
    add_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: add_1, 1: group_input.outputs["Phase Offset"]})
    
    voronoi_texture = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'W': add_2, 'Scale': 8.0, 'Randomness': 0.7125},
        attrs={'voronoi_dimensions': '1D'})
    
    length = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: vector_rotate},
        attrs={'operation': 'LENGTH'})
    
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: 0.05, 1: length.outputs["Value"]},
        attrs={'operation': 'MULTIPLY', 'use_clamp': True})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: 0.08, 1: multiply_2},
        attrs={'operation': 'SUBTRACT', 'use_clamp': True})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': voronoi_texture.outputs["Distance"], 2: subtract, 3: 1.0, 4: 0.0})
    
    absolute_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["X Modulated"]},
        attrs={'operation': 'ABSOLUTE'})
    
    subtract_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["Y"], 1: 0.0},
        attrs={'operation': 'SUBTRACT'})
    
    multiply_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract_1, 1: group_input.outputs["Anneal"]},
        attrs={'operation': 'MULTIPLY'})
    
    less_than = nw.new_node(Nodes.Math,
        input_kwargs={0: absolute_1, 1: multiply_3},
        attrs={'operation': 'LESS_THAN'})
    
    multiply_4 = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range.outputs["Result"], 1: less_than},
        attrs={'operation': 'MULTIPLY'})
    
    less_than_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: add, 1: group_input.outputs["Start"]},
        attrs={'operation': 'LESS_THAN'})
    
    multiply_5 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_4, 1: less_than_1},
        attrs={'operation': 'MULTIPLY'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Result': multiply_5})

@node_utils.to_nodegroup('nodegroup_leaf_shader', singleton=False, type='ShaderNodeTree')
def nodegroup_leaf_shader(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketColor', 'Color', (0.8, 0.8, 0.8, 1.0))])
    
    diffuse_bsdf = nw.new_node('ShaderNodeBsdfDiffuse',
        input_kwargs={'Color': group_input.outputs["Color"]})
    
    glossy_bsdf = nw.new_node('ShaderNodeBsdfGlossy',
        input_kwargs={'Color': group_input.outputs["Color"], 'Roughness': 0.3})
    
    mix_shader = nw.new_node(Nodes.MixShader,
        input_kwargs={'Fac': 0.2, 1: diffuse_bsdf, 2: glossy_bsdf})
    
    translucent_bsdf = nw.new_node(Nodes.TranslucentBSDF,
        input_kwargs={'Color': group_input.outputs["Color"]})
    
    mix_shader_1 = nw.new_node(Nodes.MixShader,
        input_kwargs={'Fac': 0.3, 1: mix_shader, 2: translucent_bsdf})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Shader': mix_shader_1})

@node_utils.to_nodegroup('nodegroup_node_group_002', singleton=False, type='GeometryNodeTree')
def nodegroup_node_group_002(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    position = nw.new_node(Nodes.InputPosition)
    
    length = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: position},
        attrs={'operation': 'LENGTH'})
    
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'Shape', 0.5)])
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: length.outputs["Value"], 1: group_input.outputs["Shape"]},
        attrs={'operation': 'MULTIPLY'})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': multiply, 1: -1.0, 2: 0.0, 3: -0.1, 4: 0.1},
        attrs={'clamp': False})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Result': map_range_1.outputs["Result"]})

@node_utils.to_nodegroup('nodegroup_nodegroup_sub_vein', singleton=False, type='GeometryNodeTree')
def nodegroup_nodegroup_sub_vein(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'X', 0.5),
            ('NodeSocketFloat', 'Y', 0.0)])
    
    absolute = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["X"]},
        attrs={'operation': 'ABSOLUTE'})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': absolute, 'Y': group_input.outputs["Y"]})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': combine_xyz})
    
    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.9, 'Color1': noise_texture.outputs["Color"], 'Color2': combine_xyz})
    
    voronoi_texture = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': mix, 'Scale': 30.0})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': voronoi_texture.outputs["Distance"], 2: 0.1, 4: 2.0},
        attrs={'clamp': False})
    
    voronoi_texture_1 = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': mix, 'Scale': 150.0},
        attrs={'feature': 'DISTANCE_TO_EDGE'})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': voronoi_texture_1.outputs["Distance"], 2: 0.1})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range.outputs["Result"], 1: map_range_1.outputs["Result"]})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: add, 1: -1.0},
        attrs={'operation': 'MULTIPLY'})
    
    map_range_3 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': map_range_1.outputs["Result"], 4: -1.0})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Value': multiply, 'Color Value': map_range_3.outputs["Result"]})

@node_utils.to_nodegroup('nodegroup_midrib', singleton=False, type='GeometryNodeTree')
def nodegroup_midrib(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'Vector', (0.0, 0.0, 0.0)),
            ('NodeSocketFloatAngle', 'Angle', 0.8238),
            ('NodeSocketFloatAngle', 'vein Angle', 0.7854),
            ('NodeSocketFloat', 'vein Length', 0.2),
            ('NodeSocketFloat', 'vein Start', -0.2),
            ('NodeSocketFloat', 'Anneal', 0.4),
            ('NodeSocketFloat', 'Phase Offset', 0.0)])
    
    vector_rotate_1 = nw.new_node(Nodes.VectorRotate,
        input_kwargs={'Vector': group_input.outputs["Vector"], 'Angle': group_input.outputs["Angle"]},
        attrs={'rotation_type': 'Z_AXIS'})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': vector_rotate_1})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_xyz.outputs["Y"]})
    
    float_curve = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': map_range_1.outputs["Result"]})
    node_utils.assign_curve(float_curve.mapping.curves[0], [(0.0, 0.5), (0.1432, 0.5406), (0.2591, 0.5062), (0.3705, 0.5406), (0.4591, 0.425), (0.5932, 0.4562), (0.7432, 0.3562), (0.8727, 0.5062), (1.0, 0.5)])
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.1
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: float_curve, 1: value},
        attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: multiply})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: value},
        attrs={'operation': 'MULTIPLY'})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: add, 1: multiply_1},
        attrs={'operation': 'SUBTRACT'})
    
    vein = nw.new_node(nodegroup_vein().name,
        input_kwargs={'Vector': vector_rotate_1, 'Angle': group_input.outputs["vein Angle"], 'Length': group_input.outputs["vein Length"], 'Start': group_input.outputs["vein Start"], 'X Modulated': subtract, 'Anneal': group_input.outputs["Anneal"], 'Phase Offset': group_input.outputs["Phase Offset"]})
    
    absolute = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract},
        attrs={'operation': 'ABSOLUTE'})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': vector_rotate_1, 'Scale': 10.0})
    
    subtract_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: noise_texture.outputs["Fac"]},
        attrs={'operation': 'SUBTRACT'})
    
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract_1, 1: 0.01},
        attrs={'operation': 'MULTIPLY'})
    
    add_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: absolute, 1: multiply_2})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': add_1, 2: 0.01, 3: 1.0, 4: 0.0})
    
    greater_than = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: 0.0},
        attrs={'operation': 'GREATER_THAN'})
    
    multiply_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range.outputs["Result"], 1: greater_than},
        attrs={'operation': 'MULTIPLY'})
    
    maximum = nw.new_node(Nodes.Math,
        input_kwargs={0: vein, 1: multiply_3},
        attrs={'operation': 'MAXIMUM'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Result': maximum, 'Vector': vector_rotate_1})

@node_utils.to_nodegroup('nodegroup_valid_area', singleton=False, type='GeometryNodeTree')
def nodegroup_valid_area(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'Value', 0.5)])
    
    sign = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Value"]},
        attrs={'operation': 'SIGN'})
    
    map_range_4 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': sign, 1: -1.0, 3: 1.0, 4: 0.0})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Result': map_range_4.outputs["Result"]})

@node_utils.to_nodegroup('nodegroup_maple_shape', singleton=False, type='GeometryNodeTree')
def nodegroup_maple_shape(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'Coordinate', (0.0, 0.0, 0.0)),
            ('NodeSocketFloat', 'Multiplier', 1.96),
            ('NodeSocketFloat', 'Noise Level', 0.02)])
    
    multiply = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Coordinate"], 1: (0.9, 1.0, 0.0)},
        attrs={'operation': 'MULTIPLY'})
    
    length = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: multiply.outputs["Vector"]},
        attrs={'operation': 'LENGTH'})
    
    gradient_texture = nw.new_node(Nodes.GradientTexture,
        input_kwargs={'Vector': group_input.outputs["Coordinate"]},
        attrs={'gradient_type': 'RADIAL'})
    
    pingpong = nw.new_node(Nodes.Math,
        input_kwargs={0: gradient_texture.outputs["Fac"]},
        attrs={'operation': 'PINGPONG'})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: pingpong, 1: group_input.outputs["Multiplier"]},
        attrs={'operation': 'MULTIPLY'})
    
    float_curve = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': multiply_1})
    node_utils.assign_curve(float_curve.mapping.curves[0], [(0.0, 0.0), (0.1156, 0.075), (0.2109, 0.2719), (0.2602, 0.2344), (0.3633, 0.2625), (0.4171, 0.5545), (0.4336, 0.5344), (0.4568, 0.7094), (0.4749, 0.6012), (0.4882, 0.6636), (0.5352, 0.4594), (0.5484, 0.4375), (0.5648, 0.4469), (0.6366, 0.7331), (0.6719, 0.6562), (0.7149, 0.8225), (0.768, 0.6344), (0.7928, 0.6853), (0.8156, 0.5125), (0.8297, 0.4906), (0.85, 0.5125), (0.8988, 0.747), (0.9297, 0.6937), (0.9648, 0.8937), (0.9797, 0.8656), (0.9883, 0.8938), (1.0, 1.0)], handles=['AUTO', 'AUTO', 'VECTOR', 'AUTO', 'AUTO', 'VECTOR', 'AUTO', 'VECTOR', 'AUTO', 'VECTOR', 'AUTO', 'AUTO', 'AUTO', 'VECTOR', 'AUTO', 'VECTOR', 'AUTO', 'VECTOR', 'AUTO', 'AUTO', 'AUTO', 'VECTOR', 'AUTO', 'VECTOR', 'AUTO', 'VECTOR', 'AUTO'])
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: length.outputs["Value"], 1: float_curve},
        attrs={'operation': 'SUBTRACT'})
    
    subtract_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract, 1: 0.06},
        attrs={'operation': 'SUBTRACT'})
    
    float_curve_1 = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': multiply_1})
    node_utils.assign_curve(float_curve_1.mapping.curves[0], [(0.0, 0.0), (0.1156, 0.075), (0.2109, 0.2719), (0.2602, 0.2344), (0.3633, 0.2625), (0.4336, 0.5344), (0.4568, 0.7094), (0.4749, 0.6012), (0.5352, 0.4594), (0.5484, 0.4375), (0.5648, 0.4469), (0.6719, 0.6562), (0.7149, 0.8225), (0.768, 0.6344), (0.8156, 0.5125), (0.8297, 0.4906), (0.85, 0.5125), (0.9297, 0.6937), (0.9883, 0.8938), (1.0, 1.0)], handles=['AUTO', 'AUTO', 'VECTOR', 'AUTO', 'AUTO', 'AUTO', 'VECTOR', 'AUTO', 'AUTO', 'AUTO', 'AUTO', 'AUTO', 'VECTOR', 'AUTO', 'AUTO', 'AUTO', 'AUTO', 'AUTO', 'VECTOR', 'AUTO'])
    
    subtract_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: length.outputs["Value"], 1: float_curve_1},
        attrs={'operation': 'SUBTRACT'})
    
    subtract_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract_2, 1: 0.06},
        attrs={'operation': 'SUBTRACT'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Shape': subtract_1, 'Displacement': subtract_3})

@node_utils.to_nodegroup('nodegroup_maple_stem', singleton=False, type='GeometryNodeTree')
def nodegroup_maple_stem(nw: NodeWrangler, stem_curve_control_points):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'Coordinate', (0.0, 0.0, 0.0)),
            ('NodeSocketFloat', 'Length', 0.64),
            ('NodeSocketFloat', 'Value', 0.005)])
    
    add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Coordinate"], 1: (0.0, 0.08, 0.0)})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': add.outputs["Vector"]})
    
    map_range_2 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_xyz.outputs["Y"], 1: -1.0, 2: 0.0})
    
    float_curve_1 = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': map_range_2.outputs["Result"]})
    node_utils.assign_curve(float_curve_1.mapping.curves[0], stem_curve_control_points)
    
    map_range_3 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': float_curve_1, 3: -1.0})
    
    add_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_3.outputs["Result"], 1: separate_xyz.outputs["X"]})
    
    absolute = nw.new_node(Nodes.Math,
        input_kwargs={0: add_1},
        attrs={'operation': 'ABSOLUTE'})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_xyz.outputs["Y"], 1: -1.72, 2: -0.35, 3: 0.03, 4: 0.008},
        attrs={'interpolation_type': 'SMOOTHSTEP'})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: absolute, 1: map_range.outputs["Result"]},
        attrs={'operation': 'SUBTRACT'})
    
    add_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: group_input.outputs["Length"]})
    
    absolute_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: add_2},
        attrs={'operation': 'ABSOLUTE'})
    
    subtract_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: absolute_1, 1: group_input.outputs["Length"]},
        attrs={'operation': 'SUBTRACT'})
    
    smooth_max = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract, 1: subtract_1, 2: 0.02},
        attrs={'operation': 'SMOOTH_MAX'})
    
    subtract_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: smooth_max, 1: group_input.outputs["Value"]},
        attrs={'operation': 'SUBTRACT'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Stem': subtract_2, 'Stem Raw': absolute})

@node_utils.to_nodegroup('nodegroup_move_to_origin', singleton=False, type='GeometryNodeTree')
def nodegroup_move_to_origin(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None)])
    
    position = nw.new_node(Nodes.InputPosition)
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': position})
    
    attribute_statistic = nw.new_node(Nodes.AttributeStatistic,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 2: separate_xyz.outputs["Y"]})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: 0.0, 1: attribute_statistic.outputs["Min"]},
        attrs={'operation': 'SUBTRACT'})
    
    attribute_statistic_1 = nw.new_node(Nodes.AttributeStatistic,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 2: separate_xyz.outputs["Z"]})
    
    subtract_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: 0.0, 1: attribute_statistic_1.outputs["Max"]},
        attrs={'operation': 'SUBTRACT'})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Y': subtract, 'Z': subtract_1})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Offset': combine_xyz})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_position})

def shader_material(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    texture_coordinate = nw.new_node(Nodes.TextureCoord)
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': texture_coordinate.outputs["Object"], 'Detail': 10.0, 'Roughness': 0.7})
    
    separate_rgb = nw.new_node(Nodes.SeparateRGB,
        input_kwargs={'Image': noise_texture.outputs["Color"]})
    
    map_range_4 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_rgb.outputs["G"], 1: 0.4, 2: 0.7, 3: 0.48, 4: 0.55},
        attrs={'interpolation_type': 'SMOOTHSTEP'})
    
    map_range_6 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_rgb.outputs["B"], 1: 0.4, 2: 0.7, 3: 0.4},
        attrs={'interpolation_type': 'SMOOTHSTEP'})
    
    attribute = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'vein'})
    
    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': attribute.outputs["Color"], 'Color1': kwargs['color_vein'], 'Color2': kwargs['color_base']})
    
    hue_saturation_value = nw.new_node('ShaderNodeHueSaturation',
        input_kwargs={'Hue': map_range_4.outputs["Result"], 'Value': map_range_6.outputs["Result"], 'Color': mix})
    
    group = nw.new_node(nodegroup_leaf_shader().name,
        input_kwargs={'Color': hue_saturation_value})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': group})

def geo_leaf_maple(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None)])
    
    # subdivide_mesh_1 = nw.new_node(Nodes.SubdivideMesh,
    #     input_kwargs={'Mesh': group_input.outputs["Geometry"]})
    
    subdivide_mesh = nw.new_node(Nodes.SubdivideMesh,
        input_kwargs={'Mesh': group_input.outputs["Geometry"], 'Level': 11})
    
    position = nw.new_node(Nodes.InputPosition)
    
    maplestem = nw.new_node(nodegroup_maple_stem(stem_curve_control_points=kwargs['stem_curve_control_points']).name,
        input_kwargs={'Coordinate': position, 'Length': 0.32, 'Value': 0.005})
    
    vector_rotate_1 = nw.new_node(Nodes.VectorRotate,
        input_kwargs={'Vector': position, 'Angle': deg2rad(kwargs['angle'])},
        attrs={'rotation_type': 'Z_AXIS'})
    
    vector_rotate = nw.new_node(Nodes.VectorRotate,
        input_kwargs={'Vector': vector_rotate_1, 'Angle': -1.5708},
        attrs={'rotation_type': 'Z_AXIS'})
    
    mapleshape = nw.new_node(nodegroup_maple_shape().name,
        input_kwargs={'Coordinate': vector_rotate, 'Multiplier': kwargs['multiplier'], 'Noise Level': 0.04})
    
    smooth_min = nw.new_node(Nodes.Math,
        input_kwargs={0: maplestem.outputs["Stem"], 1: mapleshape.outputs["Shape"], 2: 0.0},
        attrs={'operation': 'SMOOTH_MIN'})
    
    stem_length = nw.new_node(Nodes.Compare,
        input_kwargs={0: smooth_min},
        label='stem length')
    
    delete_geometry = nw.new_node(Nodes.DeleteGeom,
        input_kwargs={'Geometry': subdivide_mesh, 'Selection': stem_length})
    
    validarea = nw.new_node(nodegroup_valid_area().name,
        input_kwargs={'Value': mapleshape.outputs["Shape"]})
    
    midrib = nw.new_node(nodegroup_midrib().name,
        input_kwargs={'Vector': vector_rotate_1, 'Angle': 1.693, 'vein Length': 0.12, 'vein Start': -0.12, 'Phase Offset': uniform(0, 100)})
    
    midrib_1 = nw.new_node(nodegroup_midrib().name,
        input_kwargs={'Vector': vector_rotate_1, 'Angle': -1.7279, 'vein Length': 0.12, 'vein Start': -0.12, 'Phase Offset': uniform(0, 100)})
    
    maximum = nw.new_node(Nodes.Math,
        input_kwargs={0: midrib.outputs["Result"], 1: midrib_1.outputs["Result"]},
        attrs={'operation': 'MAXIMUM'})
    
    midrib_2 = nw.new_node(nodegroup_midrib().name,
        input_kwargs={'Vector': vector_rotate_1, 'Angle': 0.8901, 'vein Length': 0.2, 'vein Start': 0.0, 'Phase Offset': uniform(0, 100)})
    
    midrib_3 = nw.new_node(nodegroup_midrib().name,
        input_kwargs={'Vector': vector_rotate_1, 'Angle': -0.9041, 'vein Start': 0.0, 'Phase Offset': uniform(0, 100)})
    
    maximum_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: midrib_2.outputs["Result"], 1: midrib_3.outputs["Result"]},
        attrs={'operation': 'MAXIMUM'})
    
    maximum_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: maximum, 1: maximum_1},
        attrs={'operation': 'MAXIMUM'})
    
    midrib_4 = nw.new_node(nodegroup_midrib().name,
        input_kwargs={'Vector': vector_rotate_1, 'Angle': 0.0, 'vein Length': 1.64, 'vein Start': -0.12, 'Phase Offset': uniform(0, 100)})
    
    midrib_5 = nw.new_node(nodegroup_midrib().name,
        input_kwargs={'Vector': vector_rotate_1, 'Angle': 3.1416, 'vein Angle': 0.761, 'vein Length': -10.56, 'vein Start': 0.02, 'Anneal': 10.0, 'Phase Offset': uniform(0, 100)})
    
    maximum_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: midrib_4.outputs["Result"], 1: midrib_5.outputs["Result"]},
        attrs={'operation': 'MAXIMUM'})
    
    maximum_4 = nw.new_node(Nodes.Math,
        input_kwargs={0: maximum_2, 1: maximum_3},
        attrs={'operation': 'MAXIMUM'})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': position})
    
    nodegroup_sub_vein = nw.new_node(nodegroup_nodegroup_sub_vein().name,
        input_kwargs={'X': separate_xyz.outputs["X"], 'Y': separate_xyz.outputs["Y"]})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': nodegroup_sub_vein.outputs["Color Value"], 2: -0.94, 3: 1.0, 4: 0.0})
    
    maximum_5 = nw.new_node(Nodes.Math,
        input_kwargs={0: maximum_4, 1: map_range.outputs["Result"]},
        attrs={'operation': 'MAXIMUM'})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: 1.0, 1: maximum_5},
        attrs={'operation': 'SUBTRACT'})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: validarea, 1: subtract},
        attrs={'operation': 'MULTIPLY'})
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': delete_geometry, 2: multiply})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: nodegroup_sub_vein.outputs["Value"], 1: -0.03},
        attrs={'operation': 'MULTIPLY'})
    
    maximum_6 = nw.new_node(Nodes.Math,
        input_kwargs={0: maximum_4, 1: multiply_1},
        attrs={'operation': 'MAXIMUM'})
    
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: maximum_6, 1: 0.015},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_2, 1: -1.0},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_4 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_3, 1: validarea},
        attrs={'operation': 'MULTIPLY'})
    
    validarea_1 = nw.new_node(nodegroup_valid_area().name,
        input_kwargs={'Value': maplestem.outputs["Stem"]})
    
    subtract_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: maplestem.outputs["Stem Raw"], 1: 0.01},
        attrs={'operation': 'SUBTRACT'})
    
    multiply_5 = nw.new_node(Nodes.Math,
        input_kwargs={0: validarea_1, 1: subtract_1},
        attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_4, 1: multiply_5})
    
    multiply_6 = nw.new_node(Nodes.Math,
        input_kwargs={0: add},
        attrs={'operation': 'MULTIPLY'})
    
    nodegroup_002 = nw.new_node(nodegroup_node_group_002().name,
        input_kwargs={'Shape': mapleshape.outputs["Displacement"]})
    
    add_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_6, 1: nodegroup_002})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Z': add_1})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': capture_attribute.outputs["Geometry"], 'Offset': combine_xyz})
    
    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': vector_rotate_1})

    move_to_origin = nw.new_node(nodegroup_move_to_origin().name,
        input_kwargs={'Geometry': set_position})
    
    apply_wave = nw.new_node(nodegroup_apply_wave(y_wave_control_points=kwargs['y_wave_control_points'], x_wave_control_points=kwargs['x_wave_control_points']).name,
        input_kwargs={'Geometry': move_to_origin, 'Wave Scale X': 0.5, 'Wave Scale Y': 1.0, 'X Modulated': separate_xyz_1.outputs["X"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': apply_wave, 'Vein': capture_attribute.outputs[2]})

class LeafFactoryMaple(AssetFactory):

    scale = 0.5

    def __init__(self, factory_seed, season='autumn', coarse=False):
        super().__init__(factory_seed, coarse=coarse)

        with FixedSeed(factory_seed):
            self.genome = self.sample_geo_genome()

            t = uniform(0.0, 1.0)

            if season=='autumn':
                hsvcol_blade = [uniform(0.0, 0.20), 0.85, 0.9]
                hsvcol_vein = np.copy(hsvcol_blade)
                hsvcol_vein[2] = 0.7

            elif season=='summer' or season=='spring':
                hsvcol_blade = [uniform(0.28, 0.32), uniform(0.6, 0.7), 0.9]
                hsvcol_vein = np.copy(hsvcol_blade)
                hsvcol_blade[2] = uniform(0.1, 0.5)
                hsvcol_vein[2] = uniform(0.1, 0.5)

            elif season=='winter':
                hsvcol_blade = [uniform(0.0, 0.10), uniform(0.2, 0.6), uniform(0.0, 0.1)]
                hsvcol_vein = [uniform(0.0, 0.10), uniform(0.2, 0.6), uniform(0.0, 0.1)]

            else:
                raise NotImplementedError

            self.blade_color = hsvcol_blade
            self.vein_color = hsvcol_vein

            self.color_randomness = uniform(0.05, 0.10)
            
            # if t < 0.5:
            #     self.blade_color = np.array((0.2346, 0.4735, 0.0273, 1.0))
            # else:
            #     self.blade_color = np.array((1.000, 0.855, 0.007, 1.0))
            
    @staticmethod
    def sample_geo_genome():
        return {
            'midrib_length': uniform(0.0, 0.8),
            'midrib_width': uniform(0.5, 1.0),
            'stem_length': uniform(0.7, 0.9),
            'vein_asymmetry': uniform(0.0, 1.0),
            'vein_angle': uniform(0.2, 2.0),
            'vein_density': uniform(5.0, 20.0),
            'subvein_scale': uniform(10.0, 20.0),
            'jigsaw_scale': uniform(5.0, 20.0),
            'jigsaw_depth': uniform(0.0, 2.0),
            'midrib_shape_control_points': [(0.0, 0.5), (0.25, uniform(0.48, 0.52)), (0.75, uniform(0.48, 0.52)), (1.0, 0.5)],
            'leaf_shape_control_points': [(0.0, 0.0), (uniform(0.2, 0.4), uniform(0.1, 0.4)), (uniform(0.6, 0.8), uniform(0.1, 0.4)), (1.0, 0.0)],
            'vein_shape_control_points': [(0.0, 0.0), (0.25, uniform(0.1, 0.4)), (0.75, uniform(0.6, 0.9)), (1.0, 1.0)],
        }

    def create_asset(self, **params):

        bpy.ops.mesh.primitive_plane_add(
            size=4, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        obj = bpy.context.active_object

        # add noise to the genotype output
        #hue_noise = np.random.randn() * 0
        #hsv_blade = self.hsv_blade + hue_noise
        #hsv_vein = self.hsv_vein + hue_noise

        phenome = self.genome.copy()

        phenome['y_wave_control_points'] = [(0.0, 0.5), (uniform(0.25, 0.75), uniform(0.50, 0.60)), (1.0, 0.5)]
        x_wave_val = np.random.uniform(0.50, 0.58)
        phenome['x_wave_control_points'] = [(0.0, 0.5), (0.4, x_wave_val), (0.5, 0.5), (0.6, x_wave_val), (1.0, 0.5)]

        phenome['stem_curve_control_points'] = [(0.0, 0.5), 
            (uniform(0.2, 0.3), uniform(0.45, 0.55)), 
            (uniform(0.7, 0.8), uniform(0.45, 0.55)), 
            (1.0, 0.5)]
        phenome['shape_curve_control_points'] = [(0.0, 0.0), (0.523, 0.1156), (0.5805, 0.7469), (0.7742, 0.7719), (0.9461, 0.7531), (1.0, 0.0)]
        phenome['vein_length'] = uniform(0.4, 0.5)
        phenome['angle'] = uniform(-15.0, 15.0)
        phenome['multiplier'] = uniform(1.92, 2.00)

        phenome['scale_vein'] = uniform(70.0, 90.0)
        phenome['scale_wave'] = uniform(4.0, 6.0)
        phenome['scale_margin'] = uniform(5.5, 7.5)

        material_kwargs = phenome.copy()
        material_kwargs['color_base'] = np.copy(self.blade_color) # (0.2346, 0.4735, 0.0273, 1.0), 
        material_kwargs['color_base'][0] += np.random.normal(0.0, 0.02)
        material_kwargs['color_base'][1] += np.random.normal(0.0, self.color_randomness)
        material_kwargs['color_base'][2] += np.random.normal(0.0, self.color_randomness)
        material_kwargs['color_base'] = hsv2rgba(material_kwargs['color_base'])

        material_kwargs['color_vein'] = np.copy(self.vein_color) # (0.2346, 0.4735, 0.0273, 1.0), 
        material_kwargs['color_vein'][0] += np.random.normal(0.0, 0.02)
        material_kwargs['color_vein'][1] += np.random.normal(0.0, self.color_randomness)
        material_kwargs['color_vein'][2] += np.random.normal(0.0, self.color_randomness)
        material_kwargs['color_vein'] = hsv2rgba(material_kwargs['color_vein'])

        surface.add_geomod(obj, geo_leaf_maple, apply=False, attributes=['vein'], input_kwargs=phenome)
        surface.add_material(obj, shader_material, reuse=False, input_kwargs=material_kwargs)

        bpy.ops.object.convert(target='MESH')

        obj = bpy.context.object
        obj.scale *= normal(1, 0.1) * self.scale
        butil.apply_transform(obj)
        tag_object(obj, 'leaf_maple')

        return obj