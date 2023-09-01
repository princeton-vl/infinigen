# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo


import colorsys
import logging

import numpy as np
from numpy.random import uniform, normal

import bpy

from infinigen.core import surface
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes

from infinigen.core.util.math import FixedSeed
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.color import color_category


import bpy
import mathutils
from numpy.random import uniform, normal
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

@node_utils.to_nodegroup('shader_nodegroup_sub_vein', singleton=False, type='ShaderNodeTree')
def shader_nodegroup_sub_vein(nw):
    # Code generated using version 2.3.2 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'X Modulated', 0.5),
            ('NodeSocketFloat', 'Y', 0.0)])
    
    absolute = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["X Modulated"]},
        attrs={'operation': 'ABSOLUTE', 'use_clamp': True})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': absolute, 'Y': group_input.outputs["Y"]})
    
    voronoi_texture = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': combine_xyz, 'Scale': 30.0, 'Randomness': 0.754},
        attrs={'feature': 'DISTANCE_TO_EDGE'})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': voronoi_texture.outputs["Distance"], 2: 0.1, 4: 3.0})
    
    voronoi_texture_1 = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': combine_xyz, 'Scale': 10.0, 'Randomness': 0.754},
        attrs={'feature': 'DISTANCE_TO_EDGE'})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': voronoi_texture_1.outputs["Distance"], 2: 0.1, 4: 3.0})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range.outputs["Result"], 1: map_range_1.outputs["Result"]},
        attrs={'operation': 'MULTIPLY'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Value': multiply})

@node_utils.to_nodegroup('shader_nodegroup_midrib', singleton=False, type='ShaderNodeTree')
def shader_nodegroup_midrib(nw, midrib_curve_control_points=[(0.0, 0.5), (0.2809, 0.4868), (0.7448, 0.5164), (1.0, 0.5)]):
    # Code generated using version 2.3.2 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'X', 0.5),
            ('NodeSocketFloat', 'Y', -0.6),
            ('NodeSocketFloat', 'Midrib Length', 0.4),
            ('NodeSocketFloat', 'Midrib Width', 1.0),
            ('NodeSocketFloat', 'Stem Length', 0.8)
            ])
    
    map_range_6 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': group_input.outputs["Y"], 1: -0.6, 2: 0.6})
    
    stem_shape = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': map_range_6.outputs["Result"]},
        label='Stem shape')
    node_utils.assign_curve(stem_shape.mapping.curves[0], midrib_curve_control_points)
    
    map_range_7 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': stem_shape, 3: -1.0})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_7.outputs["Result"], 1: group_input.outputs["X"]},
        attrs={'operation': 'SUBTRACT'})
    
    map_range_8 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': group_input.outputs["Y"], 1: -70.0, 2: group_input.outputs["Midrib Length"], 3: group_input.outputs["Midrib Width"], 4: 0.0})
    
    absolute = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract},
        attrs={'operation': 'ABSOLUTE'})
    
    subtract_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_8.outputs["Result"], 1: absolute},
        attrs={'operation': 'SUBTRACT'})
    
    absolute_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Y"]},
        attrs={'operation': 'ABSOLUTE'})
    
    map_range_9 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': absolute_1, 2: group_input.outputs["Stem Length"], 3: 1.0, 4: 0.0})
    
    smooth_min = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract_1, 1: map_range_9.outputs["Result"], 2: 0.06},
        attrs={'operation': 'SMOOTH_MIN'})
    
    divide = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_8.outputs["Result"], 1: smooth_min},
        attrs={'operation': 'DIVIDE', 'use_clamp': True})
    
    map_range_11 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': divide, 1: 0.001, 2: 0.03, 3: 1.0, 4: 0.0})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'X Modulated': subtract, 'Midrib Value': map_range_11.outputs["Result"]})

@node_utils.to_nodegroup('shader_nodegroup_vein_coord', singleton=False, type='ShaderNodeTree')
def shader_nodegroup_vein_coord(nw, vein_curve_control_points=[(0.0, 0.0), (0.3608, 0.2434), (0.7454, 0.4951), (1.0, 1.0)]):
    # Code generated using version 2.3.2 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'X Modulated', 0.5),
            ('NodeSocketFloat', 'Y', 0.5),
            ('NodeSocketFloat', 'Vein Asymmetry', 0.0),
            ('NodeSocketFloat', 'Vein Angle', 2.0)])
    
    sign = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["X Modulated"]},
        attrs={'operation': 'SIGN'})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: sign, 1: group_input.outputs["Vein Asymmetry"]},
        attrs={'operation': 'MULTIPLY'})
    
    map_range_13 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': group_input.outputs["Y"], 1: -1.0})
    
    absolute = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["X Modulated"]},
        attrs={'operation': 'ABSOLUTE', 'use_clamp': True})
    
    vein__shape = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': absolute},
        label='Vein Shape')
    node_utils.assign_curve(vein__shape.mapping.curves[0], vein_curve_control_points)
    
    map_range_4 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': vein__shape, 2: 0.9, 4: 1.9})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_4.outputs["Result"], 1: group_input.outputs["Vein Angle"]},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_13.outputs["Result"], 1: multiply_1},
        attrs={'operation': 'MULTIPLY'})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_2, 1: group_input.outputs["Y"]},
        attrs={'operation': 'SUBTRACT'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply, 1: subtract})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Vein Coord': add})

@node_utils.to_nodegroup('shader_nodegroup_shape', singleton=False, type='ShaderNodeTree')
def shader_nodegroup_shape(nw, shape_curve_control_points=[(0.0, 0.0), (0.3454, 0.2336), (1.0, 0.0)]):
    # Code generated using version 2.3.2 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'X Modulated', 0.0),
            ('NodeSocketFloat', 'Y', 0.0)])
    
    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': group_input.outputs["X Modulated"], 'Y': group_input.outputs["Y"]})
    
    clamp = nw.new_node('ShaderNodeClamp',
        input_kwargs={'Value': group_input.outputs["Y"], 'Min': -0.6, 'Max': 0.6})
    
    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Y': clamp})
    
    subtract = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: combine_xyz_2, 1: combine_xyz_1},
        attrs={'operation': 'SUBTRACT'})
    
    length = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: subtract.outputs["Vector"]},
        attrs={'operation': 'LENGTH'})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': group_input.outputs["Y"], 1: -0.6, 2: 0.6})
    
    leaf_shape = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': map_range_1.outputs["Result"]},
        label='Leaf shape')
    node_utils.assign_curve(leaf_shape.mapping.curves[0], shape_curve_control_points)
    
    subtract_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: length.outputs["Value"], 1: leaf_shape},
        attrs={'operation': 'SUBTRACT'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Leaf Shape': subtract_1})

@node_utils.to_nodegroup('shader_nodegroup_apply_vein_midrib', singleton=False, type='ShaderNodeTree')
def shader_nodegroup_apply_vein_midrib(nw):
    # Code generated using version 2.3.2 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'Vein Coord', 0.0),
            ('NodeSocketFloat', 'Midrib Value', 0.5),
            ('NodeSocketFloat', 'Leaf Shape', 1.0),
            ('NodeSocketFloat', 'Vein Density', 6.0)])
    
    map_range_5 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': group_input.outputs["Leaf Shape"], 1: -0.3, 2: 0.0, 3: 0.015, 4: 0.0})
    
    vein = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'W': group_input.outputs["Vein Coord"], 'Scale': group_input.outputs["Vein Density"], 'Randomness': 0.2},
        label='Vein',
        attrs={'voronoi_dimensions': '1D'})
    
    map_range_3 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': vein.outputs["Distance"], 1: 0.001, 2: 0.05, 3: 1.0, 4: 0.0})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_5.outputs["Result"], 1: map_range_3.outputs["Result"]},
        attrs={'operation': 'MULTIPLY'})
    
    map_range_10 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': multiply, 1: 0.001, 2: 0.03, 3: 1.0, 4: 0.0})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Midrib Value"], 1: map_range_10.outputs["Result"]},
        attrs={'operation': 'MULTIPLY'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Vein Value': multiply_1})

@node_utils.to_nodegroup('shader_nodegroup_leaf_gen', singleton=False, type='ShaderNodeTree')
def shader_nodegroup_leaf_gen(nw, midrib_curve_control_points, vein_curve_control_points, shape_curve_control_points):
    # Code generated using version 2.3.2 of the node_transpiler
    input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Mesh', None),
            ('NodeSocketFloat', 'Displancement scale', 0.01),
            ('NodeSocketFloat', 'Vein Asymmetry', 0.8),
            ('NodeSocketFloat', 'Vein Density', 10.0),
            ('NodeSocketFloat', 'Jigsaw Scale', 18.0),
            ('NodeSocketFloat', 'Jigsaw Depth', 1.0),
            ('NodeSocketFloat', 'Vein Angle', 1.0),
            ('NodeSocketFloat', 'Sub-vein Displacement', 0.5),
            ('NodeSocketFloat', 'Sub-vein Scale', 20.0),
            ('NodeSocketFloat', 'Wave Displacement', 0.05),
             ('NodeSocketFloat', 'Midrib Length', 0.4),
            ('NodeSocketFloat', 'Midrib Width', 1.0),
            ('NodeSocketFloat', 'Stem Length', 0.8),
            ])

    coordinate = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'coordinate'})

    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': coordinate.outputs["Vector"]})
    
    midrib = nw.new_node(shader_nodegroup_midrib(midrib_curve_control_points=midrib_curve_control_points).name,
        input_kwargs={'X': separate_xyz.outputs["X"], 'Y': separate_xyz.outputs["Y"],
        'Midrib Length': input.outputs["Midrib Length"], 'Midrib Width': input.outputs["Midrib Width"], 'Stem Length': input.outputs["Stem Length"]
        })

    veincoord = nw.new_node(shader_nodegroup_vein_coord(vein_curve_control_points=vein_curve_control_points).name,
        input_kwargs={'X Modulated': midrib.outputs["X Modulated"], 'Y': separate_xyz.outputs["Y"], 
        'Vein Asymmetry': input.outputs["Vein Asymmetry"], 'Vein Angle': input.outputs["Vein Angle"]})

    shape = nw.new_node(shader_nodegroup_shape(shape_curve_control_points=shape_curve_control_points).name,
        input_kwargs={'X Modulated': midrib.outputs["X Modulated"], 'Y': separate_xyz.outputs["Y"]})

    applyveinmidrib = nw.new_node(shader_nodegroup_apply_vein_midrib().name,
        input_kwargs={'Vein Coord': veincoord, 'Midrib Value': midrib.outputs["Midrib Value"], 
        'Leaf Shape': shape, 'Vein Density': input.outputs["Vein Density"]})

    subvein = nw.new_node(shader_nodegroup_sub_vein().name,
        input_kwargs={'X Modulated': midrib.outputs["X Modulated"], 'Y': veincoord})

    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Vein Value': applyveinmidrib, 'Sub Vein Value': subvein})

    
@node_utils.to_nodegroup('nodegroup_shape_with_jigsaw', singleton=False, type='GeometryNodeTree')
def nodegroup_shape_with_jigsaw(nw):
    # Code generated using version 2.3.2 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'Midrib Value', 1.0),
            ('NodeSocketFloat', 'Vein Coord', 0.0),
            ('NodeSocketFloat', 'Leaf Shape', 0.5),
            ('NodeSocketFloat', 'Jigsaw Scale', 18.0),
            ('NodeSocketFloat', 'Jigsaw Depth', 0.5)])
    
    map_range_12 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': group_input.outputs["Midrib Value"], 3: 1.0, 4: 0.0})
    
    jigsaw = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'W': group_input.outputs["Vein Coord"], 'Scale': group_input.outputs["Jigsaw Scale"]},
        label='Jigsaw',
        attrs={'voronoi_dimensions': '1D'})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Jigsaw Depth"], 1: 0.05},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_add = nw.new_node(Nodes.Math,
        input_kwargs={0: jigsaw.outputs["Distance"], 1: multiply, 2: group_input.outputs["Leaf Shape"]},
        attrs={'operation': 'MULTIPLY_ADD', 'use_clamp': True})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': multiply_add, 1: 0.001, 2: 0.002, 3: 1.0, 4: 0.0})
    
    maximum = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_12.outputs["Result"], 1: map_range.outputs["Result"]},
        attrs={'operation': 'MAXIMUM'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Value': maximum})

@node_utils.to_nodegroup('nodegroup_shape', singleton=False, type='GeometryNodeTree')
def nodegroup_shape(nw, shape_curve_control_points=[(0.0, 0.0), (0.3454, 0.2336), (1.0, 0.0)]):
    # Code generated using version 2.3.2 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'X Modulated', 0.0),
            ('NodeSocketFloat', 'Y', 0.0)])
    
    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': group_input.outputs["X Modulated"], 'Y': group_input.outputs["Y"]})
    
    clamp = nw.new_node('ShaderNodeClamp',
        input_kwargs={'Value': group_input.outputs["Y"], 'Min': -0.6, 'Max': 0.6})
    
    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Y': clamp})
    
    subtract = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: combine_xyz_2, 1: combine_xyz_1},
        attrs={'operation': 'SUBTRACT'})
    
    length = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: subtract.outputs["Vector"]},
        attrs={'operation': 'LENGTH'})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': group_input.outputs["Y"], 1: -0.6, 2: 0.6})
    
    leaf_shape = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': map_range_1.outputs["Result"]},
        label='Leaf shape')
    node_utils.assign_curve(leaf_shape.mapping.curves[0], shape_curve_control_points)
    
    subtract_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: length.outputs["Value"], 1: leaf_shape},
        attrs={'operation': 'SUBTRACT'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Leaf Shape': subtract_1})

@node_utils.to_nodegroup('nodegroup_midrib', singleton=False, type='GeometryNodeTree')
def nodegroup_midrib(nw, midrib_curve_control_points=[(0.0, 0.5), (0.2809, 0.4868), (0.7448, 0.5164), (1.0, 0.5)]):
    # Code generated using version 2.3.2 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'X', 0.5),
            ('NodeSocketFloat', 'Y', -0.6),
            ('NodeSocketFloat', 'Midrib Length', 0.4),
            ('NodeSocketFloat', 'Midrib Width', 1.0),
            ('NodeSocketFloat', 'Stem Length', 0.8)
            ])
    
    map_range_6 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': group_input.outputs["Y"], 1: -0.6, 2: 0.6})
    
    stem_shape = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': map_range_6.outputs["Result"]},
        label='Stem shape')
    node_utils.assign_curve(stem_shape.mapping.curves[0], midrib_curve_control_points)
    
    map_range_7 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': stem_shape, 3: -1.0})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_7.outputs["Result"], 1: group_input.outputs["X"]},
        attrs={'operation': 'SUBTRACT'})
    
    map_range_8 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': group_input.outputs["Y"], 1: -70.0, 2: group_input.outputs["Midrib Length"], 3: group_input.outputs["Midrib Width"], 4: 0.0})
    
    absolute = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract},
        attrs={'operation': 'ABSOLUTE'})
    
    subtract_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_8.outputs["Result"], 1: absolute},
        attrs={'operation': 'SUBTRACT'})
    
    absolute_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Y"]},
        attrs={'operation': 'ABSOLUTE'})
    
    map_range_9 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': absolute_1, 2: group_input.outputs["Stem Length"], 3: 1.0, 4: 0.0})
    
    smooth_min = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract_1, 1: map_range_9.outputs["Result"], 2: 0.06},
        attrs={'operation': 'SMOOTH_MIN'})
    
    divide = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_8.outputs["Result"], 1: smooth_min},
        attrs={'operation': 'DIVIDE', 'use_clamp': True})
    
    map_range_11 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': divide, 1: 0.001, 2: 0.03, 3: 1.0, 4: 0.0})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'X Modulated': subtract, 'Midrib Value': map_range_11.outputs["Result"]})

@node_utils.to_nodegroup('nodegroup_vein_coord', singleton=False, type='GeometryNodeTree')
def nodegroup_vein_coord(nw, vein_curve_control_points=[(0.0, 0.0), (0.3608, 0.2434), (0.7454, 0.4951), (1.0, 1.0)]):
    # Code generated using version 2.3.2 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'X Modulated', 0.5),
            ('NodeSocketFloat', 'Y', 0.5),
            ('NodeSocketFloat', 'Vein Asymmetry', 0.0),
            ('NodeSocketFloat', 'Vein Angle', 2.0)])
    
    sign = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["X Modulated"]},
        attrs={'operation': 'SIGN'})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: sign, 1: group_input.outputs["Vein Asymmetry"]},
        attrs={'operation': 'MULTIPLY'})
    
    map_range_13 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': group_input.outputs["Y"], 1: -1.0})
    
    absolute = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["X Modulated"]},
        attrs={'operation': 'ABSOLUTE', 'use_clamp': True})
    
    vein__shape = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': absolute},
        label='Vein Shape')
    node_utils.assign_curve(vein__shape.mapping.curves[0], vein_curve_control_points)
    
    map_range_4 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': vein__shape, 2: 0.9, 4: 1.9})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_4.outputs["Result"], 1: group_input.outputs["Vein Angle"]},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_13.outputs["Result"], 1: multiply_1},
        attrs={'operation': 'MULTIPLY'})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_2, 1: group_input.outputs["Y"]},
        attrs={'operation': 'SUBTRACT'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply, 1: subtract})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Vein Coord': add})

@node_utils.to_nodegroup('nodegroup_apply_vein_midrib', singleton=False, type='GeometryNodeTree')
def nodegroup_apply_vein_midrib(nw):
    # Code generated using version 2.3.2 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'Vein Coord', 0.0),
            ('NodeSocketFloat', 'Midrib Value', 0.5),
            ('NodeSocketFloat', 'Leaf Shape', 1.0),
            ('NodeSocketFloat', 'Vein Density', 6.0)])
    
    map_range_5 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': group_input.outputs["Leaf Shape"], 1: -0.3, 2: 0.0, 3: 0.015, 4: 0.0})
    
    vein = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'W': group_input.outputs["Vein Coord"], 'Scale': group_input.outputs["Vein Density"], 'Randomness': 0.2},
        label='Vein',
        attrs={'voronoi_dimensions': '1D'})
    
    map_range_3 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': vein.outputs["Distance"], 1: 0.001, 2: 0.05, 3: 1.0, 4: 0.0})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_5.outputs["Result"], 1: map_range_3.outputs["Result"]},
        attrs={'operation': 'MULTIPLY'})
    
    map_range_10 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': multiply, 1: 0.001, 2: 0.01, 3: 1.0, 4: 0.0})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Midrib Value"], 1: map_range_10.outputs["Result"]},
        attrs={'operation': 'MULTIPLY'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Vein Value': multiply_1})

@node_utils.to_nodegroup('nodegroup_leaf_gen', singleton=False, type='GeometryNodeTree')
def nodegroup_leaf_gen(nw, midrib_curve_control_points, vein_curve_control_points, shape_curve_control_points):
    # Code generated using version 2.3.2 of the node_transpiler

    geometry = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Mesh', None),
            ('NodeSocketFloat', 'Displancement scale', 0.5),
            ('NodeSocketFloat', 'Vein Asymmetry', 0.0),
            ('NodeSocketFloat', 'Vein Density', 6.0),
            ('NodeSocketFloat', 'Jigsaw Scale', 18.0),
            ('NodeSocketFloat', 'Jigsaw Depth', 0.07),
            ('NodeSocketFloat', 'Vein Angle', 1.0),
            ('NodeSocketFloat', 'Sub-vein Displacement', 0.5),
            ('NodeSocketFloat', 'Sub-vein Scale', 50.0),
            ('NodeSocketFloat', 'Wave Displacement', 0.1),
            ('NodeSocketFloat', 'Midrib Length', 0.4),
            ('NodeSocketFloat', 'Midrib Width', 1.0),
            ('NodeSocketFloat', 'Stem Length', 0.8),
            ])
    
    position = nw.new_node(Nodes.InputPosition)
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': position})
    
    midrib = nw.new_node(nodegroup_midrib(midrib_curve_control_points=midrib_curve_control_points).name,
        input_kwargs={'X': separate_xyz.outputs["X"], 'Y': separate_xyz.outputs["Y"], 
        'Midrib Length': geometry.outputs["Midrib Length"], 'Midrib Width': geometry.outputs["Midrib Width"], 'Stem Length': geometry.outputs["Stem Length"]
        })
    
    veincoord = nw.new_node(nodegroup_vein_coord(vein_curve_control_points=vein_curve_control_points).name,
        input_kwargs={'X Modulated': midrib.outputs["X Modulated"], 'Y': separate_xyz.outputs["Y"], 'Vein Asymmetry': geometry.outputs["Vein Asymmetry"], 'Vein Angle': geometry.outputs["Vein Angle"]})
    
    shape = nw.new_node(nodegroup_shape(shape_curve_control_points=shape_curve_control_points).name,
        input_kwargs={'X Modulated': midrib.outputs["X Modulated"], 'Y': separate_xyz.outputs["Y"]})
    
    applyveinmidrib = nw.new_node(nodegroup_apply_vein_midrib().name,
        input_kwargs={'Vein Coord': veincoord, 'Midrib Value': midrib.outputs["Midrib Value"], 'Leaf Shape': shape, 'Vein Density': geometry.outputs["Vein Density"]})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: geometry.outputs["Displancement scale"], 1: applyveinmidrib},
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Z': multiply})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': geometry.outputs["Mesh"], 'Offset': combine_xyz})
    
    shapewithjigsaw = nw.new_node(nodegroup_shape_with_jigsaw().name,
        input_kwargs={'Midrib Value': midrib.outputs["Midrib Value"], 'Vein Coord': veincoord, 'Leaf Shape': shape, 'Jigsaw Scale': geometry.outputs["Jigsaw Scale"], 'Jigsaw Depth': geometry.outputs["Jigsaw Depth"]})
    
    less_than = nw.new_node(Nodes.Compare,
        input_kwargs={0: shapewithjigsaw, 1: 0.5},
        attrs={'operation': 'LESS_THAN'})
    
    delete_geometry = nw.new_node('GeometryNodeDeleteGeometry',
        input_kwargs={'Geometry': set_position, 'Selection': less_than})
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': delete_geometry, 2: applyveinmidrib})
        
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Mesh': capture_attribute, 'Attribute': capture_attribute.outputs[2], 'X Modulated': midrib.outputs["X Modulated"], 'Vein Coord': veincoord})


@node_utils.to_nodegroup('nodegroup_sub_vein', singleton=False, type='GeometryNodeTree')
def nodegroup_sub_vein(nw):
    # Code generated using version 2.3.2 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'X', 0.5),
            ('NodeSocketFloat', 'Y', 0.0)])
    
    absolute = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["X"]},
        attrs={'operation': 'ABSOLUTE', 'use_clamp': True})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': absolute, 'Y': group_input.outputs["Y"]})
    
    voronoi_texture = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': combine_xyz, 'Scale': 30.0, 'Randomness': 0.754},
        attrs={'feature': 'DISTANCE_TO_EDGE'})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': voronoi_texture.outputs["Distance"], 2: 0.1})
    
    voronoi_texture_1 = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': combine_xyz, 'Scale': 10.0, 'Randomness': 0.754},
        attrs={'feature': 'DISTANCE_TO_EDGE'})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': voronoi_texture_1.outputs["Distance"], 2: 0.1})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range.outputs["Result"], 1: map_range_1.outputs["Result"]},
        attrs={'operation': 'MULTIPLY'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Value': multiply})


@node_utils.to_nodegroup('nodegroup_add_noise', singleton=False, type='GeometryNodeTree')
def nodegroup_add_noise(nw):
    # Code generated using version 2.3.2 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketFloat', 'Displacement', 0.05),
            ('NodeSocketFloat', 'Scale', 10.0)])
    
    position_1 = nw.new_node(Nodes.InputPosition)
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': position_1, 'Scale': group_input.outputs["Scale"]})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: noise_texture.outputs["Fac"], 1: group_input.outputs["Displacement"]},
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Z': multiply})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Offset': combine_xyz})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_position})


@node_utils.to_nodegroup('nodegroup_apply_wave', singleton=False, type='GeometryNodeTree')
def nodegroup_apply_wave(nw, y_wave_control_points, x_wave_control_points):
    # Code generated using version 2.3.2 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketFloat', 'Wave Scale Y', 1.0),
            ('NodeSocketFloat', 'Wave Scale X', 1.0),
            ('NodeSocketFloat', 'X Modulated', None),
            ])
    
    position = nw.new_node(Nodes.InputPosition)
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': position})
    
    position_1 = nw.new_node(Nodes.InputPosition)
    
    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': position_1})
    
    attribute_statistic = nw.new_node(Nodes.AttributeStatistic,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 2: separate_xyz_1.outputs["Y"]})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_xyz.outputs["Y"], 1: attribute_statistic.outputs["Min"], 2: attribute_statistic.outputs["Max"]})

    float_curves = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': map_range.outputs["Result"]})
    node_utils.assign_curve(float_curves.mapping.curves[0], y_wave_control_points)
    
    map_range_2 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': float_curves, 3: -1.0})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_2.outputs["Result"], 1: group_input.outputs["Wave Scale Y"]},
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Z': multiply})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Offset': combine_xyz})
    
    attribute_statistic_1 = nw.new_node(Nodes.AttributeStatistic,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 2: group_input.outputs['X Modulated']})
    
    map_range_7 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': group_input.outputs['X Modulated'], 1: attribute_statistic_1.outputs["Min"], 2: attribute_statistic_1.outputs["Max"]})
    
    float_curves_2 = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': map_range_7.outputs["Result"]}
        )
    node_utils.assign_curve(float_curves_2.mapping.curves[0], x_wave_control_points)
    float_curves_2.mapping.curves[0].points[2].handle_type = 'VECTOR'
    
    map_range_4 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': float_curves_2, 3: -1.0})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_4.outputs["Result"], 1: group_input.outputs["Wave Scale X"]},
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Z': multiply_1})
    
    set_position_1 = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': set_position, 'Offset': combine_xyz_1})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_position_1})

@node_utils.to_nodegroup('nodegroup_move_to_origin', singleton=False, type='GeometryNodeTree')
def nodegroup_move_to_origin(nw):
    # Code generated using version 2.3.2 of the node_transpiler

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
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Y': subtract})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Offset': combine_xyz})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_position})

@node_utils.to_nodegroup('nodegroup_blight', singleton=False, type='ShaderNodeTree')
def nodegroup_blight(nw):
    # Code generated using version 2.3.2 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'Coordinate', (0.0, 0.0, 0.0)),
            ('NodeSocketColor', 'Leaf Color', (0.5, 0.5, 0.5, 1.0)),
            ('NodeSocketColor', 'Blight Color', (0.5, 0.3992, 0.035, 1.0)),
            ('NodeSocketFloat', 'Random Seed', 18.3),
            ('NodeSocketFloat', 'Offset', 0.5)])
    
    musgrave_texture = nw.new_node(Nodes.MusgraveTexture,
        input_kwargs={'Vector': group_input.outputs["Coordinate"], 'W': group_input.outputs["Random Seed"], 'Scale': 4.0, 'Detail': 10.0, 'Dimension': 10.0, 'Lacunarity': 5.0, 'Offset': group_input.outputs["Offset"]},
        attrs={'musgrave_dimensions': '4D', 'musgrave_type': 'HETERO_TERRAIN'})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': musgrave_texture, 4: 0.8})
    
    mix_4 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': map_range_1.outputs["Result"], 'Color1': group_input.outputs["Leaf Color"], 'Color2': group_input.outputs["Blight Color"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Color': mix_4})

@node_utils.to_nodegroup('nodegroup_dotted_blight', singleton=False, type='ShaderNodeTree')
def nodegroup_dotted_blight(nw):
    # Code generated using version 2.3.2 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'Coord', (0.0, 0.0, 0.0)),
            ('NodeSocketColor', 'Leaf Color', (0.5, 0.5, 0.5, 1.0)),
            ('NodeSocketColor', 'Blight Color', (0.4969, 0.2831, 0.0273, 1.0))])
    
    voronoi_texture = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': group_input.outputs["Coord"], 'Scale': 20.0},
        attrs={'voronoi_dimensions': '2D'})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': voronoi_texture.outputs["Distance"], 2: 0.15, 3: 1.0, 4: 0.0})
    
    mix_5 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': map_range.outputs["Result"], 'Color1': group_input.outputs["Blight Color"], 'Color2': (0.0, 0.0, 0.0, 1.0)})
    
    mix_3 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': map_range.outputs["Result"], 'Color1': group_input.outputs["Leaf Color"], 'Color2': mix_5})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Color': mix_3})


def shader_leaf_new(nw, **kwargs):
    # Code generated using version 2.3.2 of the node_transpiler
    leafgen = nw.new_node(shader_nodegroup_leaf_gen(midrib_curve_control_points=kwargs['midrib_shape_control_points'], 
        vein_curve_control_points=kwargs['vein_shape_control_points'], 
        shape_curve_control_points=kwargs['leaf_shape_control_points']).name,
        input_kwargs={'Displancement scale': 0.01, 
        'Vein Asymmetry': kwargs['vein_asymmetry'], 
        'Vein Angle': kwargs['vein_angle'], 
        'Vein Density': kwargs['vein_density'], 
        'Jigsaw Scale': kwargs['jigsaw_scale'], 
        'Jigsaw Depth': kwargs['jigsaw_depth'],
        'Midrib Length': kwargs['midrib_length'],
        'Midrib Width': kwargs['midrib_width'],
        'Stem Length': kwargs['stem_length']
        })

    rgb = nw.new_node(Nodes.RGB)
    rgb.outputs[0].default_value = kwargs['blade_color']
    
    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': kwargs['vein_color_mix_factor'], 'Color1': rgb, 'Color2': (0.35, 0.35, 0.35, 1.0)})
    
    mix_1 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': leafgen.outputs["Sub Vein Value"], 'Color1': mix, 'Color2': rgb})
    
    mix_2 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': leafgen.outputs["Vein Value"], 'Color1': mix, 'Color2': mix_1})
    
    texture_coordinate = nw.new_node(Nodes.TextureCoord)
    
    rgb_1 = nw.new_node(Nodes.RGB)
    rgb_1.outputs[0].default_value = kwargs['blight_color']
    
    group_1 = nw.new_node(nodegroup_dotted_blight().name,
        input_kwargs={'Coord': texture_coordinate.outputs["Generated"], 'Leaf Color': mix_2, 'Blight Color': rgb_1})
    
    mix_3 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': kwargs['dotted_blight_weight'], 'Color1': mix_2, 'Color2': group_1})
    
    group_2 = nw.new_node(nodegroup_blight().name,
        input_kwargs={'Coordinate': texture_coordinate.outputs["Generated"], 'Leaf Color': mix_3, 'Blight Color': rgb_1, 'Random Seed': kwargs['blight_random_seed'], 'Offset': kwargs['blight_area_factor']})
    
    mix_4 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': kwargs['blight_weight'], 'Color1': mix_3, 'Color2': group_2})
    
    translucent_bsdf = nw.new_node(Nodes.TranslucentBSDF,
        input_kwargs={'Color': mix_4})
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': mix_4})
    
    mix_shader = nw.new_node(Nodes.MixShader,
        input_kwargs={'Fac': 0.7, 1: translucent_bsdf, 2: principled_bsdf})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': mix_shader})
    


def geo_leaf_v2(nw, **kwargs):
    # Code generated using version 2.3.2 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None)])
    
    subdivide_mesh = nw.new_node(Nodes.SubdivideMesh,
        input_kwargs={'Mesh': group_input.outputs["Geometry"], 'Level': 10})
    
    position = nw.new_node(Nodes.InputPosition)
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': subdivide_mesh, 1: position},
        attrs={'data_type': 'FLOAT_VECTOR'})

    leafgen = nw.new_node(nodegroup_leaf_gen(midrib_curve_control_points=kwargs['midrib_shape_control_points'], 
    vein_curve_control_points=kwargs['vein_shape_control_points'], 
    shape_curve_control_points=kwargs['leaf_shape_control_points']).name,
        input_kwargs={'Mesh': capture_attribute.outputs["Geometry"], 
        'Displancement scale': 0.005, 
        'Vein Asymmetry': kwargs['vein_asymmetry'], 
        'Vein Angle': kwargs['vein_angle'], 
        'Vein Density': kwargs['vein_density'], 
        'Jigsaw Scale': kwargs['jigsaw_scale'], 
        'Jigsaw Depth': kwargs['jigsaw_depth'],
        'Midrib Length': kwargs['midrib_length'],
        'Midrib Width': kwargs['midrib_width'],
        'Stem Length': kwargs['stem_length'],
        })

    # addnoise = nw.new_node(nodegroup_add_noise().name, 
    #     input_kwargs={'Geometry': leafgen.outputs["Mesh"], 'Displacement': 0.03, 'Scale': 10.0})

    subvein = nw.new_node(nodegroup_sub_vein().name,
        input_kwargs={'X': leafgen.outputs["X Modulated"], 'Y': leafgen.outputs["Vein Coord"]})

    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: subvein, 1: 0.001},
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Z': multiply})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': leafgen.outputs["Mesh"], 'Offset': combine_xyz})
    
    logging.warning(f'Disabling set_position to avoid LeafV2 segfault')
    set_position = leafgen.outputs["Mesh"]

    applywave = nw.new_node(nodegroup_apply_wave(y_wave_control_points=kwargs['y_wave_control_points'], x_wave_control_points=kwargs['x_wave_control_points']).name,
        input_kwargs={'Geometry': set_position, 'Wave Scale X': 0.15, 'Wave Scale Y': 1.5, 'X Modulated': leafgen.outputs["X Modulated"]})
    
    movetoorigin = nw.new_node(nodegroup_move_to_origin().name,
        input_kwargs={'Geometry': applywave})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': movetoorigin, 'Attribute': leafgen.outputs["Attribute"], 'Coordinate': capture_attribute.outputs["Attribute"]})

class LeafFactoryV2(AssetFactory):
    
    scale = 0.5

    def __init__(self, factory_seed, coarse=False):
        super(LeafFactoryV2, self).__init__(factory_seed, coarse=coarse)

        with FixedSeed(factory_seed):
            self.genome = self.sample_geo_genome()

            t = uniform(0.0, 1.0)

            if t < 0.8:
                self.blade_color = color_category('greenery')
            elif t < 0.9:
                self.blade_color = color_category('yellowish')
            else:
                self.blade_color = color_category('red')

            self.blight_color = color_category('yellowish')
            self.vein_color_mix_factor = uniform(0.2, 0.6)
            
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
            size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        obj = bpy.context.active_object

        # add noise to the genotype output
        #hue_noise = np.random.randn() * 0
        #hsv_blade = self.hsv_blade + hue_noise
        #hsv_vein = self.hsv_vein + hue_noise

        phenome = self.genome.copy()

        phenome['y_wave_control_points'] = [(0.0, 0.5), (np.random.uniform(0.25, 0.75), np.random.uniform(0.50, 0.60)), (1.0, 0.5)]
        x_wave_val = np.random.uniform(0.50, 0.58)
        phenome['x_wave_control_points'] = [(0.0, 0.5), (0.4, x_wave_val), (0.5, 0.5), (0.6, x_wave_val), (1.0, 0.5)]

        material_kwargs = phenome.copy()
        material_kwargs['blade_color'] = self.blade_color
        material_kwargs['blade_color'][0] += np.random.normal(0.0, 0.03)
        material_kwargs['blade_color'][1] += np.random.normal(0.0, 0.03)
        material_kwargs['blade_color'][2] += np.random.normal(0.0, 0.03)

        material_kwargs['blight_color'] = self.blight_color

        material_kwargs['vein_color_mix_factor'] = self.vein_color_mix_factor
        material_kwargs['blight_weight'] = np.random.binomial(1, 0.1)
        material_kwargs['dotted_blight_weight'] = np.random.binomial(1, 0.1)
        material_kwargs['blight_random_seed'] = np.random.uniform(0.0, 100.0)
        material_kwargs['blight_area_factor'] = np.random.uniform(0.2, 0.8)

        # TODO: add more phenome attributes
        
        surface.add_geomod(obj, geo_leaf_v2, apply=False,
                       attributes=['offset', 'coordinate'], input_kwargs=phenome)
        surface.add_material(obj, shader_leaf_new,
                            reuse=False, input_kwargs=material_kwargs)

        bpy.ops.object.convert(target='MESH')

        obj = bpy.context.object
        obj.scale *= normal(1, 0.05) * self.scale
        butil.apply_transform(obj)
        tag_object(obj, 'leaf')

        return obj
