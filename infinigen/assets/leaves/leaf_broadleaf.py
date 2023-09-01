# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo
# Acknowledgment: This file draws inspiration from https://www.youtube.com/watch?v=pfOKB1GKJHM by Dr. Blender

import numpy as np
import bpy
import mathutils
from numpy.random import uniform, normal, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category, hsv2rgba
from infinigen.core import surface
from infinigen.assets.leaves.leaf_v2 import nodegroup_apply_wave, nodegroup_move_to_origin
from infinigen.assets.leaves.leaf_maple import nodegroup_leaf_shader

from infinigen.core.util.math import FixedSeed
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.assets.utils.tag import tag_object, tag_nodegroup


@node_utils.to_nodegroup('nodegroup_random_mask_vein', singleton=False, type='GeometryNodeTree')
def nodegroup_random_mask_vein(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'Coord', 0.0),
            ('NodeSocketFloat', 'Shape', 0.5),
            ('NodeSocketFloat', 'Density', 0.5),
            ('NodeSocketFloat', 'Random Scale Seed', 0.5)])
    
    vein = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'W': group_input.outputs["Coord"], 'Scale': group_input.outputs["Density"], 'Randomness': 0.2},
        label='Vein',
        attrs={'voronoi_dimensions': '1D'})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Density"], 1: group_input.outputs["Random Scale Seed"]},
        attrs={'operation': 'MULTIPLY'})
    
    vein_1 = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'W': group_input.outputs["Coord"], 'Scale': multiply},
        label='Vein',
        attrs={'voronoi_dimensions': '1D'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: vein_1.outputs["Distance"], 1: 0.35})
    
    round = nw.new_node(Nodes.Math,
        input_kwargs={0: add},
        attrs={'operation': 'ROUND'})
    
    add_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: vein.outputs["Distance"], 1: round})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': add_1, 2: 0.02, 3: 0.95, 4: 0.0})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Shape"], 1: map_range_1.outputs["Result"]},
        attrs={'operation': 'MULTIPLY'})
    
    map_range_2 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': multiply_1, 1: 0.001, 2: 0.005, 3: 1.0, 4: 0.0})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Result': map_range_2.outputs["Result"]})

@node_utils.to_nodegroup('nodegroup_nodegroup_vein_coord_001', singleton=False, type='GeometryNodeTree')
def nodegroup_nodegroup_vein_coord_001(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'X Modulated', 0.5),
            ('NodeSocketFloat', 'Y', 0.5),
            ('NodeSocketFloat', 'Vein Asymmetry', 0.0),
            ('NodeSocketFloat', 'Vein Angle', 2.0),
            ('NodeSocketFloat', 'Leaf Shape', 0.0)])
    
    sign = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["X Modulated"]},
        attrs={'operation': 'SIGN'})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Vein Asymmetry"], 1: sign},
        attrs={'operation': 'MULTIPLY'})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': group_input.outputs["Y"], 1: -1.0})
    
    vein_shape = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': group_input.outputs["X Modulated"]},
        label='Vein Shape')
    node_utils.assign_curve(vein_shape.mapping.curves[0], [(0.0, 0.0), (0.0182, 0.05), (0.3364, 0.2386), (0.7227, 0.75), (1.0, 1.0)])
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': vein_shape, 4: 1.9})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_1.outputs["Result"], 1: group_input.outputs["Vein Angle"]},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range.outputs["Result"], 1: multiply_1},
        attrs={'operation': 'MULTIPLY'})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_2, 1: group_input.outputs["Y"]},
        attrs={'operation': 'SUBTRACT'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply, 1: subtract})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Vein Coord': add})

@node_utils.to_nodegroup('nodegroup_nodegroup_shape_with_jigsaw', singleton=False, type='GeometryNodeTree')
def nodegroup_nodegroup_shape_with_jigsaw(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'Midrib Value', 1.0),
            ('NodeSocketFloat', 'Vein Coord', 0.0),
            ('NodeSocketFloat', 'Leaf Shape', 0.5),
            ('NodeSocketFloat', 'Jigsaw Scale', 18.0),
            ('NodeSocketFloat', 'Jigsaw Depth', 0.5)])
    
    map_range = nw.new_node(Nodes.MapRange,
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
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': multiply_add, 1: 0.001, 2: 0.002, 3: 1.0, 4: 0.0})
    
    maximum = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range.outputs["Result"], 1: map_range_1.outputs["Result"]},
        attrs={'operation': 'MAXIMUM'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Value': maximum})


@node_utils.to_nodegroup('nodegroup_nodegroup_vein_coord', singleton=False, type='GeometryNodeTree')
def nodegroup_nodegroup_vein_coord(nw: NodeWrangler, vein_curve_control_points, vein_curve_control_handles):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'X Modulated', 0.5),
            ('NodeSocketFloat', 'Y', 0.5),
            ('NodeSocketFloat', 'Vein Asymmetry', 0.0),
            ('NodeSocketFloat', 'Vein Angle', 2.0),
            ('NodeSocketFloat', 'Leaf Shape', 0.0)])
    
    sign = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["X Modulated"]},
        attrs={'operation': 'SIGN'})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Vein Asymmetry"], 1: sign},
        attrs={'operation': 'MULTIPLY'})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': group_input.outputs["Y"], 1: -1.0})
    
    absolute = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["X Modulated"]},
        attrs={'operation': 'ABSOLUTE', 'use_clamp': True})
    
    divide = nw.new_node(Nodes.Math,
        input_kwargs={0: absolute, 1: group_input.outputs["Leaf Shape"]},
        attrs={'operation': 'DIVIDE', 'use_clamp': True})
    
    vein_shape = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': divide},
        label='Vein Shape')
    node_utils.assign_curve(vein_shape.mapping.curves[0], vein_curve_control_points, vein_curve_control_handles)
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': vein_shape, 4: 1.9})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_1.outputs["Result"], 1: group_input.outputs["Vein Angle"]},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range.outputs["Result"], 1: multiply_1},
        attrs={'operation': 'MULTIPLY'})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_2, 1: group_input.outputs["Y"]},
        attrs={'operation': 'SUBTRACT'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply, 1: subtract})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Vein Coord': add})

@node_utils.to_nodegroup('nodegroup_nodegroup_shape', singleton=False, type='GeometryNodeTree')
def nodegroup_nodegroup_shape(nw: NodeWrangler, shape_curve_control_points=[(0.0, 0.0), (0.15, 0.2), (0.3864, 0.2625), (0.6227, 0.2), (0.7756, 0.1145), (0.8955, 0.0312), (1.0, 0.0)]):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'X Modulated', 0.0),
            ('NodeSocketFloat', 'Y', 0.0)])
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': group_input.outputs["X Modulated"], 'Y': group_input.outputs["Y"]})
    
    clamp = nw.new_node(Nodes.Clamp,
        input_kwargs={'Value': group_input.outputs["Y"], 'Min': -0.6, 'Max': 0.6})
    
    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Y': clamp})
    
    subtract = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: combine_xyz, 1: combine_xyz_1},
        attrs={'operation': 'SUBTRACT'})
    
    length = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: subtract.outputs["Vector"]},
        attrs={'operation': 'LENGTH'})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': group_input.outputs["Y"], 1: -0.6, 2: 0.6})
    
    leaf_shape = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': map_range.outputs["Result"]},
        label='Leaf shape')
    node_utils.assign_curve(leaf_shape.mapping.curves[0], shape_curve_control_points)
    
    subtract_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: length.outputs["Value"], 1: leaf_shape},
        attrs={'operation': 'SUBTRACT'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Leaf Shape': subtract_1, 'Value': leaf_shape})

@node_utils.to_nodegroup('nodegroup_nodegroup_midrib', singleton=False, type='GeometryNodeTree')
def nodegroup_nodegroup_midrib(nw: NodeWrangler, midrib_curve_control_points=[(0.0, 0.5), (0.2455, 0.5078), (0.5, 0.4938), (0.75, 0.503), (0.8773, 0.5125), (1.0, 0.5)]):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'X', 0.5),
            ('NodeSocketFloat', 'Y', -0.6),
            ('NodeSocketFloat', 'Midrib Length', 0.4),
            ('NodeSocketFloat', 'Midrib Width', 1.0),
            ('NodeSocketFloat', 'Stem Length', 0.8)])
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': group_input.outputs["Y"], 1: -0.6, 2: 0.6})
    
    stem_shape = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': map_range.outputs["Result"]},
        label='Stem shape')
    node_utils.assign_curve(stem_shape.mapping.curves[0], midrib_curve_control_points)
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': stem_shape, 3: -1.0})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_1.outputs["Result"], 1: group_input.outputs["X"]},
        attrs={'operation': 'SUBTRACT'})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Scale': 20.0})
    
    map_range_5 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': noise_texture.outputs["Fac"], 3: -1.0})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_5.outputs["Result"], 1: 0.01},
        attrs={'operation': 'MULTIPLY'})
    
    map_range_2 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': group_input.outputs["Y"], 1: -70.0, 2: group_input.outputs["Midrib Length"], 3: group_input.outputs["Midrib Width"], 4: 0.0})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply, 1: map_range_2.outputs["Result"]})
    
    absolute = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract},
        attrs={'operation': 'ABSOLUTE'})
    
    subtract_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: add, 1: absolute},
        attrs={'operation': 'SUBTRACT'})
    
    absolute_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Y"]},
        attrs={'operation': 'ABSOLUTE'})
    
    map_range_3 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': absolute_1, 2: group_input.outputs["Stem Length"], 3: 1.0, 4: 0.0})
    
    smooth_min = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract_1, 1: map_range_3.outputs["Result"], 2: 0.06},
        attrs={'operation': 'SMOOTH_MIN'})
    
    divide = nw.new_node(Nodes.Math,
        input_kwargs={0: 1.0, 1: smooth_min},
        attrs={'operation': 'DIVIDE', 'use_clamp': True})
    
    map_range_4 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': divide, 1: 0.001, 2: 0.03, 3: 1.0, 4: 0.0})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'X Modulated': subtract, 'Midrib Value': map_range_4.outputs["Result"]})

@node_utils.to_nodegroup('nodegroup_nodegroup_apply_vein_midrib', singleton=False, type='GeometryNodeTree')
def nodegroup_nodegroup_apply_vein_midrib(nw: NodeWrangler, random_scale_seed=1.08):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'Midrib Value', 0.5),
            ('NodeSocketFloat', 'Leaf Shape', 1.0),
            ('NodeSocketFloat', 'Vein Density', 6.0),
            ('NodeSocketFloat', 'Vein Coord - main', 0.0),
            ('NodeSocketFloat', 'Vein Coord - 1', 0.0),
            ('NodeSocketFloat', 'Vein Coord - 2', 0.0)])
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': group_input.outputs["Leaf Shape"], 1: -0.3, 2: 0.05, 3: 0.015, 4: 0.0})
    
    nodegroup = nw.new_node(nodegroup_random_mask_vein().name,
        input_kwargs={'Coord': group_input.outputs["Vein Coord - 2"], 'Shape': map_range.outputs["Result"], 'Density': group_input.outputs["Vein Density"], 'Random Scale Seed': random_scale_seed*2.7})
    
    nodegroup_1 = nw.new_node(nodegroup_random_mask_vein().name,
        input_kwargs={'Coord': group_input.outputs["Vein Coord - 1"], 'Shape': map_range.outputs["Result"], 'Density': group_input.outputs["Vein Density"], 'Random Scale Seed': random_scale_seed})
    
    vein = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'W': group_input.outputs["Vein Coord - main"], 'Scale': group_input.outputs["Vein Density"], 'Randomness': 0.2},
        label='Vein',
        attrs={'voronoi_dimensions': '1D'})
    
    position = nw.new_node(Nodes.InputPosition)
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': position, 'Scale': 20.0})
    
    map_range_3 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': noise_texture.outputs["Fac"], 3: -1.0})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_3.outputs["Result"], 1: 0.02},
        attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: vein.outputs["Distance"], 1: multiply})
    
    map_range_4 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': add, 2: 0.03, 3: 1.0, 4: 0.0})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range.outputs["Result"], 1: map_range_4.outputs["Result"]},
        attrs={'operation': 'MULTIPLY'})
    
    map_range_5 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': multiply_1, 1: 0.001, 2: 0.01, 3: 1.0, 4: 0.0})
    
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: nodegroup_1, 1: map_range_5.outputs["Result"]},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: nodegroup, 1: multiply_2},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_4 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Midrib Value"], 1: multiply_3},
        attrs={'operation': 'MULTIPLY'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Vein Value': multiply_4})

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
    
    voronoi_texture = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': combine_xyz, 'Scale': 30.0})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': voronoi_texture.outputs["Distance"], 2: 0.1, 4: 2.0},
        attrs={'clamp': False})
    
    voronoi_texture_1 = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': combine_xyz, 'Scale': 150.0},
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

@node_utils.to_nodegroup('nodegroup_nodegroup_leaf_gen', singleton=False, type='GeometryNodeTree')
def nodegroup_nodegroup_leaf_gen(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
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
            ('NodeSocketFloat', 'Stem Length', 0.8),])
    
    position = nw.new_node(Nodes.InputPosition)
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': position})
    
    nodegroup_midrib = nw.new_node(nodegroup_nodegroup_midrib(midrib_curve_control_points=kwargs['midrib_curve_control_points']).name,
        input_kwargs={'X': separate_xyz.outputs["X"], 'Y': separate_xyz.outputs["Y"], 'Midrib Length': group_input.outputs["Midrib Length"], 
        'Midrib Width': group_input.outputs["Midrib Width"], 
        'Stem Length': group_input.outputs["Stem Length"]})
    
    nodegroup_shape = nw.new_node(nodegroup_nodegroup_shape(shape_curve_control_points=kwargs['shape_curve_control_points']).name,
        input_kwargs={'X Modulated': nodegroup_midrib.outputs["X Modulated"], 'Y': separate_xyz.outputs["Y"]})
    
    nodegroup_vein_coord = nw.new_node(nodegroup_nodegroup_vein_coord(
            vein_curve_control_points=[(0.0, 0.0), (0.0182, 0.05), (0.3364, 0.2386), (0.6045, 0.4812), (0.7, 0.725), (0.8273, 0.8437), (1.0, 1.0)], 
            vein_curve_control_handles=['AUTO', 'AUTO', 'AUTO', 'VECTOR', 'AUTO', 'AUTO', 'AUTO']).name,
        input_kwargs={'X Modulated': nodegroup_midrib.outputs["X Modulated"], 'Y': separate_xyz.outputs["Y"], 
        'Vein Asymmetry': group_input.outputs["Vein Asymmetry"], 'Vein Angle': group_input.outputs["Vein Angle"], 'Leaf Shape': nodegroup_shape.outputs["Value"]})

    nodegroup_vein_coord_002 = nw.new_node(nodegroup_nodegroup_vein_coord(
            vein_curve_control_points=[(0.0, 0.0), (0.0182, 0.05), (0.3364, 0.2386), (0.8091, 0.7312), (1.0, 0.9937)], 
            vein_curve_control_handles=['AUTO', 'AUTO', 'AUTO', 'AUTO', 'AUTO']).name,
        input_kwargs={'X Modulated': nodegroup_midrib.outputs["X Modulated"], 'Y': separate_xyz.outputs["Y"], 
        'Vein Asymmetry': group_input.outputs["Vein Asymmetry"], 'Vein Angle': group_input.outputs["Vein Angle"], 'Leaf Shape': nodegroup_shape.outputs["Value"]})

    nodegroup_vein_coord_003 = nw.new_node(nodegroup_nodegroup_vein_coord(
            vein_curve_control_points=[(0.0, 0.0), (0.0182, 0.05), (0.2909, 0.2199), (0.4182, 0.3063), (0.7045, 0.3), (1.0, 0.8562)], 
            vein_curve_control_handles=['AUTO', 'AUTO', 'AUTO', 'VECTOR', 'AUTO', 'AUTO']).name,
        input_kwargs={'X Modulated': nodegroup_midrib.outputs["X Modulated"], 'Y': separate_xyz.outputs["Y"], 
        'Vein Asymmetry': group_input.outputs["Vein Asymmetry"], 'Vein Angle': group_input.outputs["Vein Angle"], 'Leaf Shape': nodegroup_shape.outputs["Value"]})
    
    nodegroup_apply_vein_midrib = nw.new_node(nodegroup_nodegroup_apply_vein_midrib(random_scale_seed=kwargs['vein_mask_random_seed']).name,
        input_kwargs={'Midrib Value': nodegroup_midrib.outputs["Midrib Value"], 'Leaf Shape': nodegroup_shape.outputs["Leaf Shape"], 'Vein Density': group_input.outputs["Vein Density"], 'Vein Coord - main': nodegroup_vein_coord_002, 'Vein Coord - 1': nodegroup_vein_coord, 'Vein Coord - 2': nodegroup_vein_coord_003})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Displancement scale"], 1: nodegroup_apply_vein_midrib},
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Z': multiply})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["Mesh"], 'Offset': combine_xyz})
    
    nodegroup_shape_with_jigsaw = nw.new_node(nodegroup_nodegroup_shape_with_jigsaw().name,
        input_kwargs={'Midrib Value': nodegroup_midrib.outputs["Midrib Value"], 'Vein Coord': nodegroup_vein_coord_002, 'Leaf Shape': nodegroup_shape.outputs["Leaf Shape"], 'Jigsaw Scale': group_input.outputs["Jigsaw Scale"], 'Jigsaw Depth': group_input.outputs["Jigsaw Depth"]})
    
    less_than = nw.new_node(Nodes.Compare,
        input_kwargs={0: nodegroup_shape_with_jigsaw, 1: 0.5},
        attrs={'operation': 'LESS_THAN'})
    
    delete_geometry = nw.new_node(Nodes.DeleteGeom,
        input_kwargs={'Geometry': set_position, 'Selection': less_than})
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': delete_geometry, 2: nodegroup_apply_vein_midrib})
    
    position_1 = nw.new_node(Nodes.InputPosition)
    
    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': position_1})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_xyz_1.outputs["Y"], 1: -0.6, 2: 0.6})
    
    float_curve_1 = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': map_range_1.outputs["Result"]})
    node_utils.assign_curve(float_curve_1.mapping.curves[0], [(0.0, 0.0), (0.5182, 1.0), (1.0, 1.0)])
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': nodegroup_shape.outputs["Leaf Shape"], 2: -1.0})
    
    float_curve = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': map_range.outputs["Result"]})
    node_utils.assign_curve(float_curve.mapping.curves[0], [(0.0045, 0.0063), (0.0409, 0.0375), (0.4182, 0.05), (1.0, 0.0)])
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: float_curve_1, 1: float_curve},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_1, 1: 0.7},
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Z': multiply_2})
    
    set_position_1 = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': capture_attribute.outputs["Geometry"], 'Offset': combine_xyz_1})
    
    nodegroup_vein_coord_001 = nw.new_node(nodegroup_nodegroup_vein_coord_001().name,
        input_kwargs={'X Modulated': nodegroup_midrib.outputs["X Modulated"], 'Y': separate_xyz.outputs["Y"], 'Vein Asymmetry': group_input.outputs["Vein Asymmetry"], 'Vein Angle': group_input.outputs["Vein Angle"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Mesh': set_position_1, 'Attribute': capture_attribute.outputs[2], 'X Modulated': nodegroup_midrib.outputs["X Modulated"], 'Vein Coord': nodegroup_vein_coord_001, 'Vein Value': nodegroup_apply_vein_midrib})

def shader_material(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    attribute_1 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'vein value'})
    
    # rgb_3 = nw.new_node(Nodes.RGB)
    # rgb_3.outputs[0].default_value = (0.9823, 0.8388, 0.117, 1.0)
    
    texture_coordinate = nw.new_node(Nodes.TextureCoord)
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': texture_coordinate.outputs["Object"], 'Scale': 6.8, 'Detail': 10.0, 'Roughness': 0.7})
    
    separate_rgb = nw.new_node(Nodes.SeparateRGB,
        input_kwargs={'Image': noise_texture.outputs["Color"]})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_rgb.outputs["G"], 1: 0.4, 2: 0.7, 3: 0.48, 4: 0.52})
    
    map_range_2 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_rgb.outputs["B"], 1: 0.4, 2: 0.7, 3: 0.8, 4: 1.2})
    
    attribute = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'subvein offset'})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': attribute.outputs["Color"], 2: -0.94})
    
    # rgb_1 = nw.new_node(Nodes.RGB)
    # rgb_1.outputs[0].default_value = (0.1878, 0.305, 0.0762, 1.0)
    
    # rgb = nw.new_node(Nodes.RGB)
    # rgb.outputs[0].default_value = (0.0762, 0.1441, 0.0529, 1.0)

    hue_saturation_value_1 = nw.new_node('ShaderNodeHueSaturation',
        input_kwargs={'Value': 2.0, 'Color': kwargs['color_base']})
    
    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': map_range.outputs["Result"], 'Color1': hue_saturation_value_1, 'Color2': kwargs['color_base']})
    
    hue_saturation_value = nw.new_node('ShaderNodeHueSaturation',
        input_kwargs={'Hue': map_range_1.outputs["Result"], 'Value': map_range_2.outputs["Result"], 'Color': mix})
    
    mix_1 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': attribute_1.outputs["Color"], 'Color1': kwargs['color_vein'], 'Color2': hue_saturation_value})

    leaf_shader = nw.new_node(nodegroup_leaf_shader().name,
        input_kwargs={'Color': mix_1})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': leaf_shader})

def geo_leaf_broadleaf(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None)])
    
    subdivide_mesh = nw.new_node(Nodes.SubdivideMesh,
        input_kwargs={'Mesh': group_input.outputs["Geometry"], 'Level': 10})
    
    # subdivide_mesh_1 = nw.new_node(Nodes.SubdivideMesh,
    #     input_kwargs={'Mesh': subdivide_mesh})
    
    position = nw.new_node(Nodes.InputPosition)
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': subdivide_mesh, 1: position},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    nodegroup_leaf_gen = nw.new_node(nodegroup_nodegroup_leaf_gen(**kwargs).name,
        input_kwargs={'Mesh': capture_attribute.outputs["Geometry"], 
            'Displancement scale': 0.005, 
            'Vein Asymmetry': kwargs['vein_asymmetry'], # 0.3023 
            'Vein Density': kwargs['vein_density'], # 7.0
            'Jigsaw Scale': kwargs['jigsaw_scale'], # 50
            'Jigsaw Depth': kwargs['jigsaw_depth'], # 0.3
            'Vein Angle': kwargs['vein_angle'], # 0.3
            'Midrib Length': kwargs['midrib_length'], # 0.3336 
            'Midrib Width': kwargs['midrib_length'], # 0.6302,
            'Stem Length': kwargs['stem_length'],
            })
    
    nodegroup_sub_vein = nw.new_node(nodegroup_nodegroup_sub_vein().name,
        input_kwargs={'X': nodegroup_leaf_gen.outputs["X Modulated"], 'Y': nodegroup_leaf_gen.outputs["Vein Coord"]})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: nodegroup_sub_vein.outputs["Value"], 1: 0.0002},
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Z': multiply})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': nodegroup_leaf_gen.outputs["Mesh"], 'Offset': combine_xyz})
    
    capture_attribute_1 = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': set_position, 2: nodegroup_sub_vein.outputs["Color Value"]})
    
    capture_attribute_2 = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': capture_attribute_1.outputs["Geometry"], 2: nodegroup_leaf_gen.outputs["Vein Value"]})
    
    apply_wave = nw.new_node(nodegroup_apply_wave(y_wave_control_points=kwargs['y_wave_control_points'], x_wave_control_points=kwargs['x_wave_control_points']).name,
        input_kwargs={'Geometry': capture_attribute_2.outputs["Geometry"], 'Wave Scale X': 0.2, 'Wave Scale Y': 1.0, 'X Modulated': nodegroup_leaf_gen.outputs["X Modulated"]})
    
    move_to_origin = nw.new_node(nodegroup_move_to_origin().name,
        input_kwargs={'Geometry': apply_wave})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': move_to_origin, 'Offset': nodegroup_leaf_gen.outputs["Attribute"], 'Coordinate': capture_attribute.outputs["Attribute"], 'subvein offset': capture_attribute_1.outputs[2], 'vein value': capture_attribute_2.outputs[2]})



class LeafFactoryBroadleaf(AssetFactory):

    scale = 0.5

    def __init__(self, factory_seed, season='autumn', coarse=False):
        super(LeafFactoryBroadleaf, self).__init__(factory_seed, coarse=coarse)

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
        leaf_width_1 = uniform(0.2, 0.4)
        leaf_width_2 = uniform(0.1, leaf_width_1)

        leaf_offset_1 = uniform(0.49, 0.51)

        return {
            'midrib_length': uniform(0.0, 0.8),
            'midrib_width': uniform(0.5, 1.0),
            'stem_length': uniform(0.7, 0.9),
            'vein_asymmetry': uniform(0.0, 1.0),
            'vein_angle': uniform(0.4, 1.0),
            'vein_density': uniform(3.0, 8.0),
            'subvein_scale': uniform(10.0, 20.0),
            'jigsaw_scale': uniform(30.0, 70.0),
            'jigsaw_depth': uniform(0.0, 0.6),
            'vein_mask_random_seed': uniform(0.0, 100.0),
            'midrib_curve_control_points': [(0.0, 0.5), (0.25, leaf_offset_1), (0.75, 1.0-leaf_offset_1), (1.0, 0.5)],
            'shape_curve_control_points': [(0.0, 0.0), (uniform(0.2, 0.4), leaf_width_1), (uniform(0.6, 0.8), leaf_width_2), (1.0, 0.0)],
            'vein_curve_control_points': [(0.0, 0.0), (0.25, uniform(0.1, 0.4)), (0.75, uniform(0.6, 0.9)), (1.0, 1.0)],      
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

        phenome['y_wave_control_points'] = [(0.0, 0.5), (uniform(0.25, 0.75), uniform(0.50, 0.60)), (1.0, 0.5)]
        x_wave_val = np.random.uniform(0.50, 0.58)
        phenome['x_wave_control_points'] = [(0.0, 0.5), (0.4, x_wave_val), (0.5, 0.5), (0.6, x_wave_val), (1.0, 0.5)]

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

        surface.add_geomod(obj, geo_leaf_broadleaf, apply=False, attributes=['offset', 'coordinate', 'subvein offset', 'vein value'], input_kwargs=phenome)
        surface.add_material(obj, shader_material, reuse=False, input_kwargs=material_kwargs)

        bpy.ops.object.convert(target='MESH')

        obj = bpy.context.object
        obj.scale *= normal(1, 0.1) * self.scale
        butil.apply_transform(obj)
        tag_object(obj, 'leaf_broadleaf')

        return obj