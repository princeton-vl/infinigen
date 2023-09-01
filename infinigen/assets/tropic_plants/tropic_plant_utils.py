# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Beining Han


import bpy
import mathutils
from numpy.random import uniform, normal, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core import surface
from infinigen.core.placement.factory import AssetFactory
import numpy as np
from infinigen.core.util.color import hsv2rgba
from infinigen.core.util import blender as butil


@node_utils.to_nodegroup('nodegroup_node_group', singleton=False, type='GeometryNodeTree')
def nodegroup_node_group(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'Coord', 0.0),
                                            ('NodeSocketFloat', 'Shape', 0.5),
                                            ('NodeSocketFloat', 'Density', 0.5),
                                            ('NodeSocketFloat', 'Random Scale Seed', 0.5)])

    vein = nw.new_node(Nodes.VoronoiTexture,
                       input_kwargs={'W': group_input.outputs["Coord"], 'Scale': group_input.outputs["Density"],
                                     'Randomness': 0.2},
                       label='Vein',
                       attrs={'voronoi_dimensions': '1D'})

    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: group_input.outputs["Density"],
                                         1: group_input.outputs["Random Scale Seed"]},
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
    node_utils.assign_curve(vein_shape.mapping.curves[0],
                            [(0.0, 0.0), (0.0182, 0.05), (0.3364, 0.2386), (0.7227, 0.75), (1.0, 1.0)])

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
                         input_kwargs={'W': group_input.outputs["Vein Coord"],
                                       'Scale': group_input.outputs["Jigsaw Scale"]},
                         label='Jigsaw',
                         attrs={'voronoi_dimensions': '1D'})

    colorramp = nw.new_node(Nodes.ColorRamp,
                            input_kwargs={'Fac': jigsaw.outputs["Distance"]})
    colorramp.color_ramp.elements[0].position = 0.4795
    colorramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp.color_ramp.elements[1].position = 0.5545
    colorramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: group_input.outputs["Jigsaw Depth"], 1: 0.0},
                           attrs={'operation': 'MULTIPLY'})

    multiply_add = nw.new_node(Nodes.Math,
                               input_kwargs={0: colorramp.outputs["Color"], 1: multiply,
                                             2: group_input.outputs["Leaf Shape"]},
                               attrs={'operation': 'MULTIPLY_ADD'})

    map_range_1 = nw.new_node(Nodes.MapRange,
                              input_kwargs={'Value': multiply_add, 1: 0.001, 2: 0.002, 3: 1.0, 4: 0.0})

    maximum = nw.new_node(Nodes.Math,
                          input_kwargs={0: map_range.outputs["Result"], 1: map_range_1.outputs["Result"]},
                          attrs={'operation': 'MAXIMUM'})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Value': maximum})


@node_utils.to_nodegroup('nodegroup_nodegroup_vein_coord_003', singleton=False, type='GeometryNodeTree')
def nodegroup_nodegroup_vein_coord_003(nw: NodeWrangler):
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
    node_utils.assign_curve(vein_shape.mapping.curves[0],
                            [(0.0, 0.0), (0.0182, 0.05), (0.2909, 0.2199), (0.4182, 0.3063), (0.7045, 0.3),
                             (1.0, 0.8562)], handles=['AUTO', 'AUTO', 'AUTO', 'VECTOR', 'AUTO', 'AUTO'])

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


@node_utils.to_nodegroup('nodegroup_nodegroup_vein_coord', singleton=False, type='GeometryNodeTree')
def nodegroup_nodegroup_vein_coord(nw: NodeWrangler):
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
    node_utils.assign_curve(vein_shape.mapping.curves[0],
                            [(0.0, 0.0), (0.0182, 0.05), (0.3364, 0.2386), (0.6045, 0.4812), (0.7, 0.725),
                             (0.8273, 0.8437), (1.0, 1.0)],
                            handles=['AUTO', 'AUTO', 'AUTO', 'VECTOR', 'AUTO', 'AUTO', 'AUTO'])

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


@node_utils.to_nodegroup('nodegroup_nodegroup_vein_coord_002', singleton=False, type='GeometryNodeTree')
def nodegroup_nodegroup_vein_coord_002(nw: NodeWrangler):
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
    node_utils.assign_curve(vein_shape.mapping.curves[0],
                            [(0.0, 0.0), (0.0182, 0.05), (0.3364, 0.2386), (0.8091, 0.7312), (1.0, 0.9937)])

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
def nodegroup_nodegroup_shape(nw: NodeWrangler, leaf_contour_control_points=None):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'X Modulated', 0.0),
                                            ('NodeSocketFloat', 'Y', 0.0),
                                            ('NodeSocketFloat', 'scale', 0.0)])

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

    if leaf_contour_control_points is not None:
        node_utils.assign_curve(leaf_shape.mapping.curves[0],
                                [(0.0, 0.0), (0.1, leaf_contour_control_points[0]),
                                 (0.25, leaf_contour_control_points[1]),
                                 (0.4, leaf_contour_control_points[2]),
                                 (0.55, leaf_contour_control_points[3]),
                                 (0.7, leaf_contour_control_points[4]),
                                 (0.85, leaf_contour_control_points[5]), (1.0, 0.0)])
    else:
        node_utils.assign_curve(leaf_shape.mapping.curves[0],
                                [(0.0, 0.0), (0.15, 0.25), (0.3818, 0.35), (0.6273, 0.3625), (0.7802, 0.2957),
                                 (0.8955, 0.2), (1.0, 0.0)])

    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: leaf_shape, 1: group_input.outputs["scale"]},
                           attrs={'operation': 'MULTIPLY'})

    subtract_1 = nw.new_node(Nodes.Math,
                             input_kwargs={0: length.outputs["Value"], 1: multiply},
                             attrs={'operation': 'SUBTRACT'})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Leaf Shape': subtract_1, 'Value': multiply})


@node_utils.to_nodegroup('nodegroup_nodegroup_leaf_gen', singleton=False, type='GeometryNodeTree')
def nodegroup_nodegroup_leaf_gen(nw: NodeWrangler, leaf_contour_control_points=None):
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
                                            ('NodeSocketFloat', 'Stem Length', 0.8),
                                            ('NodeSocketFloat', 'Leaf Width Scale', 0.0)])

    position = nw.new_node(Nodes.InputPosition)

    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
                               input_kwargs={'Vector': position})

    nodegroup_midrib = nw.new_node(nodegroup_nodegroup_midrib().name,
                                   input_kwargs={'X': separate_xyz.outputs["X"], 'Y': separate_xyz.outputs["Y"],
                                                 'Midrib Length': group_input.outputs["Midrib Length"],
                                                 'Midrib Width': group_input.outputs["Midrib Width"],
                                                 'Stem Length': group_input.outputs["Stem Length"]})

    nodegroup_shape = nw.new_node(nodegroup_nodegroup_shape(leaf_contour_control_points).name,
                                  input_kwargs={'X Modulated': nodegroup_midrib.outputs["X Modulated"],
                                                'Y': separate_xyz.outputs["Y"],
                                                'scale': group_input.outputs["Leaf Width Scale"]})

    nodegroup_vein_coord_002 = nw.new_node(nodegroup_nodegroup_vein_coord_002().name,
                                           input_kwargs={'X Modulated': nodegroup_midrib.outputs["X Modulated"],
                                                         'Y': separate_xyz.outputs["Y"],
                                                         'Vein Asymmetry': group_input.outputs["Vein Asymmetry"],
                                                         'Vein Angle': group_input.outputs["Vein Angle"],
                                                         'Leaf Shape': nodegroup_shape.outputs["Value"]})

    nodegroup_vein_coord = nw.new_node(nodegroup_nodegroup_vein_coord().name,
                                       input_kwargs={'X Modulated': nodegroup_midrib.outputs["X Modulated"],
                                                     'Y': separate_xyz.outputs["Y"],
                                                     'Vein Asymmetry': group_input.outputs["Vein Asymmetry"],
                                                     'Vein Angle': group_input.outputs["Vein Angle"],
                                                     'Leaf Shape': nodegroup_shape.outputs["Value"]})

    nodegroup_vein_coord_003 = nw.new_node(nodegroup_nodegroup_vein_coord_003().name,
                                           input_kwargs={'X Modulated': nodegroup_midrib.outputs["X Modulated"],
                                                         'Y': separate_xyz.outputs["Y"],
                                                         'Vein Asymmetry': group_input.outputs["Vein Asymmetry"],
                                                         'Vein Angle': group_input.outputs["Vein Angle"],
                                                         'Leaf Shape': nodegroup_shape.outputs["Value"]})

    nodegroup_apply_vein_midrib = nw.new_node(nodegroup_nodegroup_apply_vein_midrib().name,
                                              input_kwargs={'Midrib Value': nodegroup_midrib.outputs["Midrib Value"],
                                                            'Leaf Shape': nodegroup_shape.outputs["Leaf Shape"],
                                                            'Vein Density': group_input.outputs["Vein Density"],
                                                            'Vein Coord - main': nodegroup_vein_coord_002,
                                                            'Vein Coord - 1': nodegroup_vein_coord,
                                                            'Vein Coord - 2': nodegroup_vein_coord_003})

    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: group_input.outputs["Displancement scale"], 1: nodegroup_apply_vein_midrib},
                           attrs={'operation': 'MULTIPLY'})

    combine_xyz = nw.new_node(Nodes.CombineXYZ,
                              input_kwargs={'Z': multiply})

    set_position = nw.new_node(Nodes.SetPosition,
                               input_kwargs={'Geometry': group_input.outputs["Mesh"], 'Offset': combine_xyz})

    nodegroup_shape_with_jigsaw = nw.new_node(nodegroup_nodegroup_shape_with_jigsaw().name,
                                              input_kwargs={'Midrib Value': nodegroup_midrib.outputs["Midrib Value"],
                                                            'Vein Coord': nodegroup_vein_coord_002,
                                                            'Leaf Shape': nodegroup_shape.outputs["Leaf Shape"],
                                                            'Jigsaw Scale': group_input.outputs["Jigsaw Scale"],
                                                            'Jigsaw Depth': group_input.outputs["Jigsaw Depth"]})

    less_than = nw.new_node(Nodes.Compare,
                            input_kwargs={0: nodegroup_shape_with_jigsaw, 1: 0.5},
                            attrs={'operation': 'LESS_THAN'})

    delete_geometry = nw.new_node(Nodes.DeleteGeometry,
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
    node_utils.assign_curve(float_curve.mapping.curves[0],
                            [(0.0045, 0.0063), (0.0409, 0.0375), (0.4182, 0.05), (1.0, 0.0)])

    multiply_1 = nw.new_node(Nodes.Math,
                             input_kwargs={0: float_curve_1, 1: float_curve},
                             attrs={'operation': 'MULTIPLY'})

    multiply_2 = nw.new_node(Nodes.Math,
                             input_kwargs={0: multiply_1, 1: 0.7},
                             attrs={'operation': 'MULTIPLY'})

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
                                input_kwargs={'Z': multiply_2})

    set_position_1 = nw.new_node(Nodes.SetPosition,
                                 input_kwargs={'Geometry': capture_attribute.outputs["Geometry"],
                                               'Offset': combine_xyz_1})

    nodegroup_vein_coord_001 = nw.new_node(nodegroup_nodegroup_vein_coord_001().name,
                                           input_kwargs={'X Modulated': nodegroup_midrib.outputs["X Modulated"],
                                                         'Y': separate_xyz.outputs["Y"],
                                                         'Vein Asymmetry': group_input.outputs["Vein Asymmetry"],
                                                         'Vein Angle': group_input.outputs["Vein Angle"]})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Mesh': set_position_1, 'Attribute': capture_attribute.outputs[2],
                                             'X Modulated': nodegroup_midrib.outputs["X Modulated"],
                                             'Vein Coord': nodegroup_vein_coord_001,
                                             'Vein Value': nodegroup_apply_vein_midrib})


@node_utils.to_nodegroup('nodegroup_nodegroup_midrib', singleton=False, type='GeometryNodeTree')
def nodegroup_nodegroup_midrib(nw: NodeWrangler):
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
    node_utils.assign_curve(stem_shape.mapping.curves[0],
                            [(0.0, 0.5), (0.25, 0.4828), (0.5, 0.4938), (0.75, 0.503), (0.8773, 0.5125), (1.0, 0.5)])

    map_range_1 = nw.new_node(Nodes.MapRange,
                              input_kwargs={'Value': stem_shape, 3: -1.0})

    subtract = nw.new_node(Nodes.Math,
                           input_kwargs={0: map_range_1.outputs["Result"], 1: group_input.outputs["X"]},
                           attrs={'operation': 'SUBTRACT'})

    noise_texture = nw.new_node(Nodes.NoiseTexture)

    map_range_5 = nw.new_node(Nodes.MapRange,
                              input_kwargs={'Value': noise_texture.outputs["Fac"], 3: -1.0})

    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: map_range_5.outputs["Result"], 1: 0.01},
                           attrs={'operation': 'MULTIPLY'})

    map_range_2 = nw.new_node(Nodes.MapRange,
                              input_kwargs={'Value': group_input.outputs["Y"], 1: -70.0,
                                            2: group_input.outputs["Midrib Length"],
                                            3: group_input.outputs["Midrib Width"], 4: 0.0})

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
def nodegroup_nodegroup_apply_vein_midrib(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'Midrib Value', 0.5),
                                            ('NodeSocketFloat', 'Leaf Shape', 1.0),
                                            ('NodeSocketFloat', 'Vein Density', 6.0),
                                            ('NodeSocketFloat', 'Vein Coord - main', 0.0),
                                            ('NodeSocketFloat', 'Vein Coord - 1', 0.0),
                                            ('NodeSocketFloat', 'Vein Coord - 2', 0.0)])

    map_range = nw.new_node(Nodes.MapRange,
                            input_kwargs={'Value': group_input.outputs["Leaf Shape"], 1: -0.3, 2: 0.05, 3: 0.015,
                                          4: 0.0})

    nodegroup = nw.new_node(nodegroup_node_group().name,
                            input_kwargs={'Coord': group_input.outputs["Vein Coord - 2"],
                                          'Shape': map_range.outputs["Result"],
                                          'Density': group_input.outputs["Vein Density"], 'Random Scale Seed': 3.57})

    nodegroup_1 = nw.new_node(nodegroup_node_group().name,
                              input_kwargs={'Coord': group_input.outputs["Vein Coord - 1"],
                                            'Shape': map_range.outputs["Result"],
                                            'Density': group_input.outputs["Vein Density"], 'Random Scale Seed': 1.08})

    vein = nw.new_node(Nodes.VoronoiTexture,
                       input_kwargs={'W': group_input.outputs["Vein Coord - main"],
                                     'Scale': group_input.outputs["Vein Density"], 'Randomness': 0.2},
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

@node_utils.to_nodegroup('nodegroup_nodegroup_move_to_origin', singleton=False, type='GeometryNodeTree')
def nodegroup_nodegroup_move_to_origin(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Geometry', None)])

    position = nw.new_node(Nodes.InputPosition)

    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
                               input_kwargs={'Vector': position})

    attribute_statistic = nw.new_node(Nodes.AttributeStatistic,
                                      input_kwargs={'Geometry': group_input.outputs["Geometry"],
                                                    2: separate_xyz.outputs["Y"]})

    subtract = nw.new_node(Nodes.Math,
                           input_kwargs={0: 0.0, 1: attribute_statistic.outputs["Min"]},
                           attrs={'operation': 'SUBTRACT'})

    combine_xyz = nw.new_node(Nodes.CombineXYZ,
                              input_kwargs={'Y': subtract})

    set_position = nw.new_node(Nodes.SetPosition,
                               input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Offset': combine_xyz})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': set_position})


@node_utils.to_nodegroup('nodegroup_nodegroup_leaf_rotate_x', singleton=False, type='GeometryNodeTree')
def nodegroup_nodegroup_leaf_rotate_x(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Geometry', None),
                                            ('NodeSocketFloat', 'To Max', -0.4)])

    position_1 = nw.new_node(Nodes.InputPosition)

    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
                               input_kwargs={'Vector': position_1})

    map_range = nw.new_node(Nodes.MapRange,
                            input_kwargs={'Value': separate_xyz.outputs["Y"], 4: group_input.outputs["To Max"]},
                            attrs={'clamp': False})

    vector_rotate = nw.new_node(Nodes.VectorRotate,
                                input_kwargs={'Vector': position_1, 'Angle': map_range.outputs["Result"]},
                                attrs={'rotation_type': 'X_AXIS'})

    set_position_1 = nw.new_node(Nodes.SetPosition,
                                 input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Position': vector_rotate})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': set_position_1})


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


@node_utils.to_nodegroup('nodegroup_nodegroup_leaf_shader', singleton=False, type='ShaderNodeTree')
def nodegroup_nodegroup_leaf_shader(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketColor', 'Color', (0.8, 0.8, 0.8, 1.0))])

    diffuse_bsdf = nw.new_node(Nodes.DiffuseBSDF,
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


def shader_stem_material(nw: NodeWrangler, stem_color_hsv=None):
    # Code generated using version 2.4.3 of the node_transpiler

    if stem_color_hsv is None:
        stem_color_hsv = (uniform(0.25, 0.32), uniform(0.6, 0.9), uniform(0.2, 0.6))
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
                                  input_kwargs={'Base Color': hsv2rgba(stem_color_hsv)})

    material_output = nw.new_node(Nodes.MaterialOutput,
                                  input_kwargs={'Surface': principled_bsdf})




