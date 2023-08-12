# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Beining Han
# Acknowledgement: This file draws inspiration from https://www.youtube.com/watch?v=MGxNuS_-bpo by Bad Normals


import bpy
import mathutils

import gin
import numpy as np
from numpy.random import uniform, normal, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler

from infinigen.assets.small_plants import leaf_general as Leaf

from infinigen.core.nodes import node_utils
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core import surface
from infinigen.assets.materials import simple_greenery

from infinigen.assets.utils.tag import tag_object, tag_nodegroup

def random_pinnae_level2_curvature():
    z_max_curvature = uniform(0.3, 0.45, (1,))[0]
    y_curvature_noise = np.clip(np.abs(normal(0., 0.2, (1,))), a_min=0.0, a_max=0.3)[0]
    y_curvature_k = uniform(-0.04, 0.2, (1,))[0]
    z_curvature, y_curvature = [0.25], [0.5]
    for k in range(1, 6):
        z_curvature.append(0.25 + z_max_curvature * k / 5.)
        y_curvature.append(0.5 + y_curvature_k + y_curvature_noise * k / 5.)
    x_curvature = [0.0 for _ in range(6)]
    return x_curvature, y_curvature, z_curvature


@node_utils.to_nodegroup('nodegroup_pinnae_level1_yaxis_rotation', singleton=False, type='GeometryNodeTree')
def nodegroup_pinnae_level1_yaxis_rotation(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler
    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'From Max', 1.0),
                                            ('NodeSocketFloat', 'Value', 1.0)])
    map_range = nw.new_node(Nodes.MapRange,
                            input_kwargs={'Value': group_input.outputs["Value"], 2: group_input.outputs["From Max"]})
    curvature = np.clip(normal(0, 0.3, 1), a_min=-0.4, a_max=0.4)
    float_curve = nw.new_node(Nodes.FloatCurve, input_kwargs={'Value': map_range.outputs["Result"]})
    node_utils.assign_curve(float_curve.mapping.curves[0],
                            [(0.0, 0.5), (0.1, curvature / 5. + 0.5), (0.25, curvature / 2.5 + 0.5),
                             (0.45, curvature / 1.5 + 0.5), (0.6, curvature / 1.2 + 0.5), (1.0, curvature + 0.5)])
    add = nw.new_node(Nodes.Math, input_kwargs={0: float_curve, 1: -0.5}, attrs={'operation': 'ADD'})
    multiply = nw.new_node(Nodes.Math, input_kwargs={0: add, 1: 1.0}, attrs={'operation': 'MULTIPLY'})
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Value': multiply})


@node_utils.to_nodegroup('nodegroup_pinnae_level1_zaxis_rotation', singleton=False, type='GeometryNodeTree')
def nodegroup_pinnae_level1_zaxis_rotation(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler
    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'From Max', 1.0), ('NodeSocketFloat', 'Value', 1.0)])
    map_range = nw.new_node(Nodes.MapRange,
                            input_kwargs={'Value': group_input.outputs["Value"], 2: group_input.outputs["From Max"]})
    curvature = normal(0, 0.2, 1)
    float_curve = nw.new_node(Nodes.FloatCurve, input_kwargs={'Value': map_range.outputs["Result"]})
    node_utils.assign_curve(float_curve.mapping.curves[0],
                            [(0.0, 0.5), (0.1, curvature / 5. + 0.5),
                             (0.25, curvature / 2.5 + 0.5), (0.45, curvature / 1.5 + 0.5),
                             (0.6, curvature / 1.2 + 0.5), (1.0, curvature + 0.5)])
    add = nw.new_node(Nodes.Math, input_kwargs={0: float_curve, 1: -0.5}, attrs={'operation': 'ADD'})
    multiply = nw.new_node(Nodes.Math, input_kwargs={0: add, 1: 1.0}, attrs={'operation': 'MULTIPLY'})
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Value': multiply})


@node_utils.to_nodegroup('nodegroup_pinnae_level1_gravity_rotation', singleton=False, type='GeometryNodeTree')
def nodegroup_pinnae_level1_gravity_rotation(nw: NodeWrangler, gravity_rotation=1.):
    # Code generated using version 2.4.3 of the node_transpiler
    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'From Max', 1.0), ('NodeSocketFloat', 'Value', 1.0)])
    map_range = nw.new_node(Nodes.MapRange,
                            input_kwargs={'Value': group_input.outputs["Value"], 2: group_input.outputs["From Max"]})
    curvature = uniform(0.25, 0.42, size=(1,))[0] * gravity_rotation
    float_curve = nw.new_node(Nodes.FloatCurve, input_kwargs={'Value': map_range.outputs["Result"]})
    node_utils.assign_curve(float_curve.mapping.curves[0],
                            [(0.0, 0.5), (0.1, curvature / 5. + 0.5),
                             (0.25, curvature / 2.5 + 0.5), (0.45, curvature / 1.67 + 0.5),
                             (0.6, curvature / 1.25 + 0.5), (1.0, curvature + 0.5)])
    add = nw.new_node(Nodes.Math, input_kwargs={0: float_curve, 1: -0.5}, attrs={'operation': 'ADD'})
    multiply = nw.new_node(Nodes.Math, input_kwargs={0: add, 1: 1}, attrs={'operation': 'MULTIPLY'})
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Value': multiply})


@node_utils.to_nodegroup('nodegroup_pinnae_level1_xaxis_rotation', singleton=False, type='GeometryNodeTree')
def nodegroup_pinnae_level1_xaxis_rotation(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'From Max', 1.0000),
                                            ('NodeSocketFloat', 'Value1', 1.0000),
                                            ('NodeSocketFloat', 'Value2', 1.0000)])

    map_range_1 = nw.new_node(Nodes.MapRange,
                              input_kwargs={'Value': group_input.outputs["Value1"], 2: group_input.outputs["From Max"]},
                              attrs={'clamp': False})

    float_curve = nw.new_node(Nodes.FloatCurve, input_kwargs={'Value': map_range_1.outputs["Result"]})
    node_utils.assign_curve(float_curve.mapping.curves[0],
                            [(0.0000, 0.0000), (0.2000, 0.2563), (0.4843, 0.4089), (0.7882, 0.3441), (1.0000, 0.0000)])

    map_range = nw.new_node(Nodes.MapRange, input_kwargs={'Value': group_input.outputs['Value2'], 3: -1.5000, 4: 0.0000})

    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: float_curve, 1: map_range.outputs["Result"]},
                           attrs={'operation': 'MULTIPLY'})

    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Value': multiply}, attrs={'is_active_output': True})


@node_utils.to_nodegroup('nodegroup_pinnae_level1_stein', singleton=False, type='GeometryNodeTree')
def nodegroup_pinnae_level1_stein(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Mesh', None),
                                            ('NodeSocketFloat', 'Value1', 0.5),
                                            ('NodeSocketFloat', 'Value2', 0.5)])
    mesh_to_curve = nw.new_node(Nodes.MeshToCurve, input_kwargs={'Mesh': group_input.outputs["Mesh"]})
    multiply = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs['Value2'], 1: 0.01},
                           attrs={'operation': 'MULTIPLY'})
    set_curve_radius = nw.new_node(Nodes.SetCurveRadius, input_kwargs={'Curve': mesh_to_curve, 'Radius': multiply})
    multiply_1 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Value1"], 1: 15.0},
                             attrs={'operation': 'MULTIPLY'})
    curve_circle = nw.new_node(Nodes.CurveCircle, input_kwargs={'Radius': multiply_1, 'Resolution': 10})
    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
                                input_kwargs={'Curve': set_curve_radius,
                                              'Profile Curve': curve_circle.outputs["Curve"]})
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Mesh': curve_to_mesh})


@node_utils.to_nodegroup('nodegroup_pinnae_level1_scale', singleton=False, type='GeometryNodeTree')
def nodegroup_pinnae_level1_scale(nw: NodeWrangler, pinnae_contour):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'Value1', 1.0),
                                            ('NodeSocketFloat', 'Value2', 1.0)])

    pinnae_contour_float_curve = nw.new_node(Nodes.FloatCurve, input_kwargs={'Value': group_input.outputs["Value1"]},
                                             label='PinnaeContourFloatCurve')
    node_utils.assign_curve(pinnae_contour_float_curve.mapping.curves[0],
                            [(0.0, pinnae_contour[0]), (0.2, pinnae_contour[1]), (0.4, pinnae_contour[2]),
                             (0.55, pinnae_contour[3]), (0.7, pinnae_contour[4]), (0.8, pinnae_contour[5]),
                             (0.9, pinnae_contour[6]), (1.0, pinnae_contour[7])])
    map_range = nw.new_node(Nodes.MapRange, input_kwargs={'Value': group_input.outputs['Value2'], 3: 1.0, 4: 3.0})
    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: pinnae_contour_float_curve, 1: map_range.outputs["Result"]},
                           attrs={'operation': 'MULTIPLY'})
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Value': multiply})


@node_utils.to_nodegroup('nodegroup_pinnae_level1_instance_rotation', singleton=False, type='GeometryNodeTree')
def nodegroup_pinnae_level1_instance_rotation(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'Value1', 0.5),
                                            ('NodeSocketFloat', 'Value2', 1.0)])
    map_range_8 = nw.new_node(Nodes.MapRange, input_kwargs={'Value': group_input.outputs['Value2'], 3: 2, 4: 3.1})
    add = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Value1"], 1: map_range_8.outputs["Result"]})
    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': add})
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Vector': combine_xyz})


@node_utils.to_nodegroup('nodegroup_pinnae_level1_rotation', singleton=False, type='GeometryNodeTree')
def nodegroup_pinnae_level1_rotation(nw: NodeWrangler, gravity_rotation=1):
    # Code generated using version 2.4.3 of the node_transpiler

    position = nw.new_node(Nodes.InputPosition)
    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Geometry', None),
                                            ('NodeSocketFloat', 'Value1', 1.0),
                                            ('NodeSocketFloat', 'Value2', 0.5)])
    bounding_box = nw.new_node(Nodes.BoundingBox, input_kwargs={'Geometry': group_input.outputs["Geometry"]})
    multiply = nw.new_node(Nodes.VectorMath, input_kwargs={0: bounding_box.outputs["Max"], 1: (0.0, 0.0, 1.0)},
                           attrs={'operation': 'MULTIPLY'})
    add = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs['Value2'], 1: 0.0})
    pinnae_index = nw.new_node(Nodes.Index, label='PinnaeIndex')
    pinnaelevel1xaxisrotation = nw.new_node(nodegroup_pinnae_level1_xaxis_rotation().name,
                                            input_kwargs={'From Max': add, 1: pinnae_index,
                                                          2: group_input.outputs["Value1"]})
    vector_rotate = nw.new_node(Nodes.VectorRotate,
                                input_kwargs={'Vector': position, 'Center': (0, 0, 0),
                                              'Angle': pinnaelevel1xaxisrotation},
                                attrs={'rotation_type': 'X_AXIS'})
    pinnaelevel1gravityrotation = nw.new_node(nodegroup_pinnae_level1_gravity_rotation(gravity_rotation=gravity_rotation).name,
                                              input_kwargs={'From Max': add, 'Value': pinnae_index})
    vector_rotate_1 = nw.new_node(Nodes.VectorRotate,
                                  input_kwargs={'Vector': vector_rotate, 'Center': (0, 0, 0),
                                                'Angle': pinnaelevel1gravityrotation},
                                  attrs={'rotation_type': 'X_AXIS'})
    pinnaelevel1zaxisrotation = nw.new_node(nodegroup_pinnae_level1_zaxis_rotation().name,
                                            input_kwargs={'From Max': add, 'Value': pinnae_index})
    vector_rotate_2 = nw.new_node(Nodes.VectorRotate,
                                  input_kwargs={'Vector': vector_rotate_1, 'Center': multiply.outputs["Vector"],
                                                'Angle': pinnaelevel1zaxisrotation},
                                  attrs={'rotation_type': 'Z_AXIS'})
    pinnaelevel1yaxisrotation = nw.new_node(nodegroup_pinnae_level1_yaxis_rotation().name,
                                            input_kwargs={'From Max': add, 'Value': pinnae_index})
    vector_rotate_3 = nw.new_node(Nodes.VectorRotate,
                                  input_kwargs={'Vector': vector_rotate_2, 'Center': multiply.outputs["Vector"],
                                                'Angle': pinnaelevel1yaxisrotation},
                                  attrs={'rotation_type': 'Y_AXIS'})
    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Vector': vector_rotate_3, 'Value': pinnaelevel1xaxisrotation})


@node_utils.to_nodegroup('nodegroup_pinnae_level1_instance_position', singleton=False, type='GeometryNodeTree')
def nodegroup_pinnae_level1_instance_position(nw: NodeWrangler, pinnae_contour):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'Value1', 1.0),
                                            ('NodeSocketFloat', 'From Max', 1.0),
                                            ('NodeSocketFloat', 'Value2', 1.0)])

    map_range_3 = nw.new_node(Nodes.MapRange,
                              input_kwargs={'Value': group_input.outputs["Value1"], 2: group_input.outputs["From Max"],
                                            3: 1.0, 4: 0.0})

    float_curve_2 = nw.new_node(Nodes.FloatCurve, input_kwargs={'Value': map_range_3.outputs["Result"]})
    node_utils.assign_curve(float_curve_2.mapping.curves[0],
                            [(0.0, pinnae_contour[0]), (0.2, pinnae_contour[1]), (0.4, pinnae_contour[2]),
                             (0.55, pinnae_contour[3]), (0.7, pinnae_contour[4]), (0.8, pinnae_contour[5]),
                             (0.9, pinnae_contour[6]), (1.0, pinnae_contour[7])])
    accumulate_field_1 = nw.new_node(Nodes.AccumulateField, input_kwargs={1: float_curve_2})
    # pinnae scale w.r.t fern age
    map_range_5 = nw.new_node(Nodes.MapRange, input_kwargs={'Value': group_input.outputs['Value2'], 3: 0.3, 4: 4.5})
    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: accumulate_field_1.outputs[4], 1: map_range_5.outputs["Result"]},
                           attrs={'operation': 'MULTIPLY'})
    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Z': multiply})
    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Vector': combine_xyz_1, 'Result': map_range_3.outputs["Result"]})


@node_utils.to_nodegroup('nodegroup_pinnae_level2_rotation', singleton=False, type='GeometryNodeTree')
def nodegroup_pinnae_level2_rotation(nw: NodeWrangler, z_axis_rotate, y_axis_rotate, x_axis_rotate):
    # Code generated using version 2.4.3 of the node_transpiler

    position_1 = nw.new_node(Nodes.InputPosition)
    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Geometry', None),
                                            ('NodeSocketFloat', 'Value1', 1.0),
                                            ('NodeSocketFloat', 'Value2', 0.5),
                                            ('NodeSocketFloat', 'Value3', 0.5)])
    add = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs['Value2'], 1: 0.0})
    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs['Value3'], 1: 0.0})
    map_range_2 = nw.new_node(Nodes.MapRange, input_kwargs={'Value': add, 'From Max': add_1})
    float_curve_1 = nw.new_node(Nodes.FloatCurve, input_kwargs={'Value': map_range_2.outputs["Result"]})
    node_utils.assign_curve(float_curve_1.mapping.curves[0],
                            [(0.0, z_axis_rotate[0]), (0.1, z_axis_rotate[1]), (0.25, z_axis_rotate[2]),
                             (0.45, z_axis_rotate[3]), (0.6, z_axis_rotate[4]), (1.0, z_axis_rotate[5])])
    add_2 = nw.new_node(Nodes.Math, input_kwargs={0: float_curve_1, 1: -0.25})

    # pinna z-axis curvature w.r.t the fern age
    map_range_7 = nw.new_node(Nodes.MapRange, input_kwargs={'Value': group_input.outputs['Value1'], 3: 1.2, 4: 0.0})

    multiply_1 = nw.new_node(Nodes.Math, input_kwargs={0: add_2, 1: map_range_7.outputs["Result"]},
                             attrs={'operation': 'MULTIPLY'})
    vector_rotate_1 = nw.new_node(Nodes.VectorRotate,
                                  input_kwargs={'Vector': position_1, 'Center': (0, 0, 0),
                                                'Angle': multiply_1}, attrs={'rotation_type': 'Z_AXIS'})
    map_range = nw.new_node(Nodes.MapRange, input_kwargs={'Value': add, 2: add_1})
    float_curve = nw.new_node(Nodes.FloatCurve, input_kwargs={'Value': map_range.outputs["Result"]})
    node_utils.assign_curve(float_curve.mapping.curves[0],
                            [(0.0, y_axis_rotate[0]), (0.1, y_axis_rotate[1]), (0.25, y_axis_rotate[2]),
                             (0.45, y_axis_rotate[3]), (0.6, y_axis_rotate[4]), (1.0, y_axis_rotate[5])])

    add_3 = nw.new_node(Nodes.Math, input_kwargs={0: float_curve, 1: -0.5})
    multiply_2 = nw.new_node(Nodes.Math, input_kwargs={0: add_3, 1: 1.0}, attrs={'operation': 'MULTIPLY'})
    vector_rotate = nw.new_node(Nodes.VectorRotate,
                                input_kwargs={'Vector': vector_rotate_1, 'Angle': multiply_2},
                                attrs={'rotation_type': 'Y_AXIS'})
    map_range_1 = nw.new_node(Nodes.MapRange, input_kwargs={'Value': add, 2: add_1})
    float_curve_2 = nw.new_node(Nodes.FloatCurve, input_kwargs={'Value': map_range_1.outputs["Result"]})
    node_utils.assign_curve(float_curve_2.mapping.curves[0],
                            [(0.0, x_axis_rotate[0]), (0.1, x_axis_rotate[1]), (0.25, x_axis_rotate[2]),
                             (0.45, x_axis_rotate[3]), (0.6, x_axis_rotate[4]), (1.0, x_axis_rotate[5])])
    multiply_3 = nw.new_node(Nodes.Math, input_kwargs={0: float_curve_2, 1: 1.0}, attrs={'operation': 'MULTIPLY'})
    vector_rotate_2 = nw.new_node(Nodes.VectorRotate, input_kwargs={'Vector': vector_rotate, 'Angle': multiply_3},
                                  attrs={'rotation_type': 'X_AXIS'})
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Vector': vector_rotate_2})


@node_utils.to_nodegroup('nodegroup_pinnae_level2_set_point', singleton=False, type='GeometryNodeTree')
def nodegroup_pinnae_level2_set_point(nw: NodeWrangler, pinna_contour):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'Value1', 1.0),
                                            ('NodeSocketFloat', 'From Max', 1.0),
                                            ('NodeSocketFloat', 'Value2', 1.0)])
    map_range_4 = nw.new_node(Nodes.MapRange,
                              input_kwargs={'Value': group_input.outputs["Value1"], 2: group_input.outputs["From Max"],
                                            3: 1.0, 4: 0.0})
    float_curve = nw.new_node(Nodes.FloatCurve, input_kwargs={'Value': map_range_4.outputs["Result"]})
    node_utils.assign_curve(float_curve.mapping.curves[0], [(0.0, pinna_contour[0]), (0.38, pinna_contour[1]),
                                                            (0.55, pinna_contour[2]), (0.75, pinna_contour[3]),
                                                            (0.9, pinna_contour[4]), (1.0, pinna_contour[5])])
    accumulate_field_2 = nw.new_node(Nodes.AccumulateField, input_kwargs={1: float_curve})

    # pinna scale w.r.t fern age
    map_range_6 = nw.new_node(Nodes.MapRange, input_kwargs={'Value': group_input.outputs['Value2'], 3: 0.5, 4: 2.0})
    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: accumulate_field_2.outputs[4], 1: map_range_6.outputs["Result"]},
                           attrs={'operation': 'MULTIPLY'})
    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': multiply})
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Vector': combine_xyz_2, 'Value': float_curve,
                                                                'Result': map_range_4.outputs["Result"]})


@node_utils.to_nodegroup('nodegroup_pinnae_level2_instance_on_points', singleton=False, type='GeometryNodeTree')
def nodegroup_pinnae_level2_instance_on_points(nw: NodeWrangler, leaf, pinna_contour):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Points', None),
                                            ('NodeSocketFloat', 'Value1', 1.0),
                                            ('NodeSocketFloat', 'Value2', 0.5),
                                            ('NodeSocketFloat', 'Value3', 1.0)])
    index = nw.new_node(Nodes.Index)
    object_info_2 = nw.new_node(Nodes.ObjectInfo, input_kwargs={'Object': leaf})
    transform = nw.new_node(Nodes.Transform,
                            input_kwargs={'Geometry': object_info_2.outputs["Geometry"], 'Scale': (1.2, -1.0, 1.0)})
    transform_2 = nw.new_node(Nodes.Transform,
                              input_kwargs={'Geometry': object_info_2.outputs["Geometry"], 'Scale': (1.2, 1.0, 1.0)})
    join_geometry = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [transform, transform_2]})
    add = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs['Value2'], 1: -0.3})
    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': 1.57, 'Z': add})
    float_curve_6 = nw.new_node(Nodes.FloatCurve, input_kwargs={'Value': group_input.outputs["Value1"]})
    node_utils.assign_curve(float_curve_6.mapping.curves[0], [(0.0, pinna_contour[0]), (0.38, pinna_contour[1]),
                                                              (0.55, pinna_contour[2]), (0.75, pinna_contour[3]),
                                                              (0.9, pinna_contour[4]), (1.0, pinna_contour[5])])
    # pinna leaf size w.r.t the fern age
    map_range = nw.new_node(Nodes.MapRange, input_kwargs={'Value': group_input.outputs['Value3'], 3: 6, 4: 8})
    multiply = nw.new_node(Nodes.VectorMath, input_kwargs={0: float_curve_6, 1: map_range.outputs["Result"]},
                           attrs={'operation': 'MULTIPLY'})
    instance_on_points_2 = nw.new_node(Nodes.InstanceOnPoints,
                                       input_kwargs={'Points': group_input.outputs["Points"], 'Selection': index,
                                                     'Instance': join_geometry, 'Rotation': combine_xyz_3,
                                                     'Scale': multiply.outputs["Vector"]})
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Instances': instance_on_points_2})


@node_utils.to_nodegroup('nodegroup_pinnae_level2_stein', singleton=False, type='GeometryNodeTree')
def nodegroup_pinnae_level2_stein(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'Value1', 0.5),
                                            ('NodeSocketFloat', 'Value2', 0.5),
                                            ('NodeSocketGeometry', 'Mesh', None)])
    mesh_to_curve_1 = nw.new_node(Nodes.MeshToCurve, input_kwargs={'Mesh': group_input.outputs["Mesh"]})

    multiply = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Value1"], 1: 0.1},
                           attrs={'operation': 'MULTIPLY'})
    set_curve_radius_1 = nw.new_node(Nodes.SetCurveRadius, input_kwargs={'Curve': mesh_to_curve_1, 'Radius': multiply})
    multiply_1 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs['Value2'], 1: 0.5},
                             attrs={'operation': 'MULTIPLY'})
    curve_circle_1 = nw.new_node(Nodes.CurveCircle, input_kwargs={'Radius': multiply_1, 'Resolution': 10})
    curve_to_mesh_1 = nw.new_node(Nodes.CurveToMesh,
                                  input_kwargs={'Curve': set_curve_radius_1,
                                                'Profile Curve': curve_circle_1.outputs["Curve"]})
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Mesh': curve_to_mesh_1})


@node_utils.to_nodegroup('nodegroup_pinnae', singleton=False, type='GeometryNodeTree')
def geometry_pinnae_nodes(nw: NodeWrangler, leaf, leaf_num_param=18, age_param=0.4, pinna_num_param=40,
                          version_num_param=4, gravity_rotation=1):
    # Code generated using version 2.4.3 of the node_transpiler

    # Define Input Node
    leaf_index = nw.new_node(Nodes.Index, label='LeafIndex')
    pinna_index = nw.new_node(Nodes.Index, label='PinnaIndex')
    pinna_num = nw.new_node(Nodes.Integer,  label='PinnaNum', attrs={'integer': 10})
    pinna_num.integer = pinna_num_param
    age = nw.new_node(Nodes.Value, label='Age')
    age.outputs[0].default_value = age_param

    mesh_lines_left, selections_left = [], []
    mesh_lines_right, selections_right = [], []

    # Generate Random Pinnae Contour, Two Modes: Linear+Noise, StepwiseLinear+Noise
    mode_random_bit = randint(0, 2, size=(1,))[0]
    if mode_random_bit:
        pinnae_contour = [0, 0.2, 0.6, 1.4, 3.0, 4.0, 5.0, 6.0]
        for i in range(8):
            pinnae_contour[i] = (pinnae_contour[i] + normal(0, 0.04 * i, (1,))[0]) / 6.
    else:
        pinnae_contour = [0, 0.2, 0.6, 1.4, 3.0, 4.0, 5.0, 4.2]
        for i in range(8):
            pinnae_contour[i] = (pinnae_contour[i] + normal(0, 0.04 * i, (1,))[0]) / 6.

    # Common Components
    pinnaelevel1instanceposition = nw.new_node(nodegroup_pinnae_level1_instance_position(pinnae_contour).name,
                                               input_kwargs={0: pinna_index, 'From Max': pinna_num, 2: age})
    left_noise, right_noise = nw.new_node(Nodes.WhiteNoiseTexture), nw.new_node(Nodes.WhiteNoiseTexture)
    pinnaelevel1scale = nw.new_node(nodegroup_pinnae_level1_scale(pinnae_contour).name,
                                    input_kwargs={0: pinnaelevel1instanceposition.outputs["Result"], 1: age})

    # Left & Right Instance Point Selections for each Version
    random_bit = randint(2, size=(1,))[0]
    for i in range(version_num_param):
        index = nw.new_node(Nodes.Index)
        greater_equal = nw.new_node(Nodes.Compare,
                                    input_kwargs={0: left_noise.outputs["Value"], 1: i / version_num_param},
                                    attrs={'operation': 'GREATER_EQUAL'})
        less_equal = nw.new_node(Nodes.Compare,
                                 input_kwargs={0: left_noise.outputs["Value"], 1: (i+1) / version_num_param},
                                 attrs={'operation': 'LESS_EQUAL'})
        op_and = nw.new_node(Nodes.BooleanMath, input_kwargs={0: greater_equal, 1: less_equal})

        greater_than = nw.new_node(Nodes.Math, input_kwargs={0: index, 1: 2.0},
                                   attrs={'operation': 'GREATER_THAN'})
        modulo = nw.new_node(Nodes.Math, input_kwargs={0: index, 1: 2.0},
                             attrs={'operation': 'MODULO'})
        if random_bit:
            modulo = nw.new_node(Nodes.Math, input_kwargs={0: 1, 1: modulo}, attrs={'operation': 'SUBTRACT'})
        op_and_1 = nw.new_node(Nodes.BooleanMath, input_kwargs={0: greater_than, 1: modulo})
        op_and_2 = nw.new_node(Nodes.BooleanMath, input_kwargs={0: op_and, 1: op_and_1})
        selections_left.append(op_and_2)

    random_bit = randint(2, size=(1,))[0]
    for i in range(version_num_param):
        greater_equal = nw.new_node(Nodes.Compare,
                                    input_kwargs={0: right_noise.outputs["Value"], 1: i / version_num_param},
                                    attrs={'operation': 'GREATER_EQUAL'})
        less_equal = nw.new_node(Nodes.Compare,
                                 input_kwargs={0: right_noise.outputs["Value"], 1: (i+1) / version_num_param},
                                 attrs={'operation': 'LESS_EQUAL'})
        op_and = nw.new_node(Nodes.BooleanMath, input_kwargs={0: greater_equal, 1: less_equal})
        index = nw.new_node(Nodes.Index)
        greater_than = nw.new_node(Nodes.Math, input_kwargs={0: index, 1: 2.0},
                                   attrs={'operation': 'GREATER_THAN'})
        modulo = nw.new_node(Nodes.Math, input_kwargs={0: index, 1: 2.0},
                             attrs={'operation': 'MODULO'})
        if random_bit:
            modulo = nw.new_node(Nodes.Math, input_kwargs={0: 1, 1: modulo}, attrs={'operation': 'SUBTRACT'})
        op_and_1 = nw.new_node(Nodes.BooleanMath, input_kwargs={0: greater_than, 1: modulo})
        op_and_2 = nw.new_node(Nodes.BooleanMath, input_kwargs={0: op_and, 1: op_and_1})
        selections_right.append(op_and_2)

    # Each Pinna Version
    rotation, pinnaelevel1rotation = True, None
    for i in range(version_num_param):
        # Define the Pinna Contour of each Version
        pinna_contour = []
        k = uniform(0.5, 0.58, size=(1,))[0]
        for j in range(6):
            pinna_contour.append(k * np.clip(j * (1. + normal(0, 0.1, (1,))[0]) / 5. + 0.08, 0, 0.7))
        # Define the Num Leaf of each Version
        integer_2 = nw.new_node(Nodes.Integer, attrs={'integer': 10})
        integer_2.integer = leaf_num_param + randint(-1, 2, (1,))[0]

        mesh_line_pinna = nw.new_node(Nodes.MeshLine, input_kwargs={'Count': pinna_num, 'Offset': (0.0, 0.0, 0.0)})
        set_position_pinna = nw.new_node(Nodes.SetPosition,
                                         input_kwargs={'Geometry': mesh_line_pinna,
                                                       'Position': pinnaelevel1instanceposition.outputs["Vector"]})
        if rotation:
            pinnaelevel1rotation = nw.new_node(nodegroup_pinnae_level1_rotation(gravity_rotation=gravity_rotation).name,
                                               input_kwargs={'Geometry': set_position_pinna, 1: age, 2: pinna_num})
            rotation = False
        pinnaelevel1instancerotation = nw.new_node(nodegroup_pinnae_level1_instance_rotation().name,
                                                   input_kwargs={0: pinnaelevel1rotation.outputs["Value"], 1: age})
        set_rotation_pinna = nw.new_node(Nodes.SetPosition,
                                         input_kwargs={'Geometry': set_position_pinna,
                                                       'Position': pinnaelevel1rotation.outputs["Vector"]})
        mesh_line_leaf = nw.new_node(Nodes.MeshLine, input_kwargs={'Count': integer_2, 'Offset': (0.0, 0.0, 0.0)})
        pinnaelevel2setpoint = nw.new_node(nodegroup_pinnae_level2_set_point(pinna_contour=pinna_contour).name,
                                           input_kwargs={0: leaf_index, 'From Max': integer_2, 2: age})
        set_position_leaf = nw.new_node(Nodes.SetPosition,
                                        input_kwargs={'Geometry': mesh_line_leaf,
                                                      'Position': pinnaelevel2setpoint.outputs["Vector"]})

        x_curvature, y_curvature, z_curvature = random_pinnae_level2_curvature()
        pinnaelevel2rotation = nw.new_node(nodegroup_pinnae_level2_rotation(z_axis_rotate=z_curvature,
                                                                            y_axis_rotate=y_curvature,
                                                                            x_axis_rotate=x_curvature).name,
                                           input_kwargs={'Geometry': set_position_leaf, 1: age,
                                                         2: leaf_index, 3: integer_2})
        set_rotation_leaf = nw.new_node(Nodes.SetPosition,
                                        input_kwargs={'Geometry': set_position_leaf, 'Position': pinnaelevel2rotation})
        pinna_on_pinnae = nw.new_node(Nodes.InstanceOnPoints,
                                      input_kwargs={'Points': set_rotation_pinna,
                                                    'Selection': selections_left[i],
                                                    'Instance': set_rotation_leaf,
                                                    'Rotation': pinnaelevel1instancerotation,
                                                    'Scale': pinnaelevel1scale})
        rotate_instances = nw.new_node(Nodes.RotateInstances,
                                       input_kwargs={'Instances': pinna_on_pinnae,
                                                     'Rotation': (-0.1571, 0.0, 0.0)})
        scale_instances = nw.new_node(Nodes.ScaleInstances,
                                      input_kwargs={'Instances': rotate_instances, 'Scale': (-1.0, 1.0, 1.0)})
        pinnaelevel2stein = nw.new_node(nodegroup_pinnae_level2_stein().name,
                                        input_kwargs={0: pinnaelevel2setpoint.outputs["Result"],
                                                      'Mesh': scale_instances})
        pinnaelevel2instanceonpoints = nw.new_node(
            nodegroup_pinnae_level2_instance_on_points(leaf=leaf, pinna_contour=pinna_contour).name,
            input_kwargs={'Points': scale_instances, 1: pinnaelevel2setpoint.outputs["Result"], 2: 0.0, 3: age})
        join_geometry = nw.new_node(Nodes.JoinGeometry,
                                    input_kwargs={'Geometry': [pinnaelevel2stein, pinnaelevel2instanceonpoints]})

        mesh_lines_left.append(join_geometry)
        if i == version_num_param - 1:
            pinnaelevel1stein = nw.new_node(nodegroup_pinnae_level1_stein().name,
                                            input_kwargs={'Mesh': set_rotation_pinna, 1: age,
                                                          2: pinnaelevel1instanceposition.outputs["Result"]})
            mesh_lines_left.append(pinnaelevel1stein)

    for i in range(version_num_param):
        # Define the Pinna Contour of each Version
        pinna_contour = []
        k = uniform(0.5, 0.58, size=(1,))[0]
        for j in range(6):
            pinna_contour.append(k * np.clip(j * (1. + normal(0, 0.1, (1,))[0]) / 5. + 0.08, 0, 0.7))
        # Define the Num Leaf of each Version
        integer_2 = nw.new_node(Nodes.Integer, attrs={'integer': 10})
        integer_2.integer = leaf_num_param + randint(-1, 2, (1,))[0]

        mesh_line_pinna = nw.new_node(Nodes.MeshLine, input_kwargs={'Count': pinna_num, 'Offset': (0.0, 0.0, 0.0)})
        set_position_pinna = nw.new_node(Nodes.SetPosition,
                                         input_kwargs={'Geometry': mesh_line_pinna,
                                                       'Position': pinnaelevel1instanceposition.outputs["Vector"]})
        pinnaelevel1instancerotation = nw.new_node(nodegroup_pinnae_level1_instance_rotation().name,
                                                   input_kwargs={0: pinnaelevel1rotation.outputs["Value"], 1: age})
        set_rotation_pinna = nw.new_node(Nodes.SetPosition,
                                         input_kwargs={'Geometry': set_position_pinna,
                                                       'Position': pinnaelevel1rotation.outputs["Vector"]})
        mesh_line_leaf = nw.new_node(Nodes.MeshLine,
                                     input_kwargs={'Count': integer_2, 'Offset': (0.0, 0.0, 0.0)})

        pinnaelevel2setpoint = nw.new_node(nodegroup_pinnae_level2_set_point(pinna_contour=pinna_contour).name,
                                           input_kwargs={0: leaf_index, 'From Max': integer_2, 2: age})

        set_position_leaf = nw.new_node(Nodes.SetPosition,
                                        input_kwargs={'Geometry': mesh_line_leaf,
                                                      'Position': pinnaelevel2setpoint.outputs["Vector"]})
        x_curvature, y_curvature, z_curvature = random_pinnae_level2_curvature()
        pinnaelevel2rotation = nw.new_node(nodegroup_pinnae_level2_rotation(z_axis_rotate=z_curvature,
                                                                            y_axis_rotate=y_curvature,
                                                                            x_axis_rotate=x_curvature).name,
                                           input_kwargs={'Geometry': set_position_leaf, 1: age, 2: leaf_index,
                                                         3: integer_2})
        set_rotation_leaf = nw.new_node(Nodes.SetPosition,
                                        input_kwargs={'Geometry': set_position_leaf, 'Position': pinnaelevel2rotation})
        pinna_on_pinnae = nw.new_node(Nodes.InstanceOnPoints,
                                      input_kwargs={'Points': set_rotation_pinna, 'Selection': selections_right[i],
                                                    'Instance': set_rotation_leaf, 'Scale': pinnaelevel1scale,
                                                    'Rotation': pinnaelevel1instancerotation})
        rotate_instances = nw.new_node(Nodes.RotateInstances, input_kwargs={'Instances': pinna_on_pinnae,
                                                                            'Rotation': (-0.1571, 0.0, 0.0)})
        scale_instances = nw.new_node(Nodes.ScaleInstances,
                                      input_kwargs={'Instances': rotate_instances, 'Scale': (1.0, 1.0, 1.0)})
        pinnaelevel2stein = nw.new_node(nodegroup_pinnae_level2_stein().name,
                                        input_kwargs={0: pinnaelevel2setpoint.outputs["Result"],
                                                      'Mesh': scale_instances})
        pinnaelevel2instanceonpoints = nw.new_node(
            nodegroup_pinnae_level2_instance_on_points(leaf=leaf, pinna_contour=pinna_contour).name,
            input_kwargs={'Points': scale_instances, 1: pinnaelevel2setpoint.outputs["Result"], 2: 0.0, 3: age})
        join_geometry = nw.new_node(Nodes.JoinGeometry,
                                    input_kwargs={'Geometry': [pinnaelevel2stein, pinnaelevel2instanceonpoints]})
        mesh_lines_right.append(join_geometry)

    join_geometry_whole = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': mesh_lines_left + mesh_lines_right})
    realize_instances = nw.new_node(Nodes.RealizeInstances, input_kwargs={'Geometry': join_geometry_whole})
    noise_texture = nw.new_node(Nodes.NoiseTexture, input_kwargs={'Scale': 0.4, 'Roughness': 0.2})
    set_positions = nw.new_node(Nodes.SetPosition, input_kwargs={'Geometry': realize_instances,
                                                                 'Offset': noise_texture.outputs["Color"]})
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': set_positions})


def check_vicinity(rotation, pinnae_rs):
    for r in pinnae_rs:
        if abs(rotation[1] - r[1]) < 0.1 and abs(rotation[2] - r[2]) < 0.15:
            return True
    return False


def geo_fern(nw: NodeWrangler, **kwargs):
    pinnaes = []
    # Two modes: Random Like and Flatten Like
    fern_mode = kwargs["fern_mode"]
    pinnae_num = kwargs["pinnae_num"]
    scale = kwargs["scale"]
    version_num = kwargs["version_num"]
    leaf = kwargs["leaf"]
    if fern_mode == "young_and_grownup":
        rotates = [] # Horizontal grownup pinnae
        # Generate non-overlapping pinnae orientations
        for i in range(pinnae_num):
            flip_bit = randint(0, 3, (1,))[0]
            if flip_bit:
                rotate_z = uniform(2.74, 3.54, (1,))[0]
            else:
                rotate_z = uniform(-0.4, 0.4, (1,))[0]
            rotate_x = uniform(0.8, 1.1, (1,))[0]
            rotate_z2 = uniform(0, 6.28, (1,))[0]
            if flip_bit:
                gravity_dir = 1
            else:
                gravity_dir = -1
            rotate = (rotate_z, rotate_x, rotate_z2, gravity_dir)
            if check_vicinity(rotate, rotates):
                continue
            else:
                rotates.append(rotate)
        # Generate pinnae
        for r in rotates:
            random_age = uniform(0.7, 0.95, (1,))[0]
            random_leaf_num = randint(15, 25, (1,))[0]
            random_pinna_num = randint(60, 80, (1,))[0]
            shape = nw.new_node(geometry_pinnae_nodes(leaf, leaf_num_param=random_leaf_num, age_param=random_age,
                                                      pinna_num_param=random_pinna_num,
                                                      version_num_param=version_num,
                                                      gravity_rotation=r[3]).name)
            z_transform = nw.new_node(Nodes.Transform,
                                      input_kwargs={'Geometry': shape, 'Rotation': (0., 0., r[0])})
            x_transform = nw.new_node(Nodes.Transform,
                                      input_kwargs={'Geometry': z_transform, 'Rotation': (-r[1], 0., 0.)})
            z2_transform = nw.new_node(Nodes.Transform,
                                       input_kwargs={'Geometry': x_transform, 'Rotation': (0., 0., r[2])})
            pinnaes.append(z2_transform)

        # Verticle young pinnae
        young_num = randint(0, 5, size=(1,))[0]
        for i in range(young_num):
            random_age = uniform(0.2, 0.5, (1,))[0]
            random_leaf_num = randint(14, 20, (1,))[0]
            random_pinna_num = randint(60, 100, (1,))[0]
            rotate_z = uniform(0, 6.28, (1,))
            rotate_x = uniform(0, 0.4, (1,))
            rotate_z2 = uniform(0, 6.28, (1,))
            shape = nw.new_node(geometry_pinnae_nodes(leaf, leaf_num_param=random_leaf_num, age_param=random_age,
                                                      pinna_num_param=random_pinna_num,
                                                      version_num_param=version_num, gravity_rotation=0).name)
            z_transform = nw.new_node(Nodes.Transform,
                                      input_kwargs={'Geometry': shape, 'Rotation': (0., 0., rotate_z[0])})
            x_transform = nw.new_node(Nodes.Transform,
                                      input_kwargs={'Geometry': z_transform, 'Rotation': (-rotate_x[0], 0., 0.)})
            z2_transform = nw.new_node(Nodes.Transform,
                                       input_kwargs={'Geometry': x_transform, 'Rotation': (0., 0., rotate_z2[0])})
            pinnaes.append(z2_transform)
    elif fern_mode == 'all_grownup':
        # Random grownup pinnae
        rotates = []
        for i in range(pinnae_num):
            rotate_z = normal(3.14, 0.2, (1,))[0]
            rotate_x = uniform(0.5, 1.1, (1,))[0]
            rotate_z2 = uniform(0, 6.28, (1,))[0]
            rotate = (rotate_z, rotate_x, rotate_z2, 1)
            if check_vicinity(rotate, rotates):
                continue
            else:
                rotates.append(rotate)

        for r in rotates:
            random_age = uniform(0.7, 0.9, (1,))[0]
            random_leaf_num = randint(16, 25, (1,))[0]
            random_pinna_num = randint(60, 80, (1,))[0]
            shape = nw.new_node(geometry_pinnae_nodes(leaf, leaf_num_param=random_leaf_num, age_param=random_age,
                                                      pinna_num_param=random_pinna_num,
                                                      version_num_param=version_num,
                                                      gravity_rotation=r[3]).name)
            z_transform = nw.new_node(Nodes.Transform,
                                      input_kwargs={'Geometry': shape, 'Rotation': (0., 0., r[0])})
            x_transform = nw.new_node(Nodes.Transform,
                                      input_kwargs={'Geometry': z_transform, 'Rotation': (-r[1], 0., 0.)})
            z2_transform = nw.new_node(Nodes.Transform,
                                       input_kwargs={'Geometry': x_transform, 'Rotation': (0., 0., r[2])})
            pinnaes.append(z2_transform)
    elif fern_mode == 'single_pinnae':
        shape = nw.new_node(geometry_pinnae_nodes(leaf,
                                                  leaf_num_param=20,
                                                  age_param=kwargs["age"],
                                                  pinna_num_param=60,
                                                  version_num_param=version_num).name)
        pinnaes.append(shape)
    else:
        raise NotImplementedError

    join_geometry = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': pinnaes})
    geometry = nw.new_node(Nodes.Transform, input_kwargs={'Geometry': join_geometry, 'Scale': (scale, scale, scale)})
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geometry})

@gin.register
class FernFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super(FernFactory, self).__init__(factory_seed, coarse=coarse)

    def create_asset(self, **params):
        bpy.ops.mesh.primitive_plane_add(
            size=1, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        obj = bpy.context.active_object

        if "fern_mode" not in params:

            type_bit = randint(0, 2, (1, ))[0]
            if type_bit:
                params["fern_mode"] = "young_and_grownup"
            else:
                params["fern_mode"] = "all_grownup"

        if "scale" not in params:
            params["scale"] = 0.02

        if "version_num" not in params:
            params["version_num"] = 5

        if "pinnae_num" not in params:
            params["pinnae_num"] = randint(12, 30, size=(1,))[0]

        # Make the Leaf and Delete It Later
        lf_seed = randint(0, 1000, size=(1,))[0]
        leaf_model = Leaf.LeafFactory(genome={"leaf_width": 0.4, "width_rand": 0.04}, factory_seed=lf_seed)
        leaf = leaf_model.create_asset(material=False)
        params["leaf"] = leaf

        surface.add_geomod(obj, geo_fern, apply=True, attributes=[], input_kwargs=params)
        butil.delete([leaf])
        with butil.SelectObjects(obj):
            bpy.ops.object.material_slot_remove()
            bpy.ops.object.shade_flat()

        simple_greenery.apply(obj)
            
        return obj

    def debug_asset(self, **params):
        bpy.ops.mesh.primitive_plane_add(
            size=1, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        obj = bpy.context.active_object
        params["fern_mode"] = "single_pinnae"
        params["scale"] = 1.0
        params["version_num"] = 5
        params["pinnae_num"] = 1
        params["age"] = uniform(0.5, 0.9)

        leaf_model = Leaf.LeafFactory(genome={"leaf_width": 0.4, "width_rand": 0.04}, factory_seed=0)
        leaf = leaf_model.create_asset(material=False)
        params["leaf"] = leaf
        surface.add_geomod(obj, geo_fern, apply=True, attributes=[], input_kwargs=params)

        bpy.ops.object.convert(target='MESH')
        butil.delete([leaf])
        tag_object(obj, 'fern')
        return obj



# if __name__ == '__main__':
#     fern = FernFactory(0)
#     obj = fern.debug_asset()
#     simple_greenery.apply([obj])