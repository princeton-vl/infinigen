# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Beining Han
# Acknowledgement: This file draws inspiration from https://www.youtube.com/watch?v=61Sk8j1Ml9c by BradleyAnimation

import bpy
import mathutils
from numpy.random import uniform, normal, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core import surface
from infinigen.assets.materials import simple_greenery
from infinigen.assets.materials import simple_whitish
from infinigen.assets.materials import simple_brownish
from infinigen.core.placement.factory import AssetFactory
import numpy as np
from infinigen.core.util import blender as butil
from infinigen.assets.utils.tag import tag_object, tag_nodegroup


@node_utils.to_nodegroup('nodegroup_pedal_stem_head_geometry', singleton=False, type='GeometryNodeTree')
def nodegroup_pedal_stem_head_geometry(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketVectorTranslation', 'Translation', (0.0, 0.0, 1.0)),
                                            ('NodeSocketFloatDistance', 'Radius', 0.04)])

    uv_sphere_1 = nw.new_node(Nodes.MeshUVSphere,
                              input_kwargs={'Segments': 64, 'Radius': group_input.outputs["Radius"]})

    transform_1 = nw.new_node(Nodes.Transform,
                              input_kwargs={'Geometry': uv_sphere_1, 'Translation': group_input.outputs["Translation"]})

    set_material = nw.new_node(Nodes.SetMaterial,
                               input_kwargs={'Geometry': transform_1,
                                             'Material': surface.shaderfunc_to_material(simple_brownish.shader_simple_brown)})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': set_material})


@node_utils.to_nodegroup('nodegroup_pedal_stem_end_geometry', singleton=False, type='GeometryNodeTree')
def nodegroup_pedal_stem_end_geometry(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Points', None)])

    endpoint_selection = nw.new_node('GeometryNodeCurveEndpointSelection',
                                     input_kwargs={'End Size': 0})

    uv_sphere = nw.new_node(Nodes.MeshUVSphere,
                            input_kwargs={'Segments': 64, 'Radius': 0.04})

    vector = nw.new_node(Nodes.Vector)
    vector.vector = (uniform(0.45, 0.7), uniform(0.45, 0.7), uniform(2, 3))

    transform = nw.new_node(Nodes.Transform,
                            input_kwargs={'Geometry': uv_sphere, 'Scale': vector})

    cone = nw.new_node('GeometryNodeMeshCone', input_kwargs={'Radius Bottom': 0.0040, 'Depth': 0.0040})

    normal = nw.new_node(Nodes.InputNormal)

    align_euler_to_vector_1 = nw.new_node(Nodes.AlignEulerToVector, input_kwargs={'Vector': normal},
                                          attrs={'axis': 'Z'})

    instance_on_points_1 = nw.new_node(Nodes.InstanceOnPoints,
                                       input_kwargs={'Points': transform, 'Instance': cone.outputs["Mesh"],
                                                     'Rotation': align_euler_to_vector_1})

    join_geometry = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [instance_on_points_1, transform]})

    set_material = nw.new_node(Nodes.SetMaterial,
                               input_kwargs={'Geometry': join_geometry,
                                             'Material': surface.shaderfunc_to_material(simple_brownish.shader_simple_brown)})

    geometry_to_instance = nw.new_node('GeometryNodeGeometryToInstance',
                                       input_kwargs={'Geometry': set_material})

    curve_tangent = nw.new_node(Nodes.CurveTangent)

    align_euler_to_vector = nw.new_node(Nodes.AlignEulerToVector,
                                        input_kwargs={'Vector': curve_tangent},
                                        attrs={'axis': 'Z'})

    instance_on_points = nw.new_node(Nodes.InstanceOnPoints,
                                     input_kwargs={'Points': group_input.outputs["Points"],
                                                   'Selection': endpoint_selection, 'Instance': geometry_to_instance,
                                                   'Rotation': align_euler_to_vector})

    realize_instances = nw.new_node(Nodes.RealizeInstances,
                                    input_kwargs={'Geometry': instance_on_points})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': realize_instances})


@node_utils.to_nodegroup('nodegroup_pedal_stem_branch_shape', singleton=False, type='GeometryNodeTree')
def nodegroup_pedal_stem_branch_shape(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    pedal_stem_branches_num = nw.new_node(Nodes.Integer, label='pedal_stem_branches_num')
    pedal_stem_branches_num.integer = 40

    group_input = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketFloatDistance', 'Radius', 0.0100)])

    curve_circle_1 = nw.new_node(Nodes.CurveCircle,
                                 input_kwargs={'Resolution': pedal_stem_branches_num,
                                               'Radius': group_input.outputs["Radius"]})

    pedal_stem_branch_length = nw.new_node(Nodes.Value, label='pedal_stem_branch_length')
    pedal_stem_branch_length.outputs[0].default_value = 0.5000

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': pedal_stem_branch_length})

    curve_line_1 = nw.new_node(Nodes.CurveLine, input_kwargs={'End': combine_xyz_1})

    resample_curve = nw.new_node(Nodes.ResampleCurve, input_kwargs={'Curve': curve_line_1, 'Count': 40})

    spline_parameter = nw.new_node(Nodes.SplineParameter)

    float_curve = nw.new_node(Nodes.FloatCurve, input_kwargs={'Value': spline_parameter.outputs["Factor"]})
    node_utils.assign_curve(float_curve.mapping.curves[0],
                            [(0.0000, 0.0000),
                             (0.2, 0.08 * np.random.normal(1., 0.15)),
                             (0.4, 0.22 * np.random.normal(1., 0.2)),
                             (0.6, 0.45 * np.random.normal(1., 0.2)),
                             (0.8, 0.7 * np.random.normal(1., 0.1)),
                             (1.0000, 1.0000)])

    multiply = nw.new_node(Nodes.Math, input_kwargs={0: float_curve, 1: uniform(0.15, 0.4)}, attrs={'operation': 'MULTIPLY'})

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Z': multiply})

    set_position = nw.new_node(Nodes.SetPosition, input_kwargs={'Geometry': resample_curve, 'Offset': combine_xyz})

    normal = nw.new_node(Nodes.InputNormal)

    align_euler_to_vector = nw.new_node(Nodes.AlignEulerToVector, input_kwargs={'Vector': normal})

    instance_on_points = nw.new_node(Nodes.InstanceOnPoints,
                                     input_kwargs={'Points': curve_circle_1.outputs["Curve"], 'Instance': set_position,
                                                   'Rotation': align_euler_to_vector})

    random_value_1 = nw.new_node(Nodes.RandomValue, input_kwargs={2: -0.2000, 3: 0.2000, 'Seed': 2})

    random_value_2 = nw.new_node(Nodes.RandomValue, input_kwargs={2: -0.2000, 3: 0.2000, 'Seed': 1})

    random_value = nw.new_node(Nodes.RandomValue, input_kwargs={2: -0.2000, 3: 0.2000})

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ,
                                input_kwargs={'X': random_value_1.outputs[1], 'Y': random_value_2.outputs[1],
                                              'Z': random_value.outputs[1]})

    rotate_instances = nw.new_node(Nodes.RotateInstances,
                                   input_kwargs={'Instances': instance_on_points, 'Rotation': combine_xyz_2})

    random_value_3 = nw.new_node(Nodes.RandomValue, input_kwargs={2: 0.8000})

    scale_instances = nw.new_node(Nodes.ScaleInstances,
                                  input_kwargs={'Instances': rotate_instances, 'Scale': random_value_3.outputs[1]})

    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Instances': scale_instances},
                               attrs={'is_active_output': True})


@node_utils.to_nodegroup('nodegroup_pedal_stem_branch_contour', singleton=False, type='GeometryNodeTree')
def nodegroup_pedal_stem_branch_contour(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Geometry', None)])

    realize_instances = nw.new_node(Nodes.RealizeInstances,
                                    input_kwargs={'Geometry': group_input.outputs["Geometry"]})

    pedal_stem_branch_rsample = nw.new_node(Nodes.Value,
                                            label='pedal_stem_branch_rsample')
    pedal_stem_branch_rsample.outputs[0].default_value = 10.0

    resample_curve = nw.new_node(Nodes.ResampleCurve,
                                 input_kwargs={'Curve': realize_instances, 'Count': pedal_stem_branch_rsample})

    index = nw.new_node(Nodes.Index)

    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
                                    input_kwargs={'Geometry': resample_curve, 5: index},
                                    attrs={'domain': 'CURVE', 'data_type': 'INT'})

    spline_parameter = nw.new_node(Nodes.SplineParameter)

    float_curve = nw.new_node(Nodes.FloatCurve,
                              input_kwargs={'Value': spline_parameter.outputs["Factor"]})

    # generate pedal branch contour
    dist = uniform(-0.05, -0.25)
    node_utils.assign_curve(float_curve.mapping.curves[0],
                            [(0.0, 0.0), (0.2, 0.2 + (dist + normal(0, 0.05)) / 2.),
                             (0.4, 0.4 + (dist + normal(0, 0.05))),
                             (0.6, 0.6 + (dist + normal(0, 0.05)) / 1.2),
                             (0.8, 0.8 + (dist + normal(0, 0.05)) / 2.4), (1.0, 0.95 + normal(0, 0.05))])

    random_value = nw.new_node(Nodes.RandomValue,
                               input_kwargs={2: 0.05, 3: 0.35, 'ID': capture_attribute.outputs[5]})

    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: float_curve, 1: random_value.outputs[1]},
                           attrs={'operation': 'MULTIPLY'})

    combine_xyz = nw.new_node(Nodes.CombineXYZ,
                              input_kwargs={'Z': multiply})

    set_position = nw.new_node(Nodes.SetPosition,
                               input_kwargs={'Geometry': capture_attribute.outputs["Geometry"], 'Offset': combine_xyz})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': set_position})


@node_utils.to_nodegroup('nodegroup_pedal_stem_branch_geometry', singleton=False, type='GeometryNodeTree')
def nodegroup_pedal_stem_branch_geometry(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Curve', None),
                                            ('NodeSocketVectorTranslation', 'Translation', (0.0, 0.0, 1.0))])

    set_curve_radius_1 = nw.new_node(Nodes.SetCurveRadius,
                                     input_kwargs={'Curve': group_input.outputs["Curve"], 'Radius': 1.0})

    curve_circle_2 = nw.new_node(Nodes.CurveCircle,
                                 input_kwargs={'Radius': uniform(0.001, 0.0025), 'Resolution': 4})

    curve_to_mesh_1 = nw.new_node(Nodes.CurveToMesh,
                                  input_kwargs={'Curve': set_curve_radius_1,
                                                'Profile Curve': curve_circle_2.outputs["Curve"], 'Fill Caps': True})

    transform_2 = nw.new_node(Nodes.Transform,
                              input_kwargs={'Geometry': curve_to_mesh_1,
                                            'Translation': group_input.outputs["Translation"]})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': transform_2})


@node_utils.to_nodegroup('nodegroup_pedal_stem_geometry', singleton=False, type='GeometryNodeTree')
def nodegroup_pedal_stem_geometry(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketVectorTranslation', 'End', (0.0, 0.0, 1.0)),
                                            ('NodeSocketVectorTranslation', 'Middle', (0.0, 0.0, 0.5)),
                                            ('NodeSocketFloatDistance', 'Radius', 0.05)])

    quadratic_bezier = nw.new_node(Nodes.QuadraticBezier,
                                   input_kwargs={'Start': (0.0, 0.0, 0.0), 'Middle': group_input.outputs["Middle"],
                                                 'End': group_input.outputs["End"]})

    set_curve_radius = nw.new_node(Nodes.SetCurveRadius,
                                   input_kwargs={'Curve': quadratic_bezier, 'Radius': group_input.outputs["Radius"]})

    curve_circle = nw.new_node(Nodes.CurveCircle,
                               input_kwargs={'Radius': 0.2, 'Resolution': 8})

    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
                                input_kwargs={'Curve': set_curve_radius, 'Profile Curve': curve_circle.outputs["Curve"],
                                              'Fill Caps': True})

    set_material_2 = nw.new_node(Nodes.SetMaterial,
                                 input_kwargs={'Geometry': curve_to_mesh,
                                               'Material': surface.shaderfunc_to_material(simple_whitish.shader_simple_white)})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': set_material_2, 'Curve': quadratic_bezier})


@node_utils.to_nodegroup('nodegroup_pedal_selection', singleton=False, type='GeometryNodeTree')
def nodegroup_pedal_selection(nw: NodeWrangler, params):
    # Code generated using version 2.4.3 of the node_transpiler

    random_value = nw.new_node(Nodes.RandomValue,
                               input_kwargs={5: 1})

    greater_than = nw.new_node(Nodes.Math,
                               input_kwargs={0: params["random_dropout"], 1: random_value.outputs[1]},
                               attrs={'operation': 'GREATER_THAN'})

    index_1 = nw.new_node(Nodes.Index)

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'num_segments', 0.5)])

    divide = nw.new_node(Nodes.Math,
                         input_kwargs={0: index_1, 1: group_input.outputs["num_segments"]},
                         attrs={'operation': 'DIVIDE'})

    less_than = nw.new_node(Nodes.Math,
                            input_kwargs={0: divide, 1: params["row_less_than"]},
                            attrs={'operation': 'LESS_THAN'})

    greater_than_1 = nw.new_node(Nodes.Math,
                                 input_kwargs={0: divide, 1: params["row_great_than"]},
                                 attrs={'operation': 'GREATER_THAN'})

    op_and = nw.new_node(Nodes.BooleanMath,
                         input_kwargs={0: less_than, 1: greater_than_1})

    modulo = nw.new_node(Nodes.Math,
                         input_kwargs={0: index_1, 1: group_input.outputs["num_segments"]},
                         attrs={'operation': 'MODULO'})

    less_than_1 = nw.new_node(Nodes.Math,
                              input_kwargs={0: modulo, 1: params["col_less_than"]},
                              attrs={'operation': 'LESS_THAN'})

    greater_than_2 = nw.new_node(Nodes.Math,
                                 input_kwargs={0: modulo, 1: params["col_great_than"]},
                                 attrs={'operation': 'GREATER_THAN'})

    op_and_1 = nw.new_node(Nodes.BooleanMath,
                           input_kwargs={0: less_than_1, 1: greater_than_2})

    nand = nw.new_node(Nodes.BooleanMath,
                       input_kwargs={0: op_and, 1: op_and_1},
                       attrs={'operation': 'NAND'})

    op_and_2 = nw.new_node(Nodes.BooleanMath,
                           input_kwargs={0: greater_than, 1: nand})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Boolean': op_and_2})


@node_utils.to_nodegroup('nodegroup_stem_geometry', singleton=False, type='GeometryNodeTree')
def nodegroup_stem_geometry(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Curve', None)])

    spline_parameter = nw.new_node(Nodes.SplineParameter)

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = uniform(0.2, 0.4)

    map_range = nw.new_node(Nodes.MapRange,
                            input_kwargs={'Value': spline_parameter.outputs["Factor"], 3: 0.4, 4: value})

    set_curve_radius_2 = nw.new_node(Nodes.SetCurveRadius,
                                     input_kwargs={'Curve': group_input.outputs["Curve"],
                                                   'Radius': map_range.outputs["Result"]})

    stem_radius = nw.new_node(Nodes.Value,
                              label='stem_radius')
    stem_radius.outputs[0].default_value = uniform(0.01, 0.024)

    curve_circle_3 = nw.new_node(Nodes.CurveCircle,
                                 input_kwargs={'Radius': stem_radius})

    curve_to_mesh_2 = nw.new_node(Nodes.CurveToMesh,
                                  input_kwargs={'Curve': set_curve_radius_2,
                                                'Profile Curve': curve_circle_3.outputs["Curve"], 'Fill Caps': True})

    set_material = nw.new_node(Nodes.SetMaterial,
                               input_kwargs={'Geometry': curve_to_mesh_2,
                                             'Material': surface.shaderfunc_to_material(simple_greenery.shader_simple_greenery)})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Mesh': tag_nodegroup(nw, set_material, 'stem')})


@node_utils.to_nodegroup('nodegroup_pedal_stem', singleton=False, type='GeometryNodeTree')
def nodegroup_pedal_stem(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    pedal_stem_top_point = nw.new_node(Nodes.Vector,
                                       label='pedal_stem_top_point')
    pedal_stem_top_point.vector = (0.0, 0.0, 1.0)

    pedal_stem_mid_point = nw.new_node(Nodes.Vector,
                                       label='pedal_stem_mid_point')
    pedal_stem_mid_point.vector = (normal(0, 0.05), normal(0, 0.05), 0.5)

    pedal_stem_radius = nw.new_node(Nodes.Value,
                                    label='pedal_stem_radius')
    pedal_stem_radius.outputs[0].default_value = uniform(0.02, 0.045)

    pedal_stem_geometry = nw.new_node(nodegroup_pedal_stem_geometry().name,
                                      input_kwargs={'End': pedal_stem_top_point, 'Middle': pedal_stem_mid_point,
                                                    'Radius': pedal_stem_radius})

    pedal_stem_top_radius = nw.new_node(Nodes.Value,
                                        label='pedal_stem_top_radius')
    pedal_stem_top_radius.outputs[0].default_value = uniform(0.005, 0.008)

    pedal_stem_branch_shape = nw.new_node(nodegroup_pedal_stem_branch_shape().name,
                                          input_kwargs={'Radius': pedal_stem_top_radius})

    pedal_stem_branch_geometry = nw.new_node(nodegroup_pedal_stem_branch_geometry().name,
                                             input_kwargs={'Curve': pedal_stem_branch_shape,
                                                           'Translation': pedal_stem_top_point})

    set_material_3 = nw.new_node(Nodes.SetMaterial,
                                 input_kwargs={'Geometry': pedal_stem_branch_geometry,
                                               'Material': surface.shaderfunc_to_material(simple_whitish.shader_simple_white)})

    resample_curve = nw.new_node(Nodes.ResampleCurve,
                                 input_kwargs={'Curve': pedal_stem_geometry.outputs["Curve"]})

    pedal_stem_end_geometry = nw.new_node(nodegroup_pedal_stem_end_geometry().name,
                                          input_kwargs={'Points': resample_curve})

    pedal_stem_head_geometry = nw.new_node(nodegroup_pedal_stem_head_geometry().name,
                                           input_kwargs={'Translation': pedal_stem_top_point,
                                                         'Radius': pedal_stem_top_radius})

    join_geometry = nw.new_node(Nodes.JoinGeometry,
                                input_kwargs={'Geometry': [pedal_stem_geometry.outputs["Geometry"], set_material_3,
                                                           pedal_stem_end_geometry, pedal_stem_head_geometry]})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': join_geometry})


@node_utils.to_nodegroup('nodegroup_flower_geometry', singleton=False, type='GeometryNodeTree')
def nodegroup_flower_geometry(nw: NodeWrangler, params):
    # Code generated using version 2.4.3 of the node_transpiler

    num_core_segments = nw.new_node(Nodes.Integer,
                                    label='num_core_segments',
                                    attrs={'integer': 10})
    num_core_segments.integer = randint(8, 25)

    num_core_rings = nw.new_node(Nodes.Integer,
                                 label='num_core_rings',
                                 attrs={'integer': 10})
    num_core_rings.integer = randint(8, 20)

    uv_sphere_2 = nw.new_node(Nodes.MeshUVSphere,
                              input_kwargs={'Segments': num_core_segments, 'Rings': num_core_rings,
                                            'Radius': uniform(0.02, 0.05)})

    flower_core_shape = nw.new_node(Nodes.Vector,
                                    label='flower_core_shape')
    flower_core_shape.vector = (uniform(0.8, 1.2), uniform(0.8, 1.2), uniform(0.5, 0.8))

    transform = nw.new_node(Nodes.Transform,
                            input_kwargs={'Geometry': uv_sphere_2, 'Scale': flower_core_shape})

    selection_params = {
        "random_dropout": params["random_dropout"],
        "row_less_than": int(params["row_less_than"] * num_core_rings.integer),
        "row_great_than": int(params["row_great_than"] * num_core_rings.integer),
        "col_less_than": int(params["col_less_than"] * num_core_segments.integer),
        "col_great_than": int(params["col_less_than"] * num_core_segments.integer)
    }
    pedal_selection = nw.new_node(nodegroup_pedal_selection(params=selection_params).name,
                                  input_kwargs={'num_segments': num_core_segments})

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Instance', None)])

    normal_1 = nw.new_node(Nodes.InputNormal)

    align_euler_to_vector_1 = nw.new_node(Nodes.AlignEulerToVector,
                                          input_kwargs={'Vector': normal_1},
                                          attrs={'axis': 'Z'})

    random_value_1 = nw.new_node(Nodes.RandomValue,
                                 input_kwargs={2: 0.4, 3: 0.7})

    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: random_value_1.outputs[1]},
                           attrs={'operation': 'MULTIPLY'})

    instance_on_points_1 = nw.new_node(Nodes.InstanceOnPoints,
                                       input_kwargs={'Points': transform, 'Selection': pedal_selection,
                                                     'Instance': group_input.outputs["Instance"],
                                                     'Rotation': align_euler_to_vector_1, 'Scale': multiply})

    realize_instances_1 = nw.new_node(Nodes.RealizeInstances,
        input_kwargs={'Geometry': instance_on_points_1})   

    set_material = nw.new_node(Nodes.SetMaterial,
                               input_kwargs={'Geometry': transform,
                                             'Material': surface.shaderfunc_to_material(simple_whitish.shader_simple_white)})

    join_geometry_1 = nw.new_node(Nodes.JoinGeometry,
                                  input_kwargs={'Geometry': [realize_instances_1, set_material]})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': tag_nodegroup(nw, join_geometry_1, 'flower')})


@node_utils.to_nodegroup('nodegroup_flower_on_stem', singleton=False, type='GeometryNodeTree')
def nodegroup_flower_on_stem(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Points', None),
                                            ('NodeSocketGeometry', 'Instance', None)])

    endpoint_selection = nw.new_node('GeometryNodeCurveEndpointSelection',
                                     input_kwargs={'Start Size': 0})

    curve_tangent = nw.new_node(Nodes.CurveTangent)

    align_euler_to_vector_2 = nw.new_node(Nodes.AlignEulerToVector,
                                          input_kwargs={'Vector': curve_tangent},
                                          attrs={'axis': 'Z'})

    instance_on_points_2 = nw.new_node(Nodes.InstanceOnPoints,
                                       input_kwargs={'Points': group_input.outputs["Points"],
                                                     'Selection': endpoint_selection,
                                                     'Instance': group_input.outputs["Instance"],
                                                     'Rotation': align_euler_to_vector_2})

    realize_instances_2 = nw.new_node(Nodes.RealizeInstances,
        input_kwargs={'Geometry': instance_on_points_2})
    

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Instances': realize_instances_2})


def geometry_dandelion_nodes(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    quadratic_bezier_1 = nw.new_node(Nodes.QuadraticBezier,
                                     input_kwargs={'Start': (0.0, 0.0, 0.0),
                                                   'Middle': (normal(0, 0.1), normal(0, 0.1), 0.5),
                                                   'End': (normal(0, 0.1), normal(0, 0.1), 1.0)})

    resample_curve = nw.new_node(Nodes.ResampleCurve,
                                 input_kwargs={'Curve': quadratic_bezier_1})

    pedal_stem = nw.new_node(nodegroup_pedal_stem().name)

    geometry_to_instance = nw.new_node('GeometryNodeGeometryToInstance',
                                       input_kwargs={'Geometry': pedal_stem})

    flower_geometry = nw.new_node(nodegroup_flower_geometry(kwargs).name,
                                  input_kwargs={'Instance': geometry_to_instance})

    geometry_to_instance_1 = nw.new_node('GeometryNodeGeometryToInstance',
                                         input_kwargs={'Geometry': flower_geometry})

    value_2 = nw.new_node(Nodes.Value)
    value_2.outputs[0].default_value = uniform(-0.15, -0.5)

    transform_3 = nw.new_node(Nodes.Transform,
                              input_kwargs={'Geometry': geometry_to_instance_1, 'Scale': value_2})

    flower_on_stem = nw.new_node(nodegroup_flower_on_stem().name,
                                 input_kwargs={'Points': resample_curve, 'Instance': transform_3})

    stem_geometry = nw.new_node(nodegroup_stem_geometry().name,
                                input_kwargs={'Curve': quadratic_bezier_1})

    join_geometry_2 = nw.new_node(Nodes.JoinGeometry,
                                  input_kwargs={'Geometry': [flower_on_stem, stem_geometry]})

    realize_instances = nw.new_node(Nodes.RealizeInstances,
                                    input_kwargs={'Geometry': join_geometry_2})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': realize_instances})


def geometry_dandelion_seed_nodes(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    pedal_stem = nw.new_node(nodegroup_pedal_stem().name)

    geometry_to_instance = nw.new_node('GeometryNodeGeometryToInstance',
                                       input_kwargs={'Geometry': pedal_stem})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': geometry_to_instance})


class DandelionFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super(DandelionFactory, self).__init__(factory_seed, coarse=coarse)
        self.flower_mode = ['full_flower', 'no_flower', 'top_half_flower', 'top_missing_flower', 'sparse_flower']
        self.flower_mode_pb = [0.4, 0.04, 0.23, 0.13, 0.2]

    def get_mode_params(self, mode):
        if mode == 'full_flower':
            # generate a flower with full seeds
            return {
                "random_dropout": uniform(0.5, 1.0),
                "row_less_than": 0.0,
                "row_great_than": 0.0,
                "col_less_than": 0.0,
                "col_great_than": 0.0
            }
        elif mode == 'no_flower':
            # generate a flower with no seeds
            return {
                "random_dropout": 0.0,
                "row_less_than": 1.0,
                "row_great_than": 0.0,
                "col_less_than": 1.0,
                "col_great_than": 0.0
            }
        elif mode == 'top_half_flower':
            # generate a flower with no seeds at bottom half
            return {
                "random_dropout": uniform(0.6, 1.0),
                "row_less_than": uniform(0.3, 0.5),
                "row_great_than": 0.0,
                "col_less_than": 1.0,
                "col_great_than": 0.0
            }
        elif mode == 'top_missing_flower':
            # generate a flower with no seeds at bottom half
            col = uniform(0.3, 1.0)
            return {
                "random_dropout": uniform(0.5, 0.9),
                "row_less_than": 1.0,
                "row_great_than": uniform(0.5, 0.7),
                "col_less_than": col,
                "col_great_than": col - uniform(0.2, 0.4)
            }
        elif mode == 'sparse_flower':
            # generate a flower with no seeds at bottom half
            return {
                "random_dropout": uniform(0.3, 0.5),
                "row_less_than": 0.0,
                "row_great_than": 0.0,
                "col_less_than": 0.0,
                "col_great_than": 0.0
            }
        else:
            raise NotImplementedError

    def create_asset(self, **params):
        bpy.ops.mesh.primitive_plane_add(
            size=1, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        obj = bpy.context.active_object

        mode = np.random.choice(self.flower_mode, p=self.flower_mode_pb)
        params = self.get_mode_params(mode)

        surface.add_geomod(obj, geometry_dandelion_nodes, apply=True, attributes=[], input_kwargs=params)
        tag_object(obj, 'dandelion')
        return obj


class DandelionSeedFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super(DandelionSeedFactory, self).__init__(factory_seed, coarse=coarse)

    def create_asset(self, **params):
        bpy.ops.mesh.primitive_plane_add(
            size=1, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        obj = bpy.context.active_object

        surface.add_geomod(obj, geometry_dandelion_seed_nodes, apply=True, attributes=[], input_kwargs=params)
        tag_object(obj, 'seed')
        return obj


if __name__ == '__main__':
    f = DandelionSeedFactory(0)
    obj = f.create_asset()