# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Beining Han

import bpy

import numpy as np
from numpy.random import uniform, normal, randint

from infinigen.core.nodes import Nodes, NodeWrangler, node_utils
from infinigen.core.util.color import hsv2rgba
from infinigen.core import surface
from infinigen.core.placement.factory import AssetFactory

from infinigen.assets.tropic_plants.tropic_plant_utils import (
    nodegroup_nodegroup_leaf_shader, 
    nodegroup_nodegroup_sub_vein,
    nodegroup_nodegroup_leaf_gen,
    nodegroup_nodegroup_move_to_origin,
    nodegroup_nodegroup_leaf_rotate_x, 
    shader_stem_material
)
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

@node_utils.to_nodegroup('nodegroup_nodegroup_apply_wave', singleton=False, type='GeometryNodeTree')
def nodegroup_nodegroup_apply_wave(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Geometry', None),
                                            ('NodeSocketFloat', 'Wave Scale Y', 1.0),
                                            ('NodeSocketFloat', 'Wave Scale X', 1.0),
                                            ('NodeSocketFloat', 'X Modulated', 0.0),
                                            ('NodeSocketFloat', 'Width Scale', 0.0)])

    position = nw.new_node(Nodes.InputPosition)

    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
                               input_kwargs={'Vector': position})

    position_1 = nw.new_node(Nodes.InputPosition)

    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ,
                                 input_kwargs={'Vector': position_1})

    attribute_statistic = nw.new_node(Nodes.AttributeStatistic,
                                      input_kwargs={'Geometry': group_input.outputs["Geometry"],
                                                    2: separate_xyz_1.outputs["Y"]})

    map_range = nw.new_node(Nodes.MapRange,
                            input_kwargs={'Value': separate_xyz.outputs["Y"], 1: attribute_statistic.outputs["Min"],
                                          2: attribute_statistic.outputs["Max"]})

    float_curve = nw.new_node(Nodes.FloatCurve,
                              input_kwargs={'Value': map_range.outputs["Result"]})
    node_utils.assign_curve(float_curve.mapping.curves[0],
                            [(0.0, 0.4875), (0.1091, 0.5), (0.3275, 0.4921), (0.7409, 0.5031), (1.0, 0.5063)])

    map_range_1 = nw.new_node(Nodes.MapRange,
                              input_kwargs={'Value': float_curve, 3: -1.0})

    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: map_range_1.outputs["Result"], 1: group_input.outputs["Wave Scale Y"]},
                           attrs={'operation': 'MULTIPLY'})

    combine_xyz = nw.new_node(Nodes.CombineXYZ,
                              input_kwargs={'Z': multiply})

    set_position = nw.new_node(Nodes.SetPosition,
                               input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Offset': combine_xyz})

    attribute_statistic_1 = nw.new_node(Nodes.AttributeStatistic,
                                        input_kwargs={'Geometry': group_input.outputs["Geometry"],
                                                      2: group_input.outputs["X Modulated"]})

    map_range_2 = nw.new_node(Nodes.MapRange,
                              input_kwargs={'Value': group_input.outputs["X Modulated"],
                                            1: attribute_statistic_1.outputs["Min"],
                                            2: attribute_statistic_1.outputs["Max"]})

    float_curve_1 = nw.new_node(Nodes.FloatCurve,
                                input_kwargs={'Value': map_range_2.outputs["Result"]})
    node_utils.assign_curve(float_curve_1.mapping.curves[0],
                            [(0.0, 0.1625), (0.0955, 0.2844), (0.2318, 0.3594), (0.3727, 0.451), (0.5045, 0.5094),
                             (0.6045, 0.4447), (0.7886, 0.325), (1.0, 0.1594)],
                            handles=['AUTO', 'AUTO', 'AUTO', 'AUTO', 'VECTOR', 'AUTO', 'AUTO', 'AUTO'])

    map_range_3 = nw.new_node(Nodes.MapRange,
                              input_kwargs={'Value': float_curve_1, 3: -1.0})

    multiply_1 = nw.new_node(Nodes.Math,
                             input_kwargs={0: map_range_3.outputs["Result"], 1: group_input.outputs["Wave Scale X"]},
                             attrs={'operation': 'MULTIPLY'})

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
                                input_kwargs={'Z': multiply_1})

    set_position_1 = nw.new_node(Nodes.SetPosition,
                                 input_kwargs={'Geometry': set_position, 'Offset': combine_xyz_1})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': set_position_1})


@node_utils.to_nodegroup('nodegroup_leaf_on_stem_selection', singleton=False, type='GeometryNodeTree')
def nodegroup_leaf_on_stem_selection(nw: NodeWrangler, gt, lt, th):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'Samples', 0.0),
                                            ('NodeSocketFloat', 'Random Value', 0.0)])

    greater_than = nw.new_node(Nodes.Math,
                               input_kwargs={0: group_input.outputs["Random Value"], 1: gt},
                               attrs={'operation': 'GREATER_THAN'})

    less_than = nw.new_node(Nodes.Math,
                            input_kwargs={0: group_input.outputs["Random Value"], 1: lt},
                            attrs={'operation': 'LESS_THAN'})

    op_and = nw.new_node(Nodes.BooleanMath,
                         input_kwargs={0: greater_than, 1: less_than})

    index = nw.new_node(Nodes.Index)

    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: group_input.outputs["Samples"], 1: th * uniform(0.95, 1.05)},
                           attrs={'operation': 'MULTIPLY'})

    less_than_1 = nw.new_node(Nodes.Math,
                              input_kwargs={0: index, 1: multiply},
                              attrs={'operation': 'LESS_THAN'})

    op_and_1 = nw.new_node(Nodes.BooleanMath,
                           input_kwargs={0: op_and, 1: less_than_1})

    op_not = nw.new_node(Nodes.BooleanMath,
                         input_kwargs={0: op_and_1},
                         attrs={'operation': 'NOT'})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Boolean': op_not})


@node_utils.to_nodegroup('nodegroup_leaf_on_stem_scale_up_down', singleton=False, type='GeometryNodeTree')
def nodegroup_leaf_on_stem_scale_up_down(nw: NodeWrangler, gap):
    # Code generated using version 2.4.3 of the node_transpiler

    index_2 = nw.new_node(Nodes.Index)

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'Samples', 0.0)])

    map_range_1 = nw.new_node(Nodes.MapRange,
                              input_kwargs={'Value': index_2, 2: group_input.outputs["Samples"]},
                              attrs={'clamp': False})

    float_curve_1 = nw.new_node(Nodes.FloatCurve,
                                input_kwargs={'Value': map_range_1.outputs["Result"]})
    node_utils.assign_curve(float_curve_1.mapping.curves[0],
                            [(0.0, 1.0 - gap), (0.3, 1.0 - gap / 2.), (0.6, 1.0 - gap / 5.), (1.0, 1.0)])

    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: float_curve_1},
                           attrs={'operation': 'MULTIPLY'})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Value': multiply})


@node_utils.to_nodegroup('nodegroup_leaf_on_stem_rotation_up_down', singleton=False, type='GeometryNodeTree')
def nodegroup_leaf_on_stem_rotation_up_down(nw: NodeWrangler, scale, gap):
    # Code generated using version 2.4.3 of the node_transpiler

    index_1 = nw.new_node(Nodes.Index)

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketInt', 'Samples', 0)])

    map_range = nw.new_node(Nodes.MapRange,
                            input_kwargs={'Value': index_1, 2: group_input.outputs["Samples"], 3: 1.0, 4: 0.0},
                            attrs={'clamp': False})

    float_curve = nw.new_node(Nodes.FloatCurve,
                              input_kwargs={'Value': map_range.outputs["Result"]})
    node_utils.assign_curve(float_curve.mapping.curves[0], [(0.0, 1.0 - gap), (0.7, 1.0 - gap / 2.), (1.0, 1.0)])

    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: float_curve, 1: scale},
                           attrs={'operation': 'MULTIPLY'})

    combine_xyz = nw.new_node(Nodes.CombineXYZ,
                              input_kwargs={'Z': multiply})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Vector': combine_xyz})


@node_utils.to_nodegroup('nodegroup_leaf_on_stem_rotation_in_out', singleton=False, type='GeometryNodeTree')
def nodegroup_leaf_on_stem_rotation_in_out(nw: NodeWrangler, in_out_scale=1.0):
    # Code generated using version 2.4.3 of the node_transpiler

    index_1 = nw.new_node(Nodes.Index)

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketInt', 'Samples', 0)])

    map_range = nw.new_node(Nodes.MapRange,
                            input_kwargs={'Value': index_1, 2: group_input.outputs["Samples"], 3: 1.0, 4: 0.0},
                            attrs={'clamp': False})

    float_curve = nw.new_node(Nodes.FloatCurve,
                              input_kwargs={'Value': map_range.outputs["Result"]})
    node_utils.assign_curve(float_curve.mapping.curves[0], [(0.0, 0.0), (0.5136, 0.2188), (1.0, 0.8813)])

    add = nw.new_node(Nodes.Math,
                      input_kwargs={0: float_curve, 1: -0.5})

    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: add, 1: in_out_scale},
                           attrs={'operation': 'MULTIPLY'})

    combine_xyz = nw.new_node(Nodes.CombineXYZ,
                              input_kwargs={'X': multiply})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Vector': combine_xyz})


@node_utils.to_nodegroup('nodegroup_round_tropical_leaf', singleton=False, type='GeometryNodeTree')
def nodegroup_palm_leaf_instance(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'To Max', -0.4),
                                            ('NodeSocketGeometry', 'Mesh', None),
                                            ('NodeSocketFloat', 'Wave Scale Y', 0.3),
                                            ('NodeSocketFloat', 'Wave Scale X', 0.5),
                                            ('NodeSocketFloat', 'Leaf Width Scale', 0.0)])

    subdivide_mesh = nw.new_node(Nodes.SubdivideMesh,
                                 input_kwargs={'Mesh': group_input.outputs["Mesh"], 'Level': 8})

    subdivide_mesh_1 = nw.new_node(Nodes.SubdivideMesh,
                                   input_kwargs={'Mesh': subdivide_mesh})

    position = nw.new_node(Nodes.InputPosition)

    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
                                    input_kwargs={'Geometry': subdivide_mesh_1, 1: position},
                                    attrs={'data_type': 'FLOAT_VECTOR'})

    nodegroup_leaf_gen = nw.new_node(nodegroup_nodegroup_leaf_gen().name,
                                     input_kwargs={'Mesh': capture_attribute.outputs["Geometry"],
                                                   'Displancement scale': 0.0, 'Vein Asymmetry': uniform(0.2, 0.4),
                                                   'Vein Density': 0.0, 'Jigsaw Scale': 10.0, 'Jigsaw Depth': 0.0,
                                                   'Vein Angle': 0.3, 'Wave Displacement': 0.0, 'Midrib Length': 0.3336,
                                                   'Midrib Width': uniform(0.9, 1.5), 'Stem Length': uniform(0.55, 0.65),
                                                   'Leaf Width Scale': group_input.outputs["Leaf Width Scale"]})

    nodegroup_sub_vein = nw.new_node(nodegroup_nodegroup_sub_vein().name,
                                     input_kwargs={'X': nodegroup_leaf_gen.outputs["X Modulated"]})

    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: nodegroup_sub_vein.outputs["Value"], 1: 0.0005},
                           attrs={'operation': 'MULTIPLY'})

    combine_xyz = nw.new_node(Nodes.CombineXYZ,
                              input_kwargs={'Z': multiply})

    set_position = nw.new_node(Nodes.SetPosition,
                               input_kwargs={'Geometry': nodegroup_leaf_gen.outputs["Mesh"], 'Offset': combine_xyz})

    capture_attribute_1 = nw.new_node(Nodes.CaptureAttribute,
                                      input_kwargs={'Geometry': set_position,
                                                    2: nodegroup_sub_vein.outputs["Color Value"]})

    capture_attribute_2 = nw.new_node(Nodes.CaptureAttribute,
                                      input_kwargs={'Geometry': capture_attribute_1.outputs["Geometry"],
                                                    2: nodegroup_leaf_gen.outputs["Vein Value"]})

    nodegroup_apply_wave = nw.new_node(nodegroup_nodegroup_apply_wave().name,
                                       input_kwargs={'Geometry': capture_attribute_2.outputs["Geometry"],
                                                     'Wave Scale Y': group_input.outputs["Wave Scale Y"],
                                                     'Wave Scale X': group_input.outputs["Wave Scale X"],
                                                     'X Modulated': nodegroup_leaf_gen.outputs["X Modulated"],
                                                     'Width Scale': group_input.outputs["Leaf Width Scale"]})

    nodegroup_move_to_origin = nw.new_node(nodegroup_nodegroup_move_to_origin().name,
                                           input_kwargs={'Geometry': nodegroup_apply_wave})

    nodegroup_leaf_rotate_x = nw.new_node(nodegroup_nodegroup_leaf_rotate_x().name,
                                          input_kwargs={'Geometry': nodegroup_move_to_origin,
                                                        'To Max': group_input.outputs["To Max"]})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Attribute': nodegroup_leaf_gen.outputs["Attribute"],
                                             'Coordinate': capture_attribute.outputs["Attribute"],
                                             'subvein': capture_attribute_1.outputs[2],
                                             'vein': capture_attribute_2.outputs[2],
                                             'Geometry': nodegroup_leaf_rotate_x})


@node_utils.to_nodegroup('nodegroup_leaf_on_stem', singleton=False, type='GeometryNodeTree')
def nodegroup_leaf_on_stem(nw: NodeWrangler, versions):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Points', None),
                                            ('NodeSocketGeometry', 'Instance', None),
                                            ('NodeSocketVectorXYZ', 'Scale', (1.0, 1.0, 1.0)),
                                            ('NodeSocketInt', 'Samples', 0)])

    rotation_scale, rotation_gap = uniform(0.6, 1.2), uniform(0.2, 0.6)
    scale_gap = uniform(0.2, 0.5)
    in_out_scale = normal(0., 0.7)
    leaves = []
    for L in [-1, 1]:
        curve_tangent_1 = nw.new_node(Nodes.CurveTangent)

        align_euler_to_vector_1 = nw.new_node(Nodes.AlignEulerToVector,
                                              input_kwargs={'Vector': curve_tangent_1},
                                              attrs={'pivot_axis': 'Y'})

        instance_on_points_2 = nw.new_node(Nodes.InstanceOnPoints,
                                           input_kwargs={'Points': group_input.outputs["Points"],
                                                         'Instance': group_input.outputs["Instance"],
                                                         'Rotation': align_euler_to_vector_1})

        scale_instances_4 = nw.new_node(Nodes.ScaleInstances,
                                        input_kwargs={'Instances': instance_on_points_2, 'Scale': (1.0, L, 1.0)})

        index_1 = nw.new_node(Nodes.Index)

        random_value_4 = nw.new_node(Nodes.RandomValue,
                                     input_kwargs={'ID': index_1, 'Seed': L + 1})

        leaf_on_stem_selection_1 = nw.new_node(nodegroup_leaf_on_stem_selection(0, 0, 0).name,
                                               input_kwargs={'Samples': group_input.outputs["Samples"],
                                                             'Random Value': random_value_4.outputs[1]})

        value_1 = nw.new_node(Nodes.Value)
        value_1.outputs[0].default_value = 1.0

        scale_instances_3 = nw.new_node(Nodes.ScaleInstances,
                                        input_kwargs={'Instances': scale_instances_4, 'Selection': leaf_on_stem_selection_1,
                                                      'Scale': value_1})

        join_geometry_2 = nw.new_node(Nodes.JoinGeometry,
                                      input_kwargs={'Geometry': scale_instances_3})

        leaf_on_stem_rotation_up_down = nw.new_node(nodegroup_leaf_on_stem_rotation_up_down(rotation_scale * L, rotation_gap).name,
                                                        input_kwargs={'Samples': group_input.outputs["Samples"]})

        rotate_instances_6 = nw.new_node(Nodes.RotateInstances,
                                         input_kwargs={'Instances': join_geometry_2,
                                                       'Rotation': leaf_on_stem_rotation_up_down})

        leaf_on_stem_rotation_in_out_001 = nw.new_node(nodegroup_leaf_on_stem_rotation_in_out(in_out_scale=in_out_scale).name,
                                                       input_kwargs={'Samples': group_input.outputs["Samples"]})

        rotate_instances_7 = nw.new_node(Nodes.RotateInstances,
                                         input_kwargs={'Instances': rotate_instances_6,
                                                       'Rotation': leaf_on_stem_rotation_in_out_001})

        leaf_on_stem_scale_up_down_1 = nw.new_node(nodegroup_leaf_on_stem_scale_up_down(scale_gap).name,
                                                   input_kwargs={'Samples': group_input.outputs["Samples"]})

        scale_instances_9 = nw.new_node(Nodes.ScaleInstances,
                                        input_kwargs={'Instances': rotate_instances_7,
                                                      'Scale': leaf_on_stem_scale_up_down_1})
        leaves.append(scale_instances_9)
    join_geometry = nw.new_node(Nodes.JoinGeometry,
                                input_kwargs={'Geometry': leaves})

    random_value_1 = nw.new_node(Nodes.RandomValue,
                                 input_kwargs={2: -0.3, 3: 0.3})

    random_value_3 = nw.new_node(Nodes.RandomValue,
                                 input_kwargs={2: -0.3, 3: 0.3})

    combine_xyz = nw.new_node(Nodes.CombineXYZ,
                              input_kwargs={'X': random_value_1.outputs[1], 'Y': random_value_3.outputs[1]})

    rotate_instances = nw.new_node(Nodes.RotateInstances,
                                   input_kwargs={'Instances': join_geometry, 'Rotation': combine_xyz})

    random_value_2 = nw.new_node(Nodes.RandomValue,
                                 input_kwargs={2: 0.7})

    scale_instances_6 = nw.new_node(Nodes.ScaleInstances,
                                    input_kwargs={'Instances': rotate_instances, 'Scale': random_value_2.outputs[1]})

    realize_instances = nw.new_node(Nodes.RealizeInstances,
                                    input_kwargs={'Geometry': scale_instances_6})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': realize_instances})


@node_utils.to_nodegroup('nodegroup_stem_curvature', singleton=False, type='GeometryNodeTree')
def nodegroup_stem_curvature(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Curve', None),
                                            ('NodeSocketFloat', 'Y Stem Rotate', 0.2),
                                            ('NodeSocketFloat', 'Stem Count', 0.0),
                                            ('NodeSocketFloat', 'X Stem Rotate', -0.2)])

    resample_curve = nw.new_node(Nodes.ResampleCurve,
                                 input_kwargs={'Curve': group_input.outputs["Curve"],
                                               'Count': group_input.outputs["Stem Count"]})

    position_2 = nw.new_node(Nodes.InputPosition)

    spline_parameter_1 = nw.new_node(Nodes.SplineParameter)

    map_range_1 = nw.new_node(Nodes.MapRange,
                              input_kwargs={'Value': spline_parameter_1.outputs["Factor"],
                                            3: group_input.outputs["Y Stem Rotate"], 4: 0.0})

    vector_rotate = nw.new_node(Nodes.VectorRotate,
                                input_kwargs={'Vector': position_2, 'Center': (0.0, 0.0, 2.0),
                                              'Angle': map_range_1.outputs["Result"]},
                                attrs={'rotation_type': 'Y_AXIS'})

    set_position_1 = nw.new_node(Nodes.SetPosition,
                                 input_kwargs={'Geometry': resample_curve, 'Position': vector_rotate})

    position_1 = nw.new_node(Nodes.InputPosition)

    spline_parameter_2 = nw.new_node(Nodes.SplineParameter)

    map_range_2 = nw.new_node(Nodes.MapRange,
                              input_kwargs={'Value': spline_parameter_2.outputs["Factor"],
                                            3: group_input.outputs["X Stem Rotate"], 4: 0.0})

    vector_rotate_1 = nw.new_node(Nodes.VectorRotate,
                                  input_kwargs={'Vector': position_1, 'Angle': map_range_2.outputs["Result"]},
                                  attrs={'rotation_type': 'X_AXIS'})

    set_position_2 = nw.new_node(Nodes.SetPosition,
                                 input_kwargs={'Geometry': set_position_1, 'Position': vector_rotate_1})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': set_position_2})


@node_utils.to_nodegroup('nodegroup_stem_geometry', singleton=False, type='GeometryNodeTree')
def nodegroup_stem_geometry(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Curve', None)])

    spline_parameter = nw.new_node(Nodes.SplineParameter)

    map_range = nw.new_node(Nodes.MapRange,
                            input_kwargs={'Value': spline_parameter.outputs["Factor"], 3: uniform(0.1, 0.3), 4: 0.8},
                            attrs={'interpolation_type': 'SMOOTHSTEP'})

    set_curve_radius = nw.new_node(Nodes.SetCurveRadius,
                                   input_kwargs={'Curve': group_input.outputs["Curve"],
                                                 'Radius': map_range.outputs["Result"]})

    curve_circle = nw.new_node(Nodes.CurveCircle,
                               input_kwargs={'Radius': uniform(0.03, 0.06)})

    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
                                input_kwargs={'Curve': set_curve_radius, 'Profile Curve': curve_circle.outputs["Curve"],
                                              'Fill Caps': True})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Mesh': curve_to_mesh})


def shader_leaf_material(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    attribute = nw.new_node(Nodes.Attribute,
                            attrs={'attribute_name': 'vein'})

    texture_coordinate = nw.new_node(Nodes.TextureCoord)

    noise_texture = nw.new_node(Nodes.NoiseTexture,
                                input_kwargs={'Vector': texture_coordinate.outputs["Object"], 'Scale': 6.8,
                                              'Detail': 10.0, 'Roughness': 0.7})

    separate_rgb = nw.new_node(Nodes.SeparateRGB,
                               input_kwargs={'Image': noise_texture.outputs["Color"]})

    map_range = nw.new_node(Nodes.MapRange,
                            input_kwargs={'Value': separate_rgb.outputs["G"], 1: 0.4, 2: 0.7, 3: 0.48, 4: 0.52})

    map_range_1 = nw.new_node(Nodes.MapRange,
                              input_kwargs={'Value': separate_rgb.outputs["B"], 1: 0.4, 2: 0.7, 3: 0.8, 4: 1.2})

    attribute_1 = nw.new_node(Nodes.Attribute,
                              attrs={'attribute_name': 'subvein offset'})

    map_range_2 = nw.new_node(Nodes.MapRange,
                              input_kwargs={'Value': attribute_1.outputs["Color"], 2: -0.94})

    main_leaf_hsv = (uniform(0.3, 0.36), uniform(0.8, 1.0), uniform(0.25, 0.45))
    hue_saturation_value = nw.new_node('ShaderNodeHueSaturation',
                                       input_kwargs={'Value': 2.0, 'Color': hsv2rgba(main_leaf_hsv)})

    main_leaf_hsv_2 = (main_leaf_hsv[0] + normal(0.0, 0.005),) + main_leaf_hsv[1:]
    mix = nw.new_node(Nodes.MixRGB,
                      input_kwargs={'Fac': map_range_2.outputs["Result"], 'Color1': hue_saturation_value,
                                    'Color2': hsv2rgba(main_leaf_hsv_2)})

    hue_saturation_value_1 = nw.new_node('ShaderNodeHueSaturation',
                                         input_kwargs={'Hue': map_range.outputs["Result"],
                                                       'Value': map_range_1.outputs["Result"], 'Color': mix})

    stem_color_hsv = main_leaf_hsv[:-1] + (main_leaf_hsv[-1] - uniform(0.05, 0.15),)
    mix_1 = nw.new_node(Nodes.MixRGB,
                        input_kwargs={'Fac': attribute.outputs["Color"], 'Color1': hsv2rgba(stem_color_hsv),
                                      'Color2': hue_saturation_value_1})

    group = nw.new_node(nodegroup_nodegroup_leaf_shader().name,
                        input_kwargs={'Color': mix_1})

    material_output = nw.new_node(Nodes.MaterialOutput,
                                  input_kwargs={'Surface': group})


def geometry_palm_tree_leaf_nodes(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    curve_line_1 = nw.new_node(Nodes.CurveLine,
                               input_kwargs={'Start': (0.0, 0.0, 2.0), 'End': (0.0, 0.0, 0.0)})

    leaf_x_curvature = nw.new_node(Nodes.Value,
                                   label='leaf_x_curvature')
    leaf_x_curvature.outputs[0].default_value = kwargs['leaf_x_curvature']

    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: leaf_x_curvature, 1: kwargs['leaf_instance_curvature_ratio']},
        attrs={'operation': 'MULTIPLY'})

    integer_1 = nw.new_node(Nodes.Integer,
                            attrs={'integer': 50})
    integer_1.integer = kwargs['num_leaf_samples']

    stem_x_curvature = nw.new_node(Nodes.Value,
                                   label='stem_x_curvature')
    stem_x_curvature.outputs[0].default_value = normal(0., 0.15)

    stem_curvature = nw.new_node(nodegroup_stem_curvature().name,
                                 input_kwargs={'Curve': curve_line_1, 'Y Stem Rotate': leaf_x_curvature,
                                               'Stem Count': integer_1, 'X Stem Rotate': stem_x_curvature})

    stem_geometry = nw.new_node(nodegroup_stem_geometry().name,
                                input_kwargs={'Curve': stem_curvature})

    set_material = nw.new_node(Nodes.SetMaterial,
                               input_kwargs={'Geometry': stem_geometry,
                                             'Material': surface.shaderfunc_to_material(shader_stem_material)})

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Geometry', None)])

    wave_x_scale = nw.new_node(Nodes.Value,
                               label='wave_x_scale')
    wave_x_scale.outputs[0].default_value = 0.0

    wave_y_scale = nw.new_node(Nodes.Value,
                               label='wave_y_scale')
    wave_y_scale.outputs[0].default_value = 0.0

    leaf_width_scale = nw.new_node(Nodes.Value,
                                   label='leaf_width_scale')
    leaf_width_scale.outputs[0].default_value = kwargs['leaf_instance_width']

    palm_leaf_instance = nw.new_node(nodegroup_palm_leaf_instance().name,
                                      input_kwargs={'To Max': multiply, 'Mesh': group_input.outputs["Geometry"],
                                                    'Wave Scale Y': wave_x_scale, 'Wave Scale X': wave_y_scale,
                                                    'Leaf Width Scale': leaf_width_scale})

    leaf_scale = nw.new_node(Nodes.Value, label='leaf_scale')
    leaf_scale.outputs[0].default_value = uniform(0.5, 0.7)

    leaf_on_stem = nw.new_node(nodegroup_leaf_on_stem(kwargs['versions']).name,
                               input_kwargs={'Points': stem_curvature,
                                             'Instance': palm_leaf_instance.outputs["Geometry"], 'Scale': leaf_scale,
                                             'Samples': integer_1})

    join_geometry_1 = nw.new_node(Nodes.JoinGeometry,
                                  input_kwargs={'Geometry': [set_material, leaf_on_stem]})

    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': join_geometry_1,
                      'Translation': kwargs['plant_translation'],
                      'Rotation': (0.0, 0.0, kwargs['plant_z_rotate']),
                      'Scale': kwargs['plant_scale']})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': transform,
                                             'Attribute': palm_leaf_instance.outputs["Attribute"],
                                             'Coordinate': palm_leaf_instance.outputs["Coordinate"],
                                             'subvein offset': palm_leaf_instance.outputs["subvein"],
                                             'vein': palm_leaf_instance.outputs["vein"]})


class LeafPalmTreeFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super(LeafPalmTreeFactory, self).__init__(factory_seed, coarse=coarse)

    def update_params(self, params):
        if params.get('leaf_x_curvature', None) is None:
            params['leaf_x_curvature'] = uniform(0.0, 0.8)
        if params.get('leaf_instance_curvature_ratio', None) is None:
            params['leaf_instance_curvature_ratio'] = uniform(0.3, 0.6)
        if params.get('leaf_instance_width', None) is None:
            params['leaf_instance_width'] = uniform(0.07, 0.15)
        if params.get('num_leaf_samples', None) is None:
            params['num_leaf_samples'] = int(randint(6, 10) / params['leaf_instance_width'])
        if params.get('plant_translation', None) is None:
            params['plant_translation'] = (0.0, 0.0, 0.0)
        if params.get('plant_z_rotate', None) is None:
            params['plant_z_rotate'] = uniform(-0.4, 0.4)
        if params.get('versions', None) is None:
            params['versions'] = 3
        if params.get('plant_scale', None) is None:
            s = uniform(0.8, 1.5)
            params['plant_scale'] = (s, s, s)
        return params

    def create_asset(self, params={}, **kwargs):
        bpy.ops.mesh.primitive_plane_add(
            size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        obj = bpy.context.active_object

        params = self.update_params(params)
        surface.add_geomod(obj, geometry_palm_tree_leaf_nodes, apply=True,
                           attributes=['Attribute', 'Coordinate',
                                       'subvein offset', 'vein'], input_kwargs=params)
        surface.add_material(obj, shader_leaf_material, selection=None)

        tag_object(obj, 'leaf_palm_tree')
        return obj


if __name__ == '__main__':
    fac = LeafPalmTreeFactory(0)
    fac.create_asset()