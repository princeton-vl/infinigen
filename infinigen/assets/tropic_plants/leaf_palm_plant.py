# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Beining Han


import bpy
import mathutils
import numpy as np
from numpy.random import uniform, normal, randint
from infinigen.assets.tropic_plants.tropic_plant_utils import (
    nodegroup_nodegroup_leaf_gen,
    nodegroup_nodegroup_leaf_rotate_x,
    nodegroup_nodegroup_leaf_shader,
    nodegroup_nodegroup_move_to_origin,
    nodegroup_nodegroup_sub_vein,
    hsv2rgba,
    shader_stem_material,
)
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.nodes import Nodes, NodeWrangler, node_utils
from infinigen.core.util import blender as butil
from infinigen.core import surface


@node_utils.to_nodegroup('nodegroup_nodegroup_apply_wave', singleton=False, type='GeometryNodeTree')
def nodegroup_nodegroup_apply_wave(nw: NodeWrangler, leaf_h_wave_control_points):
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
                            [(0.0, .5),
                             (0.2, leaf_h_wave_control_points[0] + .5),
                             (0.4, leaf_h_wave_control_points[1] + .5),
                             (0.6, leaf_h_wave_control_points[2] + .5),
                             (0.8, leaf_h_wave_control_points[3] + .5),
                             (1.0, leaf_h_wave_control_points[4] + .5)])

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


@node_utils.to_nodegroup('nodegroup_palm_leaf_assemble', singleton=False, type='GeometryNodeTree')
def nodegroup_palm_leaf_assemble(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Points', None),
                                            ('NodeSocketGeometry', 'Instance', None),
                                            ('NodeSocketFloat', 'Resolution', 0.0)])

    index = nw.new_node(Nodes.Index)

    divide = nw.new_node(Nodes.Math,
                         input_kwargs={0: group_input.outputs["Resolution"], 1: 2.0},
                         attrs={'operation': 'DIVIDE'})

    less_than = nw.new_node(Nodes.Math,
                            input_kwargs={0: index, 1: divide},
                            attrs={'operation': 'LESS_THAN'})

    greater_than = nw.new_node(Nodes.Math,
                               input_kwargs={0: index, 1: 0.0},
                               attrs={'operation': 'GREATER_THAN'})

    op_and = nw.new_node(Nodes.BooleanMath,
                         input_kwargs={0: less_than, 1: greater_than})

    curve_tangent = nw.new_node(Nodes.CurveTangent)

    align_euler_to_vector = nw.new_node(Nodes.AlignEulerToVector,
                                        input_kwargs={'Vector': curve_tangent})

    random_value = nw.new_node(Nodes.RandomValue,
                               input_kwargs={2: 0.9, 3: 1.1, 'Seed': 2})

    instance_on_points_1 = nw.new_node(Nodes.InstanceOnPoints,
                                       input_kwargs={'Points': group_input.outputs["Points"], 'Selection': op_and,
                                                     'Instance': group_input.outputs["Instance"],
                                                     'Rotation': align_euler_to_vector,
                                                     'Scale': random_value.outputs[1]})

    realize_instances_1 = nw.new_node(Nodes.RealizeInstances,
                                      input_kwargs={'Geometry': instance_on_points_1})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': realize_instances_1})


@node_utils.to_nodegroup('nodegroup_round_tropical_leaf', singleton=False, type='GeometryNodeTree')
def nodegroup_leaf_palm_instance(nw: NodeWrangler, leaf_h_wave_control_points):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'To Max', -0.4),
                                            ('NodeSocketGeometry', 'Mesh', None),
                                            ('NodeSocketFloat', 'Wave Scale Y', 0.3),
                                            ('NodeSocketFloat', 'Wave Scale X', 0.5),
                                            ('NodeSocketFloat', 'Leaf Width Scale', 0.0)])

    subdivide_mesh = nw.new_node(Nodes.SubdivideMesh,
                                 input_kwargs={'Mesh': group_input.outputs["Mesh"], 'Level': 10})

    subdivide_mesh_1 = nw.new_node(Nodes.SubdivideMesh,
                                   input_kwargs={'Mesh': subdivide_mesh})

    position = nw.new_node(Nodes.InputPosition)

    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
                                    input_kwargs={'Geometry': subdivide_mesh_1, 1: position},
                                    attrs={'data_type': 'FLOAT_VECTOR'})

    nodegroup_leaf_gen = nw.new_node(nodegroup_nodegroup_leaf_gen().name,
                                     input_kwargs={'Mesh': capture_attribute.outputs["Geometry"],
                                                   'Displancement scale': 0.0, 'Vein Asymmetry': 0.3023,
                                                   'Vein Density': 0.0, 'Jigsaw Scale': 10.0, 'Jigsaw Depth': 0.0,
                                                   'Vein Angle': 0.3, 'Wave Displacement': 0.0, 'Midrib Length': 0.3336,
                                                   'Midrib Width': 1.3, 'Stem Length': 0.6,
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

    nodegroup_apply_wave = nw.new_node(nodegroup_nodegroup_apply_wave(leaf_h_wave_control_points).name,
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


@node_utils.to_nodegroup('nodegroup_palmleafsector', singleton=False, type='GeometryNodeTree')
def nodegroup_palmleafsector(nw: NodeWrangler, leaf_h_wave_control_points):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'To Max', -0.4),
                                            ('NodeSocketGeometry', 'Mesh', None),
                                            ('NodeSocketFloat', 'Wave Scale Y', 0.3),
                                            ('NodeSocketFloat', 'Wave Scale X', 0.5),
                                            ('NodeSocketFloat', 'Leaf Width Scale', 0.0),
                                            ('NodeSocketInt', 'Resolution1', 26),
                                            ('NodeSocketFloat', 'Resolution2', 0.0)])

    round_tropical_leaf = nw.new_node(nodegroup_leaf_palm_instance(leaf_h_wave_control_points).name,
                                      input_kwargs={'To Max': group_input.outputs["To Max"],
                                                    'Mesh': group_input.outputs["Mesh"],
                                                    'Wave Scale Y': group_input.outputs["Wave Scale Y"],
                                                    'Wave Scale X': group_input.outputs["Wave Scale X"],
                                                    'Leaf Width Scale': group_input.outputs["Leaf Width Scale"]})

    curve_circle = nw.new_node(Nodes.CurveCircle,
                               input_kwargs={'Resolution': group_input.outputs["Resolution1"], 'Radius': 0.01})

    palm_leaf_assemble = nw.new_node(nodegroup_palm_leaf_assemble().name,
                                     input_kwargs={'Points': curve_circle.outputs["Curve"],
                                                   'Instance': round_tropical_leaf.outputs["Geometry"],
                                                   'Resolution': group_input.outputs[6]})

    join_geometry = nw.new_node(Nodes.JoinGeometry,
                                input_kwargs={'Geometry': palm_leaf_assemble})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Attribute': round_tropical_leaf.outputs["Attribute"],
                                             'Coordinate': round_tropical_leaf.outputs["Coordinate"],
                                             'subvein': round_tropical_leaf.outputs["subvein"],
                                             'vein': round_tropical_leaf.outputs["vein"], 'Geometry': join_geometry})


@node_utils.to_nodegroup('nodegroup_leaf_on_stem', singleton=False, type='GeometryNodeTree')
def nodegroup_leaf_on_stem(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Points', None),
                                            ('NodeSocketGeometry', 'Instance', None),
                                            ('NodeSocketVectorXYZ', 'Scale', (1.0, 1.0, 1.0))])

    endpoint_selection = nw.new_node('GeometryNodeCurveEndpointSelection',
                                     input_kwargs={'End Size': 0})

    curve_tangent = nw.new_node(Nodes.CurveTangent)

    align_euler_to_vector = nw.new_node(Nodes.AlignEulerToVector,
                                        input_kwargs={'Vector': curve_tangent},
                                        attrs={'axis': 'Z'})

    instance_on_points_1 = nw.new_node(Nodes.InstanceOnPoints,
                                       input_kwargs={'Points': group_input.outputs["Points"],
                                                     'Selection': endpoint_selection,
                                                     'Instance': group_input.outputs["Instance"],
                                                     'Rotation': align_euler_to_vector,
                                                     'Scale': group_input.outputs["Scale"]})

    rotate_instances = nw.new_node(Nodes.RotateInstances,
                                   input_kwargs={'Instances': instance_on_points_1, 'Rotation': (1.5708, 0.0, 3.1416)})

    realize_instances_1 = nw.new_node(Nodes.RealizeInstances,
                                      input_kwargs={'Geometry': rotate_instances})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': realize_instances_1})


@node_utils.to_nodegroup('nodegroup_stem_curvature', singleton=False, type='GeometryNodeTree')
def nodegroup_stem_curvature(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Curve', None),
                                            ('NodeSocketFloat', 'Y Stem Rotate', 0.2),
                                            ('NodeSocketFloat', 'X Stem Rotate', -0.2)])

    resample_curve = nw.new_node(Nodes.ResampleCurve,
                                 input_kwargs={'Curve': group_input.outputs["Curve"], 'Count': 100})

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
                            input_kwargs={'Value': spline_parameter.outputs["Factor"], 3: 0.4, 4: 0.8},
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

    main_leaf_hsv = (uniform(0.3, 0.36), uniform(0.6, 0.7), uniform(0.2, 0.3))
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


def geometry_plant_nodes(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    curve_line_1 = nw.new_node(Nodes.CurveLine,
                               input_kwargs={'Start': (0.0, 0.0, kwargs['plant_stem_length']), 'End': (0.0, 0.0, 0.0)})

    stem_y_curvature = nw.new_node(Nodes.Value,
                                   label='stem_y_curvature')
    stem_y_curvature.outputs[0].default_value = kwargs['stem_y_curvature']

    stem_x_curvature = nw.new_node(Nodes.Value,
                                   label='stem_x_curvature')
    stem_x_curvature.outputs[0].default_value = kwargs['stem_x_curvature']

    stem_curvature = nw.new_node(nodegroup_stem_curvature().name,
                                 input_kwargs={'Curve': curve_line_1, 'Y Stem Rotate': stem_y_curvature,
                                               'X Stem Rotate': stem_x_curvature})

    stem_geometry = nw.new_node(nodegroup_stem_geometry().name,
                                input_kwargs={'Curve': stem_curvature})

    set_material = nw.new_node(Nodes.SetMaterial,
                               input_kwargs={'Geometry': stem_geometry,
                                             'Material': surface.shaderfunc_to_material(shader_stem_material)})

    leaf_x_curvature = nw.new_node(Nodes.Value,
                                   label='leaf_x_curvature')
    leaf_x_curvature.outputs[0].default_value = kwargs['leaf_x_curvature']

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Geometry', None)])

    wave_x_scale = nw.new_node(Nodes.Value,
                               label='wave_x_scale')
    wave_x_scale.outputs[0].default_value = kwargs['leaf_h_wave_scale']

    wave_y_scale = nw.new_node(Nodes.Value,
                               label='wave_y_scale')
    wave_y_scale.outputs[0].default_value = 0.0

    leaf_width_scale = nw.new_node(Nodes.Value,
                                   label='leaf_width_scale')
    leaf_width_scale.outputs[0].default_value = uniform(0.15, 0.2)

    integer = nw.new_node(Nodes.Integer,
                          attrs={'integer': 24})
    integer.integer = randint(20, 30)

    palmleafsector = nw.new_node(nodegroup_palmleafsector(leaf_h_wave_control_points=
                                                          kwargs['leaf_h_wave_control_points']).name,
                                 input_kwargs={'To Max': leaf_x_curvature, 'Mesh': group_input.outputs["Geometry"],
                                               'Wave Scale Y': wave_x_scale, 'Wave Scale X': wave_y_scale,
                                               'Leaf Width Scale': leaf_width_scale, 5: integer, 6: integer})

    leaf_scale = nw.new_node(Nodes.Value,
                             label='leaf_scale')
    leaf_scale.outputs[0].default_value = uniform(0.85, 1.25)

    leaf_on_stem = nw.new_node(nodegroup_leaf_on_stem().name,
                               input_kwargs={'Points': stem_curvature, 'Instance': palmleafsector.outputs["Geometry"],
                                             'Scale': leaf_scale})

    join_geometry_1 = nw.new_node(Nodes.JoinGeometry,
                                  input_kwargs={'Geometry': [set_material, leaf_on_stem]})

    combine_xyz = nw.new_node(Nodes.CombineXYZ,
                              input_kwargs={'X': leaf_x_curvature})

    transform_1 = nw.new_node(Nodes.Transform,
                              input_kwargs={'Geometry': join_geometry_1, 'Rotation': combine_xyz})

    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': transform_1,
                      'Translation': kwargs['plant_translation'],
                      'Rotation': (0.0, 0.0, kwargs['plant_z_rotate']),
                      'Scale': kwargs['plant_scale']})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': transform, 'Attribute': palmleafsector.outputs["Attribute"],
                                             'Coordinate': palmleafsector.outputs["Coordinate"],
                                             'subvein offset': palmleafsector.outputs["subvein"],
                                             'vein': palmleafsector.outputs["vein"]})


class LeafPalmPlantFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super(LeafPalmPlantFactory, self).__init__(factory_seed, coarse=coarse)

    def get_h_wave_contour(self, mode):
        if mode == 'flat':
            return [normal(0., 0.03) for _ in range(6)]
        elif mode == 's':
            return [-0.5 + normal(0., 0.01), 0. + normal(0., 0.01),
                    0.05 + normal(0., 0.01), 0. + normal(0., 0.01),
                    -0.05 + normal(0., 0.01)]
        else:
            raise NotImplementedError

    def update_params(self, params):
        if params.get('leaf_h_wave_control_points', None) is None:
            mode = np.random.choice(['flat', 's'], p=[0.7, 0.3])
            params['leaf_h_wave_control_points'] = self.get_h_wave_contour(mode)
        if params.get('leaf_h_wave_scale', None) is None:
            params['leaf_h_wave_scale'] = uniform(0.01, 0.15)
        if params.get('leaf_x_curvature', None) is None:
            params['leaf_x_curvature'] = uniform(0.0, 0.5)
        if params.get('stem_x_curvature', None) is None:
            params['stem_x_curvature'] = uniform(-0.1, 0.4)
        if params.get('stem_y_curvature', None) is None:
            params['stem_y_curvature'] = uniform(-0.15, 0.15)
        if params.get('plant_translation', None) is None:
            params['plant_translation'] = (0.0, 0.0, 0.0)
        if params.get('plant_z_rotate', None) is None:
            params['plant_z_rotate'] = uniform(-0.4, 0.4)
        if params.get('plant_stem_length', None) is None:
            params['plant_stem_length'] = uniform(1.5, 2.2)
        if params.get('plant_scale', None) is None:
            s = uniform(0.8, 1.3)
            params['plant_scale'] = (s, s, s)
        return params

    def create_asset(self, params={}, **kwargs):
        bpy.ops.mesh.primitive_plane_add(
            size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        obj = bpy.context.active_object

        params = self.update_params(params)
        surface.add_geomod(obj, geometry_plant_nodes, apply=False,
                           attributes=['Attribute', 'Coordinate',
                                       'subvein offset', 'vein'], input_kwargs=params)
        surface.add_material(obj, shader_leaf_material, selection=None)

        return obj

