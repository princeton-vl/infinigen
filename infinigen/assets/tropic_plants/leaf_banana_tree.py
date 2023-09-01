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
from infinigen.assets.tropic_plants.tropic_plant_utils import (
    nodegroup_nodegroup_leaf_gen,
    nodegroup_nodegroup_leaf_rotate_x,
    nodegroup_nodegroup_leaf_shader,
    nodegroup_nodegroup_move_to_origin,
    nodegroup_nodegroup_sub_vein,
    shader_stem_material
)
from infinigen.core.util import blender as butil
from infinigen.assets.utils.tag import tag_object, tag_nodegroup


@node_utils.to_nodegroup('nodegroup_nodegroup_apply_wave', singleton=False, type='GeometryNodeTree')
def nodegroup_nodegroup_apply_wave(nw: NodeWrangler, leaf_h_wave_control_points,
                                   leaf_w_wave_control_points, leaf_edge_wave_control_points):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Geometry', None),
                                            ('NodeSocketFloat', 'Wave Scale Y', 1.0),
                                            ('NodeSocketFloat', 'Wave Scale X', 1.0),
                                            ('NodeSocketFloat', 'X Modulated', 0.0),
                                            ('NodeSocketFloat', 'Width Scale', 0.0),
                                            ('NodeSocketFloat', 'Wave Scale E', 1.0)])

    position = nw.new_node(Nodes.InputPosition)

    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
                               input_kwargs={'Vector': position})

    map_range_6 = nw.new_node(Nodes.MapRange,
                              input_kwargs={'Value': separate_xyz.outputs["Y"], 1: -0.6, 2: 0.6})

    float_curve_3 = nw.new_node(Nodes.FloatCurve,
                                input_kwargs={'Value': map_range_6.outputs["Result"]})
    node_utils.assign_curve(float_curve_3.mapping.curves[0],
                            [(0.0, 0.5),
                             (0.1, leaf_edge_wave_control_points[0] + .5),
                             (0.2, leaf_edge_wave_control_points[1] + .5),
                             (0.3, leaf_edge_wave_control_points[2] + .5),
                             (0.4, leaf_edge_wave_control_points[3] + .5),
                             (0.5, leaf_edge_wave_control_points[4] + .5),
                             (0.6, leaf_edge_wave_control_points[5] + .5),
                             (0.7, leaf_edge_wave_control_points[6] + .5),
                             (0.8, leaf_edge_wave_control_points[7] + .5),
                             (0.9, leaf_edge_wave_control_points[8] + .5),
                             (1.0, 0.5)])

    map_range_7 = nw.new_node(Nodes.MapRange,
                              input_kwargs={'Value': float_curve_3, 3: -1.0})

    absolute = nw.new_node(Nodes.Math,
                           input_kwargs={0: separate_xyz.outputs["X"]},
                           attrs={'operation': 'ABSOLUTE'})

    map_range_4 = nw.new_node(Nodes.MapRange,
                              input_kwargs={'Value': absolute, 2: group_input.outputs["Width Scale"]})

    colorramp = nw.new_node(Nodes.ColorRamp,
                            input_kwargs={'Fac': map_range_4.outputs["Result"]})
    colorramp.color_ramp.elements[0].position = 0.015
    colorramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp.color_ramp.elements[1].position = uniform(0.3, 0.5)
    colorramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: map_range_7.outputs["Result"], 1: colorramp.outputs["Color"]},
                           attrs={'operation': 'MULTIPLY'})

    multiply_1 = nw.new_node(Nodes.Math,
                             input_kwargs={0: multiply, 1: group_input.outputs["Wave Scale E"]},
                             attrs={'operation': 'MULTIPLY'})

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ,
                                input_kwargs={'Z': multiply_1})

    set_position_2 = nw.new_node(Nodes.SetPosition,
                                 input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Offset': combine_xyz_3})

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

    multiply_2 = nw.new_node(Nodes.Math,
                             input_kwargs={0: map_range_1.outputs["Result"], 1: group_input.outputs["Wave Scale Y"]},
                             attrs={'operation': 'MULTIPLY'})

    combine_xyz = nw.new_node(Nodes.CombineXYZ,
                              input_kwargs={'Z': multiply_2})

    set_position = nw.new_node(Nodes.SetPosition,
                               input_kwargs={'Geometry': set_position_2, 'Offset': combine_xyz})

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
                            [(0.0, leaf_w_wave_control_points[0] + .5 + normal(0., 0.02)),
                             (0.1, leaf_w_wave_control_points[1] + .5 + normal(0., 0.02)),
                             (0.25, leaf_w_wave_control_points[2] + .5 + normal(0., 0.02)),
                             (0.4, leaf_w_wave_control_points[3] + .5 + normal(0., 0.02)),
                             (0.5, 0.5),
                             (0.6, leaf_w_wave_control_points[3] + .5 + normal(0., 0.02)),
                             (0.75, leaf_w_wave_control_points[2] + .5 + normal(0., 0.02)),
                             (0.9, leaf_w_wave_control_points[1] + .5 + normal(0., 0.02)),
                             (1.0, leaf_w_wave_control_points[0] + .5 + normal(0., 0.02))],
                            handles=['AUTO', 'AUTO', 'AUTO', 'AUTO', 'VECTOR', 'AUTO', 'AUTO', 'AUTO', 'AUTO'])

    map_range_3 = nw.new_node(Nodes.MapRange,
                              input_kwargs={'Value': float_curve_1, 3: -1.0})

    multiply_3 = nw.new_node(Nodes.Math,
                             input_kwargs={0: map_range_3.outputs["Result"], 1: group_input.outputs["Wave Scale X"]},
                             attrs={'operation': 'MULTIPLY'})

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
                                input_kwargs={'Z': multiply_3})

    set_position_1 = nw.new_node(Nodes.SetPosition,
                                 input_kwargs={'Geometry': set_position, 'Offset': combine_xyz_1})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': set_position_1})


def shader_leaf_material(nw: NodeWrangler, stem_color_hsv):
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

    main_leaf_hsv = (uniform(0.26, 0.37), uniform(0.8, 1.0), uniform(0.15, 0.55))
    hue_saturation_value = nw.new_node('ShaderNodeHueSaturation',
                                       input_kwargs={'Value': 2.0, 'Color': hsv2rgba(main_leaf_hsv)})

    main_leaf_hsv_2 = (main_leaf_hsv[0] + normal(0.0, 0.02),) + main_leaf_hsv[1:]
    mix = nw.new_node(Nodes.MixRGB,
                      input_kwargs={'Fac': map_range_2.outputs["Result"], 'Color1': hue_saturation_value,
                                    'Color2': hsv2rgba(main_leaf_hsv_2)})

    hue_saturation_value_1 = nw.new_node('ShaderNodeHueSaturation',
                                         input_kwargs={'Hue': map_range.outputs["Result"],
                                                       'Value': map_range_1.outputs["Result"], 'Color': mix})

    mix_1 = nw.new_node(Nodes.MixRGB,
                        input_kwargs={'Fac': attribute.outputs["Color"], 'Color1': hsv2rgba(stem_color_hsv),
                                      'Color2': hue_saturation_value_1})

    group = nw.new_node(nodegroup_nodegroup_leaf_shader().name,
                        input_kwargs={'Color': mix_1})

    material_output = nw.new_node(Nodes.MaterialOutput,
                                  input_kwargs={'Surface': group})


@node_utils.to_nodegroup('nodegroup_round_tropical_leaf', singleton=False, type='GeometryNodeTree')
def nodegroup_round_tropical_leaf(nw: NodeWrangler, jigsaw_depth, leaf_h_wave_control_points,
                                  leaf_w_wave_control_points, leaf_edge_wave_control_points,
                                  leaf_contour_control_points):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'To Max', -0.4),
                                            ('NodeSocketGeometry', 'Mesh', None),
                                            ('NodeSocketFloat', 'Wave Scale Y', 0.3),
                                            ('NodeSocketFloat', 'Wave Scale X', 0.5),
                                            ('NodeSocketFloat', 'Wave Scale E', 0.5),
                                            ('NodeSocketFloat', 'Leaf Width Scale', 0.0)])

    subdivide_mesh = nw.new_node(Nodes.SubdivideMesh,
                                 input_kwargs={'Mesh': group_input.outputs["Mesh"], 'Level': 10})

    subdivide_mesh_1 = nw.new_node(Nodes.SubdivideMesh,
                                   input_kwargs={'Mesh': subdivide_mesh})

    position = nw.new_node(Nodes.InputPosition)

    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
                                    input_kwargs={'Geometry': subdivide_mesh_1, 1: position},
                                    attrs={'data_type': 'FLOAT_VECTOR'})

    nodegroup_leaf_gen = nw.new_node(nodegroup_nodegroup_leaf_gen(leaf_contour_control_points).name,
                                     input_kwargs={'Mesh': capture_attribute.outputs["Geometry"],
                                                   'Displancement scale': 0.0, 'Vein Asymmetry': 0.3023,
                                                   'Vein Density': 0.0, 'Jigsaw Scale': uniform(5.0, 20.0),
                                                   'Jigsaw Depth': jigsaw_depth,
                                                   'Vein Angle': 0.3, 'Wave Displacement': 0.0, 'Midrib Length': 0.333,
                                                   'Stem Length': 0.6, 'Midrib Width': uniform(0.8, 1.4),
                                                   'Leaf Width Scale': group_input.outputs["Leaf Width Scale"]})

    nodegroup_sub_vein = nw.new_node(nodegroup_nodegroup_sub_vein().name,
                                     input_kwargs={'X': 0.0, 'Y': nodegroup_leaf_gen.outputs["Vein Coord"]})

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

    nodegroup_apply_wave = nw.new_node(nodegroup_nodegroup_apply_wave(leaf_h_wave_control_points,
                                                                      leaf_w_wave_control_points,
                                                                      leaf_edge_wave_control_points).name,
                                       input_kwargs={'Geometry': capture_attribute_2.outputs["Geometry"],
                                                     'Wave Scale Y': group_input.outputs["Wave Scale Y"],
                                                     'Wave Scale X': group_input.outputs["Wave Scale X"],
                                                     'X Modulated': nodegroup_leaf_gen.outputs["X Modulated"],
                                                     'Wave Scale E': group_input.outputs["Wave Scale E"]})

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
                                   input_kwargs={'Instances': instance_on_points_1, 'Rotation': (-1.5708, 0.0, 0.0)})

    realize_instances_1 = nw.new_node(Nodes.RealizeInstances,
                                      input_kwargs={'Geometry': rotate_instances})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': realize_instances_1})


@node_utils.to_nodegroup('nodegroup_stem_curvature', singleton=False, type='GeometryNodeTree')
def nodegroup_stem_curvature(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Curve', None),
                                            ('NodeSocketFloat', 'To Min1', 0.2),
                                            ('NodeSocketFloat', 'To Min2', -0.2)])

    resample_curve = nw.new_node(Nodes.ResampleCurve,
                                 input_kwargs={'Curve': group_input.outputs["Curve"], 'Count': 100})

    position_2 = nw.new_node(Nodes.InputPosition)

    spline_parameter_1 = nw.new_node(Nodes.SplineParameter)

    map_range_1 = nw.new_node(Nodes.MapRange,
                              input_kwargs={'Value': spline_parameter_1.outputs["Factor"],
                                            3: group_input.outputs["To Min1"], 4: 0.0})

    vector_rotate = nw.new_node(Nodes.VectorRotate,
                                input_kwargs={'Vector': position_2, 'Center': (0.0, 0.0, 2.0),
                                              'Angle': map_range_1.outputs["Result"]},
                                attrs={'rotation_type': 'Y_AXIS'})

    set_position_1 = nw.new_node(Nodes.SetPosition,
                                 input_kwargs={'Geometry': resample_curve, 'Position': vector_rotate})

    position_1 = nw.new_node(Nodes.InputPosition)

    spline_parameter_2 = nw.new_node(Nodes.SplineParameter)

    map_range_2 = nw.new_node(Nodes.MapRange,
                              input_kwargs={'Value': spline_parameter_2.outputs["Factor"], 3: group_input.outputs[2],
                                            4: 0.0})

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
                               input_kwargs={'Radius': 0.02})

    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
                                input_kwargs={'Curve': set_curve_radius, 'Profile Curve': curve_circle.outputs["Curve"],
                                              'Fill Caps': True})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Mesh': curve_to_mesh})


def geometry_leaf_nodes(nw: NodeWrangler, **kwargs):

    leaf_x_curvature = nw.new_node(Nodes.Value,
                                   label='leaf_x_curvature')
    leaf_x_curvature.outputs[0].default_value = -kwargs['leaf_x_curvature']

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Geometry', None)])

    wave_x_scale = nw.new_node(Nodes.Value,
                               label='wave_x_scale')
    wave_x_scale.outputs[0].default_value = kwargs['leaf_h_wave_scale']

    wave_y_scale = nw.new_node(Nodes.Value,
                               label='wave_y_scale')
    wave_y_scale.outputs[0].default_value = kwargs['leaf_w_wave_scale']

    wave_e_scale = nw.new_node(Nodes.Value,
                               label='wave_e_scale')
    wave_e_scale.outputs[0].default_value = kwargs['leaf_edge_wave_scale']

    leaf_width_scale = nw.new_node(Nodes.Value,
                                   label='leaf_width_scale')
    leaf_width_scale.outputs[0].default_value = kwargs['leaf_width']

    leaf_h_wave_control_points = kwargs['leaf_h_wave_control_points']
    leaf_w_wave_control_points = kwargs['leaf_w_wave_control_points']
    leaf_edge_wave_control_points = kwargs['leaf_edge_wave_control_points']
    leaf_contour_control_points = kwargs['leaf_contour_control_points']
    leaf_jigsaw_depth = kwargs['leaf_jigsaw_depth']

    round_tropical_leaf = nw.new_node(nodegroup_round_tropical_leaf(leaf_jigsaw_depth,
                                                                    leaf_h_wave_control_points,
                                                                    leaf_w_wave_control_points,
                                                                    leaf_edge_wave_control_points,
                                                                    leaf_contour_control_points).name,
                                      input_kwargs={'To Max': leaf_x_curvature, 'Mesh': group_input.outputs["Geometry"],
                                                    'Wave Scale Y': wave_x_scale, 'Wave Scale X': wave_y_scale,
                                                    'Leaf Width Scale': leaf_width_scale, 'Wave Scale E': wave_e_scale})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': round_tropical_leaf.outputs["Geometry"],
                                             'Attribute': round_tropical_leaf.outputs["Attribute"],
                                             'Coordinate': round_tropical_leaf.outputs["Coordinate"],
                                             'subvein offset': round_tropical_leaf.outputs["subvein"],
                                             'vein': round_tropical_leaf.outputs["vein"]})


def geometry_plant_nodes(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    curve_line_1 = nw.new_node(Nodes.CurveLine,
                               input_kwargs={'Start': (0.0, 0.0, 2.0), 'End': (0.0, 0.0, 0.0)})

    stem_y_curvature = nw.new_node(Nodes.Value,
                                   label='stem_y_curvature')
    stem_y_curvature.outputs[0].default_value = uniform(-0.5, 0.5)

    stem_x_curvature = nw.new_node(Nodes.Value,
                                   label='stem_x_curvature')
    stem_x_curvature.outputs[0].default_value = -kwargs['leaf_x_curvature']

    stem_curvature = nw.new_node(nodegroup_stem_curvature().name,
                                 input_kwargs={'Curve': curve_line_1, 1: stem_y_curvature, 2: stem_x_curvature})

    stem_geometry = nw.new_node(nodegroup_stem_geometry().name,
                                input_kwargs={'Curve': stem_curvature})

    set_material = nw.new_node(Nodes.SetMaterial,
                               input_kwargs={'Geometry': stem_geometry,
                                             'Material': surface.shaderfunc_to_material(
                                                 lambda x: shader_stem_material(x, stem_color_hsv=
                                                 kwargs['stem_color_hsv']))})

    leaf_x_curvature = nw.new_node(Nodes.Value,
                                   label='leaf_x_curvature')
    leaf_x_curvature.outputs[0].default_value = -kwargs['leaf_x_curvature']

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Geometry', None)])

    wave_x_scale = nw.new_node(Nodes.Value,
                               label='wave_x_scale')
    wave_x_scale.outputs[0].default_value = kwargs['leaf_h_wave_scale']

    wave_y_scale = nw.new_node(Nodes.Value,
                               label='wave_y_scale')
    wave_y_scale.outputs[0].default_value = kwargs['leaf_w_wave_scale']

    wave_e_scale = nw.new_node(Nodes.Value,
                               label='wave_edge_scale')
    wave_e_scale.outputs[0].default_value = kwargs['leaf_edge_wave_scale']

    leaf_width_scale = nw.new_node(Nodes.Value,
                                   label='leaf_width_scale')
    leaf_width_scale.outputs[0].default_value = kwargs['leaf_width']

    leaf_h_wave_control_points = kwargs['leaf_h_wave_control_points']
    leaf_w_wave_control_points = kwargs['leaf_w_wave_control_points']
    leaf_edge_wave_control_points = kwargs['leaf_edge_wave_control_points']
    leaf_contour_control_points = kwargs['leaf_contour_control_points']
    leaf_jigsaw_depth = kwargs['leaf_jigsaw_depth']

    round_tropical_leaf = nw.new_node(nodegroup_round_tropical_leaf(leaf_jigsaw_depth,
                                                                    leaf_h_wave_control_points,
                                                                    leaf_w_wave_control_points,
                                                                    leaf_edge_wave_control_points,
                                                                    leaf_contour_control_points).name,
                                      input_kwargs={'To Max': leaf_x_curvature, 'Mesh': group_input.outputs["Geometry"],
                                                    'Wave Scale Y': wave_x_scale, 'Wave Scale X': wave_y_scale,
                                                    'Leaf Width Scale': leaf_width_scale, 'Wave Scale E': wave_e_scale})

    leaf_scale = nw.new_node(Nodes.Value,
                             label='leaf_scale')
    leaf_scale.outputs[0].default_value = normal(1.0, 0.3)

    leaf_on_stem = nw.new_node(nodegroup_leaf_on_stem().name,
                               input_kwargs={'Points': stem_curvature,
                                             'Instance': round_tropical_leaf.outputs["Geometry"], 'Scale': leaf_scale})

    join_geometry = nw.new_node(Nodes.JoinGeometry,
                                input_kwargs={'Geometry': [set_material, leaf_on_stem]})

    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': join_geometry,
                      'Translation': kwargs['plant_translation'],
                      'Rotation': (0.0, 0.0, kwargs['plant_z_rotate']),
                      'Scale': kwargs['plant_scale']})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': transform,
                                             'Attribute': round_tropical_leaf.outputs["Attribute"],
                                             'Coordinate': round_tropical_leaf.outputs["Coordinate"],
                                             'subvein offset': round_tropical_leaf.outputs["subvein"],
                                             'vein': round_tropical_leaf.outputs["vein"]})


class LeafBananaTreeFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super(LeafBananaTreeFactory, self).__init__(factory_seed, coarse=coarse)

    def get_leaf_contour(self, mode):
        if mode == 'oval':
            return [0.13, 0.275, 0.35, 0.365, 0.32, 0.21]
        elif mode == 'pear':
            return [0.30, 0.46, 0.46, 0.43, 0.37, 0.23]
        else:
            return NotImplementedError

    def get_h_wave_contour(self, mode):
        if mode == 'flat':
            return [normal(0., 0.03) for _ in range(6)]
        elif mode == 's':
            return [-0.1 + normal(0., 0.02), 0. + normal(0., 0.02),
                    0.08 + normal(0., 0.02), 0. + normal(0., 0.02),
                    -0.05 + normal(0., 0.01)]
        elif mode == 'w':
            return [-0.08 + normal(0., 0.02), 0.07 + normal(0., 0.02),
                    -0.08 + normal(0., 0.02), 0.08 + normal(0., 0.02),
                    -0.05 + normal(0, 0.02)]
        else:
            raise NotImplementedError

    def get_w_wave_contour(self, mode):
        if mode == 'fold':
            return [-0.28 + normal(0., 0.02), -0.2 + normal(0., 0.02),
                    -0.13 + normal(0., 0.01), -0.06 + normal(0., 0.01)], uniform(0.1, 0.3)
        elif mode == 'wing':
            return [0.0 + normal(0., 0.02), 0.06 + normal(0., 0.02),
                    0.07 + normal(0., 0.01), 0.04 + normal(0., 0.01)], uniform(0.0, 0.3)
        else:
            raise NotImplementedError

    def get_e_wave_contour(self, mode):
        if mode == 'wavy':
            return [-0.06 + normal(0., 0.01), 0.06 + normal(0., 0.01), -0.06 + normal(0., 0.01),
                    0.06 + normal(0., 0.01), -0.06 + normal(0., 0.01), 0.06 + normal(0., 0.01),
                    -0.06 + normal(0., 0.01), 0.06 + normal(0., 0.01), -0.06 + normal(0., 0.01)], 10
        elif mode == 'flat':
            return [0.0 for _ in range(9)], 0.0
        else:
            raise NotImplementedError

    def update_params(self, **params):
        if params.get('leaf_h_wave_control_points', None) is None:
            mode = np.random.choice(['flat', 'w', 's'], p=[0.4, 0.3, 0.3])
            params['leaf_h_wave_control_points'] = self.get_h_wave_contour(mode)

        if params.get('leaf_w_wave_control_points', None) is None:
            mode = np.random.choice(['fold', 'wing'], p=[0.2, 0.8])
            params['leaf_w_wave_control_points'], params['leaf_w_wave_scale'] = self.get_w_wave_contour(mode)

        if params.get('leaf_edge_wave_control_points', None) is None:
            mode = np.random.choice(['wavy', 'flat'], p=[1.0, 0.0]) # 0.6, 0.4
            params['leaf_edge_wave_control_points'], params['leaf_edge_wave_scale'] = self.get_e_wave_contour(mode)

        if params.get('leaf_contour_control_points', None) is None:
            mode = np.random.choice(['oval', 'pear'], p=[0.5, 0.5])
            params['leaf_contour_control_points'] = self.get_leaf_contour(mode)

        if params.get('leaf_jigsaw_depth', None) is None:
            mode = np.random.choice([0, 1], p=[0.4, 0.6])
            params['leaf_jigsaw_depth'] = mode * uniform(0.8, 1.7)

        if params.get('leaf_width', None) is None:
            params['leaf_width'] = uniform(0.5, 0.85)

        if params.get('leaf_h_wave_scale', None) is None:
            params['leaf_h_wave_scale'] = uniform(0.02, 0.2)

        if params.get('leaf_w_wave_scale', None) is None:
            params['leaf_w_wave_scale'] = uniform(0.05, 0.25)

        if params.get('leaf_x_curvature', None) is None:
            params['leaf_x_curvature'] = uniform(0.0, 0.1)

        if params.get('stem_color_hsv', None) is None:
            params['stem_color_hsv'] = (uniform(0.25, 0.32), uniform(0.8, 1.0), uniform(0.8, 1.0))

        return params

    def create_asset(self, **params):
        bpy.ops.mesh.primitive_plane_add(
            size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        obj = bpy.context.active_object

        params = self.update_params(**params)
        surface.add_geomod(obj, geometry_leaf_nodes, apply=True,
                           attributes=['Attribute', 'Coordinate',
                                       'subvein offset', 'vein'], input_kwargs=params)
        surface.add_material(obj, lambda x: shader_leaf_material(x, stem_color_hsv=params['stem_color_hsv']),
                             selection=None)

        tag_object(obj, 'leaf_banana_tree')
        return obj


class PlantBananaTreeFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super(PlantBananaTreeFactory, self).__init__(factory_seed, coarse=coarse)
        self.leaf_tropical_factory = LeafBananaTreeFactory(factory_seed)

    def update_params(self, **params):
        params = self.leaf_tropical_factory.update_params(**params)
        # Add new params update
        if params.get('plant_translation', None) is None:
            params['plant_translation'] = (0.0, 0.0, 0.0)
        if params.get('plant_z_rotate', None) is None:
            params['plant_z_rotate'] = uniform(-0.4, 0.4)
        if params.get('plant_scale', None) is None:
            s = uniform(0.8, 1.5)
            params['plant_scale'] = (s, s, s)
        return params

    def create_asset(self, **params):
        bpy.ops.mesh.primitive_plane_add(
            size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        obj = bpy.context.active_object

        params = self.update_params(**params)
        surface.add_geomod(obj, geometry_plant_nodes, apply=True,
                           attributes=['Attribute', 'Coordinate',
                                       'subvein offset', 'vein'], input_kwargs=params)
        surface.add_material(obj, lambda x: shader_leaf_material(x, stem_color_hsv=params['stem_color_hsv']),
                             selection=None)

        tag_object(obj, 'leaf_banana_tree')
        return obj


if __name__ == '__main__':
    fac = LeafBananaTreeFactory(0)
    fac.create_asset()