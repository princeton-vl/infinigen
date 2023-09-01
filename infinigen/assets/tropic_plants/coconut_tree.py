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
from infinigen.assets.tropic_plants.leaf_palm_tree import LeafPalmTreeFactory
from infinigen.assets.fruits.coconutgreen import FruitFactoryCoconutgreen


@node_utils.to_nodegroup('nodegroup_pedal_cross_contour_top', singleton=False, type='GeometryNodeTree')
def nodegroup_pedal_cross_contour_top(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    normal_2 = nw.new_node(Nodes.InputNormal)

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'Y', 0.0),
                                            ('NodeSocketFloat', 'X', 0.0)])

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ,
                                input_kwargs={'X': group_input.outputs["X"], 'Y': group_input.outputs["Y"]})

    multiply = nw.new_node(Nodes.VectorMath,
                           input_kwargs={0: normal_2, 1: combine_xyz_3},
                           attrs={'operation': 'MULTIPLY'})

    index_1 = nw.new_node(Nodes.Index)

    greater_than = nw.new_node(Nodes.Math,
                               input_kwargs={0: index_1, 1: 63.0},
                               attrs={'operation': 'GREATER_THAN'})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Vector': multiply.outputs["Vector"], 'Value': greater_than})


@node_utils.to_nodegroup('nodegroup_pedal_cross_contour_bottom', singleton=False, type='GeometryNodeTree')
def nodegroup_pedal_cross_contour_bottom(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    normal = nw.new_node(Nodes.InputNormal)

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'Y', 0.0),
                                            ('NodeSocketFloat', 'X', 0.0)])

    combine_xyz = nw.new_node(Nodes.CombineXYZ,
                              input_kwargs={'X': group_input.outputs["X"], 'Y': group_input.outputs["Y"]})

    multiply = nw.new_node(Nodes.VectorMath,
                           input_kwargs={0: normal, 1: combine_xyz},
                           attrs={'operation': 'MULTIPLY'})

    index = nw.new_node(Nodes.Index)

    less_than = nw.new_node(Nodes.Math,
                            input_kwargs={0: index, 1: 64.0},
                            attrs={'operation': 'LESS_THAN'})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Vector': multiply.outputs["Vector"], 'Value': less_than})


@node_utils.to_nodegroup('nodegroup_trunk_radius_001', singleton=False, type='GeometryNodeTree')
def nodegroup_trunk_radius_001(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    random_value = nw.new_node(Nodes.RandomValue,
                               input_kwargs={2: 0.01, 3: 0.05})

    spline_parameter = nw.new_node(Nodes.SplineParameter)

    map_range = nw.new_node(Nodes.MapRange,
                            input_kwargs={'Value': spline_parameter.outputs["Factor"], 3: 1.0, 4: 0.0},
                            attrs={'clamp': False})

    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: spline_parameter.outputs["Factor"], 1: 10000.0},
                           attrs={'operation': 'MULTIPLY'})

    floor = nw.new_node(Nodes.Math,
                        input_kwargs={0: multiply},
                        attrs={'operation': 'FLOOR'})

    subtract = nw.new_node(Nodes.Math,
                           input_kwargs={0: multiply, 1: floor},
                           attrs={'operation': 'SUBTRACT'})

    float_curve = nw.new_node(Nodes.FloatCurve,
                              input_kwargs={'Value': subtract})
    node_utils.assign_curve(float_curve.mapping.curves[0],
                            [(0.0, 0.0156), (0.2545, 0.2), (0.5182, 0.0344), (0.7682, 0.2375), (1.0, 0.0)])

    multiply_1 = nw.new_node(Nodes.Math,
                             input_kwargs={0: float_curve, 1: 1.0},
                             attrs={'operation': 'MULTIPLY'})

    multiply_2 = nw.new_node(Nodes.Math,
                             input_kwargs={0: map_range.outputs["Result"], 1: multiply_1},
                             attrs={'operation': 'MULTIPLY'})

    add = nw.new_node(Nodes.Math,
                      input_kwargs={0: map_range.outputs["Result"], 1: multiply_2})

    add_1 = nw.new_node(Nodes.Math,
                        input_kwargs={0: random_value.outputs[1], 1: add})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Value': add_1})


@node_utils.to_nodegroup('nodegroup_coutour_cross_geometry', singleton=False, type='GeometryNodeTree')
def nodegroup_coutour_cross_geometry(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    curve_circle = nw.new_node(Nodes.CurveCircle,
                               input_kwargs={'Resolution': 128, 'Radius': 0.05})

    pedal_cross_coutour_x = nw.new_node(Nodes.Value,
                                        label='pedal_cross_coutour_x')
    pedal_cross_coutour_x.outputs[0].default_value = 0.3

    pedal_cross_contour_bottom = nw.new_node(nodegroup_pedal_cross_contour_bottom().name,
                                             input_kwargs={'X': pedal_cross_coutour_x})

    set_position_1 = nw.new_node(Nodes.SetPosition,
                                 input_kwargs={'Geometry': curve_circle.outputs["Curve"],
                                               'Selection': pedal_cross_contour_bottom.outputs["Value"],
                                               'Offset': pedal_cross_contour_bottom.outputs["Vector"]})

    pedal_cross_coutour_y = nw.new_node(Nodes.Value,
                                        label='pedal_cross_coutour_y')
    pedal_cross_coutour_y.outputs[0].default_value = 0.3

    pedal_cross_contour_top = nw.new_node(nodegroup_pedal_cross_contour_top().name,
                                          input_kwargs={'Y': pedal_cross_coutour_y, 'X': pedal_cross_coutour_x})

    set_position_2 = nw.new_node(Nodes.SetPosition,
                                 input_kwargs={'Geometry': set_position_1,
                                               'Selection': pedal_cross_contour_top.outputs["Value"],
                                               'Offset': pedal_cross_contour_top.outputs["Vector"]})

    noise_texture_2 = nw.new_node(Nodes.NoiseTexture,
                                  input_kwargs={'W': 7.0, 'Detail': 15.0},
                                  attrs={'noise_dimensions': '4D'})

    scale = nw.new_node(Nodes.VectorMath,
                        input_kwargs={0: noise_texture_2.outputs["Fac"], 'Scale': 0.0},
                        attrs={'operation': 'SCALE'})

    set_position_5 = nw.new_node(Nodes.SetPosition,
                                 input_kwargs={'Geometry': set_position_2, 'Offset': scale.outputs["Vector"]})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': set_position_5})


@node_utils.to_nodegroup('nodegroup_pedal_z_contour', singleton=False, type='GeometryNodeTree')
def nodegroup_pedal_z_contour(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    spline_parameter = nw.new_node(Nodes.SplineParameter)

    float_curve = nw.new_node(Nodes.FloatCurve,
                              input_kwargs={'Value': spline_parameter.outputs["Factor"]})
    node_utils.assign_curve(float_curve.mapping.curves[0],
                            [(0.0, 0.4094), (0.1773, 0.475), (0.3795, 0.5062), (0.5864, 0.5187), (0.7202, 0.5084),
                             (0.8636, 0.4781), (1.0, 0.375)])

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'Value', 0.5)])

    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: float_curve, 1: group_input.outputs["Value"]},
                           attrs={'operation': 'MULTIPLY'})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Value': multiply})


@node_utils.to_nodegroup('nodegroup_pedal_stem_curvature', singleton=False, type='GeometryNodeTree')
def nodegroup_pedal_stem_curvature(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    position_3 = nw.new_node(Nodes.InputPosition)

    spline_parameter_1 = nw.new_node(Nodes.SplineParameter)

    float_curve_1 = nw.new_node(Nodes.FloatCurve,
                                input_kwargs={'Value': spline_parameter_1.outputs["Factor"]})
    node_utils.assign_curve(float_curve_1.mapping.curves[0],
                            [(0.0, 0.0688), (0.2545, 0.2281), (0.5023, 0.2563), (0.9773, 0.2656)])

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'Value', 0.2)])

    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: float_curve_1, 1: group_input.outputs["Value"]},
                           attrs={'operation': 'MULTIPLY'})

    vector_rotate = nw.new_node(Nodes.VectorRotate,
                                input_kwargs={'Vector': position_3, 'Center': (0.0, 0.0, 0.2), 'Angle': multiply},
                                attrs={'rotation_type': 'X_AXIS'})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Vector': vector_rotate})


@node_utils.to_nodegroup('nodegroup_node_group_002', singleton=False, type='ShaderNodeTree')
def nodegroup_node_group_002(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    texture_coordinate = nw.new_node(Nodes.TextureCoord)

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketColor', 'Color', (0.8, 0.8, 0.8, 1.0)),
                                            ('NodeSocketFloat', 'attribute', 0.0),
                                            ('NodeSocketFloat', 'voronoi scale', 50.0),
                                            ('NodeSocketFloatFactor', 'voronoi randomness', 1.0),
                                            ('NodeSocketFloat', 'seed', 0.0),
                                            ('NodeSocketFloat', 'noise scale', 10.0),
                                            ('NodeSocketFloat', 'noise amount', 1.4),
                                            ('NodeSocketFloat', 'hue min', 0.6),
                                            ('NodeSocketFloat', 'hue max', 1.085)])

    add = nw.new_node(Nodes.VectorMath,
                      input_kwargs={0: texture_coordinate.outputs["Object"], 1: group_input.outputs["seed"]})

    noise_texture = nw.new_node(Nodes.NoiseTexture,
                                input_kwargs={'Vector': add.outputs["Vector"],
                                              'Scale': group_input.outputs["noise scale"], 'Detail': 1.0})

    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: noise_texture.outputs["Fac"], 1: group_input.outputs["noise amount"]},
                           attrs={'operation': 'MULTIPLY'})

    voronoi_texture = nw.new_node(Nodes.VoronoiTexture,
                                  input_kwargs={'W': group_input.outputs["attribute"],
                                                'Scale': group_input.outputs["voronoi scale"],
                                                'Randomness': group_input.outputs["voronoi randomness"]},
                                  attrs={'voronoi_dimensions': '1D'})

    add_1 = nw.new_node(Nodes.Math,
                        input_kwargs={0: multiply, 1: voronoi_texture.outputs["Distance"]})

    map_range = nw.new_node(Nodes.MapRange,
                            input_kwargs={'Value': add_1, 3: group_input.outputs["hue min"],
                                          4: group_input.outputs["hue max"]})

    hue_saturation_value = nw.new_node('ShaderNodeHueSaturation',
                                       input_kwargs={'Value': map_range.outputs["Result"],
                                                     'Color': group_input.outputs["Color"]})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Color': hue_saturation_value})


@node_utils.to_nodegroup('nodegroup_coconutvein', singleton=False, type='GeometryNodeTree')
def nodegroup_coconutvein(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    index_2 = nw.new_node(Nodes.Index)

    map_range = nw.new_node(Nodes.MapRange,
                            input_kwargs={'Value': index_2, 1: 400.0, 2: 0.0},
                            attrs={'clamp': False})

    float_curve = nw.new_node(Nodes.FloatCurve,
                              input_kwargs={'Factor': map_range.outputs["Result"]})
    node_utils.assign_curve(float_curve.mapping.curves[0],
                            [(0.0, 0.0), (0.2455, 0.0), (0.5091, 0.0), (0.7636, 0.1625), (1.0, 0.4688)])

    noise_texture = nw.new_node(Nodes.NoiseTexture,
                                input_kwargs={'Scale': 1.0},
                                attrs={'noise_dimensions': '4D'})

    multiply = nw.new_node(Nodes.VectorMath,
                           input_kwargs={0: float_curve, 1: noise_texture.outputs["Color"]},
                           attrs={'operation': 'MULTIPLY'})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Vector': multiply.outputs["Vector"]})


@node_utils.to_nodegroup('nodegroup_tree_trunk_geometry_001', singleton=False, type='GeometryNodeTree')
def nodegroup_tree_trunk_geometry_001(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Curve', None)])

    trunkradius_001 = nw.new_node(nodegroup_trunk_radius_001().name)

    set_curve_radius = nw.new_node(Nodes.SetCurveRadius,
                                   input_kwargs={'Curve': group_input.outputs["Curve"], 'Radius': trunkradius_001})

    trunk_resolution = nw.new_node(Nodes.Integer,
                                   label='TrunkResolution',
                                   attrs={'integer': 32})
    trunk_resolution.integer = 32

    trunk_radius = nw.new_node(Nodes.Value,
                               label='TrunkRadius')
    trunk_radius.outputs[0].default_value = 0.02

    curve_circle = nw.new_node(Nodes.CurveCircle,
                               input_kwargs={'Resolution': trunk_resolution, 'Radius': trunk_radius})

    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
                                input_kwargs={'Curve': set_curve_radius, 'Profile Curve': curve_circle.outputs["Curve"],
                                              'Fill Caps': True})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Mesh': curve_to_mesh, 'Integer': trunk_resolution})


@node_utils.to_nodegroup('nodegroup_truncated_leaf_selection', singleton=False, type='GeometryNodeTree')
def nodegroup_truncated_leaf_selection(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    index_3 = nw.new_node(Nodes.Index)

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'Value', 0.5)])

    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: 1600.0, 1: group_input.outputs["Value"]},
                           attrs={'operation': 'MULTIPLY'})

    multiply_1 = nw.new_node(Nodes.Math,
                             input_kwargs={0: multiply, 1: uniform(0.92, 0.98)},
                             attrs={'operation': 'MULTIPLY'})

    greater_than = nw.new_node(Nodes.Math,
                               input_kwargs={0: index_3, 1: multiply_1},
                               attrs={'operation': 'GREATER_THAN'})

    multiply_2 = nw.new_node(Nodes.Math,
                             input_kwargs={0: multiply, 1: np.clip(normal(0.8, 0.1), 0.7, 0.9)},
                             attrs={'operation': 'MULTIPLY'})

    less_than = nw.new_node(Nodes.Math,
                            input_kwargs={0: index_3, 1: multiply_2},
                            attrs={'operation': 'LESS_THAN'})

    op_or = nw.new_node(Nodes.BooleanMath,
                        input_kwargs={0: greater_than, 1: less_than},
                        attrs={'operation': 'OR'})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Boolean': op_or})


@node_utils.to_nodegroup('nodegroup_random_rotate', singleton=False, type='GeometryNodeTree')
def nodegroup_random_rotate(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    random_value_1 = nw.new_node(Nodes.RandomValue,
                                 input_kwargs={2: -0.2, 3: 0.2})

    random_value_2 = nw.new_node(Nodes.RandomValue,
                                 input_kwargs={2: -0.5, 3: 0.5, 'Seed': 1})

    random_value_3 = nw.new_node(Nodes.RandomValue,
                                 input_kwargs={2: -0.2, 3: 0.2, 'Seed': 3})

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
                                input_kwargs={'X': random_value_1.outputs[1], 'Y': random_value_2.outputs[1],
                                              'Z': random_value_3.outputs[1]})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Vector': combine_xyz_1})


@node_utils.to_nodegroup('nodegroup_leaf_truncated_rotate', singleton=False, type='GeometryNodeTree')
def nodegroup_leaf_truncated_rotate(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    index_1 = nw.new_node(Nodes.Index)

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'Value', 0.5)])

    add = nw.new_node(Nodes.Math,
                      input_kwargs={0: group_input.outputs["Value"], 1: 0.0})

    modulo = nw.new_node(Nodes.Math,
                         input_kwargs={0: index_1, 1: add},
                         attrs={'operation': 'MODULO'})

    divide = nw.new_node(Nodes.Math,
                         input_kwargs={0: modulo, 1: add},
                         attrs={'operation': 'DIVIDE'})

    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: divide, 1: 6.28},
                           attrs={'operation': 'MULTIPLY'})

    combine_xyz = nw.new_node(Nodes.CombineXYZ,
                              input_kwargs={'Z': multiply})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Vector': combine_xyz})


@node_utils.to_nodegroup('nodegroup_truncated_leaf_stem', singleton=False, type='GeometryNodeTree')
def nodegroup_truncated_leaf_stem(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    curve_line = nw.new_node(Nodes.CurveLine,
                             input_kwargs={'End': (0.0, 0.0, 0.15)})

    integer = nw.new_node(Nodes.Integer,
                          attrs={'integer': 64})
    integer.integer = 64

    resample_curve_1 = nw.new_node(Nodes.ResampleCurve,
                                   input_kwargs={'Curve': curve_line, 'Count': integer})

    pedal_stem_curvature_scale = nw.new_node(Nodes.Value,
                                             label='pedal_stem_curvature_scale')
    pedal_stem_curvature_scale.outputs[0].default_value = 0.2

    pedal_stem_curvature = nw.new_node(nodegroup_pedal_stem_curvature().name,
                                       input_kwargs={'Value': pedal_stem_curvature_scale})

    set_position_4 = nw.new_node(Nodes.SetPosition,
                                 input_kwargs={'Geometry': resample_curve_1, 'Offset': pedal_stem_curvature})

    pedal_z_coutour_scale = nw.new_node(Nodes.Value,
                                        label='pedal_z_coutour_scale')
    pedal_z_coutour_scale.outputs[0].default_value = uniform(0.2, 0.4)

    pedal_z_contour = nw.new_node(nodegroup_pedal_z_contour().name,
                                  input_kwargs={'Value': pedal_z_coutour_scale})

    set_curve_radius_1 = nw.new_node(Nodes.SetCurveRadius,
                                     input_kwargs={'Curve': set_position_4, 'Radius': pedal_z_contour})

    coutour_cross_geometry = nw.new_node(nodegroup_coutour_cross_geometry().name)

    curve_to_mesh_1 = nw.new_node(Nodes.CurveToMesh,
                                  input_kwargs={'Curve': set_curve_radius_1, 'Profile Curve': coutour_cross_geometry,
                                                'Fill Caps': True})

    set_material_2 = nw.new_node(Nodes.SetMaterial,
                                 input_kwargs={'Geometry': curve_to_mesh_1,
                                               'Material': surface.shaderfunc_to_material(shader_top_core)})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': set_material_2})


@node_utils.to_nodegroup('nodegroup_trunk_radius', singleton=False, type='GeometryNodeTree')
def nodegroup_trunk_radius(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    random_value = nw.new_node(Nodes.RandomValue,
                               input_kwargs={2: 0.01, 3: 0.05})

    spline_parameter = nw.new_node(Nodes.SplineParameter)

    map_range = nw.new_node(Nodes.MapRange,
                            input_kwargs={'Value': spline_parameter.outputs["Factor"], 3: 1.0, 4: 0.2},
                            attrs={'clamp': False})

    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: spline_parameter.outputs["Factor"], 1: 10000.0},
                           attrs={'operation': 'MULTIPLY'})

    floor = nw.new_node(Nodes.Math,
                        input_kwargs={0: multiply},
                        attrs={'operation': 'FLOOR'})

    subtract = nw.new_node(Nodes.Math,
                           input_kwargs={0: multiply, 1: floor},
                           attrs={'operation': 'SUBTRACT'})

    float_curve = nw.new_node(Nodes.FloatCurve,
                              input_kwargs={'Value': subtract})
    node_utils.assign_curve(float_curve.mapping.curves[0], [(0.0, 0.0969), (0.5864, 0.1406), (1.0, 0.2906)])

    multiply_1 = nw.new_node(Nodes.Math,
                             input_kwargs={0: float_curve, 1: uniform(0.1, 0.25)},
                             attrs={'operation': 'MULTIPLY'})

    multiply_2 = nw.new_node(Nodes.Math,
                             input_kwargs={0: map_range.outputs["Result"], 1: multiply_1},
                             attrs={'operation': 'MULTIPLY'})

    add = nw.new_node(Nodes.Math,
                      input_kwargs={0: map_range.outputs["Result"], 1: multiply_2})

    add_1 = nw.new_node(Nodes.Math,
                        input_kwargs={0: random_value.outputs[1], 1: add})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Value': add_1})


@node_utils.to_nodegroup('nodegroup_tree_cracks', singleton=False, type='GeometryNodeTree')
def nodegroup_tree_cracks(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Geometry', None)])

    spline_parameter = nw.new_node(Nodes.SplineParameter)

    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
                                    input_kwargs={'Geometry': group_input.outputs["Geometry"],
                                                  2: spline_parameter.outputs["Length"]})

    position = nw.new_node(Nodes.InputPosition)

    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
                               input_kwargs={'Vector': position})

    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: capture_attribute.outputs[2], 1: uniform(0.1, 0.25)},
                           attrs={'operation': 'MULTIPLY'})

    combine_xyz = nw.new_node(Nodes.CombineXYZ,
                              input_kwargs={'X': separate_xyz.outputs["X"], 'Y': separate_xyz.outputs["Y"],
                                            'Z': multiply})

    voronoi_texture = nw.new_node(Nodes.VoronoiTexture,
                                  input_kwargs={'Vector': combine_xyz, 'Scale': 400.0, 'Randomness': 10.0},
                                  attrs={'voronoi_dimensions': '4D', 'distance': 'CHEBYCHEV'})

    colorramp = nw.new_node(Nodes.ColorRamp,
                            input_kwargs={'Fac': voronoi_texture.outputs["Distance"]})
    colorramp.color_ramp.elements[0].position = 0.6091
    colorramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp.color_ramp.elements[1].position = 0.6818
    colorramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    normal = nw.new_node(Nodes.InputNormal)

    multiply_1 = nw.new_node(Nodes.VectorMath,
                             input_kwargs={0: colorramp.outputs["Color"], 1: normal},
                             attrs={'operation': 'MULTIPLY'})

    multiply_2 = nw.new_node(Nodes.VectorMath,
                             input_kwargs={0: multiply_1.outputs["Vector"], 1: (-0.01, -0.01, -0.01)},
                             attrs={'operation': 'MULTIPLY'})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': capture_attribute.outputs["Geometry"],
                                             'Vector': multiply_2.outputs["Vector"]})


@node_utils.to_nodegroup('nodegroup_leaf_instance_selection_bottom_remove', singleton=False, type='GeometryNodeTree')
def nodegroup_leaf_instance_selection_bottom_remove(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    index_1 = nw.new_node(Nodes.Index)

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'Ring', 10.0),
                                            ('NodeSocketFloat', 'Segment', 0.5)])

    divide = nw.new_node(Nodes.Math,
                         input_kwargs={0: index_1, 1: group_input.outputs["Ring"]},
                         attrs={'operation': 'DIVIDE'})

    subtract = nw.new_node(Nodes.Math,
                           input_kwargs={0: group_input.outputs["Segment"], 1: 4.0},
                           attrs={'operation': 'SUBTRACT'})

    greater_than = nw.new_node(Nodes.Math,
                               input_kwargs={0: divide, 1: subtract},
                               attrs={'operation': 'GREATER_THAN'})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Value': greater_than})


@node_utils.to_nodegroup('nodegroup_leaf_random_rotate', singleton=False, type='GeometryNodeTree')
def nodegroup_leaf_random_rotate(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    random_value_1 = nw.new_node(Nodes.RandomValue,
                                 input_kwargs={2: -0.2, 3: 0.2})

    random_value_3 = nw.new_node(Nodes.RandomValue,
                                 input_kwargs={2: -0.2, 3: 0.2})

    random_value_2 = nw.new_node(Nodes.RandomValue,
                                 input_kwargs={2: -0.2, 3: 0.2})

    combine_xyz = nw.new_node(Nodes.CombineXYZ,
                              input_kwargs={'X': random_value_1.outputs[1], 'Y': random_value_3.outputs[1],
                                            'Z': random_value_2.outputs[1]})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Vector': combine_xyz})


@node_utils.to_nodegroup('nodegroup_leaf_rotate_downward', singleton=False, type='GeometryNodeTree')
def nodegroup_leaf_rotate_downward(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    index = nw.new_node(Nodes.Index)

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'Value', 0.5)])

    add = nw.new_node(Nodes.Math,
                      input_kwargs={0: group_input.outputs["Value"], 1: 0.0})

    modulo = nw.new_node(Nodes.Math,
                         input_kwargs={0: index, 1: add},
                         attrs={'operation': 'MODULO'})

    divide = nw.new_node(Nodes.Math,
                         input_kwargs={0: modulo, 1: add},
                         attrs={'operation': 'DIVIDE'})

    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: divide, 1: 6.28},
                           attrs={'operation': 'MULTIPLY'})

    combine_xyz = nw.new_node(Nodes.CombineXYZ,
                              input_kwargs={'Z': multiply})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Vector': combine_xyz})


def shader_coconut_green_shader(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    texture_coordinate_1 = nw.new_node(Nodes.TextureCoord)

    noise_texture_1 = nw.new_node(Nodes.NoiseTexture,
                                  input_kwargs={'Vector': texture_coordinate_1.outputs["Object"], 'Scale': 1.0,
                                                'Detail': 10.0, 'Roughness': 0.7})

    separate_rgb = nw.new_node(Nodes.SeparateColor,
                               input_kwargs={'Color': noise_texture_1.outputs["Color"]})

    map_range_1 = nw.new_node(Nodes.MapRange,
                              input_kwargs={'Value': separate_rgb.outputs["Green"], 1: 0.4, 2: 0.7, 3: 0.48, 4: 0.52},
                              attrs={'interpolation_type': 'SMOOTHSTEP'})

    map_range_2 = nw.new_node(Nodes.MapRange,
                              input_kwargs={'Value': separate_rgb.outputs["Blue"], 1: 0.4, 2: 0.7, 3: 0.6},
                              attrs={'interpolation_type': 'SMOOTHSTEP'})

    attribute_1 = nw.new_node(Nodes.Attribute,
                              attrs={'attribute_name': 'spline parameter'})

    colorramp = nw.new_node(Nodes.ColorRamp,
                            input_kwargs={'Fac': attribute_1.outputs["Fac"]})
    colorramp.color_ramp.elements.new(0)
    colorramp.color_ramp.elements[0].position = 0.0
    colorramp.color_ramp.elements[0].color = (0.0908, 0.2664, 0.013, 1.0)
    colorramp.color_ramp.elements[1].position = 0.01
    colorramp.color_ramp.elements[1].color = (0.0908, 0.2664, 0.013, 1.0)
    colorramp.color_ramp.elements[2].position = 1.0
    colorramp.color_ramp.elements[2].color = (0.2462, 0.4125, 0.0044, 1.0)

    hue_saturation_value_1 = nw.new_node('ShaderNodeHueSaturation',
                                         input_kwargs={'Hue': map_range_1.outputs["Result"],
                                                       'Value': map_range_2.outputs["Result"],
                                                       'Color': colorramp.outputs["Color"]})

    attribute_2 = nw.new_node(Nodes.Attribute,
                              attrs={'attribute_name': 'cross section parameter'})

    group = nw.new_node(nodegroup_node_group_002().name,
                        input_kwargs={'Color': hue_saturation_value_1, 'attribute': attribute_2.outputs["Fac"],
                                      'seed': 10.0})

    group_1 = nw.new_node(nodegroup_node_group_002().name,
                          input_kwargs={'Color': group, 'attribute': attribute_1.outputs["Fac"], 'voronoi scale': 10.0,
                                        'voronoi randomness': 0.6446, 'seed': -10.0, 'noise amount': 0.48,
                                        'hue min': 1.32, 'hue max': 0.9})

    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
                                  input_kwargs={'Base Color': group_1, 'Specular': 0.4773, 'Roughness': 0.4455})

    material_output = nw.new_node(Nodes.MaterialOutput,
                                  input_kwargs={'Surface': principled_bsdf})


@node_utils.to_nodegroup('nodegroup_coconut_vein_geometry', singleton=False, type='GeometryNodeTree')
def nodegroup_coconut_vein_geometry(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    quadratic_bezier = nw.new_node(Nodes.QuadraticBezier,
                                   input_kwargs={'Resolution': 400, 'Start': (0.0, 0.0, 0.0), 'Middle': (0.0, 0.2, 0.5),
                                                 'End': (0.0, 0.0, 1.0)})

    coconutvein = nw.new_node(nodegroup_coconutvein().name)

    set_position = nw.new_node(Nodes.SetPosition,
                               input_kwargs={'Geometry': quadratic_bezier, 'Offset': coconutvein})

    treetrunkgeometry_001 = nw.new_node(nodegroup_tree_trunk_geometry_001().name,
                                        input_kwargs={'Curve': set_position})

    transform_1 = nw.new_node(Nodes.Transform,
                              input_kwargs={'Geometry': treetrunkgeometry_001.outputs["Mesh"],
                                            'Translation': (0.0, 0.0, -0.1)})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': transform_1})


@node_utils.to_nodegroup('nodegroup_coconut_random_rotate', singleton=False, type='GeometryNodeTree')
def nodegroup_coconut_random_rotate(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    random_value_2 = nw.new_node(Nodes.RandomValue,
                                 input_kwargs={2: -0.2, 3: 0.2})

    random_value_3 = nw.new_node(Nodes.RandomValue,
                                 input_kwargs={2: -0.2, 3: 0.2})

    random_value_4 = nw.new_node(Nodes.RandomValue,
                                 input_kwargs={2: -0.2, 3: 0.2})

    combine_xyz = nw.new_node(Nodes.CombineXYZ,
                              input_kwargs={'X': random_value_2.outputs[1], 'Y': random_value_3.outputs[1],
                                            'Z': random_value_4.outputs[1]})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Vector': combine_xyz})


@node_utils.to_nodegroup('nodegroup_truncated_stem_geometry', singleton=False, type='GeometryNodeTree')
def nodegroup_truncated_stem_geometry(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Points', None),
                                            ('NodeSocketFloat', 'Value1', 0.5),
                                            ('NodeSocketFloat', 'Value2', 0.5)])

    truncated_leaf_stem = nw.new_node(nodegroup_truncated_leaf_stem().name)

    normal_1 = nw.new_node(Nodes.InputNormal)

    align_euler_to_vector_1 = nw.new_node(Nodes.AlignEulerToVector,
                                          input_kwargs={'Vector': normal_1},
                                          attrs={'axis': 'Z'})

    instance_on_points_2 = nw.new_node(Nodes.InstanceOnPoints,
                                       input_kwargs={'Points': group_input.outputs["Points"],
                                                     'Instance': truncated_leaf_stem,
                                                     'Rotation': align_euler_to_vector_1})

    leaf_truncated_rotate = nw.new_node(nodegroup_leaf_truncated_rotate().name,
                                        input_kwargs={'Value': group_input.outputs[2]})

    rotate_instances_2 = nw.new_node(Nodes.RotateInstances,
                                     input_kwargs={'Instances': instance_on_points_2,
                                                   'Rotation': leaf_truncated_rotate})

    rotate_instances_3 = nw.new_node(Nodes.RotateInstances,
                                     input_kwargs={'Instances': rotate_instances_2, 'Rotation': (-0.9599, 0.0, 1.5708)})

    random_rotate = nw.new_node(nodegroup_random_rotate().name)

    rotate_instances_4 = nw.new_node(Nodes.RotateInstances,
                                     input_kwargs={'Instances': rotate_instances_3, 'Rotation': random_rotate})

    random_value_5 = nw.new_node(Nodes.RandomValue,
                                 input_kwargs={2: 0.6})

    scale_instances_4 = nw.new_node(Nodes.ScaleInstances,
                                    input_kwargs={'Instances': rotate_instances_4, 'Scale': random_value_5.outputs[1]})

    index_2 = nw.new_node(Nodes.Index)

    modulo = nw.new_node(Nodes.Math,
                         input_kwargs={0: index_2, 1: randint(8, 12)},
                         attrs={'operation': 'MODULO'})

    scale_instances_3 = nw.new_node(Nodes.ScaleInstances,
                                    input_kwargs={'Instances': scale_instances_4, 'Selection': modulo,
                                                  'Scale': (0.0, 0.0, 0.0)})

    truncated_leaf_selection = nw.new_node(nodegroup_truncated_leaf_selection().name,
                                           input_kwargs={'Value': group_input.outputs["Value1"]})

    scale_instances_5 = nw.new_node(Nodes.ScaleInstances,
                                    input_kwargs={'Instances': scale_instances_3, 'Selection': truncated_leaf_selection,
                                                  'Scale': (0.0, 0.0, 0.0)})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Instances': scale_instances_5})


@node_utils.to_nodegroup('nodegroup_tree_trunk_geometry', singleton=False, type='GeometryNodeTree')
def nodegroup_tree_trunk_geometry(nw: NodeWrangler, radius):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Curve', None)])

    trunkradius = nw.new_node(nodegroup_trunk_radius().name)

    set_curve_radius = nw.new_node(Nodes.SetCurveRadius,
                                   input_kwargs={'Curve': group_input.outputs["Curve"], 'Radius': trunkradius})

    treecracks = nw.new_node(nodegroup_tree_cracks().name,
                             input_kwargs={'Geometry': set_curve_radius})

    trunk_resolution = nw.new_node(Nodes.Integer,
                                   label='TrunkResolution',
                                   attrs={'integer': 32})
    trunk_resolution.integer = 32

    trunk_radius = nw.new_node(Nodes.Value,
                               label='TrunkRadius')
    trunk_radius.outputs[0].default_value = radius

    curve_circle = nw.new_node(Nodes.CurveCircle,
                               input_kwargs={'Resolution': trunk_resolution, 'Radius': trunk_radius})

    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
                                input_kwargs={'Curve': treecracks.outputs["Geometry"],
                                              'Profile Curve': curve_circle.outputs["Curve"], 'Fill Caps': True})

    subdivide_mesh = nw.new_node(Nodes.SubdivideMesh,
                                 input_kwargs={'Mesh': curve_to_mesh, 'Level': 5})

    set_position_1 = nw.new_node(Nodes.SetPosition,
                                 input_kwargs={'Geometry': subdivide_mesh, 'Offset': treecracks.outputs["Vector"]})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': set_position_1, 'Integer': trunk_resolution,
                                             'Mesh': curve_to_mesh})


@node_utils.to_nodegroup('nodegroup_leaf_on_top', singleton=False, type='GeometryNodeTree')
def nodegroup_leaf_on_top(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Points', None),
                                            ('NodeSocketFloat', 'Value', 0.5),
                                            ('NodeSocketFloat', 'Ring', 10.0),
                                            ('NodeSocketFloat', 'Segment', 0.5),
                                            ('NodeSocketGeometry', 'Instance', None)])

    normal = nw.new_node(Nodes.InputNormal)

    align_euler_to_vector = nw.new_node(Nodes.AlignEulerToVector,
                                        input_kwargs={'Vector': normal},
                                        attrs={'axis': 'Z'})

    instance_on_points_1 = nw.new_node(Nodes.InstanceOnPoints,
                                       input_kwargs={'Points': group_input.outputs["Points"],
                                                     'Instance': group_input.outputs["Instance"],
                                                     'Rotation': align_euler_to_vector})

    leafrotatedownward = nw.new_node(nodegroup_leaf_rotate_downward().name,
                                     input_kwargs={'Value': group_input.outputs["Value"]})

    rotate_instances = nw.new_node(Nodes.RotateInstances,
                                   input_kwargs={'Instances': instance_on_points_1, 'Rotation': leafrotatedownward})

    leafrandomrotate = nw.new_node(nodegroup_leaf_random_rotate().name)

    rotate_instances_1 = nw.new_node(Nodes.RotateInstances,
                                     input_kwargs={'Instances': rotate_instances, 'Rotation': leafrandomrotate})

    random_value_4 = nw.new_node(Nodes.RandomValue,
                                 input_kwargs={2: 0.9, 3: 1.2})

    scale_instances_2 = nw.new_node(Nodes.ScaleInstances,
                                    input_kwargs={'Instances': rotate_instances_1, 'Scale': random_value_4.outputs[1]})

    leafinstanceselectionbottomremove = nw.new_node(nodegroup_leaf_instance_selection_bottom_remove().name,
                                                    input_kwargs={'Ring': group_input.outputs["Ring"],
                                                                  'Segment': group_input.outputs["Segment"]})

    scale_instances = nw.new_node(Nodes.ScaleInstances,
                                  input_kwargs={'Instances': scale_instances_2,
                                                'Selection': leafinstanceselectionbottomremove,
                                                'Scale': (0.0, 0.0, 0.0)})

    random_value = nw.new_node(Nodes.RandomValue,
                               input_kwargs={5: 1},
                               attrs={'data_type': 'INT'})

    scale_instances_1 = nw.new_node(Nodes.ScaleInstances,
                                    input_kwargs={'Instances': scale_instances, 'Selection': random_value.outputs[2],
                                                  'Scale': (0.0, 0.0, 0.0)})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Instances': scale_instances_1})


@node_utils.to_nodegroup('nodegroup_coconut_instance_on_points', singleton=False, type='GeometryNodeTree')
def nodegroup_coconut_instance_on_points(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    index_1 = nw.new_node(Nodes.Index)

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'Ring', 0.5),
                                            ('NodeSocketFloat', 'Segment', 0.5)])

    add = nw.new_node(Nodes.Math,
                      input_kwargs={0: group_input.outputs["Segment"], 1: 0.0})

    divide = nw.new_node(Nodes.Math,
                         input_kwargs={0: index_1, 1: add},
                         attrs={'operation': 'DIVIDE'})

    add_1 = nw.new_node(Nodes.Math,
                        input_kwargs={0: group_input.outputs["Ring"], 1: 0.0})

    subtract = nw.new_node(Nodes.Math,
                           input_kwargs={0: add_1, 1: 4.0},
                           attrs={'operation': 'SUBTRACT'})

    greater_than = nw.new_node(Nodes.Math,
                               input_kwargs={0: divide, 1: subtract},
                               attrs={'operation': 'GREATER_THAN'})

    subtract_1 = nw.new_node(Nodes.Math,
                             input_kwargs={0: add_1, 1: 2.0},
                             attrs={'operation': 'SUBTRACT'})

    less_than = nw.new_node(Nodes.Math,
                            input_kwargs={0: divide, 1: subtract_1},
                            attrs={'operation': 'LESS_THAN'})

    op_and = nw.new_node(Nodes.BooleanMath,
                         input_kwargs={0: greater_than, 1: less_than})

    op_not = nw.new_node(Nodes.BooleanMath,
                         input_kwargs={0: op_and},
                         attrs={'operation': 'NOT'})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Boolean': op_not})


@node_utils.to_nodegroup('nodegroup_coconut_group', singleton=False, type='GeometryNodeTree')
def nodegroup_coconut_group(nw: NodeWrangler, coconut):
    # Code generated using version 2.4.3 of the node_transpiler

    uv_sphere_1 = nw.new_node(Nodes.MeshUVSphere,
                              input_kwargs={'Segments': 8, 'Rings': 6, 'Radius': 0.15})

    object_info_2 = nw.new_node(Nodes.ObjectInfo, input_kwargs={'Object': coconut})

    transform_2 = nw.new_node(Nodes.Transform,
                              input_kwargs={'Geometry': object_info_2.outputs["Geometry"],
                                            'Translation': (0.0, 0.0, -1.2)})

    normal_1 = nw.new_node(Nodes.InputNormal)

    align_euler_to_vector_1 = nw.new_node(Nodes.AlignEulerToVector,
                                          input_kwargs={'Vector': normal_1},
                                          attrs={'axis': 'Z'})

    instance_on_points_3 = nw.new_node(Nodes.InstanceOnPoints,
                                       input_kwargs={'Points': uv_sphere_1, 'Instance': transform_2,
                                                     'Rotation': align_euler_to_vector_1, 'Scale': (-1.0, -1.0, -1.0)})

    coconut_random_rotate = nw.new_node(nodegroup_coconut_random_rotate().name)

    rotate_instances = nw.new_node(Nodes.RotateInstances,
                                   input_kwargs={'Instances': instance_on_points_3, 'Rotation': coconut_random_rotate})

    random_value_2 = nw.new_node(Nodes.RandomValue,
                                 input_kwargs={2: 0.15, 3: 0.4})

    scale_instances_6 = nw.new_node(Nodes.ScaleInstances,
                                    input_kwargs={'Instances': rotate_instances, 'Scale': random_value_2.outputs[1]})

    index = nw.new_node(Nodes.Index)

    less_than = nw.new_node(Nodes.Math,
                            input_kwargs={0: index, 1: 20.0},
                            attrs={'operation': 'LESS_THAN'})

    scale_instances_2 = nw.new_node(Nodes.ScaleInstances,
                                    input_kwargs={'Instances': scale_instances_6, 'Selection': less_than,
                                                  'Scale': (0.0, 0.0, 0.0)})

    random_value_1 = nw.new_node(Nodes.RandomValue,
                                 input_kwargs={5: 2, 'Seed': 2},
                                 attrs={'data_type': 'INT'})

    scale_instances_4 = nw.new_node(Nodes.ScaleInstances,
                                    input_kwargs={'Instances': scale_instances_2,
                                                  'Selection': random_value_1.outputs[2], 'Scale': (0.0, 0.0, 0.0)})

    coconut_vein_geometry = nw.new_node(nodegroup_coconut_vein_geometry().name)

    normal_2 = nw.new_node(Nodes.InputNormal)

    align_euler_to_vector_2 = nw.new_node(Nodes.AlignEulerToVector,
                                          input_kwargs={'Vector': normal_2},
                                          attrs={'axis': 'Z'})

    instance_on_points_2 = nw.new_node(Nodes.InstanceOnPoints,
                                       input_kwargs={'Points': uv_sphere_1, 'Instance': coconut_vein_geometry,
                                                     'Rotation': align_euler_to_vector_2})

    index_2 = nw.new_node(Nodes.Index)

    less_than_1 = nw.new_node(Nodes.Math,
                              input_kwargs={0: index_2, 1: 30.0},
                              attrs={'operation': 'LESS_THAN'})

    scale_instances_3 = nw.new_node(Nodes.ScaleInstances,
                                    input_kwargs={'Instances': instance_on_points_2, 'Selection': less_than_1,
                                                  'Scale': (0.0, 0.0, 0.0)})

    random_value_5 = nw.new_node(Nodes.RandomValue,
                                 input_kwargs={5: 1, 'Seed': 4},
                                 attrs={'data_type': 'INT'})

    scale_instances_5 = nw.new_node(Nodes.ScaleInstances,
                                    input_kwargs={'Instances': scale_instances_3,
                                                  'Selection': random_value_5.outputs[2], 'Scale': (0.0, 0.0, 0.0)})

    set_material_2 = nw.new_node(Nodes.SetMaterial,
                                 input_kwargs={'Geometry': scale_instances_5,
                                               'Material': surface.shaderfunc_to_material(shader_coconut_green_shader)})

    join_geometry_2 = nw.new_node(Nodes.JoinGeometry,
                                  input_kwargs={'Geometry': [scale_instances_4, set_material_2]})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': join_geometry_2})


def shader_top_core(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    texture_coordinate = nw.new_node(Nodes.TextureCoord)

    mapping = nw.new_node(Nodes.Mapping,
                          input_kwargs={'Vector': texture_coordinate.outputs["Object"], 'Scale': (1.0, 1.0, 0.1)})

    voronoi_texture = nw.new_node(Nodes.VoronoiTexture,
                                  input_kwargs={'Vector': mapping, 'Scale': uniform(100, 400)})

    mapping_1 = nw.new_node(Nodes.Mapping,
                            input_kwargs={'Vector': texture_coordinate.outputs["Object"]})

    wave_texture = nw.new_node(Nodes.WaveTexture,
                               input_kwargs={'Vector': mapping_1, 'Scale': 2.0, 'Distortion': 5.0, 'Detail': 10.0})

    mix = nw.new_node(Nodes.MixRGB,
                      input_kwargs={'Fac': 0.4, 'Color1': voronoi_texture.outputs["Distance"],
                                    'Color2': wave_texture.outputs["Color"]})

    d_hsv = (uniform(0.02, 0.05), uniform(0.3, 0.6), uniform(0.01, 0.05))
    b_hsv = d_hsv[:1] + (uniform(0.6, 0.9), uniform(0.3, 0.6))
    colorramp = nw.new_node(Nodes.ColorRamp,
                            input_kwargs={'Fac': mix})
    colorramp.color_ramp.elements[0].position = 0.2409
    colorramp.color_ramp.elements[0].color = hsv2rgba(d_hsv)
    colorramp.color_ramp.elements[1].position = 0.6045
    colorramp.color_ramp.elements[1].color = hsv2rgba(b_hsv)

    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
                                  input_kwargs={'Base Color': colorramp.outputs["Color"],
                                                'Roughness': colorramp.outputs["Alpha"]})

    material_output = nw.new_node(Nodes.MaterialOutput,
                                  input_kwargs={'Surface': principled_bsdf})


def shader_trunk(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    texture_coordinate = nw.new_node(Nodes.TextureCoord)

    mapping = nw.new_node(Nodes.Mapping,
                          input_kwargs={'Vector': texture_coordinate.outputs["Object"]})

    voronoi_texture_1 = nw.new_node(Nodes.VoronoiTexture,
                                    input_kwargs={'Vector': mapping, 'Scale': 20.0},
                                    attrs={'voronoi_dimensions': '4D'})

    wave_texture = nw.new_node(Nodes.WaveTexture,
                               input_kwargs={'Vector': mapping, 'Scale': uniform(1.0, 3.0), 'Distortion': 5.0, 'Detail Scale': 3.0},
                               attrs={'bands_direction': 'Z'})

    mix_1 = nw.new_node(Nodes.MixRGB,
                        input_kwargs={'Color1': voronoi_texture_1.outputs["Distance"],
                                      'Color2': wave_texture.outputs["Color"]})

    d_hsv = (uniform(0.02, 0.05), uniform(0.01, 0.05) if randint(0, 2) == 1 else uniform(0.5, 0.8), uniform(0.03, 0.09))
    b_hsv = d_hsv[:-1] + (uniform(0.1, 0.3),)
    colorramp = nw.new_node(Nodes.ColorRamp,
                            input_kwargs={'Fac': mix_1})
    colorramp.color_ramp.elements[0].position = 0.4682
    colorramp.color_ramp.elements[0].color = hsv2rgba(d_hsv)
    colorramp.color_ramp.elements[1].position = 0.5591
    colorramp.color_ramp.elements[1].color = hsv2rgba(b_hsv)

    mapping_1 = nw.new_node(Nodes.Mapping,
                            input_kwargs={'Vector': texture_coordinate.outputs["Object"], 'Scale': (10.0, 10.0, 0.2)})

    voronoi_texture = nw.new_node(Nodes.VoronoiTexture,
                                  input_kwargs={'Vector': mapping_1, 'Scale': 100.0, 'Randomness': 10.0},
                                  attrs={'voronoi_dimensions': '4D', 'distance': 'CHEBYCHEV'})

    colorramp_1 = nw.new_node(Nodes.ColorRamp,
                              input_kwargs={'Fac': voronoi_texture.outputs["Distance"]})
    colorramp_1.color_ramp.elements[0].position = 0.2818
    colorramp_1.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_1.color_ramp.elements[1].position = 0.3045
    colorramp_1.color_ramp.elements[1].color = (0.5284, 0.5034, 0.4327, 1.0)

    mix = nw.new_node(Nodes.MixRGB,
                      input_kwargs={'Fac': uniform(0.1, 0.3), 'Color1': colorramp.outputs["Color"],
                                    'Color2': colorramp_1.outputs["Color"]})

    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
                                  input_kwargs={'Base Color': mix, 'Roughness': voronoi_texture.outputs["Distance"]})

    material_output = nw.new_node(Nodes.MaterialOutput,
                                  input_kwargs={'Surface': principled_bsdf})


def geometry_coconut_tree_nodes(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    leaf = kwargs["leaf"][0]
    coconut = kwargs["coconut"][0]
    radius = kwargs["trunk_radius"]

    trunk_height = nw.new_node(Nodes.Value,
                               label='trunk_height')
    trunk_height.outputs[0].default_value = 5.0

    top_x, top_y = np.random.normal(0.0, 1.), np.random.normal(0.0, 1.)
    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ,
                                input_kwargs={'X': top_x, 'Y': top_y, 'Z': trunk_height})

    quadratic_bezier = nw.new_node(Nodes.QuadraticBezier,
                                   input_kwargs={'Start': (0.0, 0.0, 0.0),
                                                 'Middle': (top_x / uniform(1.0, 2.0), top_y / uniform(1.0, 2.0), uniform(1.5, 3.0)),
                                                 'End': combine_xyz_2})

    resample_curve = nw.new_node(Nodes.ResampleCurve,
                                 input_kwargs={'Curve': quadratic_bezier, 'Length': 0.02}, #'Count': 20000
                                 attrs={'mode': 'LENGTH'})

    set_position = nw.new_node(Nodes.SetPosition,
                               input_kwargs={'Geometry': resample_curve})

    endpoint_selection = nw.new_node('GeometryNodeCurveEndpointSelection',
                                     input_kwargs={'Start Size': 0})

    top_segment = nw.new_node(Nodes.Integer,
                              label='TopSegment',
                              attrs={'integer': 12})
    top_segment.integer = randint(8, 14)

    top_ring = nw.new_node(Nodes.Integer,
                           label='TopRing',
                           attrs={'integer': 8})
    top_ring.integer = randint(8, 11)

    uv_sphere = nw.new_node(Nodes.MeshUVSphere,
                            input_kwargs={'Segments': top_segment, 'Rings': top_ring, 'Radius': uniform(0.15, 0.2)})

    transform = nw.new_node(Nodes.Transform,
                            input_kwargs={'Geometry': uv_sphere, 'Scale': (1.0, 1.0, uniform(0.8, 2.0))})

    set_material_1 = nw.new_node(Nodes.SetMaterial,
                                 input_kwargs={'Geometry': transform,
                                               'Material': surface.shaderfunc_to_material(shader_top_core)})

    coconut_group = nw.new_node(nodegroup_coconut_group(coconut=coconut).name)

    transform_1 = nw.new_node(Nodes.Transform,
                              input_kwargs={'Geometry': coconut_group, 'Scale': (-1.0, -1.0, -1.0)})

    normal = nw.new_node(Nodes.InputNormal)

    align_euler_to_vector = nw.new_node(Nodes.AlignEulerToVector,
                                        input_kwargs={'Vector': normal},
                                        attrs={'axis': 'Z'})

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.2

    instance_on_points_1 = nw.new_node(Nodes.InstanceOnPoints,
                                       input_kwargs={'Points': transform, 'Instance': transform_1,
                                                     'Rotation': align_euler_to_vector, 'Scale': value})

    random_value = nw.new_node(Nodes.RandomValue,
                               input_kwargs={5: randint(1, 3)},
                               attrs={'data_type': 'INT'})

    scale_instances = nw.new_node(Nodes.ScaleInstances,
                                  input_kwargs={'Instances': instance_on_points_1, 'Selection': random_value.outputs[2],
                                                'Scale': (0.0, 0.0, 0.0)})

    coconut_instance_on_points = nw.new_node(nodegroup_coconut_instance_on_points().name,
                                             input_kwargs={'Ring': top_ring, 'Segment': top_segment})

    scale_instances_1 = nw.new_node(Nodes.ScaleInstances,
                                    input_kwargs={'Instances': scale_instances, 'Selection': coconut_instance_on_points,
                                                  'Scale': (0.0, 0.0, 0.0)})

    object_info = nw.new_node(Nodes.ObjectInfo, input_kwargs={'Object': leaf})

    leafontop = nw.new_node(nodegroup_leaf_on_top().name,
                            input_kwargs={'Points': transform, 'Value': top_segment, 'Ring': top_segment,
                                          'Segment': top_ring, 'Instance': object_info.outputs["Geometry"]})

    join_geometry_1 = nw.new_node(Nodes.JoinGeometry,
                                  input_kwargs={'Geometry': [set_material_1, scale_instances_1, leafontop]})

    instance_on_points = nw.new_node(Nodes.InstanceOnPoints,
                                     input_kwargs={'Points': set_position, 'Selection': endpoint_selection,
                                                   'Instance': join_geometry_1})

    treetrunkgeometry = nw.new_node(nodegroup_tree_trunk_geometry(radius=radius).name,
                                    input_kwargs={'Curve': set_position})

    set_material = nw.new_node(Nodes.SetMaterial,
                               input_kwargs={'Geometry': treetrunkgeometry.outputs["Geometry"],
                                             'Material': surface.shaderfunc_to_material(shader_trunk)})

    truncatedstemgeometry = nw.new_node(nodegroup_truncated_stem_geometry().name,
                                        input_kwargs={'Points': treetrunkgeometry.outputs["Mesh"], 1: trunk_height,
                                                      2: treetrunkgeometry.outputs["Integer"]})

    geos = [instance_on_points, set_material]
    if uniform(0.0, 1.0) < 0.3:
        geos.append(truncatedstemgeometry)
    join_geometry = nw.new_node(Nodes.JoinGeometry,
                                input_kwargs={'Geometry': geos})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': join_geometry})


class CoconutTreeFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super(CoconutTreeFactory, self).__init__(factory_seed, coarse=coarse)

    def create_asset(self, params={}, **kwargs):
        bpy.ops.mesh.primitive_plane_add(
            size=1, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        obj = bpy.context.active_object

        # Make the Leaf and Delete It Later
        lf_seed = randint(0, 1000, size=(1,))[0]
        leaf_model = LeafPalmTreeFactory(factory_seed=lf_seed)
        p = {
            'leaf_x_curvature': uniform(0.3, 0.8)
        }
        leaf = leaf_model.create_asset(p)
        params["leaf"] = [leaf]

        co_seed = randint(0, 1000, size=(1,))[0]
        coconut_model = FruitFactoryCoconutgreen(factory_seed=co_seed)
        coconut = coconut_model.create_asset()
        params["coconut"] = [coconut]
        params["trunk_radius"] = uniform(0.2, 0.3)

        surface.add_geomod(obj, geometry_coconut_tree_nodes, selection=None, attributes=[], input_kwargs=params)
        butil.delete([leaf, coconut])
        with butil.SelectObjects(obj):
            bpy.ops.object.material_slot_remove()
            bpy.ops.object.shade_flat()

        return obj


if __name__ == '__main__':
    model = CoconutTreeFactory(0)
    model.create_asset()
