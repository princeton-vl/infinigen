# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Beining Han


import bpy
import mathutils
from numpy.random import uniform, normal, randint
from infinigen.assets.grassland.flower import FlowerFactory
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
import numpy as np
from infinigen.core import surface
from infinigen.assets.materials import simple_greenery
from infinigen.assets.small_plants import leaf_general as Leaf
from infinigen.assets.grassland import flower as Flower
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.assets.utils.tag import tag_object, tag_nodegroup


@node_utils.to_nodegroup('nodegroup_stem_branch_leaf_s_r', singleton=False, type='GeometryNodeTree')
def nodegroup_stem_branch_leaf_s_r(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    random_value = nw.new_node(Nodes.RandomValue,
                               input_kwargs={2: 0.2, 3: 0.7})

    curve_tangent_1 = nw.new_node(Nodes.CurveTangent)

    align_euler_to_vector_1 = nw.new_node(Nodes.AlignEulerToVector,
                                          input_kwargs={'Vector': curve_tangent_1},
                                          attrs={'axis': 'Z'})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Value': random_value.outputs[1], 'Rotation': align_euler_to_vector_1})


@node_utils.to_nodegroup('nodegroup_stem_branch_leaf_selection', singleton=False, type='GeometryNodeTree')
def nodegroup_stem_branch_leaf_selection(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    spline_parameter_1 = nw.new_node(Nodes.SplineParameter)

    colorramp_1 = nw.new_node(Nodes.ColorRamp,
                              input_kwargs={'Fac': spline_parameter_1.outputs["Factor"]})
    colorramp_1.color_ramp.interpolation = "CONSTANT"
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements[0].position = 0.0
    colorramp_1.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_1.color_ramp.elements[1].position = 0.20
    colorramp_1.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    colorramp_1.color_ramp.elements[2].position = 0.80
    colorramp_1.color_ramp.elements[2].color = (0.0, 0.0, 0.0, 1.0)

    integer = randint(10, 30, size=(1,))[0]
    random_value_3 = nw.new_node(Nodes.RandomValue,
                                 input_kwargs={5: integer},
                                 attrs={'data_type': 'INT'})

    op_not = nw.new_node(Nodes.BooleanMath,
                         input_kwargs={0: random_value_3.outputs[2]},
                         attrs={'operation': 'NOT'})

    op_and = nw.new_node(Nodes.BooleanMath,
                         input_kwargs={0: colorramp_1.outputs["Color"], 1: op_not})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Boolean': op_and})


@node_utils.to_nodegroup('nodegroup_stem_branch_leaves', singleton=False, type='GeometryNodeTree')
def nodegroup_stem_branch_leaves(nw: NodeWrangler, leaves):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Curve', None)])

    resample_curve_3 = nw.new_node(Nodes.ResampleCurve,
                                   input_kwargs={'Curve': group_input.outputs["Curve"], 'Count': 100})

    stembranchleafselection = nw.new_node(nodegroup_stem_branch_leaf_selection().name)

    leaf_id = randint(0, len(leaves), size=(1,))[0]
    leaf = leaves[leaf_id]
    object_info_2 = nw.new_node(Nodes.ObjectInfo, input_kwargs={'Object': leaf})

    stembranchleafsr = nw.new_node(nodegroup_stem_branch_leaf_s_r().name)

    instance_on_points_4 = nw.new_node(Nodes.InstanceOnPoints,
                                       input_kwargs={'Points': resample_curve_3, 'Selection': stembranchleafselection,
                                                     'Instance': object_info_2.outputs["Geometry"],
                                                     'Rotation': stembranchleafsr.outputs["Rotation"],
                                                     'Scale': stembranchleafsr.outputs["Value"]})

    random_value_3 = nw.new_node(Nodes.RandomValue,
                                 input_kwargs={'Max': (0.6, 0.6, 6.28), 'Seed': 30},
                                 attrs={'data_type': 'FLOAT_VECTOR'})

    rotate_instances_2 = nw.new_node(Nodes.RotateInstances,
                                     input_kwargs={'Instances': instance_on_points_4,
                                                   'Rotation': random_value_3.outputs["Value"]})

    realize_instances = nw.new_node(Nodes.RealizeInstances,
        input_kwargs={'Geometry': rotate_instances_2})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Instances': realize_instances})


@node_utils.to_nodegroup('nodegroup_stem_branch_geometry', singleton=False, type='GeometryNodeTree')
def nodegroup_stem_branch_geometry(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Curve', None)])

    spline_parameter = nw.new_node(Nodes.SplineParameter)

    colorramp = nw.new_node(Nodes.ColorRamp,
                            input_kwargs={'Fac': spline_parameter.outputs["Factor"]})
    colorramp.color_ramp.elements[0].position = 0.0
    colorramp.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    colorramp.color_ramp.elements[1].position = 1.0
    colorramp.color_ramp.elements[1].color = (0.4, 0.4, 0.4, 1.0)

    set_curve_radius = nw.new_node(Nodes.SetCurveRadius,
                                   input_kwargs={'Curve': group_input.outputs["Curve"],
                                                 'Radius': colorramp.outputs["Color"]})

    r = uniform(0.015, 0.022, size=(1,))[0]
    curve_circle = nw.new_node(Nodes.CurveCircle,
                               input_kwargs={'Resolution': 10, 'Radius': r})

    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
                                input_kwargs={'Curve': set_curve_radius, 'Profile Curve': curve_circle.outputs["Curve"],
                                              'Fill Caps': True})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Mesh': curve_to_mesh})


@node_utils.to_nodegroup('nodegroup_stem_branch_rotation', singleton=False, type='GeometryNodeTree')
def nodegroup_stem_branch_rotation(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    position = nw.new_node(Nodes.InputPosition)

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Geometry', None)])

    bounding_box = nw.new_node(Nodes.BoundingBox,
                               input_kwargs={'Geometry': group_input.outputs["Geometry"]})

    multiply = nw.new_node(Nodes.VectorMath,
                           input_kwargs={0: bounding_box.outputs["Max"], 1: (0.0, 0.0, 1.0)},
                           attrs={'operation': 'MULTIPLY'})

    index = nw.new_node(Nodes.Index)

    map_range = nw.new_node(Nodes.MapRange,
                            input_kwargs={'Value': index, 2: 20.0})

    curvature = uniform(-0.5, 0.5, (1,))[0]
    float_curve = nw.new_node(Nodes.FloatCurve,
                              input_kwargs={'Value': map_range.outputs["Result"]})
    node_utils.assign_curve(float_curve.mapping.curves[0],
                            [(0.0, 0.5), (0.1, curvature / 5. + 0.5),
                             (0.25, curvature / 2.5 + 0.5), (0.45, curvature / 1.5 + 0.5),
                             (0.6, curvature / 1.2 + 0.5), (1.0, curvature + 0.5)])

    add = nw.new_node(Nodes.Math,
                      input_kwargs={0: float_curve, 1: -0.5})

    multiply_1 = nw.new_node(Nodes.Math,
                             input_kwargs={0: add, 1: 1.0},
                             attrs={'operation': 'MULTIPLY'})

    vector_rotate = nw.new_node(Nodes.VectorRotate,
                                input_kwargs={'Vector': position, 'Center': multiply.outputs["Vector"],
                                              'Angle': multiply_1},
                                attrs={'rotation_type': 'X_AXIS'})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Vector': vector_rotate})


@node_utils.to_nodegroup('nodegroup_stem_leaf_s_r', singleton=False, type='GeometryNodeTree')
def nodegroup_stem_leaf_s_r(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    random_value = nw.new_node(Nodes.RandomValue,
                               input_kwargs={2: 0.3, 3: 0.6})

    curve_tangent_1 = nw.new_node(Nodes.CurveTangent)

    align_euler_to_vector_1 = nw.new_node(Nodes.AlignEulerToVector,
                                          input_kwargs={'Vector': curve_tangent_1},
                                          attrs={'axis': 'Z'})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Value': random_value.outputs[1], 'Rotation': align_euler_to_vector_1})


@node_utils.to_nodegroup('nodegroup_stem_leaf_selection', singleton=False, type='GeometryNodeTree')
def nodegroup_stem_leaf_selection(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    spline_parameter_1 = nw.new_node(Nodes.SplineParameter)

    colorramp_1 = nw.new_node(Nodes.ColorRamp,
                              input_kwargs={'Fac': spline_parameter_1.outputs["Factor"]})
    colorramp_1.color_ramp.interpolation = "CONSTANT"
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements[0].position = 0.0
    colorramp_1.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_1.color_ramp.elements[1].position = 0.30
    colorramp_1.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    colorramp_1.color_ramp.elements[2].position = 0.85
    colorramp_1.color_ramp.elements[2].color = (0.0, 0.0, 0.0, 1.0)

    integer = randint(5, 15, size=(1,))[0]
    random_value_3 = nw.new_node(Nodes.RandomValue,
                                 input_kwargs={5: integer},
                                 attrs={'data_type': 'INT'})

    op_not = nw.new_node(Nodes.BooleanMath,
                         input_kwargs={0: random_value_3.outputs[2]},
                         attrs={'operation': 'NOT'})

    op_and = nw.new_node(Nodes.BooleanMath,
                         input_kwargs={0: colorramp_1.outputs["Color"], 1: op_not})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Boolean': op_and})


@node_utils.to_nodegroup('nodegroup_stem_branch', singleton=False, type='GeometryNodeTree')
def nodegroup_stem_branch(nw: NodeWrangler, flowers, leaves):
    # Code generated using version 2.4.3 of the node_transpiler

    curve_line_2 = nw.new_node(Nodes.CurveLine)

    resample_curve_4 = nw.new_node(Nodes.ResampleCurve,
                                   input_kwargs={'Curve': curve_line_2, 'Count': 20})

    stembranchrotation = nw.new_node(nodegroup_stem_branch_rotation().name)

    set_position_2 = nw.new_node(Nodes.SetPosition,
                                 input_kwargs={'Geometry': resample_curve_4, 'Position': stembranchrotation})

    branchflowersetting = nw.new_node(nodegroup_branch_flower_setting(flowers=flowers).name)

    instance_on_points_3 = nw.new_node(Nodes.InstanceOnPoints,
                                       input_kwargs={'Points': set_position_2,
                                                     'Selection': branchflowersetting.outputs["Selection"],
                                                     'Instance': branchflowersetting.outputs["Geometry"],
                                                     'Rotation': branchflowersetting.outputs["Rotation"],
                                                     'Scale': branchflowersetting.outputs["Value"]})

    random_value = nw.new_node(Nodes.RandomValue,
                               input_kwargs={2: 0.4, 3: 0.7})

    scale_instances = nw.new_node(Nodes.ScaleInstances,
                                  input_kwargs={'Instances': instance_on_points_3, 'Scale': random_value.outputs[1]})

    stembranchgeometry = nw.new_node(nodegroup_stem_branch_geometry().name,
                                     input_kwargs={'Curve': set_position_2})

    stembranchleaves = nw.new_node(nodegroup_stem_branch_leaves(leaves=leaves).name,
                                   input_kwargs={'Curve': set_position_2})

    join_geometry_1 = nw.new_node(Nodes.JoinGeometry,
                                  input_kwargs={'Geometry': [stembranchgeometry, stembranchleaves]})

    join_geometry_1 = nw.new_node(Nodes.SetMaterial,
                                 input_kwargs={'Geometry': join_geometry_1,
                                               'Material': surface.shaderfunc_to_material(simple_greenery.shader_simple_greenery)})

    realize_instances = nw.new_node(Nodes.RealizeInstances,
        input_kwargs={'Geometry': scale_instances})
    
    join_geometry_2 = nw.new_node(Nodes.JoinGeometry,
                                  input_kwargs={'Geometry': [realize_instances, join_geometry_1]})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': join_geometry_2})


@node_utils.to_nodegroup('nodegroup_stem_branch_selection', singleton=False, type='GeometryNodeTree')
def nodegroup_stem_branch_selection(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    spline_parameter_1 = nw.new_node(Nodes.SplineParameter)

    colorramp_1 = nw.new_node(Nodes.ColorRamp,
                              input_kwargs={'Fac': spline_parameter_1.outputs["Factor"]})
    colorramp_1.color_ramp.interpolation = "CONSTANT"
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements[0].position = 0.0
    colorramp_1.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_1.color_ramp.elements[1].position = 0.50
    colorramp_1.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    colorramp_1.color_ramp.elements[2].position = 0.80
    colorramp_1.color_ramp.elements[2].color = (0.0, 0.0, 0.0, 1.0)

    seed = randint(0, 10000, size=(1,))[0]
    threshold = uniform(0.05, 0.1, size=(1,))[0]
    random_value = nw.new_node(Nodes.RandomValue,
                               input_kwargs={'Min': 0.0, 'Max': 1.0, 'Seed': seed})
    less_equal = nw.new_node(Nodes.Compare,
                             input_kwargs={0: random_value, 1: threshold},
                             attrs={'operation': 'LESS_EQUAL'})

    op_and = nw.new_node(Nodes.BooleanMath,
                         input_kwargs={0: colorramp_1.outputs["Color"], 1: less_equal})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Boolean': op_and})


@node_utils.to_nodegroup('nodegroup_stem_leaves', singleton=False, type='GeometryNodeTree')
def nodegroup_stem_leaves(nw: NodeWrangler, leaves):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Curve', None)])

    stemleafselection = nw.new_node(nodegroup_stem_leaf_selection().name)

    leaf_id = randint(0, len(leaves), size=(1,))[0]
    leaf = leaves[leaf_id]
    object_info_2 = nw.new_node(Nodes.ObjectInfo, input_kwargs={'Object': leaf})

    stemleafsr = nw.new_node(nodegroup_stem_leaf_s_r().name)

    instance_on_points_1 = nw.new_node(Nodes.InstanceOnPoints,
                                       input_kwargs={'Points': group_input.outputs["Curve"],
                                                     'Selection': stemleafselection,
                                                     'Instance': object_info_2.outputs["Geometry"],
                                                     'Rotation': stemleafsr.outputs["Rotation"],
                                                     'Scale': stemleafsr.outputs["Value"]})

    random_value_2 = nw.new_node(Nodes.RandomValue,
                                 input_kwargs={'Max': (0.5, 0.5, 6.28), 'Seed': 30},
                                 attrs={'data_type': 'FLOAT_VECTOR'})

    rotate_instances = nw.new_node(Nodes.RotateInstances,
                                   input_kwargs={'Instances': instance_on_points_1,
                                                 'Rotation': random_value_2.outputs["Value"]})
    
    realize_instances = nw.new_node(Nodes.RealizeInstances,
        input_kwargs={'Geometry': rotate_instances})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Instances': realize_instances})


@node_utils.to_nodegroup('nodegroup_main_flower_setting', singleton=False, type='GeometryNodeTree')
def nodegroup_main_flower_setting(nw: NodeWrangler, flowers):
    # Code generated using version 2.4.3 of the node_transpiler

    flower_id = randint(0, len(flowers), size=(1,))[0]
    scale = uniform(0.25, 0.45, size=(1,))[0]
    flower = flowers[flower_id]
    object_info_2 = nw.new_node(Nodes.ObjectInfo, input_kwargs={'Object': flower})
    transform = nw.new_node(Nodes.Transform, 
    input_kwargs={'Geometry': object_info_2.outputs["Geometry"], 'Scale': (scale, scale, scale)})

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.5

    endpoint_selection = nw.new_node('GeometryNodeCurveEndpointSelection',
                                     input_kwargs={'Start Size': 0})

    curve_tangent = nw.new_node(Nodes.CurveTangent)

    align_euler_to_vector = nw.new_node(Nodes.AlignEulerToVector,
                                        input_kwargs={'Vector': curve_tangent},
                                        attrs={'axis': 'Z'})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': transform, 'Value': value,
                                             'Selection': endpoint_selection, 'Rotation': align_euler_to_vector})


@node_utils.to_nodegroup('nodegroup_branch_flower_setting', singleton=False, type='GeometryNodeTree')
def nodegroup_branch_flower_setting(nw: NodeWrangler, flowers):
    # Code generated using version 2.4.3 of the node_transpiler

    flower_id = randint(0, len(flowers), size=(1,))[0]
    scale = uniform(0.4, 0.6, size=(1,))[0]
    flower = flowers[flower_id]
    object_info_2 = nw.new_node(Nodes.ObjectInfo, input_kwargs={'Object': flower})
    transform = nw.new_node(Nodes.Transform, 
    input_kwargs={'Geometry': object_info_2.outputs["Geometry"], 'Scale': (scale, scale, scale)})

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.5

    endpoint_selection = nw.new_node('GeometryNodeCurveEndpointSelection',
                                     input_kwargs={'Start Size': 0})

    curve_tangent = nw.new_node(Nodes.CurveTangent)

    align_euler_to_vector = nw.new_node(Nodes.AlignEulerToVector,
                                        input_kwargs={'Vector': curve_tangent},
                                        attrs={'axis': 'Z'})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': transform, 'Value': value,
                                             'Selection': endpoint_selection, 'Rotation': align_euler_to_vector})


@node_utils.to_nodegroup('nodegroup_stem_rotation', singleton=False, type='GeometryNodeTree')
def nodegroup_stem_rotation(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    position = nw.new_node(Nodes.InputPosition)

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Geometry', None)])

    bounding_box = nw.new_node(Nodes.BoundingBox,
                               input_kwargs={'Geometry': group_input.outputs["Geometry"]})

    multiply = nw.new_node(Nodes.VectorMath,
                           input_kwargs={0: bounding_box.outputs["Max"], 1: (0.0, 0.0, 1.0)},
                           attrs={'operation': 'MULTIPLY'})

    index = nw.new_node(Nodes.Index)

    map_range = nw.new_node(Nodes.MapRange,
                            input_kwargs={'Value': index, 2: 20.0})

    curvature = np.clip(np.abs(normal(0, 0.4, (1,))[0]), 0.0, 0.8)
    float_curve = nw.new_node(Nodes.FloatCurve,
                              input_kwargs={'Value': map_range.outputs["Result"]})
    node_utils.assign_curve(float_curve.mapping.curves[0],
                            [(0.0, 0.0), (0.1, curvature / 5.),
                             (0.25, curvature / 2.5), (0.45, curvature / 1.5),
                             (0.6, curvature / 1.2), (1.0, curvature)])

    multiply_2 = nw.new_node(Nodes.Math,
                             input_kwargs={0: float_curve, 1: 1.2}, attrs={'operation': 'MULTIPLY'})

    vector_rotate = nw.new_node(Nodes.VectorRotate,
                                input_kwargs={'Vector': position, 'Center': multiply.outputs["Vector"],
                                              'Angle': multiply_2},
                                attrs={'rotation_type': 'X_AXIS'})

    noise_texture = nw.new_node(Nodes.NoiseTexture,
                                input_kwargs={'Scale': 0.3})

    add_1 = nw.new_node(Nodes.VectorMath,
                        input_kwargs={0: (-0.5, -0.5, -0.5), 1: noise_texture.outputs["Color"]})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Rotation': vector_rotate, 'Noise': add_1.outputs["Vector"]})


@node_utils.to_nodegroup('nodegroup_stem_geometry', singleton=False, type='GeometryNodeTree')
def nodegroup_stem_geometry(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Curve', None)])

    spline_parameter = nw.new_node(Nodes.SplineParameter)

    colorramp = nw.new_node(Nodes.ColorRamp,
                            input_kwargs={'Fac': spline_parameter.outputs["Factor"]})
    colorramp.color_ramp.elements[0].position = 0.0
    colorramp.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    colorramp.color_ramp.elements[1].position = 1.0
    colorramp.color_ramp.elements[1].color = (0.4, 0.4, 0.4, 1.0)

    set_curve_radius = nw.new_node(Nodes.SetCurveRadius,
                                   input_kwargs={'Curve': group_input.outputs["Curve"],
                                                 'Radius': colorramp.outputs["Color"]})

    rad = uniform(0.01, 0.02, size=(1,))[0]
    curve_circle = nw.new_node(Nodes.CurveCircle,
                               input_kwargs={'Resolution': 10, 'Radius': rad})

    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
                                input_kwargs={'Curve': set_curve_radius, 'Profile Curve': curve_circle.outputs["Curve"],
                                              'Fill Caps': True})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Mesh': tag_nodegroup(nw, curve_to_mesh, 'stem')})


def geo_flowerplant(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.4.3 of the node_transpiler
    leaves = kwargs["leaves"]
    flowers = kwargs["flowers"]
    curve_line = nw.new_node(Nodes.CurveLine)

    resample_curve = nw.new_node(Nodes.ResampleCurve,
                                 input_kwargs={'Curve': curve_line, 'Count': 20})

    stemrotation = nw.new_node(nodegroup_stem_rotation().name,
                               input_kwargs={'Geometry': curve_line})

    set_position = nw.new_node(Nodes.SetPosition,
                               input_kwargs={'Geometry': resample_curve, 'Position': stemrotation.outputs["Rotation"],
                                             'Offset': stemrotation.outputs["Noise"]})

    stemgeometry = nw.new_node(nodegroup_stem_geometry().name,
                               input_kwargs={'Curve': set_position})

    mainflowersetting = nw.new_node(nodegroup_main_flower_setting(flowers=flowers).name)

    instance_on_points = nw.new_node(Nodes.InstanceOnPoints,
                                     input_kwargs={'Points': set_position,
                                                   'Selection': mainflowersetting.outputs["Selection"],
                                                   'Instance': mainflowersetting.outputs["Geometry"],
                                                   'Rotation': mainflowersetting.outputs["Rotation"],
                                                   'Scale': mainflowersetting.outputs["Value"]})

    resample_curve_1 = nw.new_node(Nodes.ResampleCurve,
                                   input_kwargs={'Curve': set_position, 'Count': 150})

    stemleaves = nw.new_node(nodegroup_stem_leaves(leaves=leaves).name,
                             input_kwargs={'Curve': resample_curve_1})

    join_geometry = nw.new_node(Nodes.JoinGeometry,
                                input_kwargs={'Geometry': [stemgeometry, stemleaves]})

    join_geometry = nw.new_node(Nodes.SetMaterial,
                               input_kwargs={'Geometry': join_geometry,
                                             'Material': surface.shaderfunc_to_material(simple_greenery.shader_simple_greenery)})

    num_versions = randint(0, 3, size=(1,))[0]
    branches = []
    for version in range(num_versions):
        resample_num = randint(80, 100, size=(1,))[0]
        resample_curve_2 = nw.new_node(Nodes.ResampleCurve, input_kwargs={'Curve': set_position, 'Count': resample_num})
        stembranchselection = nw.new_node(nodegroup_stem_branch_selection().name)
        stembranch = nw.new_node(nodegroup_stem_branch(flowers=flowers, leaves=leaves).name)
        random_value_1 = nw.new_node(Nodes.RandomValue, input_kwargs={'Min': (0.4, 0.4, 0.4)}, 
        attrs={'data_type': 'FLOAT_VECTOR'})
        instance_on_points_2 = nw.new_node(Nodes.InstanceOnPoints,
                                           input_kwargs={'Points': resample_curve_2, 'Selection': stembranchselection,
                                                         'Instance': stembranch, 'Scale': (random_value_1, "Value")})
        random_value_4 = nw.new_node(Nodes.RandomValue,
                                     input_kwargs={'Min': (0.15, 0.15, 0.0), 'Max': (0.45, 0.45, 6.28), 'Seed': 30},
                                     attrs={'data_type': 'FLOAT_VECTOR'})

        rotate_instances_1 = nw.new_node(Nodes.RotateInstances,
                                         input_kwargs={'Instances': instance_on_points_2,
                                                       'Rotation': (random_value_4, "Value")})
        realize_instances_1 = nw.new_node(Nodes.RealizeInstances,
            input_kwargs={'Geometry': rotate_instances_1})
        branches.append(realize_instances_1)

    realize_instances = nw.new_node(Nodes.RealizeInstances,
        input_kwargs={'Geometry': instance_on_points})
    join_geometry_1 = nw.new_node(Nodes.JoinGeometry,
                                  input_kwargs={
                                      'Geometry': [join_geometry, realize_instances] + branches})

    z_rotate = uniform(0, 6.28, size=(1,))[0]
    transform = nw.new_node(Nodes.Transform,
                            input_kwargs={'Geometry': join_geometry_1, 'Rotation': (0., 0., z_rotate)})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': transform})


class FlowerPlantFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super(FlowerPlantFactory, self).__init__(factory_seed, coarse=coarse)
        self.leaves_version_num = 4
        self.flowers_version_num = 1

    def create_asset(self, **params):
        bpy.ops.mesh.primitive_plane_add(
            size=1, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        obj = bpy.context.active_object

        # Make the Leaf and Delete It Later
        leaves = []
        for _ in range(self.leaves_version_num):
            lf_seed = randint(0, 1000, size=(1,))[0]
            leaf_model = Leaf.LeafFactory(genome={"leaf_width": 0.35, "width_rand": 0.1}, factory_seed=lf_seed)
            leaf = leaf_model.create_asset()
            leaves.append(leaf)

        flowers = []
        for _ in range(self.flowers_version_num):
            fw_seed = randint(0, 1000, size=(1,))[0]
            rad = uniform(0.4, 0.7, size=(1,))[0]
            flower_model = Flower.FlowerFactory(rad=rad, factory_seed=fw_seed)
            flower = flower_model.create_asset()
            flowers.append(flower)

        params["leaves"] = leaves
        params["flowers"] = flowers

        mod = surface.add_geomod(obj, geo_flowerplant, apply=False, attributes=[], input_kwargs=params)
        butil.delete(leaves + flowers)
        with butil.SelectObjects(obj):
            bpy.ops.object.material_slot_remove()
            bpy.ops.object.shade_flat()
        
        butil.apply_modifiers(obj)

        tag_object(obj, 'flowerplant')
        return obj
