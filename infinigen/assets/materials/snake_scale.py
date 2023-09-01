# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Hongyu Wen


import bpy
import mathutils
import random
from numpy.random import uniform, normal, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface
from infinigen.assets.materials import snake_shaders

@node_utils.to_nodegroup('nodegroup_scale_shape', singleton=False, type='GeometryNodeTree')
def nodegroup_scale_shape(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'thickness', 0.0),
            ('NodeSocketFloat', 'length', 1.0)])
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': group_input.outputs["length"]})
    
    curve_line = nw.new_node(Nodes.CurveLine,
        input_kwargs={'End': combine_xyz})
    
    resample_curve = nw.new_node(Nodes.ResampleCurve,
        input_kwargs={'Curve': curve_line})
    
    spline_parameter = nw.new_node(Nodes.SplineParameter)
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': spline_parameter.outputs["Factor"]})
    
    float_curve = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': separate_xyz.outputs["X"]})
    node_utils.assign_curve(float_curve.mapping.curves[0], [(0.0, 0.25419999999999998), (0.10680000000000001, 0.34379999999999999), (0.39479999999999998, 0.3695), (1.0, 0.0)], handles=['AUTO_CLAMPED', 'AUTO', 'AUTO', 'AUTO'])
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: float_curve, 1: group_input.outputs["length"]},
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Y': multiply})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': resample_curve, 'Offset': combine_xyz_1})
    
    position = nw.new_node(Nodes.InputPosition)
    
    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': position})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["Y"], 1: -1.0},
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': separate_xyz_1.outputs["X"], 'Y': multiply_1, 'Z': separate_xyz_1.outputs["Z"]})
    
    set_position_1 = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': set_position, 'Position': combine_xyz_2})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [set_position_1, set_position]})
    
    convex_hull = nw.new_node(Nodes.ConvexHull,
        input_kwargs={'Geometry': join_geometry})
    
    mesh_to_curve = nw.new_node(Nodes.MeshToCurve,
        input_kwargs={'Mesh': convex_hull})
    
    position_1 = nw.new_node(Nodes.InputPosition)
    
    transfer_attribute = nw.new_node(Nodes.SampleNearestSurface,
        input_kwargs={'Mesh': convex_hull, 'Value': position_1},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    separate_xyz_2 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': (transfer_attribute, "Value")})
    
    divide = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["X"], 1: group_input.outputs["length"]},
        attrs={'operation': 'DIVIDE'})
    
    float_curve_1 = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': divide})
    node_utils.assign_curve(float_curve_1.mapping.curves[0], [(0.0114, 0.46110000000000001), (0.51139999999999997, 0.30940000000000001), (1.0, 0.058099999999999999)])
    
    set_curve_radius = nw.new_node(Nodes.SetCurveRadius,
        input_kwargs={'Curve': mesh_to_curve, 'Radius': float_curve_1})
    
    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Y': group_input.outputs["thickness"]})
    
    curve_line_1 = nw.new_node(Nodes.CurveLine,
        input_kwargs={'End': combine_xyz_3})
    
    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
        input_kwargs={'Curve': set_curve_radius, 'Profile Curve': curve_line_1})
    
    convex_hull_1 = nw.new_node(Nodes.ConvexHull,
        input_kwargs={'Geometry': curve_to_mesh})
    
    join_geometry_1 = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [join_geometry, resample_curve]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Convex Hull': convex_hull_1, 'Curve': join_geometry_1})

@node_utils.to_nodegroup('nodegroup_scale', singleton=False, type='GeometryNodeTree')
def nodegroup_scale(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloatDistance', 'Radius', 0.040000000000000001),
            ('NodeSocketFloat', 'thickness', 0.10000000000000001),
            ('NodeSocketVectorEuler', 'Rotation', (0.0, -0.17449999999999999, 0.0))])
    
    nodegroup_nodegroup_nodegroup_scale_shape_011 = nw.new_node(nodegroup_scale_shape().name,
        input_kwargs={'thickness': group_input.outputs["thickness"], 'length': group_input.outputs["Radius"]})
    
    position = nw.new_node(Nodes.InputPosition)
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': position})
    
    float_curve = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': separate_xyz.outputs["X"]})
    node_utils.assign_curve(float_curve.mapping.curves[0], [(0.0, 0.0), (0.45229999999999998, 0.13439999999999999), (0.92949999999999999, 0.1875), (1.0, 0.0)])
    
    set_curve_radius = nw.new_node(Nodes.SetCurveRadius,
        input_kwargs={'Curve': nodegroup_nodegroup_nodegroup_scale_shape_011.outputs["Curve"], 'Radius': float_curve})
    
    curve_circle = nw.new_node(Nodes.CurveCircle,
        input_kwargs={'Resolution': 5, 'Radius': 0.02})
    
    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
        input_kwargs={'Curve': set_curve_radius, 'Profile Curve': curve_circle.outputs["Curve"]})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': curve_to_mesh})
    
    join_geometry_1 = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [join_geometry, nodegroup_nodegroup_nodegroup_scale_shape_011.outputs["Convex Hull"]]})
    
    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': join_geometry_1, 'Rotation': group_input.outputs["Rotation"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': transform})

def geometry_snake_scale(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None)])
    
    position = nw.new_node(Nodes.InputPosition)
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': position})
    
    greater_than = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: 0.0},
        attrs={'operation': 'GREATER_THAN'})
    
    separate_geometry = nw.new_node(Nodes.SeparateGeometry,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Selection': greater_than},
        attrs={'domain': 'EDGE'})
    
    distribute_points_on_faces = nw.new_node(Nodes.DistributePointsOnFaces,
        input_kwargs={'Mesh': separate_geometry.outputs["Selection"], 'Distance Min': 0.03, 'Density Max': 10000.0},
        attrs={'distribute_method': 'POISSON'})
    
    named_attribute = nw.new_node(Nodes.NamedAttribute,
        input_kwargs={'Name': 'corner'})
    
    named_attribute_1 = nw.new_node(Nodes.NamedAttribute,
        input_kwargs={'Name': 'inside_mouth'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: named_attribute.outputs[1], 1: named_attribute_1.outputs[1]})
    
    separate_geometry_1 = nw.new_node(Nodes.SeparateGeometry,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Selection': add})
    
    geometry_proximity = nw.new_node(Nodes.Proximity,
        input_kwargs={'Target': separate_geometry_1.outputs["Selection"]})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': geometry_proximity.outputs["Distance"], 2: 0.050000000000000003})
    
    greater_than_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range.outputs["Result"], 1: 0.10000000000000001},
        attrs={'operation': 'GREATER_THAN'})
    
    nodegroup_scale_1 = nw.new_node(nodegroup_scale().name,
        input_kwargs={'Radius': 0.14999999999999999, 'thickness': 0.01, 'Rotation': (0.0, -0.017500000000000002, 0.0)})
    
    set_shade_smooth = nw.new_node(Nodes.SetShadeSmooth,
        input_kwargs={'Geometry': nodegroup_scale_1})
    
    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': set_shade_smooth, 'Scale': (0.59999999999999998, 0.59999999999999998, 0.59999999999999998)})
    
    normal = nw.new_node(Nodes.InputNormal)
    
    transfer_attribute = nw.new_node(Nodes.SampleNearestSurface,
        input_kwargs={'Mesh': separate_geometry.outputs["Selection"], 'Value': normal},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    align_euler_to_vector = nw.new_node(Nodes.AlignEulerToVector,
        input_kwargs={'Vector': (transfer_attribute, "Value")},
        attrs={'axis': 'Z'})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': map_range.outputs["Result"], 'Y': map_range.outputs["Result"], 'Z': map_range.outputs["Result"]})
    
    instance_on_points = nw.new_node(Nodes.InstanceOnPoints,
        input_kwargs={'Points': distribute_points_on_faces.outputs["Points"], 'Selection': greater_than_1, 'Instance': transform, 'Rotation': align_euler_to_vector, 'Scale': combine_xyz})
    
    distribute_points_on_faces_1 = nw.new_node(Nodes.DistributePointsOnFaces,
        input_kwargs={'Mesh': separate_geometry.outputs["Inverted"], 'Distance Min': 0.02, 'Density Max': 10000.0},
        attrs={'distribute_method': 'POISSON'})
    
    greater_than_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range.outputs["Result"], 1: 0.10000000000000001},
        attrs={'operation': 'GREATER_THAN'})
    
    nodegroup_scale_2 = nw.new_node(nodegroup_scale().name,
        input_kwargs={'Radius': 0.070000000000000007, 'thickness': 0.0060000000000000001, 'Rotation': (0.0, -0.017500000000000002, 0.0)})
    
    set_shade_smooth_1 = nw.new_node(Nodes.SetShadeSmooth,
        input_kwargs={'Geometry': nodegroup_scale_2})
    
    normal_1 = nw.new_node(Nodes.InputNormal)
    
    transfer_attribute_1 = nw.new_node(Nodes.SampleNearestSurface,
        input_kwargs={'Mesh': separate_geometry.outputs["Inverted"], 'Value': normal_1},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    align_euler_to_vector_1 = nw.new_node(Nodes.AlignEulerToVector,
        input_kwargs={'Vector': (transfer_attribute_1, "Value")},
        attrs={'axis': 'Z'})
    
    instance_on_points_1 = nw.new_node(Nodes.InstanceOnPoints,
        input_kwargs={'Points': distribute_points_on_faces_1.outputs["Points"], 'Selection': greater_than_2, 'Instance': set_shade_smooth_1, 'Rotation': align_euler_to_vector_1, 'Scale': combine_xyz})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [group_input.outputs["Geometry"], instance_on_points, instance_on_points_1]})
    
    realize_instances = nw.new_node(Nodes.RealizeInstances,
        input_kwargs={'Geometry': join_geometry})
    
    position_1 = nw.new_node(Nodes.InputPosition)
    
    bounding_box = nw.new_node(Nodes.BoundingBox, input_kwargs={'Geometry': realize_instances})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Vector': position_1, 7: bounding_box.outputs["Min"], 8: bounding_box.outputs["Max"]},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    store_named_attribute = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': realize_instances, 'Name': 'Position', 2: map_range.outputs["Vector"]},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': store_named_attribute}, attrs={'is_active_output': True})

def apply(obj, **kwargs):
    shader = snake_shaders.shaders.choose()
    rand = uniform() > 0.3
    surface.add_geomod(obj, geometry_snake_scale)
    surface.add_material(obj, shader, input_kwargs={'rand': rand})