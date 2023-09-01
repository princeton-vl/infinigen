# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo


import bpy
import mathutils
from numpy.random import uniform, normal, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface

from infinigen.assets.creatures.insects.utils.shader_utils import shader_black_w_noise_shader
from infinigen.assets.creatures.insects.utils.geom_utils import nodegroup_shape_quadratic, nodegroup_surface_bump
from infinigen.assets.creatures.insects.parts.hair.principled_hair import nodegroup_principled_hair

@node_utils.to_nodegroup('nodegroup_leg_control', singleton=False, type='GeometryNodeTree')
def nodegroup_leg_control(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'Openness', 1.0)])
    
    reroute_2 = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': group_input.outputs["Openness"]})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': reroute_2, 3: 0.6, 4: 1.44})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': reroute_2, 3: -0.26, 4: 0.16})
    
    map_range_2 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': reroute_2, 3: 1.68, 4: 1.88})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Femur': map_range.outputs["Result"], 'Tarsus': map_range_1.outputs["Result"], 'Shoulder': map_range_2.outputs["Result"]})

@node_utils.to_nodegroup('nodegroup_dragonfly_leg', singleton=False, type='GeometryNodeTree')
def nodegroup_dragonfly_leg(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    legcrosssection = nw.new_node(nodegroup_leg_cross_section().name)
    
    shapequadraticclaw = nw.new_node(nodegroup_shape_quadratic(radius_control_points=[(0.0, 0.0031), (0.2682, 0.1906), (0.6364, 0.3594), (0.8091, 0.5031), (1.0, 0.5375)]).name,
        input_kwargs={'Profile Curve': legcrosssection, 'noise amount tilt': 0.0, 'Resolution': 16, 'Start': (0.0, 0.0, 3.0), 'Middle': (-1.2, 0.0, 1.5), 'End': (0.2, 0.0, 0.0)})
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.3
    
    transform_2 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': shapequadraticclaw, 'Translation': (-0.38, 0.0, 1.0), 'Rotation': (0.0, 0.4318, 0.0), 'Scale': value})
    
    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = 0.5
    
    transform_3 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': shapequadraticclaw, 'Translation': (0.1, 0.0, 0.04), 'Rotation': (0.0, -0.0262, 0.0), 'Scale': value_1})
    
    join_geometry_1 = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [shapequadraticclaw, transform_2, transform_3]})
    
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'Rot claw', 1.82),
            ('NodeSocketFloat', 'Rot Tarsus', 0.02),
            ('NodeSocketFloat', 'Rot Femur', 1.42)])
    
    legpart = nw.new_node(nodegroup_leg_part().name,
        input_kwargs={'NextJoint': join_geometry_1, 'NextJoint Y rot': group_input.outputs["Rot claw"], 'NextJoint Scale': 0.4, 'Num Hairs': 10})
    
    legpart_1 = nw.new_node(nodegroup_leg_part().name,
        input_kwargs={'NextJoint': legpart, 'NextJoint Y rot': group_input.outputs["Rot Tarsus"], 'NextJoint Scale': 0.45, 'Cross Section Scale': 0.8})
    
    legpart_2 = nw.new_node(nodegroup_leg_part().name,
        input_kwargs={'NextJoint': legpart_1, 'NextJoint Y rot': group_input.outputs["Rot Femur"], 'NextJoint Scale': 0.75, 'Cross Section Scale': 1.2, 'Num Hairs': 30, 'Hair Scale Max': 0.15})
    
    surfacebump = nw.new_node(nodegroup_surface_bump().name,
        input_kwargs={'Geometry': legpart_2, 'Displacement': 0.03, 'Scale': 5.0})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': surfacebump})

@node_utils.to_nodegroup('nodegroup_leg_part', singleton=False, type='GeometryNodeTree')
def nodegroup_leg_part(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    legcrosssection = nw.new_node(nodegroup_leg_cross_section().name)
    
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'NextJoint', None),
            ('NodeSocketFloat', 'NextJoint Y rot', 0.0),
            ('NodeSocketFloat', 'NextJoint Scale', 1.0),
            ('NodeSocketFloat', 'Cross Section Scale', 1.0),
            ('NodeSocketInt', 'Num Hairs', 15),
            ('NodeSocketFloat', 'Hair Scale Min', 0.18),
            ('NodeSocketFloat', 'Hair Scale Max', 0.22)])
    
    transform_4 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': legcrosssection, 'Rotation': (0.0, 0.0, 3.1416), 'Scale': group_input.outputs["Cross Section Scale"]})
    
    tarsus_end = nw.new_node(Nodes.Vector,
        label='tarsus end')
    tarsus_end.vector = (0.2, 0.0, 6.0)
    
    shapequadratictarsus = nw.new_node(nodegroup_shape_quadratic(radius_control_points=[(0.0, 0.3125), (0.0841, 0.3469), (0.45, 0.4125), (0.55, 0.3719), (0.9045, 0.325), (1.0, 0.125)]).name,
        input_kwargs={'Profile Curve': transform_4, 'noise amount tilt': 0.0, 'Resolution': 128, 'Start': (0.0, 0.0, 0.0), 'Middle': (-0.4, 0.0, 3.0), 'End': tarsus_end})
    
    spline_parameter_1 = nw.new_node(Nodes.SplineParameter)
    
    capture_attribute_1 = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': shapequadratictarsus.outputs["Mesh"], 2: spline_parameter_1.outputs["Factor"]})
    
    curve_to_points_1 = nw.new_node(Nodes.CurveToPoints,
        input_kwargs={'Curve': capture_attribute_1.outputs["Geometry"], 'Count': group_input.outputs["Num Hairs"]})
    
    greater_than = nw.new_node(Nodes.Compare,
        input_kwargs={0: capture_attribute_1.outputs[2], 1: 0.9})
    
    delete_geometry_1 = nw.new_node(Nodes.DeleteGeometry,
        input_kwargs={'Geometry': curve_to_points_1.outputs["Points"], 'Selection': greater_than})
    
    leghair = nw.new_node(nodegroup_principled_hair().name)
    
    random_value_3 = nw.new_node(Nodes.RandomValue,
        input_kwargs={2: 0.88})
    
    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Y': random_value_3.outputs[1]})
    
    random_value_2 = nw.new_node(Nodes.RandomValue,
        input_kwargs={2: group_input.outputs["Hair Scale Min"], 3: group_input.outputs["Hair Scale Max"]})
    
    instance_on_points_1 = nw.new_node(Nodes.InstanceOnPoints,
        input_kwargs={'Points': delete_geometry_1, 'Instance': leghair, 'Rotation': combine_xyz_1, 'Scale': random_value_2.outputs[1]})
    
    subtract = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: tarsus_end, 1: (0.0, 0.0, 0.05)},
        attrs={'operation': 'SUBTRACT'})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Y': group_input.outputs["NextJoint Y rot"]})
    
    transform_5 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': group_input.outputs["NextJoint"], 'Translation': subtract.outputs["Vector"], 'Rotation': combine_xyz, 'Scale': group_input.outputs["NextJoint Scale"]})
    
    join_geometry_3 = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [shapequadratictarsus.outputs["Mesh"], transform_5]})
    
    join_geometry_4 = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [instance_on_points_1, join_geometry_3]})
    
    set_material = nw.new_node(Nodes.SetMaterial,
        input_kwargs={'Geometry': join_geometry_4, 'Material': surface.shaderfunc_to_material(shader_black_w_noise_shader)})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_material})

@node_utils.to_nodegroup('nodegroup_leg_cross_section', singleton=False, type='GeometryNodeTree')
def nodegroup_leg_cross_section(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketIntUnsigned', 'Resolution', 8)])
    
    bezier_segment = nw.new_node(Nodes.CurveBezierSegment,
        input_kwargs={'Resolution': group_input.outputs["Resolution"], 'Start Handle': (-0.9, 0.7, 0.0), 'End Handle': (0.9, 0.38, 0.0)})
    
    reroute = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': bezier_segment})
    
    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': reroute, 'Scale': (1.0, -1.0, 1.0)})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [transform, reroute]})
    
    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
        input_kwargs={'Curve': join_geometry})
    
    merge_by_distance = nw.new_node(Nodes.MergeByDistance,
        input_kwargs={'Geometry': curve_to_mesh})
    
    mesh_to_curve = nw.new_node(Nodes.MeshToCurve,
        input_kwargs={'Mesh': merge_by_distance})
    
    transform_1 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': mesh_to_curve, 'Rotation': (0.0, 0.0, 1.5708), 'Scale': (0.6, 1.0, 0.6)})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': transform_1})