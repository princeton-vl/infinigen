# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import bpy
import mathutils
from numpy.random import uniform, normal, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface
from .math import nodegroup_deg2_rad
from .curve import nodegroup_warped_circle_curve, nodegroup_smooth_taper, nodegroup_profile_part

@node_utils.to_nodegroup('nodegroup_part_surface', singleton=True, type='GeometryNodeTree')
def nodegroup_part_surface(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Skeleton Curve', None),
            ('NodeSocketGeometry', 'Skin Mesh', None),
            ('NodeSocketFloatFactor', 'Length Fac', 0.0),
            ('NodeSocketVectorEuler', 'Ray Rot', (0.0, 0.0, 0.0)),
            ('NodeSocketFloat', 'Rad', 0.0)])
    
    sample_curve = nw.new_node(Nodes.SampleCurve,
        input_kwargs={'Curve': group_input.outputs["Skeleton Curve"], 'Factor': group_input.outputs["Length Fac"]},
        attrs={'mode': 'FACTOR'})
    
    vector_rotate = nw.new_node(Nodes.VectorRotate,
        input_kwargs={'Vector': sample_curve.outputs["Tangent"], 'Rotation': group_input.outputs["Ray Rot"]},
        attrs={'rotation_type': 'EULER_XYZ'})
    
    raycast = nw.new_node(Nodes.Raycast,
        input_kwargs={'Target Geometry': group_input.outputs["Skin Mesh"], 'Source Position': sample_curve.outputs["Position"], 'Ray Direction': vector_rotate, 'Ray Length': 5.0})
    
    lerp = nw.new_node(Nodes.MapRange,
        input_kwargs={'Vector': group_input.outputs["Rad"], 9: sample_curve.outputs["Position"], 10: raycast.outputs["Hit Position"]},
        label='lerp',
        attrs={'data_type': 'FLOAT_VECTOR', 'clamp': False})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Position': lerp.outputs["Vector"], 'Hit Normal': raycast.outputs["Hit Normal"], 'Tangent': sample_curve.outputs["Tangent"], 'Skeleton Pos': sample_curve.outputs["Position"]})

@node_utils.to_nodegroup('nodegroup_part_surface_simple', singleton=True, type='GeometryNodeTree')
def nodegroup_part_surface_simple(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Skeleton Curve', None),
            ('NodeSocketGeometry', 'Skin Mesh', None),
            ('NodeSocketVector', 'Length, Yaw, Rad', (0.0, 0.0, 0.0))])
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group_input.outputs["Length, Yaw, Rad"]})
    
    clamp_1 = nw.new_node(Nodes.Clamp,
        input_kwargs={'Value': separate_xyz.outputs["X"]})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': 1.5708, 'Y': separate_xyz.outputs["Y"], 'Z': 1.5708})
    
    part_surface = nw.new_node(nodegroup_part_surface().name,
        input_kwargs={'Skeleton Curve': group_input.outputs["Skeleton Curve"], 'Skin Mesh': group_input.outputs["Skin Mesh"], 'Length Fac': clamp_1, 'Ray Rot': combine_xyz, 'Rad': separate_xyz.outputs["Z"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Position': part_surface.outputs["Position"], 'Hit Normal': part_surface.outputs["Hit Normal"], 'Tangent': part_surface.outputs["Tangent"]})

@node_utils.to_nodegroup('nodegroup_raycast_rotation', singleton=True, type='GeometryNodeTree')
def nodegroup_raycast_rotation(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVectorEuler', 'Rotation', (0.0, 0.0, 0.0)),
            ('NodeSocketVector', 'Hit Normal', (0.0, 0.0, 1.0)),
            ('NodeSocketVector', 'Curve Tangent', (0.0, 0.0, 1.0)),
            ('NodeSocketBool', 'Do Normal Rot', False),
            ('NodeSocketBool', 'Do Tangent Rot', False)])
    
    align_euler_to_vector = nw.new_node(Nodes.AlignEulerToVector,
        input_kwargs={'Vector': group_input.outputs["Hit Normal"]})
    
    rotate_euler = nw.new_node(Nodes.RotateEuler,
        input_kwargs={'Rotation': group_input.outputs["Rotation"], 'Rotate By': align_euler_to_vector})
    
    if_normal_rot = nw.new_node(Nodes.Switch,
        input_kwargs={0: group_input.outputs["Do Normal Rot"], 8: group_input.outputs["Rotation"], 9: rotate_euler},
        label='if_normal_rot',
        attrs={'input_type': 'VECTOR'})
    
    align_euler_to_vector_1 = nw.new_node(Nodes.AlignEulerToVector,
        input_kwargs={'Rotation': group_input.outputs["Rotation"], 'Vector': group_input.outputs["Curve Tangent"]})
    
    rotate_euler_1 = nw.new_node(Nodes.RotateEuler,
        input_kwargs={'Rotation': align_euler_to_vector_1, 'Rotate By': group_input.outputs["Rotation"]},
        attrs={'space': 'LOCAL'})
    
    if_tangent_rot = nw.new_node(Nodes.Switch,
        input_kwargs={0: group_input.outputs["Do Tangent Rot"], 8: if_normal_rot.outputs[3], 9: rotate_euler_1},
        label='if_tangent_rot',
        attrs={'input_type': 'VECTOR'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Output': if_tangent_rot.outputs[3]})


@node_utils.to_nodegroup('nodegroup_surface_muscle', singleton=True, type='GeometryNodeTree')
def nodegroup_surface_muscle(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Skin Mesh', None),
            ('NodeSocketGeometry', 'Skeleton Curve', None),
            ('NodeSocketVector', 'Coord 0', (0.4, 0.0, 1.0)),
            ('NodeSocketVector', 'Coord 1', (0.5, 0.0, 1.0)),
            ('NodeSocketVector', 'Coord 2', (0.6, 0.0, 1.0)),
            ('NodeSocketVector', 'StartRad, EndRad, Fullness', (0.0, 0.0, 0.0)),
            ('NodeSocketVector', 'ProfileHeight, StartTilt, EndTilt', (0.0, 0.0, 0.0)),
            ('NodeSocketBool', 'Debug Points', False)])
    
    cube = nw.new_node(Nodes.MeshCube,
        input_kwargs={'Size': (0.03, 0.03, 0.03)})
    
    part_surface_simple = nw.new_node(nodegroup_part_surface_simple().name,
        input_kwargs={'Skeleton Curve': group_input.outputs["Skeleton Curve"], 'Skin Mesh': group_input.outputs["Skin Mesh"], 'Length, Yaw, Rad': group_input.outputs["Coord 0"]})
    
    transform_2 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': cube, 'Translation': part_surface_simple.outputs["Position"]})
    
    part_surface_simple_1 = nw.new_node(nodegroup_part_surface_simple().name,
        input_kwargs={'Skeleton Curve': group_input.outputs["Skeleton Curve"], 'Skin Mesh': group_input.outputs["Skin Mesh"], 'Length, Yaw, Rad': group_input.outputs["Coord 1"]})
    
    transform_1 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': cube, 'Translation': part_surface_simple_1.outputs["Position"]})
    
    part_surface_simple_2 = nw.new_node(nodegroup_part_surface_simple().name,
        input_kwargs={'Skeleton Curve': group_input.outputs["Skeleton Curve"], 'Skin Mesh': group_input.outputs["Skin Mesh"], 'Length, Yaw, Rad': group_input.outputs["Coord 2"]})
    
    transform_3 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': cube, 'Translation': part_surface_simple_2.outputs["Position"]})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [transform_2, transform_1, transform_3]})
    
    switch = nw.new_node(Nodes.Switch,
        input_kwargs={1: group_input.outputs["Debug Points"], 15: join_geometry})

    u_resolution = nw.new_node(Nodes.Integer,
        label='U Resolution')
    u_resolution.integer = 16
    
    quadratic_bezier = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': u_resolution, 'Start': part_surface_simple.outputs["Position"], 'Middle': part_surface_simple_1.outputs["Position"], 'End': part_surface_simple_2.outputs["Position"]})
    
    spline_parameter = nw.new_node(Nodes.SplineParameter)
    
    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group_input.outputs["ProfileHeight, StartTilt, EndTilt"]})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': spline_parameter.outputs["Factor"], 3: separate_xyz_1.outputs["Y"], 4: separate_xyz_1.outputs["Z"]})
    
    deg2rad = nw.new_node(nodegroup_deg2_rad().name,
        input_kwargs={'Deg': map_range_1.outputs["Result"]})
    
    set_curve_tilt = nw.new_node(Nodes.SetCurveTilt,
        input_kwargs={'Curve': quadratic_bezier, 'Tilt': deg2rad})
    
    position = nw.new_node(Nodes.InputPosition)
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': separate_xyz_1.outputs["X"], 'Y': 1.0, 'Z': 1.0})
    
    multiply = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: position, 1: combine_xyz},
        attrs={'operation': 'MULTIPLY'})
    
    v_resolution = nw.new_node(Nodes.Integer,
        label='V resolution')
    v_resolution.integer = 24
    
    warped_circle_curve = nw.new_node(nodegroup_warped_circle_curve().name,
        input_kwargs={'Position': multiply.outputs["Vector"], 'Vertices': v_resolution})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group_input.outputs["StartRad, EndRad, Fullness"]})
    
    smoothtaper = nw.new_node(nodegroup_smooth_taper().name,
        input_kwargs={'start_rad': separate_xyz.outputs["X"], 'end_rad': separate_xyz.outputs["Y"], 'fullness': separate_xyz.outputs["Z"]})
    
    profilepart = nw.new_node(nodegroup_profile_part().name,
        input_kwargs={'Skeleton Curve': set_curve_tilt, 'Profile Curve': warped_circle_curve, 'Radius Func': smoothtaper})
    
    join_geometry_1 = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [switch.outputs[6], profilepart]})
    
    switch_1 = nw.new_node(Nodes.Switch,
        input_kwargs={1: True, 15: join_geometry_1})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': switch_1.outputs[6]})

@node_utils.to_nodegroup('nodegroup_attach_part', singleton=True, type='GeometryNodeTree')
def nodegroup_attach_part(nw: NodeWrangler):
    # Code generated using version 2.4.2 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Skin Mesh', None),
            ('NodeSocketGeometry', 'Skeleton Curve', None),
            ('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketFloatFactor', 'Length Fac', 0.0),
            ('NodeSocketVectorEuler', 'Ray Rot', (0.0, 0.0, 0.0)),
            ('NodeSocketFloat', 'Rad', 0.0),
            ('NodeSocketVector', 'Part Rot', (0.0, 0.0, 0.0)),
            ('NodeSocketBool', 'Do Normal Rot', False),
            ('NodeSocketBool', 'Do Tangent Rot', False)])
    
    part_surface = nw.new_node(nodegroup_part_surface().name,
        input_kwargs={'Skeleton Curve': group_input.outputs["Skeleton Curve"], 'Skin Mesh': group_input.outputs["Skin Mesh"], 'Length Fac': group_input.outputs["Length Fac"], 'Ray Rot': group_input.outputs["Ray Rot"], 'Rad': group_input.outputs["Rad"]})
    
    deg2rad = nw.new_node(nodegroup_deg2_rad().name,
        input_kwargs={'Deg': group_input.outputs["Part Rot"]})
    
    raycast_rotation = nw.new_node(nodegroup_raycast_rotation().name,
        input_kwargs={'Rotation': deg2rad, 'Hit Normal': part_surface.outputs["Hit Normal"], 'Curve Tangent': part_surface.outputs["Tangent"], 'Do Normal Rot': group_input.outputs["Do Normal Rot"], 'Do Tangent Rot': group_input.outputs["Do Tangent Rot"]})
    
    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Translation': part_surface.outputs["Position"], 'Rotation': raycast_rotation})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': transform, 'Position': part_surface.outputs["Position"]})

