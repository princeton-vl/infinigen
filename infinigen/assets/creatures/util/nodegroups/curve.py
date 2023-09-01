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
from .math import nodegroup_polar_to_cart, nodegroup_aspect_to_dim, nodegroup_vector_sum, nodegroup_switch4

@node_utils.to_nodegroup('nodegroup_simple_tube', singleton=True, type='GeometryNodeTree')
def nodegroup_simple_tube(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'Origin', (0.0, 0.0, 0.0)),
            ('NodeSocketVector', 'Angles Deg', (30.0, -1.5, 11.0)),
            ('NodeSocketVector', 'Seg Lengths', (0.02, 0.02, 0.02)),
            ('NodeSocketFloat', 'Start Radius', 0.06),
            ('NodeSocketFloat', 'End Radius', 0.03),
            ('NodeSocketFloat', 'Fullness', 8.17),
            ('NodeSocketBool', 'Do Bezier', True),
            ('NodeSocketFloat', 'Aspect Ratio', 1.0)])
    
    polarbezier = nw.new_node(nodegroup_polar_bezier().name,
        input_kwargs={'Resolution': 25, 'Origin': group_input.outputs["Origin"], 'angles_deg': group_input.outputs["Angles Deg"], 'Seg Lengths': group_input.outputs["Seg Lengths"], 'Do Bezier': group_input.outputs["Do Bezier"]})
    
    aspect_to_dim = nw.new_node(nodegroup_aspect_to_dim().name,
        input_kwargs={'Aspect Ratio': group_input.outputs["Aspect Ratio"]})
    
    position = nw.new_node(Nodes.InputPosition)
    
    multiply = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: aspect_to_dim, 1: position},
        attrs={'operation': 'MULTIPLY'})
    
    warped_circle_curve = nw.new_node(nodegroup_warped_circle_curve().name,
        input_kwargs={'Position': multiply.outputs["Vector"], 'Vertices': 40})
    
    smoothtaper = nw.new_node(nodegroup_smooth_taper().name,
        input_kwargs={'start_rad': group_input.outputs["Start Radius"], 'end_rad': group_input.outputs["End Radius"], 'fullness': group_input.outputs["Fullness"]})
    
    profilepart = nw.new_node(nodegroup_profile_part().name,
        input_kwargs={'Skeleton Curve': polarbezier.outputs["Curve"], 'Profile Curve': warped_circle_curve, 'Radius Func': smoothtaper})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': profilepart, 'Skeleton Curve': polarbezier.outputs["Curve"], 'Endpoint': polarbezier.outputs["Endpoint"]})

@node_utils.to_nodegroup('nodegroup_simple_tube_v2', singleton=True, type='GeometryNodeTree')
def nodegroup_simple_tube_v2(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'length_rad1_rad2', (1.0, 0.5, 0.3)),
            ('NodeSocketVector', 'angles_deg', (0.0, 0.0, 0.0)),
            ('NodeSocketVector', 'proportions', (0.3333, 0.3333, 0.3333)),
            ('NodeSocketFloat', 'aspect', 1.0),
            ('NodeSocketBool', 'do_bezier', True),
            ('NodeSocketFloat', 'fullness', 4.0),
            ('NodeSocketVector', 'Origin', (0.0, 0.0, 0.0))])
    
    vector_sum = nw.new_node(nodegroup_vector_sum().name,
        input_kwargs={'Vector': group_input.outputs["proportions"]})
    
    divide = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["proportions"], 1: vector_sum},
        attrs={'operation': 'DIVIDE'})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group_input.outputs["length_rad1_rad2"]})
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: divide.outputs["Vector"], 'Scale': separate_xyz.outputs["X"]},
        attrs={'operation': 'SCALE'})
    
    polarbezier = nw.new_node(nodegroup_polar_bezier().name,
        input_kwargs={'Resolution': 25, 'Origin': group_input.outputs["Origin"], 'angles_deg': group_input.outputs["angles_deg"], 'Seg Lengths': scale.outputs["Vector"], 'Do Bezier': group_input.outputs["do_bezier"]})
    
    aspect_to_dim = nw.new_node(nodegroup_aspect_to_dim().name,
        input_kwargs={'Aspect Ratio': group_input.outputs["aspect"]})
    
    position = nw.new_node(Nodes.InputPosition)
    
    multiply = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: aspect_to_dim, 1: position},
        attrs={'operation': 'MULTIPLY'})
    
    warped_circle_curve = nw.new_node(nodegroup_warped_circle_curve().name,
        input_kwargs={'Position': multiply.outputs["Vector"], 'Vertices': 40})
    
    smoothtaper = nw.new_node(nodegroup_smooth_taper().name,
        input_kwargs={'start_rad': separate_xyz.outputs["Y"], 'end_rad': separate_xyz.outputs["Z"], 'fullness': group_input.outputs["fullness"]})
    
    profilepart = nw.new_node(nodegroup_profile_part().name,
        input_kwargs={'Skeleton Curve': polarbezier.outputs["Curve"], 'Profile Curve': warped_circle_curve, 'Radius Func': smoothtaper})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': profilepart, 'Skeleton Curve': polarbezier.outputs["Curve"], 'Endpoint': polarbezier.outputs["Endpoint"]})

@node_utils.to_nodegroup('nodegroup_smooth_taper', singleton=True, type='GeometryNodeTree')
def nodegroup_smooth_taper(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    spline_parameter = nw.new_node(Nodes.SplineParameter)
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: spline_parameter.outputs["Factor"], 1: 3.1416},
        attrs={'operation': 'MULTIPLY'})
    
    sine = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply},
        attrs={'operation': 'SINE'})
    
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'start_rad', 0.29),
            ('NodeSocketFloat', 'end_rad', 0.0),
            ('NodeSocketFloat', 'fullness', 2.5)])
    
    divide = nw.new_node(Nodes.Math,
        input_kwargs={0: 1.0, 1: group_input.outputs["fullness"]},
        attrs={'operation': 'DIVIDE'})
    
    power = nw.new_node(Nodes.Math,
        input_kwargs={0: sine, 1: divide},
        attrs={'operation': 'POWER'})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': spline_parameter.outputs["Factor"], 3: group_input.outputs["start_rad"], 4: group_input.outputs["end_rad"]},
        attrs={'clamp': False})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: power, 1: map_range.outputs["Result"]},
        attrs={'operation': 'MULTIPLY'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Value': multiply_1})

@node_utils.to_nodegroup('nodegroup_warped_circle_curve', singleton=True, type='GeometryNodeTree')
def nodegroup_warped_circle_curve(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'Position', (0.0, 0.0, 0.0)),
            ('NodeSocketInt', 'Vertices', 32)])
    
    mesh_circle = nw.new_node(Nodes.MeshCircle,
        input_kwargs={'Vertices': group_input.outputs["Vertices"]})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': mesh_circle, 'Position': group_input.outputs["Position"]})
    
    mesh_to_curve = nw.new_node(Nodes.MeshToCurve,
        input_kwargs={'Mesh': set_position})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Curve': mesh_to_curve})

@node_utils.to_nodegroup('nodegroup_polar_bezier', singleton=True, type='GeometryNodeTree')
def nodegroup_polar_bezier(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketIntUnsigned', 'Resolution', 32),
            ('NodeSocketVector', 'Origin', (0.0, 0.0, 0.0)),
            ('NodeSocketVector', 'angles_deg', (0.0, 0.0, 0.0)),
            ('NodeSocketVector', 'Seg Lengths', (0.3, 0.3, 0.3)),
            ('NodeSocketBool', 'Do Bezier', True)])
    
    mesh_line = nw.new_node(Nodes.MeshLine,
        input_kwargs={'Count': 4})
    
    index = nw.new_node(Nodes.Index)
    
    deg2_rad = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["angles_deg"], 'Scale': 0.0175},
        label='Deg2Rad',
        attrs={'operation': 'SCALE'})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': deg2_rad.outputs["Vector"]})
    
    reroute = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': separate_xyz.outputs["X"]})
    
    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group_input.outputs["Seg Lengths"]})
    
    polartocart = nw.new_node(nodegroup_polar_to_cart().name,
        input_kwargs={'Angle': reroute, 'Length': separate_xyz_1.outputs["X"], 'Origin': group_input.outputs["Origin"]})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: reroute, 1: separate_xyz.outputs["Y"]})
    
    polartocart_1 = nw.new_node(nodegroup_polar_to_cart().name,
        input_kwargs={'Angle': add, 'Length': separate_xyz_1.outputs["Y"], 'Origin': polartocart})
    
    add_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: add})
    
    polartocart_2 = nw.new_node(nodegroup_polar_to_cart().name,
        input_kwargs={'Angle': add_1, 'Length': separate_xyz_1.outputs["Z"], 'Origin': polartocart_1})
    
    switch4 = nw.new_node(nodegroup_switch4().name,
        input_kwargs={'Arg': index, 'Arg == 0': group_input.outputs["Origin"], 'Arg == 1': polartocart, 'Arg == 2': polartocart_1, 'Arg == 3': polartocart_2})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': mesh_line, 'Position': switch4})
    
    mesh_to_curve = nw.new_node(Nodes.MeshToCurve,
        input_kwargs={'Mesh': set_position})
    
    subdivide_curve_1 = nw.new_node(Nodes.SubdivideCurve,
        input_kwargs={'Curve': mesh_to_curve, 'Cuts': group_input.outputs["Resolution"]})
    
    integer = nw.new_node(Nodes.Integer,
        attrs={'integer': 2})
    integer.integer = 2
    
    bezier_segment = nw.new_node(Nodes.BezierSegment,
        input_kwargs={'Resolution': integer, 'Start': group_input.outputs["Origin"], 'Start Handle': polartocart, 'End Handle': polartocart_1, 'End': polartocart_2})
    
    divide = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Resolution"], 1: integer},
        attrs={'operation': 'DIVIDE'})
    
    subdivide_curve = nw.new_node(Nodes.SubdivideCurve,
        input_kwargs={'Curve': bezier_segment, 'Cuts': divide})
    
    switch = nw.new_node(Nodes.Switch,
        input_kwargs={1: group_input.outputs["Do Bezier"], 14: subdivide_curve_1, 15: subdivide_curve})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Curve': switch.outputs[6], 'Endpoint': polartocart_2})

@node_utils.to_nodegroup('nodegroup_simple_tube_v2', singleton=True, type='GeometryNodeTree')
def nodegroup_simple_tube_v2(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'length_rad1_rad2', (1.0, 0.5, 0.3)),
            ('NodeSocketVector', 'angles_deg', (0.0, 0.0, 0.0)),
            ('NodeSocketVector', 'proportions', (0.3333, 0.3333, 0.3333)),
            ('NodeSocketFloat', 'aspect', 1.0),
            ('NodeSocketBool', 'do_bezier', True),
            ('NodeSocketFloat', 'fullness', 4.0),
            ('NodeSocketVector', 'Origin', (0.0, 0.0, 0.0))])
    
    vector_sum = nw.new_node(nodegroup_vector_sum().name,
        input_kwargs={'Vector': group_input.outputs["proportions"]})
    
    divide = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["proportions"], 1: vector_sum},
        attrs={'operation': 'DIVIDE'})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group_input.outputs["length_rad1_rad2"]})
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: divide.outputs["Vector"], 'Scale': separate_xyz.outputs["X"]},
        attrs={'operation': 'SCALE'})
    
    polarbezier = nw.new_node(nodegroup_polar_bezier().name,
        input_kwargs={'Resolution': 25, 'Origin': group_input.outputs["Origin"], 'angles_deg': group_input.outputs["angles_deg"], 'Seg Lengths': scale.outputs["Vector"], 'Do Bezier': group_input.outputs["do_bezier"]})
    
    aspect_to_dim = nw.new_node(nodegroup_aspect_to_dim().name,
        input_kwargs={'Aspect Ratio': group_input.outputs["aspect"]})
    
    position = nw.new_node(Nodes.InputPosition)
    
    multiply = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: aspect_to_dim, 1: position},
        attrs={'operation': 'MULTIPLY'})
    
    warped_circle_curve = nw.new_node(nodegroup_warped_circle_curve().name,
        input_kwargs={'Position': multiply.outputs["Vector"], 'Vertices': 40})
    
    smoothtaper = nw.new_node(nodegroup_smooth_taper().name,
        input_kwargs={'start_rad': separate_xyz.outputs["Y"], 'end_rad': separate_xyz.outputs["Z"], 'fullness': group_input.outputs["fullness"]})
    
    profilepart = nw.new_node(nodegroup_profile_part().name,
        input_kwargs={'Skeleton Curve': polarbezier.outputs["Curve"], 'Profile Curve': warped_circle_curve, 'Radius Func': smoothtaper})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': profilepart, 'Skeleton Curve': polarbezier.outputs["Curve"], 'Endpoint': polarbezier.outputs["Endpoint"]})


@node_utils.to_nodegroup('nodegroup_smooth_taper', singleton=True, type='GeometryNodeTree')
def nodegroup_smooth_taper(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    spline_parameter = nw.new_node(Nodes.SplineParameter)
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: spline_parameter.outputs["Factor"], 1: 3.1416},
        attrs={'operation': 'MULTIPLY'})
    
    sine = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply},
        attrs={'operation': 'SINE'})
    
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'start_rad', 0.29),
            ('NodeSocketFloat', 'end_rad', 0.0),
            ('NodeSocketFloat', 'fullness', 2.5)])
    
    divide = nw.new_node(Nodes.Math,
        input_kwargs={0: 1.0, 1: group_input.outputs["fullness"]},
        attrs={'operation': 'DIVIDE'})
    
    power = nw.new_node(Nodes.Math,
        input_kwargs={0: sine, 1: divide},
        attrs={'operation': 'POWER'})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': spline_parameter.outputs["Factor"], 3: group_input.outputs["start_rad"], 4: group_input.outputs["end_rad"]},
        attrs={'clamp': False})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: power, 1: map_range.outputs["Result"]},
        attrs={'operation': 'MULTIPLY'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Value': multiply_1})

        
@node_utils.to_nodegroup('nodegroup_warped_circle_curve', singleton=True, type='GeometryNodeTree')
def nodegroup_warped_circle_curve(nw: NodeWrangler):
    # Code generated using version 2.4.2 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'Position', (0.0, 0.0, 0.0)),
            ('NodeSocketInt', 'Vertices', 32)])
    
    mesh_circle = nw.new_node(Nodes.MeshCircle,
        input_kwargs={'Vertices': group_input.outputs["Vertices"]})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': mesh_circle, 'Position': group_input.outputs["Position"]})
    
    mesh_to_curve = nw.new_node(Nodes.MeshToCurve,
        input_kwargs={'Mesh': set_position})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Curve': mesh_to_curve})

@node_utils.to_nodegroup('nodegroup_profile_part', singleton=True, type='GeometryNodeTree')
def nodegroup_profile_part(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Skeleton Curve', None),
            ('NodeSocketGeometry', 'Profile Curve', None),
            ('NodeSocketFloatDistance', 'Radius Func', 1.0)])
    
    set_curve_radius = nw.new_node(Nodes.SetCurveRadius,
        input_kwargs={'Curve': group_input.outputs["Skeleton Curve"], 'Radius': group_input.outputs["Radius Func"]})
    
    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
        input_kwargs={'Curve': set_curve_radius, 'Profile Curve': group_input.outputs["Profile Curve"], 'Fill Caps': True})
    
    set_shade_smooth = nw.new_node(Nodes.SetShadeSmooth,
        input_kwargs={'Geometry': curve_to_mesh, 'Shade Smooth': False})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_shade_smooth})