# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Hongyu Wen
# Acknowledgement: This file draws inspiration from https://www.youtube.com/watch?v=LJD3nvFXCLE by Redjam9


import bpy
import mathutils
from numpy.random import uniform, normal, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface

from infinigen.assets.creatures.util.creature import PartFactory, Part
from infinigen.assets.creatures.util.genome import Joint, IKParams
from infinigen.assets.creatures.util import part_util
from infinigen.core.util import blender as butil
from scipy.interpolate import interp1d
from infinigen.assets.creatures.util.part_util import nodegroup_to_part

from infinigen.assets.creatures.util.geometry import nurbs as nurbs_util
from infinigen.core import surface
import logging
import numpy as np

@node_utils.to_nodegroup('nodegroup_chameleon_toe', singleton=False, type='GeometryNodeTree')
def nodegroup_chameleon_toe(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    spiral = nw.new_node('GeometryNodeCurveSpiral',
        input_kwargs={'Rotations': 0.1000, 'Start Radius': 0.1000, 'End Radius': 0.3000, 'Height': 0.0000})
    
    spline_parameter = nw.new_node(Nodes.SplineParameter)
    
    float_curve = nw.new_node(Nodes.FloatCurve, input_kwargs={'Value': spline_parameter.outputs["Factor"]})
    node_utils.assign_curve(float_curve.mapping.curves[0], [(0.0000, 1.0000), (1.0000, 0.0000)])
    
    multiply = nw.new_node(Nodes.Math, input_kwargs={0: float_curve, 1: 0.4000}, attrs={'operation': 'MULTIPLY'})
    
    set_curve_radius = nw.new_node(Nodes.SetCurveRadius, input_kwargs={'Curve': spiral, 'Radius': multiply})
    
    spline_parameter_1 = nw.new_node(Nodes.SplineParameter)
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': set_curve_radius, 2: spline_parameter_1.outputs["Factor"]})
    
    curve_circle = nw.new_node(Nodes.CurveCircle, input_kwargs={'Radius': 0.1000})
    
    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
        input_kwargs={'Curve': capture_attribute.outputs["Geometry"], 'Profile Curve': curve_circle.outputs["Curve"]})
    
    store_named_attribute = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': curve_to_mesh, 'Name': 'Ridge', 'Value': capture_attribute.outputs[2]},
        attrs={'data_type': 'FLOAT', 'domain': 'POINT'})
    
    sample_curve = nw.new_node(Nodes.SampleCurve, input_kwargs={'Curve': set_curve_radius}, attrs={'mode': 'FACTOR'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': store_named_attribute, 'Position': sample_curve.outputs["Position"]},
        attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_floor_ceil', singleton=False, type='GeometryNodeTree')
def nodegroup_floor_ceil(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketFloat', 'Value', 0.0000)])
    
    float_to_integer = nw.new_node(Nodes.FloatToInt, input_kwargs={'Float': group_input.outputs["Value"]}, attrs={'rounding_mode': 'FLOOR'})
    
    float_to_integer_1 = nw.new_node(Nodes.FloatToInt,
        input_kwargs={'Float': group_input.outputs["Value"]},
        attrs={'rounding_mode': 'CEILING'})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Value"], 1: float_to_integer},
        attrs={'operation': 'SUBTRACT'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Floor': float_to_integer, 'Ceil': float_to_integer_1, 'Remainder': subtract},
        attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_clamp_or_wrap', singleton=False, type='GeometryNodeTree')
def nodegroup_clamp_or_wrap(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketInt', 'Value', 0),
            ('NodeSocketFloat', 'Max', 0.5000),
            ('NodeSocketBool', 'Use Wrap', False)])
    
    clamp = nw.new_node(Nodes.Clamp, input_kwargs={'Value': group_input.outputs["Value"], 'Max': group_input.outputs["Max"]})
    
    wrap = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Value"], 1: group_input.outputs["Max"], 2: 0.0000},
        attrs={'operation': 'WRAP'})
    
    switch = nw.new_node(Nodes.Switch,
        input_kwargs={0: group_input.outputs["Use Wrap"], 4: clamp, 5: wrap},
        attrs={'input_type': 'INT'})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Output': switch.outputs[1]}, attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_chameleon_claw_shape', singleton=False, type='GeometryNodeTree')
def nodegroup_chameleon_claw_shape(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    quadratic_bezier = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 32, 'Start': (0.0000, 0.0000, 0.0000), 'Middle': (0.5000, 0.5000, 0.0000), 'End': (0.7000, 0.3000, 0.0000)})
    
    simpletube = nw.new_node(nodegroup_simple_tube().name,
        input_kwargs={'Curve': quadratic_bezier, 'RadStartEnd': (0.2000, 0.2000, 1.0000)})
    
    quadratic_bezier_1 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 10, 'Start': (0.9500, 0.2500, 0.0000), 'Middle': (1.0000, 0.5000, 0.0000), 'End': (0.9500, 0.7500, 0.0000)})
    
    curveparametercurve = nw.new_node(nodegroup_curve_parameter_curve().name,
        input_kwargs={'Surface': simpletube.outputs["Mesh"], 'UVCurve': quadratic_bezier_1, 'CtrlptsU': 32, 'CtrlptsW': 32})
    
    curvesculpt = nw.new_node(nodegroup_curve_sculpt().name,
        input_kwargs={'Target': simpletube.outputs["Mesh"], 'Curve': curveparametercurve, 'Base Radius': 0.1000, 'Base Factor': 0.0200, 'Attr': True})
    
    chameleon_toe = nw.new_node(nodegroup_chameleon_toe().name)
    
    sample_curve = nw.new_node(Nodes.SampleCurve,
        input_kwargs={'Curve': simpletube.outputs["Curve"], 'Factor': 1.0000},
        attrs={'mode': 'FACTOR'})
    
    add = nw.new_node(Nodes.VectorMath, input_kwargs={0: sample_curve.outputs["Position"]})
    
    subtract = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: add.outputs["Vector"], 1: chameleon_toe.outputs["Position"]},
        attrs={'operation': 'SUBTRACT'})
    
    transform_1 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': chameleon_toe.outputs["Geometry"], 'Translation': subtract.outputs["Vector"], 'Rotation': (0.1745, -0.1745, 0.8727)})
    
    chameleon_toe_1 = nw.new_node(nodegroup_chameleon_toe().name)
    
    add_1 = nw.new_node(Nodes.VectorMath, input_kwargs={0: sample_curve.outputs["Position"]})
    
    subtract_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: add_1.outputs["Vector"], 1: chameleon_toe_1.outputs["Position"]},
        attrs={'operation': 'SUBTRACT'})
    
    transform_2 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': chameleon_toe_1.outputs["Geometry"], 'Translation': subtract_1.outputs["Vector"], 'Rotation': (0.0000, 0.1745, 0.8727)})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [curvesculpt.outputs["Geometry"], transform_1, transform_2]})
    
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVectorEuler', 'Rotation', (0.0000, 1.0472, 0.0000)),
            ('NodeSocketVectorXYZ', 'Scale', (0.2000, 0.2000, 0.4000))])
    
    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': join_geometry, 'Rotation': group_input.outputs["Rotation"], 'Scale': group_input.outputs["Scale"]})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': transform}, attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_u_v_param_to_vert_idxs', singleton=False, type='GeometryNodeTree')
def nodegroup_u_v_param_to_vert_idxs(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'Value', 0.5000),
            ('NodeSocketInt', 'Size', 0),
            ('NodeSocketBool', 'Cyclic', False)])
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Value"], 1: group_input.outputs["Size"]},
        attrs={'operation': 'MULTIPLY'})
    
    floorceil = nw.new_node(nodegroup_floor_ceil().name, input_kwargs={'Value': multiply})
    
    clamporwrap = nw.new_node(nodegroup_clamp_or_wrap().name,
        input_kwargs={'Value': floorceil.outputs["Floor"], 'Max': group_input.outputs["Size"], 'Use Wrap': group_input.outputs["Cyclic"]})
    
    clamporwrap_1 = nw.new_node(nodegroup_clamp_or_wrap().name,
        input_kwargs={'Value': floorceil.outputs["Ceil"], 'Max': group_input.outputs["Size"], 'Use Wrap': group_input.outputs["Cyclic"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Floor': clamporwrap, 'Ceil': clamporwrap_1, 'Remainder': floorceil.outputs["Remainder"]},
        attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_chameleon_foot_shape', singleton=False, type='GeometryNodeTree')
def nodegroup_chameleon_foot_shape(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    chameleon_claw_shape = nw.new_node(nodegroup_chameleon_claw_shape().name, input_kwargs={'Rotation': (0.0000, 0.0000, 0.0000)})
    
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVectorEuler', 'ouRotation', (0.0000, 1.0472, 0.0000)),
            ('NodeSocketVectorEuler', 'inRotation', (0.0000, 2.0944, 3.1416)),
            ('NodeSocketVectorXYZ', 'ouScale', (1.0000, 1.0000, 1.0000)),
            ('NodeSocketVectorXYZ', 'inScale', (1.0000, 1.0000, 1.0000))])
    
    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': chameleon_claw_shape, 'Rotation': group_input.outputs["ouRotation"], 'Scale': group_input.outputs["ouScale"]})
    
    chameleon_claw_shape_1 = nw.new_node(nodegroup_chameleon_claw_shape().name, input_kwargs={'Rotation': (0.0000, 0.0000, 0.0000)})
    
    transform_1 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': chameleon_claw_shape_1, 'Rotation': group_input.outputs["inRotation"], 'Scale': group_input.outputs["inScale"]})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [transform, transform_1]})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': join_geometry}, attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_bilinear_interp_index_transfer', singleton=False, type='GeometryNodeTree')
def nodegroup_bilinear_interp_index_transfer(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Source', None),
            ('NodeSocketFloat', 'U', 0.5000),
            ('NodeSocketFloat', 'V', 0.5000),
            ('NodeSocketVector', 'Attribute', (0.0000, 0.0000, 0.0000)),
            ('NodeSocketInt', 'SizeU', 0),
            ('NodeSocketInt', 'SizeV', 0),
            ('NodeSocketBool', 'CyclicU', False),
            ('NodeSocketBool', 'CyclicV', False)])
    
    uvparamtovertidxs = nw.new_node(nodegroup_u_v_param_to_vert_idxs().name,
        input_kwargs={'Value': group_input.outputs["V"], 'Size': group_input.outputs["SizeV"], 'Cyclic': group_input.outputs["CyclicV"]})
    
    uvparamtovertidxs_1 = nw.new_node(nodegroup_u_v_param_to_vert_idxs().name,
        input_kwargs={'Value': group_input.outputs["U"], 'Size': group_input.outputs["SizeU"], 'Cyclic': group_input.outputs["CyclicU"]})
    
    floor_floor = nw.new_node(Nodes.Math,
        input_kwargs={0: uvparamtovertidxs_1.outputs["Floor"], 1: group_input.outputs["SizeV"], 2: uvparamtovertidxs.outputs["Floor"]},
        label='FloorFloor',
        attrs={'operation': 'MULTIPLY_ADD'})
    
    transfer_attribute = nw.new_node(Nodes.SampleIndex,
        input_kwargs={'Geometry': group_input, 'Value': group_input.outputs["Attribute"], 'Index': floor_floor},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    ceil_floor = nw.new_node(Nodes.Math,
        input_kwargs={0: uvparamtovertidxs_1.outputs["Ceil"], 1: group_input.outputs["SizeV"], 2: uvparamtovertidxs.outputs["Floor"]},
        label='CeilFloor',
        attrs={'operation': 'MULTIPLY_ADD'})
    
    transfer_attribute_1 = nw.new_node(Nodes.SampleIndex,
        input_kwargs={'Geometry': group_input, 'Value': group_input.outputs["Attribute"], 'Index': ceil_floor},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Vector': uvparamtovertidxs_1.outputs["Remainder"], 9: (transfer_attribute, 'Value'), 10: (transfer_attribute_1, 'Value')},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    floor_ceil = nw.new_node(Nodes.Math,
        input_kwargs={0: uvparamtovertidxs_1.outputs["Floor"], 1: group_input.outputs["SizeV"], 2: uvparamtovertidxs.outputs["Ceil"]},
        label='FloorCeil',
        attrs={'operation': 'MULTIPLY_ADD'})
    
    transfer_attribute_2 = nw.new_node(Nodes.SampleIndex,
        input_kwargs={'Geometry': group_input, 'Value': group_input.outputs["Attribute"], 'Index': floor_ceil},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    ceil_ceil = nw.new_node(Nodes.Math,
        input_kwargs={0: uvparamtovertidxs_1.outputs["Ceil"], 1: group_input.outputs["SizeV"], 2: uvparamtovertidxs.outputs["Ceil"]},
        label='CeilCeil',
        attrs={'operation': 'MULTIPLY_ADD'})
    
    transfer_attribute_3 = nw.new_node(Nodes.SampleIndex,
        input_kwargs={'Geometry': group_input, 'Value': group_input.outputs["Attribute"], 'Index': ceil_ceil},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Vector': uvparamtovertidxs_1.outputs["Remainder"], 9: (transfer_attribute_2, 'Value'), 10: (transfer_attribute_3, 'Value')},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    map_range_2 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Vector': uvparamtovertidxs.outputs["Remainder"], 9: map_range.outputs["Vector"], 10: map_range_1.outputs["Vector"]},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Vector': map_range_2.outputs["Vector"]},
        attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_polar_to_cart', singleton=False, type='GeometryNodeTree')
def nodegroup_polar_to_cart(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'Angle', 0.5000),
            ('NodeSocketFloat', 'Length', 0.0000),
            ('NodeSocketVector', 'Origin', (0.0000, 0.0000, 0.0000))])
    
    cosine = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Angle"]}, attrs={'operation': 'COSINE'})
    
    sine = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Angle"]}, attrs={'operation': 'SINE'})
    
    construct_unit_vector = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': cosine, 'Z': sine}, label='Construct Unit Vector')
    
    offset_polar = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Length"], 1: construct_unit_vector, 2: group_input.outputs["Origin"]},
        label='Offset Polar',
        attrs={'operation': 'MULTIPLY_ADD'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Vector': offset_polar.outputs["Vector"]},
        attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_switch4', singleton=False, type='GeometryNodeTree')
def nodegroup_switch4(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketInt', 'Arg', 0),
            ('NodeSocketVector', 'Arg == 0', (0.0000, 0.0000, 0.0000)),
            ('NodeSocketVector', 'Arg == 1', (0.0000, 0.0000, 0.0000)),
            ('NodeSocketVector', 'Arg == 2', (0.0000, 0.0000, 0.0000)),
            ('NodeSocketVector', 'Arg == 3', (0.0000, 0.0000, 0.0000))])
    
    greater_equal = nw.new_node(Nodes.Compare,
        input_kwargs={2: group_input.outputs["Arg"], 3: 2},
        attrs={'data_type': 'INT', 'operation': 'GREATER_EQUAL'})
    
    greater_equal_1 = nw.new_node(Nodes.Compare,
        input_kwargs={2: group_input.outputs["Arg"], 3: 1},
        attrs={'data_type': 'INT', 'operation': 'GREATER_EQUAL'})
    
    switch_1 = nw.new_node(Nodes.Switch,
        input_kwargs={0: greater_equal_1, 8: group_input.outputs["Arg == 0"], 9: group_input.outputs["Arg == 1"]},
        attrs={'input_type': 'VECTOR'})
    
    greater_equal_2 = nw.new_node(Nodes.Compare,
        input_kwargs={2: group_input.outputs["Arg"], 3: 3},
        attrs={'data_type': 'INT', 'operation': 'GREATER_EQUAL'})
    
    switch_2 = nw.new_node(Nodes.Switch,
        input_kwargs={0: greater_equal_2, 8: group_input.outputs["Arg == 2"], 9: group_input.outputs["Arg == 3"]},
        attrs={'input_type': 'VECTOR'})
    
    switch = nw.new_node(Nodes.Switch,
        input_kwargs={0: greater_equal, 8: switch_1.outputs[3], 9: switch_2.outputs[3]},
        attrs={'input_type': 'VECTOR'})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Output': switch.outputs[3]}, attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_symmetric_clone', singleton=False, type='GeometryNodeTree')
def nodegroup_symmetric_clone(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketVectorXYZ', 'Scale', (1.0000, -1.0000, 1.0000))])
    
    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Scale': group_input.outputs["Scale"]})
    
    flip_faces = nw.new_node(Nodes.FlipFaces, input_kwargs={'Mesh': transform})
    
    join_geometry_2 = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [group_input.outputs["Geometry"], flip_faces]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Both': join_geometry_2, 'Orig': group_input.outputs["Geometry"], 'Inverted': flip_faces},
        attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_scale_bump', singleton=False, type='GeometryNodeTree')
def nodegroup_scale_bump(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketFloat', 'Density', 50.0000),
            ('NodeSocketFloat', 'Depth', 0.0050),
            ('NodeSocketFloat', 'Bump', 0.0100),
            ('NodeSocketInt', 'Level', 2),
            ('NodeSocketBool', 'Selection', True)])
    
    subdivide_mesh = nw.new_node(Nodes.SubdivideMesh,
        input_kwargs={'Mesh': group_input.outputs["Geometry"], 'Level': group_input.outputs["Level"]})
    
    normal = nw.new_node(Nodes.InputNormal)
    
    position = nw.new_node(Nodes.InputPosition)
    
    noise_texture_1 = nw.new_node(Nodes.NoiseTexture, input_kwargs={'Vector': position})
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: noise_texture_1.outputs["Color"], 'Scale': 0.2000},
        attrs={'operation': 'SCALE'})
    
    add = nw.new_node(Nodes.VectorMath, input_kwargs={0: scale.outputs["Vector"], 1: position})
    
    voronoi_texture_1 = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': add.outputs["Vector"], 'Scale': group_input.outputs["Density"], 'Randomness': 0.5000},
        attrs={'feature': 'DISTANCE_TO_EDGE'})
    
    colorramp_1 = nw.new_node(Nodes.ColorRamp, input_kwargs={'Fac': voronoi_texture_1.outputs["Distance"]})
    colorramp_1.color_ramp.elements[0].position = 0.0000
    colorramp_1.color_ramp.elements[0].color = [0.0000, 0.0000, 0.0000, 1.0000]
    colorramp_1.color_ramp.elements[1].position = 0.9909
    colorramp_1.color_ramp.elements[1].color = [1.0000, 1.0000, 1.0000, 1.0000]
    
    scale_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: colorramp_1.outputs["Color"], 'Scale': group_input.outputs["Bump"]},
        attrs={'operation': 'SCALE'})
    
    scale_2 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: normal, 'Scale': scale_1.outputs["Vector"]},
        attrs={'operation': 'SCALE'})
    
    colorramp = nw.new_node(Nodes.ColorRamp, input_kwargs={'Fac': voronoi_texture_1.outputs["Distance"]})
    colorramp.color_ramp.elements[0].position = 0.0000
    colorramp.color_ramp.elements[0].color = [0.0000, 0.0000, 0.0000, 1.0000]
    colorramp.color_ramp.elements[1].position = 0.0591
    colorramp.color_ramp.elements[1].color = [1.0000, 1.0000, 1.0000, 1.0000]
    
    scale_3 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: colorramp.outputs["Color"], 'Scale': group_input.outputs["Depth"]},
        attrs={'operation': 'SCALE'})
    
    scale_4 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: normal, 'Scale': scale_3.outputs["Vector"]},
        attrs={'operation': 'SCALE'})
    
    add_1 = nw.new_node(Nodes.VectorMath, input_kwargs={0: scale_2.outputs["Vector"], 1: scale_4.outputs["Vector"]})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': subdivide_mesh, 'Selection': group_input.outputs["Selection"], 'Offset': add_1.outputs["Vector"]})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': set_position}, attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_chameleon_leg_raw_shape', singleton=False, type='GeometryNodeTree')
def nodegroup_chameleon_leg_raw_shape(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'thigh_length', 0.6000),
            ('NodeSocketFloat', 'calf_length', 0.5000),
            ('NodeSocketFloat', 'thigh_body_rotation', 0.5000),
            ('NodeSocketFloat', 'calf_body_rotation', 0.5000),
            ('NodeSocketFloat', 'thigh_calf_rotation', 20.0000),
            ('NodeSocketFloat', 'toe_toe_rotation', 20.0000),
            ('NodeSocketVectorXYZ', 'thigh_scale', (1.0000, 0.6500, 1.0000)),
            ('NodeSocketVectorXYZ', 'calf_scale', (1.0000, 0.6500, 1.0000)),
            ('NodeSocketVectorXYZ', 'ouScale', (1.0000, 1.0000, 1.0000)),
            ('NodeSocketVectorXYZ', 'inScale', (1.0000, 1.0000, 1.0000))])
    
    multiply = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["thigh_length"]}, attrs={'operation': 'MULTIPLY'})
    
    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': multiply})
    
    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': group_input.outputs["thigh_length"]})
    
    quadratic_bezier = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 64, 'Start': (0.0000, 0.0000, 0.0000), 'Middle': combine_xyz_3, 'End': combine_xyz_2})
    
    simpletube = nw.new_node(nodegroup_simple_tube().name,
        input_kwargs={'Curve': quadratic_bezier, 'RadStartEnd': (0.1500, 0.2000, 0.9000), 'Resolution': 64})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["thigh_calf_rotation"], 1: -1.0000},
        attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["thigh_body_rotation"], 1: 180.0000})
    
    combine_xyz_7 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Y': multiply_1, 'Z': add})
    
    scale = nw.new_node(Nodes.VectorMath, input_kwargs={0: combine_xyz_7, 'Scale': 0.0174}, attrs={'operation': 'SCALE'})
    
    transform_geometry = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': simpletube.outputs["Mesh"], 'Rotation': scale.outputs["Vector"], 'Scale': group_input.outputs["thigh_scale"]})
    
    round_bump = nw.new_node(nodegroup_round_bump().name,
        input_kwargs={'Geometry': transform_geometry, 'Distance': 0.0070, 'Offset Scale': 0.0020})
    
    multiply_2 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["calf_length"]}, attrs={'operation': 'MULTIPLY'})
    
    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': multiply_2})
    
    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': group_input.outputs["calf_length"]})
    
    quadratic_bezier_1 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 64, 'Start': (0.0000, 0.0000, 0.0000), 'Middle': combine_xyz_4, 'End': combine_xyz_5})
    
    simpletube_1 = nw.new_node(nodegroup_simple_tube().name,
        input_kwargs={'Curve': quadratic_bezier_1, 'RadStartEnd': (0.1500, 0.1000, 0.9000), 'Resolution': 64})
    
    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["calf_body_rotation"], 1: 180.0000})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Y': group_input.outputs["thigh_calf_rotation"], 'Z': add_1})
    
    scale_1 = nw.new_node(Nodes.VectorMath, input_kwargs={0: combine_xyz, 'Scale': 0.0174}, attrs={'operation': 'SCALE'})
    
    transform_geometry_1 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': simpletube_1.outputs["Mesh"], 'Rotation': scale_1.outputs["Vector"], 'Scale': group_input.outputs["calf_scale"]})
    
    round_bump_1 = nw.new_node(nodegroup_round_bump().name,
        input_kwargs={'Geometry': transform_geometry_1, 'Distance': 0.0070, 'Offset Scale': 0.0020})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: 180.0000, 1: group_input.outputs["thigh_calf_rotation"]},
        attrs={'operation': 'SUBTRACT'})
    
    multiply_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["toe_toe_rotation"], 1: -1.0000},
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Y': subtract, 'Z': multiply_3})
    
    scale_2 = nw.new_node(Nodes.VectorMath, input_kwargs={0: combine_xyz_1, 'Scale': 0.0174}, attrs={'operation': 'SCALE'})
    
    add_2 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["toe_toe_rotation"], 1: 180.0000})
    
    combine_xyz_6 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Y': group_input.outputs["thigh_calf_rotation"], 'Z': add_2})
    
    scale_3 = nw.new_node(Nodes.VectorMath, input_kwargs={0: combine_xyz_6, 'Scale': 0.0174}, attrs={'operation': 'SCALE'})
    
    chameleon_foot_shape = nw.new_node(nodegroup_chameleon_foot_shape().name,
        input_kwargs={'ouRotation': scale_2.outputs["Vector"], 'inRotation': scale_3.outputs["Vector"], 'ouScale': group_input.outputs["ouScale"], 'inScale': group_input.outputs["inScale"]})
    
    transform_geometry_2 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': simpletube_1.outputs["Curve"], 'Rotation': scale_1.outputs["Vector"], 'Scale': (1.0000, 0.6500, 1.0000)})
    
    sample_curve = nw.new_node(Nodes.SampleCurve,
        input_kwargs={'Curve': transform_geometry_2, 'Factor': 0.8500},
        attrs={'mode': 'FACTOR'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': chameleon_foot_shape, 'Offset': sample_curve.outputs["Position"]})
    
    round_bump_2 = nw.new_node(nodegroup_round_bump().name,
        input_kwargs={'Geometry': set_position, 'Distance': 0.0050, 'Offset Scale': 0.0020})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [round_bump, round_bump_1, round_bump_2]})
    
    transform_geometry_3 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': simpletube.outputs["Curve"], 'Rotation': scale.outputs["Vector"], 'Scale': group_input.outputs["thigh_scale"]})
    
    sample_curve_1 = nw.new_node(Nodes.SampleCurve,
        input_kwargs={'Curve': transform_geometry_3, 'Factor': 1.0000},
        attrs={'mode': 'FACTOR'})
    
    scale_4 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: sample_curve_1.outputs["Position"], 'Scale': -1.0000},
        attrs={'operation': 'SCALE'})
    
    set_position_1 = nw.new_node(Nodes.SetPosition, input_kwargs={'Geometry': join_geometry, 'Offset': scale_4.outputs["Vector"]})
    
    subdivision_surface = nw.new_node(Nodes.SubdivisionSurface, input_kwargs={'Mesh': set_position_1})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Mesh': subdivision_surface}, attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_chameleon_tail_shape', singleton=False, type='GeometryNodeTree')
def nodegroup_chameleon_tail_shape(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    quadratic_bezier = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 64, 'Middle': (0.0000, 0.2000, 0.0000), 'End': (2.0000, -0.5000, 0.0000)})
    
    simpletube = nw.new_node(nodegroup_simple_tube().name,
        input_kwargs={'Curve': quadratic_bezier, 'RadStartEnd': (0.4000, 0.0000, 0.9000), 'Resolution': 64})
    
    quadratic_bezier_1 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 64, 'Start': (0.2000, 0.0000, 0.0000), 'Middle': (0.6000, 0.0000, 0.0100), 'End': (0.8000, 0.0000, 0.0200)})
    
    curveparametercurve = nw.new_node(nodegroup_curve_parameter_curve().name,
        input_kwargs={'Surface': simpletube.outputs["Mesh"], 'UVCurve': quadratic_bezier_1, 'CtrlptsU': 64, 'CtrlptsW': 64})
    
    curvesculpt = nw.new_node(nodegroup_curve_sculpt().name,
        input_kwargs={'Target': simpletube.outputs["Mesh"], 'Curve': curveparametercurve, 'Base Radius': 0.0200, 'SymmY': False, 'Attr': True})
    
    subdivision_surface = nw.new_node(Nodes.SubdivisionSurface, input_kwargs={'Mesh': curvesculpt.outputs["Geometry"], 'Level': 2})
    
    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': subdivision_surface, 'Translation': (1.0000, 0.0000, 0.1000), 'Rotation': (-1.5708, 0.0000, 0.0000)})
    
    transform_1 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': quadratic_bezier, 'Translation': (1.0000, 0.0000, 0.0000), 'Rotation': (-1.5708, 0.0000, 0.0000)})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Mesh': transform, 'Curve': transform_1},
        attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_back_bump1', singleton=False, type='GeometryNodeTree')
def nodegroup_back_bump1(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Surface', None)])
    
    quadratic_bezier_1 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 25, 'Start': (0.0000, 0.7500, 0.1000), 'Middle': (0.6000, 0.7500, 0.0000), 'End': (1.0000, 0.7500, 0.1000)})
    
    curveparametercurve = nw.new_node(nodegroup_curve_parameter_curve().name,
        input_kwargs={'Surface': group_input.outputs["Surface"], 'UVCurve': quadratic_bezier_1, 'CtrlptsU': 64, 'CtrlptsW': 64})
    
    curvesculpt = nw.new_node(nodegroup_curve_sculpt().name,
        input_kwargs={'Target': group_input.outputs["Surface"], 'Curve': curveparametercurve, 'Base Radius': 0.3000, 'Base Factor': 0.0300, 'Name': ''})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': curvesculpt.outputs["Geometry"]},
        attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_back_bump2', singleton=False, type='GeometryNodeTree')
def nodegroup_back_bump2(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Surface', None)])
    
    quadratic_bezier_1 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 64, 'Start': (0.1000, 0.7500, 0.1000), 'Middle': (0.4000, 0.7500, 0.0000), 'End': (0.9000, 0.7500, 0.1000)})
    
    curveparametercurve = nw.new_node(nodegroup_curve_parameter_curve().name,
        input_kwargs={'Surface': group_input.outputs["Surface"], 'UVCurve': quadratic_bezier_1, 'CtrlptsU': 64, 'CtrlptsW': 64})
    
    curvesculpt = nw.new_node(nodegroup_curve_sculpt().name,
        input_kwargs={'Target': group_input.outputs["Surface"], 'Curve': curveparametercurve, 'Base Radius': 0.1500, 'Base Factor': 0.1000})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': curvesculpt.outputs["Geometry"]},
        attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_back_bump3', singleton=False, type='GeometryNodeTree')
def nodegroup_back_bump3(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Surface', None)])
    
    quadratic_bezier_1 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 25, 'Start': (0.1500, 0.7500, 0.0600), 'Middle': (0.6000, 0.7500, 0.0000), 'End': (0.9000, 0.7500, 0.0600)})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': quadratic_bezier_1})
    
    curveparametercurve = nw.new_node(nodegroup_curve_parameter_curve().name,
        input_kwargs={'Surface': group_input.outputs["Surface"], 'UVCurve': join_geometry, 'CtrlptsU': 64, 'CtrlptsW': 64})
    
    curvesculpt = nw.new_node(nodegroup_curve_sculpt().name,
        input_kwargs={'Target': group_input.outputs["Surface"], 'Curve': curveparametercurve, 'Base Radius': 0.1000, 'Attr': True})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': curvesculpt.outputs["Geometry"]},
        attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_belly_sunken1', singleton=False, type='GeometryNodeTree')
def nodegroup_belly_sunken1(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Surface', None)])
    
    quadratic_bezier_1 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 30, 'Start': (0.0000, 0.2500, 0.0000), 'Middle': (0.6000, 0.2500, 0.0000), 'End': (1.0000, 0.2500, 0.0000)})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': quadratic_bezier_1})
    
    curveparametercurve = nw.new_node(nodegroup_curve_parameter_curve().name,
        input_kwargs={'Surface': group_input.outputs["Surface"], 'UVCurve': join_geometry, 'CtrlptsU': 64, 'CtrlptsW': 64})
    
    curvesculpt = nw.new_node(nodegroup_curve_sculpt().name,
        input_kwargs={'Target': group_input.outputs["Surface"], 'Curve': curveparametercurve, 'Base Radius': 0.0300, 'Base Factor': 0.0200, 'Name': ''})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': curvesculpt.outputs["Geometry"]},
        attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_shouder_sunken', singleton=False, type='GeometryNodeTree')
def nodegroup_shouder_sunken(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Surface', None)])
    
    quadratic_bezier_1 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 25, 'Start': (0.1500, 0.2500, 0.1000), 'Middle': (0.2000, 0.2500, 0.0000), 'End': (0.3000, 0.2500, 0.1000)})
    
    curveparametercurve = nw.new_node(nodegroup_curve_parameter_curve().name,
        input_kwargs={'Surface': group_input.outputs["Surface"], 'UVCurve': quadratic_bezier_1, 'CtrlptsU': 64, 'CtrlptsW': 64})
    
    curvesculpt = nw.new_node(nodegroup_curve_sculpt().name,
        input_kwargs={'Target': group_input.outputs["Surface"], 'Curve': curveparametercurve, 'Base Radius': 0.2000, 'Base Factor': -0.0300, 'SymmY': False})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': curvesculpt.outputs["Geometry"]},
        attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_neck_bump', singleton=False, type='GeometryNodeTree')
def nodegroup_neck_bump(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Surface', None)])
    
    quadratic_bezier_1 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 25, 'Start': (0.0000, 0.2500, 0.0000), 'Middle': (0.0500, 0.2500, 0.0000), 'End': (0.0700, 0.2500, 0.1000)})
    
    curveparametercurve = nw.new_node(nodegroup_curve_parameter_curve().name,
        input_kwargs={'Surface': group_input.outputs["Surface"], 'UVCurve': quadratic_bezier_1, 'CtrlptsU': 64, 'CtrlptsW': 64})
    
    curvesculpt = nw.new_node(nodegroup_curve_sculpt().name,
        input_kwargs={'Target': group_input.outputs["Surface"], 'Curve': curveparametercurve, 'Base Radius': 0.2000, 'SymmY': False})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': curvesculpt.outputs["Geometry"]},
        attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_neck_bump2', singleton=False, type='GeometryNodeTree')
def nodegroup_neck_bump2(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Surface', None)])
    
    quadratic_bezier_1 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 25, 'Start': (0.0000, 0.2500, 0.0000), 'Middle': (0.0250, 0.2500, 0.1000), 'End': (0.0500, 0.2500, 0.2000)})
    
    curveparametercurve = nw.new_node(nodegroup_curve_parameter_curve().name,
        input_kwargs={'Surface': group_input.outputs["Surface"], 'UVCurve': quadratic_bezier_1, 'CtrlptsU': 64, 'CtrlptsW': 64})
    
    curvesculpt = nw.new_node(nodegroup_curve_sculpt().name,
        input_kwargs={'Target': group_input.outputs["Surface"], 'Curve': curveparametercurve, 'Base Radius': 0.2000, 'SymmY': False})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': curvesculpt.outputs["Geometry"]},
        attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_curve_parameter_curve', singleton=False, type='GeometryNodeTree')
def nodegroup_curve_parameter_curve(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Surface', None),
            ('NodeSocketGeometry', 'UVCurve', None),
            ('NodeSocketInt', 'CtrlptsU', 0),
            ('NodeSocketInt', 'CtrlptsW', 0)])
    
    normal = nw.new_node(Nodes.InputNormal)
    
    position = nw.new_node(Nodes.InputPosition)
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': position})
    
    position_1 = nw.new_node(Nodes.InputPosition)
    
    bilinearinterpindextransfer = nw.new_node(nodegroup_bilinear_interp_index_transfer().name,
        input_kwargs={'Source': group_input.outputs["Surface"], 'U': separate_xyz.outputs["X"], 'V': separate_xyz.outputs["Y"], 'Attribute': position_1, 'SizeU': group_input.outputs["CtrlptsU"], 'SizeV': group_input.outputs["CtrlptsW"], 'CyclicV': True})
    
    transfer_attribute = nw.new_node(Nodes.SampleNearestSurface,
        input_kwargs={'Mesh': group_input.outputs["Surface"], 'Value': normal, 'Sample Position': bilinearinterpindextransfer},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    multiply_add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: (transfer_attribute, 'Value'), 1: separate_xyz.outputs["Z"], 2: bilinearinterpindextransfer},
        attrs={'operation': 'MULTIPLY_ADD'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["UVCurve"], 'Position': multiply_add.outputs["Vector"]})
    
    normal_1 = nw.new_node(Nodes.InputNormal)
    
    dot_product = nw.new_node(Nodes.VectorMath, input_kwargs={0: normal, 1: normal_1}, attrs={'operation': 'DOT_PRODUCT'})
    
    arcsine = nw.new_node(Nodes.Math, input_kwargs={0: dot_product.outputs["Value"]}, attrs={'operation': 'ARCSINE'})
    
    set_curve_tilt = nw.new_node(Nodes.SetCurveTilt, input_kwargs={'Curve': set_position, 'Tilt': arcsine})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': set_curve_tilt}, attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_polar_bezier', singleton=False, type='GeometryNodeTree')
def nodegroup_polar_bezier(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketIntUnsigned', 'Resolution', 32),
            ('NodeSocketVector', 'Origin', (0.0000, 0.0000, 0.0000)),
            ('NodeSocketVector', 'angles_deg', (0.0000, 0.0000, 0.0000)),
            ('NodeSocketVector', 'Seg Lengths', (0.3000, 0.3000, 0.3000)),
            ('NodeSocketBool', 'Do Bezier', True)])
    
    mesh_line = nw.new_node(Nodes.MeshLine, input_kwargs={'Count': 4})
    
    index = nw.new_node(Nodes.Index)
    
    deg2_rad = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["angles_deg"], 'Scale': 0.0175},
        label='Deg2Rad',
        attrs={'operation': 'SCALE'})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': deg2_rad.outputs["Vector"]})
    
    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': group_input.outputs["Seg Lengths"]})
    
    polartocart = nw.new_node(nodegroup_polar_to_cart().name,
        input_kwargs={'Angle': separate_xyz, 'Length': separate_xyz_1.outputs["X"], 'Origin': group_input.outputs["Origin"]})
    
    add = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz, 1: separate_xyz.outputs["Y"]})
    
    polartocart_1 = nw.new_node(nodegroup_polar_to_cart().name,
        input_kwargs={'Angle': add, 'Length': separate_xyz_1.outputs["Y"], 'Origin': polartocart})
    
    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz.outputs["Z"], 1: add})
    
    polartocart_2 = nw.new_node(nodegroup_polar_to_cart().name,
        input_kwargs={'Angle': add_1, 'Length': separate_xyz_1.outputs["Z"], 'Origin': polartocart_1})
    
    switch4 = nw.new_node(nodegroup_switch4().name,
        input_kwargs={'Arg': index, 'Arg == 0': group_input.outputs["Origin"], 'Arg == 1': polartocart, 'Arg == 2': polartocart_1, 'Arg == 3': polartocart_2})
    
    set_position = nw.new_node(Nodes.SetPosition, input_kwargs={'Geometry': mesh_line, 'Position': switch4})
    
    mesh_to_curve = nw.new_node(Nodes.MeshToCurve, input_kwargs={'Mesh': set_position})
    
    subdivide_curve_1 = nw.new_node(Nodes.SubdivideCurve, input_kwargs={'Curve': mesh_to_curve, 'Cuts': group_input.outputs["Resolution"]})
    
    integer = nw.new_node(Nodes.Integer)
    integer.integer = 2
    
    bezier_segment = nw.new_node(Nodes.CurveBezierSegment,
        input_kwargs={'Resolution': integer, 'Start': group_input.outputs["Origin"], 'Start Handle': polartocart, 'End Handle': polartocart_1, 'End': polartocart_2})
    
    divide = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Resolution"], 1: integer},
        attrs={'operation': 'DIVIDE'})
    
    subdivide_curve = nw.new_node(Nodes.SubdivideCurve, input_kwargs={'Curve': bezier_segment, 'Cuts': divide})
    
    switch = nw.new_node(Nodes.Switch,
        input_kwargs={1: group_input.outputs["Do Bezier"], 14: subdivide_curve_1, 15: subdivide_curve})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Curve': switch.outputs[6], 'Endpoint': polartocart_2},
        attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_simple_tube', singleton=False, type='GeometryNodeTree')
def nodegroup_simple_tube(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Curve', None),
            ('NodeSocketVector', 'RadStartEnd', (0.0500, 0.0500, 1.0000)),
            ('NodeSocketInt', 'Resolution', 32)])
    
    spline_parameter = nw.new_node(Nodes.SplineParameter)
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: 1.0000, 1: spline_parameter.outputs["Factor"]},
        attrs={'operation': 'SUBTRACT'})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract, 1: spline_parameter.outputs["Factor"]},
        attrs={'operation': 'MULTIPLY'})
    
    sqrt = nw.new_node(Nodes.Math, input_kwargs={0: multiply}, attrs={'operation': 'SQRT'})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': group_input.outputs["RadStartEnd"]})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': spline_parameter.outputs["Factor"], 3: separate_xyz.outputs["X"], 4: separate_xyz.outputs["Y"]})
    
    multiply_1 = nw.new_node(Nodes.Math, input_kwargs={0: sqrt, 1: map_range.outputs["Result"]}, attrs={'operation': 'MULTIPLY'})
    
    set_curve_radius = nw.new_node(Nodes.SetCurveRadius, input_kwargs={'Curve': group_input.outputs["Curve"], 'Radius': multiply_1})
    
    curve_circle = nw.new_node(Nodes.CurveCircle, input_kwargs={'Resolution': group_input.outputs["Resolution"]})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': 1.0000, 'Y': separate_xyz.outputs["Z"]})
    
    transform = nw.new_node(Nodes.Transform, input_kwargs={'Geometry': curve_circle.outputs["Curve"], 'Scale': combine_xyz})
    
    curve_to_mesh = nw.new_node(Nodes.CurveToMesh, input_kwargs={'Curve': set_curve_radius, 'Profile Curve': transform})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Mesh': curve_to_mesh, 'Curve': set_curve_radius},
        attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_curve_sculpt', singleton=False, type='GeometryNodeTree')
def nodegroup_curve_sculpt(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input_1 = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Target', None),
            ('NodeSocketGeometry', 'Curve', None),
            ('NodeSocketFloat', 'Base Radius', 0.0500),
            ('NodeSocketFloat', 'Base Factor', 0.0500),
            ('NodeSocketBool', 'SymmY', True),
            ('NodeSocketGeometry', 'StrokeRadFacModifier', None),
            ('NodeSocketBool', 'Switch', True),
            ('NodeSocketBool', 'Attr', False),
            ('NodeSocketString', 'Name', 'Ridge')])
    
    normal = nw.new_node(Nodes.InputNormal)
    
    symmetric_clone = nw.new_node(nodegroup_symmetric_clone().name, input_kwargs={'Geometry': group_input_1.outputs["Curve"]})
    
    switch = nw.new_node(Nodes.Switch,
        input_kwargs={1: group_input_1.outputs["SymmY"], 14: group_input_1.outputs["Curve"], 15: symmetric_clone.outputs["Both"]})
    
    curve_to_mesh = nw.new_node(Nodes.CurveToMesh, input_kwargs={'Curve': switch.outputs[6]})
    
    geometry_proximity = nw.new_node(Nodes.Proximity, input_kwargs={'Target': curve_to_mesh}, attrs={'target_element': 'POINTS'})
    
    curve_to_mesh_1 = nw.new_node(Nodes.CurveToMesh, input_kwargs={'Curve': group_input_1.outputs["StrokeRadFacModifier"]})
    
    position = nw.new_node(Nodes.InputPosition)
    
    index = nw.new_node(Nodes.Index)
    
    transfer_attribute = nw.new_node(Nodes.SampleIndex,
        input_kwargs={'Geometry': curve_to_mesh_1, 'Value': position, 'Index': index},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': (transfer_attribute, 'Value')})
    
    add = nw.new_node(Nodes.Math, input_kwargs={0: group_input_1.outputs["Base Radius"], 1: separate_xyz.outputs["X"]})
    
    map_range = nw.new_node(Nodes.MapRange, input_kwargs={'Value': geometry_proximity.outputs["Distance"], 2: add})
    
    float_curve_1 = nw.new_node(Nodes.FloatCurve, input_kwargs={'Value': map_range.outputs["Result"]})
    node_utils.assign_curve(float_curve_1.mapping.curves[0], [(0.0000, 1.0000), (0.4364, 0.9212), (0.6182, 0.0787), (1.0000, 0.0000)], handles=['VECTOR', 'AUTO', 'AUTO', 'VECTOR'])
    
    float_curve = nw.new_node(Nodes.FloatCurve, input_kwargs={'Value': map_range.outputs["Result"]})
    node_utils.assign_curve(float_curve.mapping.curves[0], [(0.0000, 1.0000), (0.2500, 0.9588), (0.7455, 0.0475), (1.0000, 0.0000)], handles=['VECTOR', 'AUTO', 'AUTO', 'VECTOR'])
    
    switch_2 = nw.new_node(Nodes.Switch,
        input_kwargs={0: group_input_1.outputs["Switch"], 2: float_curve_1, 3: float_curve},
        attrs={'input_type': 'FLOAT'})
    
    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: group_input_1.outputs["Base Factor"], 1: separate_xyz.outputs["Y"]})
    
    multiply = nw.new_node(Nodes.Math, input_kwargs={0: switch_2.outputs["Output"], 1: add_1}, attrs={'operation': 'MULTIPLY'})
    
    scale = nw.new_node(Nodes.VectorMath, input_kwargs={0: normal, 'Scale': multiply}, attrs={'operation': 'SCALE'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input_1.outputs["Target"], 'Offset': scale.outputs["Vector"]})
    
    named_attribute = nw.new_node(Nodes.NamedAttribute, input_kwargs={'Name': group_input_1.outputs["Name"]})
    
    maximum = nw.new_node(Nodes.Math,
        input_kwargs={0: named_attribute.outputs[1], 1: switch_2.outputs["Output"]},
        attrs={'use_clamp': True, 'operation': 'MAXIMUM'})
    
    store_named_attribute = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': set_position, 'Name': group_input_1.outputs["Name"], 'Value': maximum},
        attrs={'data_type': 'FLOAT', 'domain': 'POINT'})
    
    switch_3 = nw.new_node(Nodes.Switch,
        input_kwargs={1: group_input_1.outputs["Attr"], 14: set_position, 15: store_named_attribute})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': switch_3.outputs[6], 'Result': switch_2.outputs["Output"]},
        attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_chameleon_eye', singleton=False, type='GeometryNodeTree')
def nodegroup_chameleon_eye(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    polarbezier = nw.new_node(nodegroup_polar_bezier().name,
        input_kwargs={'Resolution': 1024, 'angles_deg': (0.0000, 0.0000, 10.0000), 'Seg Lengths': (0.1500, 0.1500, 0.1500)})
    
    simpletube = nw.new_node(nodegroup_simple_tube().name,
        input_kwargs={'Curve': polarbezier.outputs["Curve"], 'RadStartEnd': (0.4000, 0.4000, 1.0000), 'Resolution': 1024})
    
    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': simpletube.outputs["Mesh"], 'Scale': (4.0000, 4.5000, 4.5000)})
    
    quadratic_bezier_25 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 256, 'Start': (0.9900, 0.0000, 0.0000), 'Middle': (0.9900, 0.5000, 0.0000), 'End': (0.9900, 1.0000, 0.0000)})
    
    curveparametercurve = nw.new_node(nodegroup_curve_parameter_curve().name,
        input_kwargs={'Surface': transform, 'UVCurve': quadratic_bezier_25, 'CtrlptsU': 1024, 'CtrlptsW': 1024})
    
    curvesculpt = nw.new_node(nodegroup_curve_sculpt().name,
        input_kwargs={'Target': transform, 'Curve': curveparametercurve, 'Base Factor': 0.1000})
    
    quadratic_bezier_26 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 200, 'Start': (1.0000, 0.0000, 0.0000), 'Middle': (1.0000, 0.5000, 0.0000), 'End': (1.0000, 1.0000, 0.0000)})
    
    curveparametercurve_1 = nw.new_node(nodegroup_curve_parameter_curve().name,
        input_kwargs={'Surface': curvesculpt.outputs["Geometry"], 'UVCurve': quadratic_bezier_26, 'CtrlptsU': 1024, 'CtrlptsW': 1024})
    
    group_input = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketFloat', 'pupil_radius', 0.2200)])
    
    add = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["pupil_radius"], 1: 0.0300})
    
    curvesculpt_1 = nw.new_node(nodegroup_curve_sculpt().name,
        input_kwargs={'Target': curvesculpt.outputs["Geometry"], 'Curve': curveparametercurve_1, 'Base Radius': add, 'Base Factor': 0.0000, 'Switch': False, 'Attr': True})
    
    quadratic_bezier_27 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 256, 'Start': (1.0000, 0.0000, 0.0000), 'Middle': (1.0000, 0.5000, 0.0000), 'End': (1.0000, 1.0000, 0.0000)})
    
    curveparametercurve_2 = nw.new_node(nodegroup_curve_parameter_curve().name,
        input_kwargs={'Surface': curvesculpt_1.outputs["Geometry"], 'UVCurve': quadratic_bezier_27, 'CtrlptsU': 1024, 'CtrlptsW': 1024})
    
    curvesculpt_2 = nw.new_node(nodegroup_curve_sculpt().name,
        input_kwargs={'Target': curvesculpt_1.outputs["Geometry"], 'Curve': curveparametercurve_2, 'Base Radius': group_input.outputs["pupil_radius"], 'Base Factor': 0.0000, 'Switch': False, 'Attr': True, 'Name': 'Pupil'})
    
    op_or = nw.new_node(Nodes.BooleanMath,
        input_kwargs={0: curvesculpt_1.outputs["Result"], 1: curvesculpt_2.outputs["Result"]},
        attrs={'operation': 'OR'})
    
    op_not = nw.new_node(Nodes.BooleanMath, input_kwargs={0: op_or}, attrs={'operation': 'NOT'})
    
    scale_bump = nw.new_node(nodegroup_scale_bump().name,
        input_kwargs={'Geometry': curvesculpt_2.outputs["Geometry"], 'Density': 20.0000, 'Depth': 0.1000, 'Bump': 0.0200, 'Level': 0, 'Selection': op_not})
    
    normal = nw.new_node(Nodes.InputNormal)
    
    position = nw.new_node(Nodes.InputPosition)
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': position})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'W': separate_xyz.outputs["X"], 'Scale': 12.0000, 'Detail': 10.0000, 'Roughness': 0.0000},
        attrs={'noise_dimensions': '1D'})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: noise_texture.outputs["Fac"], 1: 0.0300},
        attrs={'use_clamp': True, 'operation': 'MULTIPLY'})
    
    scale = nw.new_node(Nodes.VectorMath, input_kwargs={0: normal, 'Scale': multiply}, attrs={'operation': 'SCALE'})
    
    set_position = nw.new_node(Nodes.SetPosition, input_kwargs={'Geometry': scale_bump, 'Offset': scale.outputs["Vector"]})
    
    set_material = nw.new_node(Nodes.SetMaterial,
        input_kwargs={'Geometry': set_position, 'Material': surface.shaderfunc_to_material(shader_chameleon_eye)})
    
    transform_1 = nw.new_node(Nodes.Transform, input_kwargs={'Geometry': set_material, 'Scale': (0.0500, 0.0600, 0.0600)})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': transform_1}, attrs={'is_active_output': True})

def shader_chameleon(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    attribute_1 = nw.new_node(Nodes.Attribute, attrs={'attribute_name': 'Ridge', 'attribute_type': 'GEOMETRY'})
    
    # map_range = nw.new_node(Nodes.MapRange, input_kwargs={'Value': attribute_1.outputs["Fac"], 2: 0.0010})
    
    colorramp_2 = nw.new_node(Nodes.ColorRamp, input_kwargs={'Fac': attribute_1.outputs["Fac"]})
    colorramp_2.color_ramp.elements[0].position = 0.0091
    colorramp_2.color_ramp.elements[0].color = [0.0000, 0.0000, 0.0000, 1.0000]
    colorramp_2.color_ramp.elements[1].position = 0.9841
    colorramp_2.color_ramp.elements[1].color = [1.0000, 1.0000, 1.0000, 1.0000]
    
    texture_coordinate = nw.new_node(Nodes.TextureCoord)

    noise_texture_1 = nw.new_node(Nodes.NoiseTexture, input_kwargs={'Vector': texture_coordinate.outputs["Generated"], 'Scale': 10.0000, 'Distortion': 2.0000})
    
    colorramp_3 = nw.new_node(Nodes.ColorRamp, input_kwargs={'Fac': noise_texture_1.outputs["Fac"]})
    colorramp_3.color_ramp.elements.new(0)
    colorramp_3.color_ramp.elements[0].position = 0.2773
    colorramp_3.color_ramp.elements[0].color = [0.0660, 0.1203, 0.0151, 1.0000]
    colorramp_3.color_ramp.elements[1].position = 0.6386
    colorramp_3.color_ramp.elements[1].color = [0.0405, 0.0397, 0.0064, 1.0000]
    colorramp_3.color_ramp.elements[2].position = 1.0000
    colorramp_3.color_ramp.elements[2].color = [0.0069, 0.0278, 0.0000, 1.0000]
    
    noise_texture = nw.new_node(Nodes.NoiseTexture, input_kwargs={'Vector': texture_coordinate.outputs["Generated"], 'W': 1.0000}, attrs={'noise_dimensions': '4D'})
    
    colorramp_1 = nw.new_node(Nodes.ColorRamp, input_kwargs={'Fac': noise_texture.outputs["Fac"]})
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements[0].position = 0.2818
    colorramp_1.color_ramp.elements[0].color = [0.3390, 0.1458, 0.0277, 1.0000]
    colorramp_1.color_ramp.elements[1].position = 0.5795
    colorramp_1.color_ramp.elements[1].color = [0.1295, 0.0542, 0.0220, 1.0000]
    colorramp_1.color_ramp.elements[2].position = 1.0000
    colorramp_1.color_ramp.elements[2].color = [0.2549, 0.1495, 0.0318, 1.0000]
    
    mix_1 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp_2.outputs["Color"], 'Color1': colorramp_3.outputs["Color"], 'Color2': colorramp_1.outputs["Color"]})
    
    separate_color = nw.new_node(Nodes.SeparateColor, input_kwargs={'Color': mix_1})
    
    texture_coordinate_1 = nw.new_node(Nodes.TextureCoord)
    
    noise_texture_2 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': texture_coordinate_1.outputs["Normal"], 'Scale': 20.0000, 'Detail': 200.0000, 'Roughness': 0.0000})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_color.outputs["Red"], 1: noise_texture_2.outputs["Fac"]},
        attrs={'use_clamp': True, 'operation': 'MULTIPLY'})
    
    combine_color = nw.new_node('ShaderNodeCombineColor',
        input_kwargs={'Red': multiply, 'Green': separate_color.outputs["Green"], 'Blue': separate_color.outputs["Blue"]})
    
    principled_bsdf_1 = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': combine_color, 'Specular': 0.3000, 'Roughness': 0.6000})
    
    material_output_1 = nw.new_node(Nodes.MaterialOutput, input_kwargs={'Surface': principled_bsdf_1}, attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_chameleon_leg_shape', singleton=False, type='GeometryNodeTree')
def nodegroup_chameleon_leg_shape(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'body_length', 0.5000),
            ('NodeSocketFloat', 'body_position', 0.1000),
            ('NodeSocketFloat', 'body_thickness', 0.0500),
            ('NodeSocketFloat', 'body_height', -0.1000),
            ('NodeSocketVectorEuler', 'Rotation', (0.0000, -0.6981, 0.0000)),
            ('NodeSocketFloat', 'thigh_length', 0.6000),
            ('NodeSocketFloat', 'calf_length', 0.5000),
            ('NodeSocketFloat', 'thigh_body_rotation', 25.0000),
            ('NodeSocketFloat', 'calf_body_rotation', 15.0000),
            ('NodeSocketFloat', 'thigh_calf_rotation', 20.0000),
            ('NodeSocketFloat', 'toe_toe_rotation', 20.0000),
            ('NodeSocketVectorXYZ', 'thigh_scale', (1.0000, 0.6500, 1.0000)),
            ('NodeSocketVectorXYZ', 'calf_scale', (1.0000, 0.6500, 1.0000)),
            ('NodeSocketVectorXYZ', 'ouScale', (1.0000, 1.0000, 1.0000)),
            ('NodeSocketVectorXYZ', 'inScale', (0.6000, 1.0000, 1.0000))])
    
    chameleon_leg_raw_shape = nw.new_node(nodegroup_chameleon_leg_raw_shape().name,
        input_kwargs={'thigh_length': group_input.outputs["thigh_length"], 'calf_length': group_input.outputs["calf_length"], 'thigh_body_rotation': group_input.outputs["thigh_body_rotation"], 'calf_body_rotation': group_input.outputs["calf_body_rotation"], 'thigh_calf_rotation': group_input.outputs["thigh_calf_rotation"], 'toe_toe_rotation': group_input.outputs["toe_toe_rotation"], 'thigh_scale': group_input.outputs["thigh_scale"], 'calf_scale': group_input.outputs["calf_scale"], 'ouScale': group_input.outputs["ouScale"], 'inScale': group_input.outputs["inScale"]})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["body_length"], 1: group_input.outputs["body_position"]},
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': multiply, 'Y': group_input.outputs["body_thickness"], 'Z': group_input.outputs["body_height"]})
    
    transform_2 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': chameleon_leg_raw_shape, 'Translation': combine_xyz, 'Rotation': group_input.outputs["Rotation"]})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': transform_2}, attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_chameleon_tail', singleton=False, type='GeometryNodeTree')
def nodegroup_chameleon_tail(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'RadStartEnd', (0.4000, 0.2500, 0.9000)),
            ('NodeSocketFloat', 'body_length', 0.5000),
            ('NodeSocketFloat', 'body_position', 0.5000)])
    
    chameleon_tail_shape = nw.new_node(nodegroup_chameleon_tail_shape().name)
    
    sample_curve = nw.new_node(Nodes.SampleCurve,
        input_kwargs={'Curve': chameleon_tail_shape.outputs["Curve"]},
        attrs={'mode': 'FACTOR'})
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: sample_curve.outputs["Position"], 'Scale': -1.0000},
        attrs={'operation': 'SCALE'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': chameleon_tail_shape.outputs["Mesh"], 'Offset': scale.outputs["Vector"]})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["body_length"], 1: group_input.outputs["body_position"]},
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': multiply, 'Z': 0.1000})
    
    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': set_position, 'Translation': combine_xyz, 'Rotation': (0.0000, 0.1745, 0.3491), 'Scale': (1.0000, 0.8000, 1.0000)})
    
    subdivide_mesh = nw.new_node(Nodes.SubdivideMesh, input_kwargs={'Mesh': transform, 'Level': 2})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': subdivide_mesh}, attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_chameleon_body_shape', singleton=False, type='GeometryNodeTree')
def nodegroup_chameleon_body_shape(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVectorXYZ', 'Scale', (0.9000, 0.7000, 0.8000)),
            ('NodeSocketFloat', 'length', 1.4000)])
    
    multiply = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["length"]}, attrs={'operation': 'MULTIPLY'})
    
    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': multiply, 'Y': 0.1000})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': group_input.outputs["length"], 'Y': 0.3000})
    
    quadratic_bezier = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 64, 'Start': (0.0000, 0.0000, 0.0000), 'Middle': combine_xyz_1, 'End': combine_xyz})
    
    simpletube = nw.new_node(nodegroup_simple_tube().name,
        input_kwargs={'Curve': quadratic_bezier, 'RadStartEnd': (0.6000, 0.6000, 1.0000), 'Resolution': 64})
    
    transform_geometry = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': simpletube.outputs["Mesh"], 'Scale': group_input.outputs["Scale"]})
    
    back_bump1 = nw.new_node(nodegroup_back_bump1().name, input_kwargs={'Surface': transform_geometry})
    
    back_bump2 = nw.new_node(nodegroup_back_bump2().name, input_kwargs={'Surface': back_bump1})
    
    back_bump3 = nw.new_node(nodegroup_back_bump3().name, input_kwargs={'Surface': back_bump2})
    
    belly_sunken1 = nw.new_node(nodegroup_belly_sunken1().name, input_kwargs={'Surface': back_bump3})
    
    shouder_sunken = nw.new_node(nodegroup_shouder_sunken().name, input_kwargs={'Surface': belly_sunken1})
    
    neck_bump = nw.new_node(nodegroup_neck_bump().name, input_kwargs={'Surface': shouder_sunken})
    
    neck_bump2 = nw.new_node(nodegroup_neck_bump2().name, input_kwargs={'Surface': neck_bump})
    
    subdivision_surface = nw.new_node(Nodes.SubdivisionSurface, input_kwargs={'Mesh': neck_bump2})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': subdivision_surface})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Mesh': join_geometry}, attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_chameleon_head_shape', singleton=False, type='GeometryNodeTree')
def nodegroup_chameleon_head_shape(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    polarbezier = nw.new_node(nodegroup_polar_bezier().name,
        input_kwargs={'Resolution': 64, 'angles_deg': (0.0000, 0.0000, -5.0000), 'Seg Lengths': (0.1000, 0.2400, 0.1000)})
    
    simpletube = nw.new_node(nodegroup_simple_tube().name,
        input_kwargs={'Curve': polarbezier.outputs["Curve"], 'RadStartEnd': (0.4000, 0.1800, 0.7800), 'Resolution': 64})
    
    group_input_2 = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'Crown', 0.2000),
            ('NodeSocketFloat', 'EyeBrow', 0.0200),
            ('NodeSocketVectorXYZ', 'Scale', (1.0000, 1.0000, 1.0000))])
    
    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': simpletube.outputs["Mesh"], 'Scale': group_input_2.outputs["Scale"]})
    
    quadratic_bezier_17 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 32, 'Start': (0.2000, 0.2500, 0.1000), 'Middle': (0.6000, 0.2500, 0.0000), 'End': (0.7900, 0.2500, 0.0000)})
    
    curveparametercurve = nw.new_node(nodegroup_curve_parameter_curve().name,
        input_kwargs={'Surface': transform, 'UVCurve': quadratic_bezier_17, 'CtrlptsU': 64, 'CtrlptsW': 64})
    
    curvesculpt = nw.new_node(nodegroup_curve_sculpt().name,
        input_kwargs={'Target': transform, 'Curve': curveparametercurve, 'Base Radius': 0.1500, 'Base Factor': 0.0200, 'SymmY': False})
    
    quadratic_bezier_22 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 32, 'Start': (0.7500, 0.7500, 0.1000), 'Middle': (0.7200, 0.7500, 0.0000), 'End': (0.7000, 0.7500, 0.0000)})
    
    curveparametercurve_1 = nw.new_node(nodegroup_curve_parameter_curve().name,
        input_kwargs={'Surface': curvesculpt.outputs["Geometry"], 'UVCurve': quadratic_bezier_22, 'CtrlptsU': 64, 'CtrlptsW': 64})
    
    curvesculpt_1 = nw.new_node(nodegroup_curve_sculpt().name,
        input_kwargs={'Target': curvesculpt.outputs["Geometry"], 'Curve': curveparametercurve_1, 'Base Radius': 0.1700, 'Base Factor': 0.0300, 'SymmY': False})
    
    quadratic_bezier_26 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 32, 'Start': (0.8000, 0.6800, 0.0300), 'Middle': (0.6500, 0.6800, 0.0000), 'End': (0.5000, 0.6000, 0.0500)})
    
    curveparametercurve_2 = nw.new_node(nodegroup_curve_parameter_curve().name,
        input_kwargs={'Surface': curvesculpt_1.outputs["Geometry"], 'UVCurve': quadratic_bezier_26, 'CtrlptsU': 64, 'CtrlptsW': 64})
    
    curvesculpt_2 = nw.new_node(nodegroup_curve_sculpt().name,
        input_kwargs={'Target': curvesculpt_1.outputs["Geometry"], 'Curve': curveparametercurve_2, 'Base Factor': 0.0300})
    
    quadratic_bezier_1 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 32, 'Start': (0.7000, 0.5500, 0.0300), 'Middle': (0.7000, 0.5500, 0.0300), 'End': (0.7500, 0.5700, -0.0200)})
    
    curveparametercurve_3 = nw.new_node(nodegroup_curve_parameter_curve().name,
        input_kwargs={'Surface': curvesculpt_2.outputs["Geometry"], 'UVCurve': quadratic_bezier_1, 'CtrlptsU': 64, 'CtrlptsW': 64})
    
    curvesculpt_3 = nw.new_node(nodegroup_curve_sculpt().name,
        input_kwargs={'Target': curvesculpt_2.outputs["Geometry"], 'Curve': curveparametercurve_3, 'Base Radius': 0.1000, 'Base Factor': -0.0200})
    
    quadratic_bezier_3 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 32, 'Start': (0.7000, 0.5800, 0.0100), 'Middle': (0.7500, 0.5800, 0.0100), 'End': (0.7700, 0.5300, 0.0100)})
    
    curveparametercurve_4 = nw.new_node(nodegroup_curve_parameter_curve().name,
        input_kwargs={'Surface': curvesculpt_3.outputs["Geometry"], 'UVCurve': quadratic_bezier_3, 'CtrlptsU': 64, 'CtrlptsW': 64})
    
    curvesculpt_4 = nw.new_node(nodegroup_curve_sculpt().name,
        input_kwargs={'Target': curvesculpt_3.outputs["Geometry"], 'Curve': curveparametercurve_4, 'Base Radius': 0.0400, 'Base Factor': -0.0100})
    
    quadratic_bezier_4 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 32, 'Start': (0.3000, 0.2500, 0.0000), 'Middle': (0.4000, 0.2500, 0.0000), 'End': (0.7000, 0.2500, 0.0000)})
    
    curveparametercurve_5 = nw.new_node(nodegroup_curve_parameter_curve().name,
        input_kwargs={'Surface': curvesculpt_4.outputs["Geometry"], 'UVCurve': quadratic_bezier_4, 'CtrlptsU': 64, 'CtrlptsW': 64})
    
    curvesculpt_5 = nw.new_node(nodegroup_curve_sculpt().name,
        input_kwargs={'Target': curvesculpt_4.outputs["Geometry"], 'Curve': curveparametercurve_5, 'Base Radius': 0.2000, 'Base Factor': 0.0100, 'SymmY': False})
    
    quadratic_bezier_9 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 32, 'Start': (0.3000, 0.2500, 0.0000), 'Middle': (0.4000, 0.2500, 0.0000), 'End': (0.5000, 0.2500, 0.0000)})
    
    curveparametercurve_6 = nw.new_node(nodegroup_curve_parameter_curve().name,
        input_kwargs={'Surface': curvesculpt_5.outputs["Geometry"], 'UVCurve': quadratic_bezier_9, 'CtrlptsU': 64, 'CtrlptsW': 64})
    
    curvesculpt_6 = nw.new_node(nodegroup_curve_sculpt().name,
        input_kwargs={'Target': curvesculpt_5.outputs["Geometry"], 'Curve': curveparametercurve_6, 'Base Radius': 0.2000, 'Base Factor': 0.0100, 'SymmY': False})
    
    quadratic_bezier_5 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 40, 'Start': (0.5000, 0.6000, 0.0000), 'Middle': (0.7000, 0.7000, 0.0000), 'End': (1.0000, 0.6500, 0.0100)})
    
    quadratic_bezier_6 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Start': (0.5000, 0.6000, 0.0000), 'Middle': (0.3000, 0.5500, 0.0000), 'End': (0.2000, 0.7000, 0.0200)})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [quadratic_bezier_5, quadratic_bezier_6]})
    
    curveparametercurve_7 = nw.new_node(nodegroup_curve_parameter_curve().name,
        input_kwargs={'Surface': curvesculpt_6.outputs["Geometry"], 'UVCurve': join_geometry, 'CtrlptsU': 64, 'CtrlptsW': 64})
    
    curvesculpt_7 = nw.new_node(nodegroup_curve_sculpt().name,
        input_kwargs={'Target': curvesculpt_6.outputs["Geometry"], 'Curve': curveparametercurve_7, 'Base Radius': 0.0150, 'Base Factor': group_input_2.outputs["EyeBrow"]})
    
    quadratic_bezier_7 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 64, 'Start': (0.6400, 0.7600, 0.0200), 'Middle': (0.4400, 0.8800, 0.0000), 'End': (0.5100, 0.9200, 0.0000)})
    
    curveparametercurve_8 = nw.new_node(nodegroup_curve_parameter_curve().name,
        input_kwargs={'Surface': curvesculpt_7.outputs["Geometry"], 'UVCurve': quadratic_bezier_7, 'CtrlptsU': 64, 'CtrlptsW': 64})
    
    curvesculpt_8 = nw.new_node(nodegroup_curve_sculpt().name,
        input_kwargs={'Target': curvesculpt_7.outputs["Geometry"], 'Curve': curveparametercurve_8, 'Base Radius': 0.1100, 'Base Factor': -0.0100})
    
    quadratic_bezier_8 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 64, 'Start': (0.6500, 0.7500, 0.0200), 'Middle': (0.3000, 0.7500, 0.0100), 'End': (0.1000, 0.7500, 0.0000)})
    
    quadratic_bezier_12 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 12, 'Start': (0.1500, 0.6000, 0.0200), 'Middle': (0.2000, 0.7000, 0.0100), 'End': (0.1000, 0.7500, 0.0050)})
    
    join_geometry_1 = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [quadratic_bezier_8, quadratic_bezier_12]})
    
    curveparametercurve_9 = nw.new_node(nodegroup_curve_parameter_curve().name,
        input_kwargs={'Surface': curvesculpt_8.outputs["Geometry"], 'UVCurve': join_geometry_1, 'CtrlptsU': 64, 'CtrlptsW': 64})
    
    curvesculpt_9 = nw.new_node(nodegroup_curve_sculpt().name,
        input_kwargs={'Target': curvesculpt_8.outputs["Geometry"], 'Curve': curveparametercurve_9, 'Base Radius': 0.0300, 'Base Factor': group_input_2.outputs["Crown"]})
    
    quadratic_bezier_18 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 200, 'Start': (0.9000, 0.2500, 0.0500), 'Middle': (0.8000, 0.2500, 0.0000), 'End': (0.6000, 0.2500, 0.0400)})
    
    curveparametercurve_10 = nw.new_node(nodegroup_curve_parameter_curve().name,
        input_kwargs={'Surface': curvesculpt_9.outputs["Geometry"], 'UVCurve': quadratic_bezier_18, 'CtrlptsU': 64, 'CtrlptsW': 64})
    
    curvesculpt_10 = nw.new_node(nodegroup_curve_sculpt().name,
        input_kwargs={'Target': curvesculpt_9.outputs["Geometry"], 'Curve': curveparametercurve_10, 'Base Radius': 0.1000, 'Base Factor': 0.0200})
    
    quadratic_bezier_16 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 32, 'Start': (0.7000, 0.3500, 0.0500), 'Middle': (0.6000, 0.4000, 0.0000), 'End': (0.4000, 0.3500, 0.0400)})
    
    curveparametercurve_11 = nw.new_node(nodegroup_curve_parameter_curve().name,
        input_kwargs={'Surface': curvesculpt_10.outputs["Geometry"], 'UVCurve': quadratic_bezier_16, 'CtrlptsU': 64, 'CtrlptsW': 64})
    
    curvesculpt_11 = nw.new_node(nodegroup_curve_sculpt().name,
        input_kwargs={'Target': curvesculpt_10.outputs["Geometry"], 'Curve': curveparametercurve_11, 'Base Radius': 0.1500, 'Base Factor': 0.0200})
    
    quadratic_bezier_15 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 20, 'Start': (0.9000, 0.2500, 0.0100), 'Middle': (0.6000, 0.2500, 0.0000), 'End': (0.2000, 0.2500, 0.0000)})
    
    curveparametercurve_12 = nw.new_node(nodegroup_curve_parameter_curve().name,
        input_kwargs={'Surface': curvesculpt_11.outputs["Geometry"], 'UVCurve': quadratic_bezier_15, 'CtrlptsU': 64, 'CtrlptsW': 64})
    
    curvesculpt_12 = nw.new_node(nodegroup_curve_sculpt().name,
        input_kwargs={'Target': curvesculpt_11.outputs["Geometry"], 'Curve': curveparametercurve_12, 'Base Radius': 0.0200, 'Base Factor': 0.0300, 'SymmY': False})
    
    quadratic_bezier_19 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 32, 'Start': (1.0000, 0.4000, 0.0100), 'Middle': (0.5000, 0.4500, 0.0000), 'End': (0.4500, 0.4000, 0.0100)})
    
    curveparametercurve_13 = nw.new_node(nodegroup_curve_parameter_curve().name,
        input_kwargs={'Surface': curvesculpt_12.outputs["Geometry"], 'UVCurve': quadratic_bezier_19, 'CtrlptsU': 64, 'CtrlptsW': 64})
    
    curvesculpt_13 = nw.new_node(nodegroup_curve_sculpt().name,
        input_kwargs={'Target': curvesculpt_12.outputs["Geometry"], 'Curve': curveparametercurve_13, 'Base Radius': 0.0200, 'Base Factor': 0.0100, 'Switch': False})
    
    quadratic_bezier_14 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 64, 'Start': (0.8000, 0.7500, 0.0000), 'Middle': (0.5000, 0.7500, 0.0000), 'End': (0.1000, 0.7500, 0.0000)})
    
    quadratic_bezier_13 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 12, 'Start': (0.1500, 0.6000, 0.0000), 'Middle': (0.2000, 0.7000, 0.0000), 'End': (0.1000, 0.7500, 0.0000)})
    
    join_geometry_2 = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [quadratic_bezier_14, quadratic_bezier_13]})
    
    curveparametercurve_14 = nw.new_node(nodegroup_curve_parameter_curve().name,
        input_kwargs={'Surface': curvesculpt_13.outputs["Geometry"], 'UVCurve': join_geometry_2, 'CtrlptsU': 64, 'CtrlptsW': 64})
    
    curvesculpt_14 = nw.new_node(nodegroup_curve_sculpt().name,
        input_kwargs={'Target': curvesculpt_13.outputs["Geometry"], 'Curve': curveparametercurve_14, 'Base Radius': 0.0300, 'Base Factor': 0.0000, 'Attr': True})
    
    quadratic_bezier_23 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 40, 'Start': (0.6000, 0.6000, 0.0000), 'Middle': (0.9000, 0.7300, 0.0000), 'End': (1.0000, 0.6500, 0.0000)})
    
    quadratic_bezier_24 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Start': (0.6000, 0.6000, 0.0000), 'Middle': (0.5000, 0.5500, 0.0000), 'End': (0.2000, 0.6200, 0.0000)})
    
    join_geometry_3 = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [quadratic_bezier_23, quadratic_bezier_24]})
    
    curveparametercurve_15 = nw.new_node(nodegroup_curve_parameter_curve().name,
        input_kwargs={'Surface': curvesculpt_14.outputs["Geometry"], 'UVCurve': join_geometry_3, 'CtrlptsU': 64, 'CtrlptsW': 64})
    
    curvesculpt_15 = nw.new_node(nodegroup_curve_sculpt().name,
        input_kwargs={'Target': curvesculpt_14.outputs["Geometry"], 'Curve': curveparametercurve_15, 'Base Radius': 0.0200, 'Base Factor': 0.0000, 'Attr': True})
    
    quadratic_bezier_25 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 64, 'Start': (1.0000, 0.4000, 0.0000), 'Middle': (0.7000, 0.4500, 0.0000), 'End': (0.4500, 0.4000, 0.0000)})
    
    curveparametercurve_16 = nw.new_node(nodegroup_curve_parameter_curve().name,
        input_kwargs={'Surface': curvesculpt_15.outputs["Geometry"], 'UVCurve': quadratic_bezier_25, 'CtrlptsU': 64, 'CtrlptsW': 64})
    
    curvesculpt_16 = nw.new_node(nodegroup_curve_sculpt().name,
        input_kwargs={'Target': curvesculpt_15.outputs["Geometry"], 'Curve': curveparametercurve_16, 'Base Radius': 0.0150, 'Base Factor': 0.0000, 'Switch': False, 'Attr': True})
    
    merge_by_distance_1 = nw.new_node(Nodes.MergeByDistance,
        input_kwargs={'Geometry': curvesculpt_16.outputs["Geometry"], 'Distance': 0.0000})
    
    subdivision_surface = nw.new_node(Nodes.SubdivisionSurface, input_kwargs={'Mesh': merge_by_distance_1, 'Level': 3})
    
    set_position = nw.new_node(Nodes.SetPosition, input_kwargs={'Geometry': subdivision_surface})
    
    set_shade_smooth = nw.new_node(Nodes.SetShadeSmooth, input_kwargs={'Geometry': set_position, 'Shade Smooth': False})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': set_shade_smooth}, attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_round_bump', singleton=False, type='GeometryNodeTree')
def nodegroup_round_bump(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketFloatDistance', 'Distance', 0.0200),
            ('NodeSocketFloat', 'Offset Scale', 0.0100),
            ('NodeSocketInt', 'Level', 1)])
    
    subdivide_mesh = nw.new_node(Nodes.SubdivideMesh,
        input_kwargs={'Mesh': group_input.outputs["Geometry"], 'Level': group_input.outputs["Level"]})
    
    merge_by_distance = nw.new_node(Nodes.MergeByDistance,
        input_kwargs={'Geometry': subdivide_mesh, 'Distance': group_input.outputs["Distance"]})
    # merge_by_distance = nw.new_node(Nodes.MergeByDistance,
    #     input_kwargs={'Geometry': subdivide_mesh, 'Distance': 2})
    
    dual_mesh = nw.new_node(Nodes.DualMesh, input_kwargs={'Mesh': merge_by_distance})
    
    split_edges = nw.new_node(Nodes.SplitEdges, input_kwargs={'Mesh': dual_mesh})
    
    scale_elements = nw.new_node(Nodes.ScaleElements, input_kwargs={'Geometry': split_edges, 'Scale': 0.9000})
    
    extrude_mesh = nw.new_node(Nodes.ExtrudeMesh,
        input_kwargs={'Mesh': scale_elements, 'Offset Scale': group_input.outputs["Offset Scale"], 'Individual': False})
    
    subdivision_surface_1 = nw.new_node(Nodes.SubdivisionSurface, input_kwargs={'Mesh': extrude_mesh.outputs["Mesh"]})
    
    set_shade_smooth = nw.new_node(Nodes.SetShadeSmooth, input_kwargs={'Geometry': subdivision_surface_1})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [set_shade_smooth, group_input.outputs["Geometry"]]})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': join_geometry}, attrs={'is_active_output': True})

def shader_chameleon_eye(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    attribute_2 = nw.new_node(Nodes.Attribute, attrs={'attribute_name': 'Pupil'})
    
    colorramp_4 = nw.new_node(Nodes.ColorRamp, input_kwargs={'Fac': attribute_2.outputs["Fac"]})
    colorramp_4.color_ramp.elements[0].position = 0.0091
    colorramp_4.color_ramp.elements[0].color = [0.0000, 0.0000, 0.0000, 1.0000]
    colorramp_4.color_ramp.elements[1].position = 0.9841
    colorramp_4.color_ramp.elements[1].color = [1.0000, 1.0000, 1.0000, 1.0000]
    
    attribute_1 = nw.new_node(Nodes.Attribute, attrs={'attribute_name': 'Ridge'})
    
    colorramp_2 = nw.new_node(Nodes.ColorRamp, input_kwargs={'Fac': attribute_1.outputs["Fac"]})
    colorramp_2.color_ramp.elements[0].position = 0.0091
    colorramp_2.color_ramp.elements[0].color = [0.0000, 0.0000, 0.0000, 1.0000]
    colorramp_2.color_ramp.elements[1].position = 0.9841
    colorramp_2.color_ramp.elements[1].color = [1.0000, 1.0000, 1.0000, 1.0000]
    
    texture_coordinate = nw.new_node(Nodes.TextureCoord)
    
    voronoi_texture = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': texture_coordinate.outputs["Generated"], 'Scale': 300.0000, 'Smoothness': 0.0000},
        attrs={'feature': 'SMOOTH_F1'})
    
    mapping_1 = nw.new_node(Nodes.Mapping, input_kwargs={'Vector': voronoi_texture.outputs["Distance"]})
    
    colorramp = nw.new_node(Nodes.ColorRamp, input_kwargs={'Fac': mapping_1})
    colorramp.color_ramp.interpolation = "CONSTANT"
    colorramp.color_ramp.elements[0].position = 0.0000
    colorramp.color_ramp.elements[0].color = [1.0000, 1.0000, 1.0000, 1.0000]
    colorramp.color_ramp.elements[1].position = 0.3159
    colorramp.color_ramp.elements[1].color = [0.0000, 0.0000, 0.0000, 1.0000]
    
    mapping = nw.new_node(Nodes.Mapping,
        input_kwargs={'Vector': texture_coordinate.outputs["Generated"], 'Location': (1.0000, 0.0000, 0.0000)})
    
    noise_texture_1 = nw.new_node(Nodes.NoiseTexture, input_kwargs={'Vector': mapping, 'Scale': 3.0000})
    
    colorramp_3 = nw.new_node(Nodes.ColorRamp, input_kwargs={'Fac': noise_texture_1.outputs["Fac"]})
    colorramp_3.color_ramp.elements.new(0)
    colorramp_3.color_ramp.elements[0].position = 0.2773
    colorramp_3.color_ramp.elements[0].color = [0.0353, 0.0942, 0.0136, 1.0000]
    colorramp_3.color_ramp.elements[1].position = 0.6000
    colorramp_3.color_ramp.elements[1].color = [0.0580, 0.0276, 0.0020, 1.0000]
    colorramp_3.color_ramp.elements[2].position = 0.6386
    colorramp_3.color_ramp.elements[2].color = [0.0405, 0.0397, 0.0064, 1.0000]
    
    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp.outputs["Color"], 'Color1': colorramp_3.outputs["Color"], 'Color2': (0.1421, 0.1015, 0.0241, 1.0000)})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture, input_kwargs={'Vector': mapping, 'W': 1.0000}, attrs={'noise_dimensions': '4D'})
    
    colorramp_1 = nw.new_node(Nodes.ColorRamp, input_kwargs={'Fac': noise_texture.outputs["Fac"]})
    colorramp_1.color_ramp.elements[0].position = 0.0000
    colorramp_1.color_ramp.elements[0].color = [0.6990, 0.5484, 0.1189, 1.0000]
    colorramp_1.color_ramp.elements[1].position = 1.0000
    colorramp_1.color_ramp.elements[1].color = [0.2549, 0.1495, 0.0318, 1.0000]
    
    mix_1 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp_2.outputs["Color"], 'Color1': mix, 'Color2': colorramp_1.outputs["Color"]})
    
    mix_2 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp_4.outputs["Color"], 'Color1': mix_1, 'Color2': (0.0082, 0.0082, 0.0082, 1.0000)})
    
    separate_color = nw.new_node(Nodes.SeparateColor, input_kwargs={'Color': mix_2})
    
    texture_coordinate_1 = nw.new_node(Nodes.TextureCoord)
    
    noise_texture_2 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': texture_coordinate_1.outputs["Normal"], 'Scale': 20.0000, 'Detail': 200.0000, 'Roughness': 0.0000})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_color.outputs["Red"], 1: noise_texture_2.outputs["Fac"]},
        attrs={'use_clamp': True, 'operation': 'MULTIPLY'})
    
    combine_color = nw.new_node('ShaderNodeCombineColor',
        input_kwargs={'Red': multiply, 'Green': separate_color.outputs["Green"], 'Blue': separate_color.outputs["Blue"]})
    
    principled_bsdf_1 = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': combine_color, 'Specular': 0.3000, 'Roughness': 0.6000})
    
    material_output_1 = nw.new_node(Nodes.MaterialOutput, input_kwargs={'Surface': principled_bsdf_1}, attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_chameleon', singleton=False, type='GeometryNodeTree')
def nodegroup_chameleon(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'body_length', 1.4000),
            ('NodeSocketFloat', 'head_crown', 0.2000),
            ('NodeSocketFloat', 'head_eyebrow', 0.0200),
            ('NodeSocketVectorXYZ', 'head_scale', (1.0000, 1.0000, 1.0000)),
            ('NodeSocketVectorEuler', 'left_eye_rotation', (0.0000, 0.0000, -1.5)),
            ('NodeSocketVectorEuler', 'right_eye_rotation', (0.0000, 0.0000, 1.5)),
            ('NodeSocketFloat', 'pupil_radius', 0.2200),
            ('NodeSocketFloat', 'front_leg_position', 0.0800),
            ('NodeSocketFloat', 'back_leg_position', 0.8500)])
    
    chameleon_head_shape = nw.new_node(nodegroup_chameleon_head_shape().name,
        input_kwargs={'Crown': group_input.outputs["head_crown"], 'EyeBrow': group_input.outputs["head_eyebrow"], 'Scale': group_input.outputs["head_scale"]})
    
    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': chameleon_head_shape, 'Translation': (0.1000, 0.0000, 0.0000), 'Rotation': (0.0000, 0.0000, 3.1416)})
    
    round_bump = nw.new_node(nodegroup_round_bump().name,
        input_kwargs={'Geometry': transform, 'Distance': 0.0080, 'Offset Scale': 0.0030})
    
    chameleon_body_shape = nw.new_node(nodegroup_chameleon_body_shape().name, input_kwargs={'length': group_input.outputs["body_length"]})
    
    round_bump_1 = nw.new_node(nodegroup_round_bump().name,
        input_kwargs={'Geometry': chameleon_body_shape.outputs["Mesh"], 'Distance': 0.0080, 'Offset Scale': 0.0030})
    
    chameleon_tail = nw.new_node(nodegroup_chameleon_tail().name,
        input_kwargs={'body_length': group_input.outputs["body_length"], 'body_position': 0.4500})
    
    round_bump_2 = nw.new_node(nodegroup_round_bump().name,
        input_kwargs={'Geometry': chameleon_tail.outputs["Geometry"], 'Distance': 0.0080, 'Offset Scale': 0.0030})
    
    chameleon_leg_shape = nw.new_node(nodegroup_chameleon_leg_shape().name,
        input_kwargs={'body_length': group_input.outputs["body_length"], 'body_position': group_input.outputs["back_leg_position"], 'body_thickness': 0.2500, 'Rotation': (0.0000, -1.0472, 3.1416), 'thigh_length': 0.4000, 'thigh_body_rotation': -35.0000, 'calf_body_rotation': -30.0000, 'thigh_calf_rotation': 10.0000, 'ouScale': (0.6000, 1.0000, 1.0000), 'inScale': (1.0000, 1.0000, 1.0000)})
    
    chameleon_leg_shape_1 = nw.new_node(nodegroup_chameleon_leg_shape().name,
        input_kwargs={'body_length': group_input.outputs["body_length"], 'body_position': group_input.outputs["back_leg_position"], 'body_thickness': 0.1500, 'Rotation': (0.0000, -1.0472, 3.1416), 'thigh_length': 0.4000, 'thigh_body_rotation': 50.0000, 'calf_body_rotation': 5.0000, 'thigh_calf_rotation': 5.0000})
    
    chameleon_leg_shape_2 = nw.new_node(nodegroup_chameleon_leg_shape().name,
        input_kwargs={'body_length': group_input.outputs["body_length"], 'body_position': group_input.outputs["front_leg_position"], 'body_thickness': 0.0800, 'thigh_body_rotation': 35.0000, 'thigh_calf_rotation': 15.0000})
    
    chameleon_leg_shape_3 = nw.new_node(nodegroup_chameleon_leg_shape().name,
        input_kwargs={'body_length': group_input.outputs["body_length"], 'body_position': group_input.outputs["front_leg_position"], 'body_thickness': -0.0300, 'thigh_body_rotation': -25.0000, 'calf_body_rotation': -15.0000, 'thigh_calf_rotation': 15.0000, 'ouScale': (0.6000, 1.0000, 1.0000), 'inScale': (1.0000, 1.0000, 1.0000)})
    
    join_geometry_2 = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [chameleon_leg_shape, chameleon_leg_shape_1, chameleon_leg_shape_2, chameleon_leg_shape_3]})
    
    join_geometry_1 = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [round_bump, round_bump_1, round_bump_2, join_geometry_2]})
    
    set_material = nw.new_node(Nodes.SetMaterial,
        input_kwargs={'Geometry': join_geometry_1, 'Material': surface.shaderfunc_to_material(shader_chameleon)})
    
    chameleon_eye = nw.new_node(nodegroup_chameleon_eye().name, input_kwargs={'pupil_radius': group_input.outputs["pupil_radius"]})
    
    transform_2 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': chameleon_eye, 'Translation': (-0.2000, -0.0300, 0.0200), 'Rotation': group_input.outputs["left_eye_rotation"]})
    
    chameleon_eye_1 = nw.new_node(nodegroup_chameleon_eye().name, input_kwargs={'pupil_radius': group_input.outputs["pupil_radius"]})
    
    transform_1 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': chameleon_eye_1, 'Translation': (-0.2000, 0.0300, 0.0200), 'Rotation': group_input.outputs["right_eye_rotation"]})
    
    join_geometry_3 = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [transform_2, transform_1]})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [set_material, join_geometry_3]})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': join_geometry}, attrs={'is_active_output': True})

class Chameleon(PartFactory):
    param_templates = {}
    tags = []

    def sample_params(self, select=None, var=1):
        return {}
    
    def make_part(self, params):
        part = nodegroup_to_part(nodegroup_chameleon, params)

        return part