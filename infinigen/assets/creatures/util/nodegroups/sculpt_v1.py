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

from infinigen.assets.creatures.util.nodegroups.math import nodegroup_floor_ceil, nodegroup_clamp_or_wrap
from infinigen.assets.creatures.util.nodegroups.geometry import nodegroup_symmetric_clone

@node_utils.to_nodegroup('nodegroup_u_v_param_to_vert_idxs', singleton=False, type='GeometryNodeTree')
def nodegroup_u_v_param_to_vert_idxs(nw: NodeWrangler):
    # Code generated using version 2.6.3 of the node_transpiler

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

@node_utils.to_nodegroup('nodegroup_bilinear_interp_index_transfer', singleton=False, type='GeometryNodeTree')
def nodegroup_bilinear_interp_index_transfer(nw: NodeWrangler):
    # Code generated using version 2.6.3 of the node_transpiler

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
    
    transfer_attribute_1 = nw.new_node(Nodes.TransferAttribute,
        input_kwargs={'Source': group_input, 1: group_input.outputs["Attribute"], 'Index': floor_floor},
        attrs={'data_type': 'FLOAT_VECTOR', 'mapping': 'INDEX'})
    
    ceil_floor = nw.new_node(Nodes.Math,
        input_kwargs={0: uvparamtovertidxs_1.outputs["Ceil"], 1: group_input.outputs["SizeV"], 2: uvparamtovertidxs.outputs["Floor"]},
        label='CeilFloor',
        attrs={'operation': 'MULTIPLY_ADD'})
    
    transfer_attribute_2 = nw.new_node(Nodes.TransferAttribute,
        input_kwargs={'Source': group_input, 1: group_input.outputs["Attribute"], 'Index': ceil_floor},
        attrs={'data_type': 'FLOAT_VECTOR', 'mapping': 'INDEX'})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Vector': uvparamtovertidxs_1.outputs["Remainder"], 9: transfer_attribute_1.outputs["Attribute"], 10: transfer_attribute_2.outputs["Attribute"]},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    floor_ceil = nw.new_node(Nodes.Math,
        input_kwargs={0: uvparamtovertidxs_1.outputs["Floor"], 1: group_input.outputs["SizeV"], 2: uvparamtovertidxs.outputs["Ceil"]},
        label='FloorCeil',
        attrs={'operation': 'MULTIPLY_ADD'})
    
    transfer_attribute_3 = nw.new_node(Nodes.TransferAttribute,
        input_kwargs={'Source': group_input, 1: group_input.outputs["Attribute"], 'Index': floor_ceil},
        attrs={'data_type': 'FLOAT_VECTOR', 'mapping': 'INDEX'})
    
    ceil_ceil = nw.new_node(Nodes.Math,
        input_kwargs={0: uvparamtovertidxs_1.outputs["Ceil"], 1: group_input.outputs["SizeV"], 2: uvparamtovertidxs.outputs["Ceil"]},
        label='CeilCeil',
        attrs={'operation': 'MULTIPLY_ADD'})
    
    transfer_attribute_4 = nw.new_node(Nodes.TransferAttribute,
        input_kwargs={'Source': group_input, 1: group_input.outputs["Attribute"], 'Index': ceil_ceil},
        attrs={'data_type': 'FLOAT_VECTOR', 'mapping': 'INDEX'})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Vector': uvparamtovertidxs_1.outputs["Remainder"], 9: transfer_attribute_3.outputs["Attribute"], 10: transfer_attribute_4.outputs["Attribute"]},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    map_range_2 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Vector': uvparamtovertidxs.outputs["Remainder"], 9: map_range.outputs["Vector"], 10: map_range_1.outputs["Vector"]},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Vector': map_range_2.outputs["Vector"]},
        attrs={'is_active_output': True})
    

@node_utils.to_nodegroup('nodegroup_curve_parameter_curve', singleton=False, type='GeometryNodeTree')
def nodegroup_curve_parameter_curve(nw: NodeWrangler):
    # Code generated using version 2.6.3 of the node_transpiler

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
    
    transfer_attribute = nw.new_node(Nodes.TransferAttribute,
        input_kwargs={'Source': group_input.outputs["Surface"], 1: normal, 'Source Position': bilinearinterpindextransfer},
        attrs={'data_type': 'FLOAT_VECTOR', 'mapping': 'NEAREST_FACE_INTERPOLATED'})
    
    multiply_add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: transfer_attribute.outputs["Attribute"], 1: separate_xyz.outputs["Z"], 2: bilinearinterpindextransfer},
        attrs={'operation': 'MULTIPLY_ADD'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["UVCurve"], 'Position': multiply_add.outputs["Vector"]})
    
    normal_1 = nw.new_node(Nodes.InputNormal)
    
    dot_product = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: transfer_attribute.outputs["Attribute"], 1: normal_1},
        attrs={'operation': 'DOT_PRODUCT'})
    
    arcsine = nw.new_node(Nodes.Math, input_kwargs={0: dot_product.outputs["Value"]}, attrs={'operation': 'ARCSINE'})
    
    set_curve_tilt = nw.new_node(Nodes.SetCurveTilt, input_kwargs={'Curve': set_position, 'Tilt': arcsine})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': set_curve_tilt}, attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_curve_sculpt', singleton=False, type='GeometryNodeTree')
def nodegroup_curve_sculpt(nw: NodeWrangler):
    # Code generated using version 2.6.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Target', None),
            ('NodeSocketGeometry', 'Curve', None),
            ('NodeSocketFloat', 'Base Radius', 0.0500),
            ('NodeSocketFloat', 'Base Factor', 0.0500),
            ('NodeSocketBool', 'SymmY', True),
            ('NodeSocketGeometry', 'StrokeRadFacModifier', None)])
    
    normal = nw.new_node(Nodes.InputNormal)
    
    symmetric_clone = nw.new_node(nodegroup_symmetric_clone().name, input_kwargs={'Geometry': group_input.outputs["Curve"]})
    
    switch = nw.new_node(Nodes.Switch,
        input_kwargs={1: group_input.outputs["SymmY"], 14: group_input.outputs["Curve"], 15: symmetric_clone.outputs["Both"]})
    
    curve_to_mesh = nw.new_node(Nodes.CurveToMesh, input_kwargs={'Curve': switch.outputs[6]})
    
    geometry_proximity = nw.new_node(Nodes.Proximity, input_kwargs={'Target': curve_to_mesh}, attrs={'target_element': 'POINTS'})
    
    curve_to_mesh_1 = nw.new_node(Nodes.CurveToMesh, input_kwargs={'Curve': group_input.outputs["StrokeRadFacModifier"]})
    
    position = nw.new_node(Nodes.InputPosition)
    
    index = nw.new_node(Nodes.Index)
    
    transfer_attribute = nw.new_node(Nodes.TransferAttribute,
        input_kwargs={'Source': curve_to_mesh_1, 1: position, 'Index': index},
        attrs={'data_type': 'FLOAT_VECTOR', 'mapping': 'INDEX'})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': transfer_attribute.outputs["Attribute"]})
    
    add = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Base Radius"], 1: separate_xyz.outputs["X"]})
    
    map_range = nw.new_node(Nodes.MapRange, input_kwargs={'Value': geometry_proximity.outputs["Distance"], 2: add})
    
    float_curve = nw.new_node(Nodes.FloatCurve, input_kwargs={'Value': map_range.outputs["Result"]})
    node_utils.assign_curve(float_curve.mapping.curves[0], [(0.0000, 1.0000), (0.2000, 0.9400), (0.8000, 0.0600), (1.0000, 0.0000)], handles=['VECTOR', 'AUTO', 'AUTO', 'VECTOR'])
    
    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Base Factor"], 1: separate_xyz.outputs["Y"]})
    
    multiply = nw.new_node(Nodes.Math, input_kwargs={0: float_curve, 1: add_1}, attrs={'operation': 'MULTIPLY'})
    
    scale = nw.new_node(Nodes.VectorMath, input_kwargs={0: normal, 'Scale': multiply}, attrs={'operation': 'SCALE'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["Target"], 'Offset': scale.outputs["Vector"]})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': set_position}, attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_simple_tube_skin', singleton=False, type='GeometryNodeTree')
def nodegroup_simple_tube_skin(nw: NodeWrangler):
    # Code generated using version 2.6.3 of the node_transpiler

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
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Mesh': curve_to_mesh}, attrs={'is_active_output': True})
