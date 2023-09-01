# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Mingzhe Wang
# Acknowledgement: This file draws inspiration from https://www.youtube.com/watch?v=EfNzAaqKHXQ by PixelicaCG, https://www.youtube.com/watch?v=JcHX4AT1vtg by CGCookie and https://www.youtube.com/watch?v=E0JyyWeptSA by CGRogue


import os, sys
import numpy as np
import math as ma
from infinigen.assets.materials.utils.surface_utils import clip, sample_range, sample_ratio, sample_color, geo_voronoi_noise
import bpy
import mathutils
from numpy.random import uniform as U, normal as N, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface

from infinigen.assets.creatures.util.creature import PartFactory
from infinigen.assets.creatures.util import part_util

@node_utils.to_nodegroup('nodegroup_circle', singleton=False, type='GeometryNodeTree')
def nodegroup_circle(nw: NodeWrangler):
    # Code generated using version 2.6.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'R', 0.5000),
            ('NodeSocketInt', 'Resolution', 512)])
    
    multiply = nw.new_node(Nodes.Math, input_kwargs={0: group_input, 1: -1.0000}, attrs={'operation': 'MULTIPLY'})
    
    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': multiply})
    
    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Z': group_input})
    
    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': group_input})
    
    curve_circle = nw.new_node(Nodes.CurveCircle,
        input_kwargs={'Resolution': group_input.outputs["Resolution"], 'Point 1': combine_xyz_4, 'Point 2': combine_xyz_3, 'Point 3': combine_xyz_5, 'Radius': 2.0000},
        attrs={'mode': 'POINTS'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Curve': curve_circle.outputs["Curve"]},
        attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_eyeball', singleton=False, type='GeometryNodeTree')
def nodegroup_eyeball(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'Value', 1.0),
            ('NodeSocketInt', 'Resolution', 32)])
    
    uv_sphere = nw.new_node(Nodes.MeshUVSphere,
        input_kwargs={'Segments': group_input.outputs["Resolution"], 'Rings': group_input.outputs["Resolution"]})
    
    position_1 = nw.new_node(Nodes.InputPosition)
    
    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': position_1})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Value"], 1: group_input.outputs["Value"]},
        attrs={'operation': 'MULTIPLY'})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: 1.0, 1: multiply},
        attrs={'operation': 'SUBTRACT'})
    
    sqrt = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract},
        attrs={'operation': 'SQRT'})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: sqrt, 1: 1.02},
        attrs={'operation': 'MULTIPLY'})
    
    subtract_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["Y"], 1: multiply_1},
        attrs={'operation': 'SUBTRACT', 'use_clamp': True})
    
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract_1, 1: 0.5},
        attrs={'operation': 'MULTIPLY'})
    
    subtract_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_2, 1: subtract_1},
        attrs={'operation': 'SUBTRACT'})
    
    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Y': subtract_2})
    
    set_position_1 = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': uv_sphere, 'Offset': combine_xyz_1})
    
    greater_than = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract_1, 1: 0.0},
        attrs={'operation': 'GREATER_THAN'})
    
    store_named_attribute = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': set_position_1, 'Name': 'Iris', 3: greater_than})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': store_named_attribute})

@node_utils.to_nodegroup('nodegroup_cornea', singleton=False, type='GeometryNodeTree')
def nodegroup_cornea(nw: NodeWrangler):
    # Code generated using version 2.6.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'ScaleX', 0.5000),
            ('NodeSocketFloat', 'Height', 2.0000),
            ('NodeSocketFloatFactor', 'ScaleZ', 0.0000),
            ('NodeSocketFloat', 'Y', 20.0000),
            ('NodeSocketInt', 'Resolution', 128)])
    
    uv_sphere_1 = nw.new_node(Nodes.MeshUVSphere,
        input_kwargs={'Segments': group_input.outputs["Resolution"], 'Rings': group_input.outputs["Resolution"]})
    
    subtract = nw.new_node(Nodes.Math, input_kwargs={0: 3.0000, 1: group_input.outputs["Height"]}, attrs={'operation': 'SUBTRACT'})
    
    divide = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["ScaleX"], 1: subtract}, attrs={'operation': 'DIVIDE'})
    
    combine_color = nw.new_node('FunctionNodeCombineColor',
        input_kwargs={'Red': group_input.outputs["ScaleX"], 'Green': divide, 'Blue': group_input.outputs["ScaleZ"]})
    
    transform = nw.new_node(Nodes.Transform, input_kwargs={'Geometry': uv_sphere_1, 'Scale': combine_color})
    
    position_2 = nw.new_node(Nodes.InputPosition)
    
    separate_xyz_2 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': position_2})
    
    greater_than = nw.new_node(Nodes.Compare, input_kwargs={0: separate_xyz_2.outputs["Y"]})
    
    separate_geometry = nw.new_node(Nodes.SeparateGeometry, input_kwargs={'Geometry': transform, 'Selection': greater_than})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture)
    
    subtract_1 = nw.new_node(Nodes.Math, input_kwargs={0: noise_texture.outputs["Fac"]}, attrs={'operation': 'SUBTRACT'})
    
    normal = nw.new_node(Nodes.InputNormal)
    
    multiply = nw.new_node(Nodes.VectorMath, input_kwargs={0: subtract_1, 1: normal}, attrs={'operation': 'MULTIPLY'})
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.0200
    
    multiply_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: multiply.outputs["Vector"], 1: value},
        attrs={'operation': 'MULTIPLY'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': separate_geometry.outputs["Selection"], 'Offset': multiply_1.outputs["Vector"]})
    
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["ScaleX"], 1: group_input.outputs["ScaleX"]},
        attrs={'operation': 'MULTIPLY'})
    
    subtract_2 = nw.new_node(Nodes.Math, input_kwargs={0: 1.0000, 1: multiply_2}, attrs={'operation': 'SUBTRACT'})
    
    sqrt = nw.new_node(Nodes.Math, input_kwargs={0: subtract_2}, attrs={'operation': 'SQRT'})
    
    multiply_3 = nw.new_node(Nodes.Math, input_kwargs={0: sqrt, 1: 0.9500}, attrs={'operation': 'MULTIPLY'})
    
    combine_color_1 = nw.new_node('FunctionNodeCombineColor', input_kwargs={'Green': multiply_3})
    
    multiply_4 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Y"], 1: -1.0000}, attrs={'operation': 'MULTIPLY'})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Y': multiply_4})
    
    transform_1 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': set_position, 'Translation': combine_color_1, 'Rotation': combine_xyz})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': transform_1}, attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_eyelid_radius', singleton=False, type='GeometryNodeTree')
def nodegroup_eyelid_radius(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    curve_line = nw.new_node(Nodes.CurveLine,
        input_kwargs={'End': (0.0, 0.8, 0.0)})
    
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'OuterControl', 0.3),
            ('NodeSocketFloat', 'InnerControl1', 5.4),
            ('NodeSocketFloat', 'InnerControl2', 0.3),
            ('NodeSocketInt', 'Resolution', 32)])
    
    resample_curve = nw.new_node(Nodes.ResampleCurve,
        input_kwargs={'Curve': curve_line, 'Count': group_input.outputs["Resolution"]})
    
    position = nw.new_node(Nodes.InputPosition)
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': position})
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': resample_curve, 2: separate_xyz.outputs["Y"]})
    
    reroute = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': separate_xyz.outputs["Y"]})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: reroute, 1: 0.4},
        attrs={'operation': 'SUBTRACT'})
    
    power = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract, 1: 2.0},
        attrs={'operation': 'POWER'})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: power, 1: -0.7},
        attrs={'operation': 'MULTIPLY'})
    
    greater_than = nw.new_node(Nodes.Math,
        input_kwargs={0: reroute, 1: group_input.outputs["InnerControl2"]},
        attrs={'operation': 'GREATER_THAN'})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply, 1: greater_than},
        attrs={'operation': 'MULTIPLY'})
    
    subtract_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: 1.0, 1: greater_than},
        attrs={'operation': 'SUBTRACT'})
    
    reroute_3 = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': group_input.outputs["OuterControl"]})
    
    subtract_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: reroute_3, 1: reroute},
        attrs={'operation': 'SUBTRACT'})
    
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract_1, 1: subtract_2},
        attrs={'operation': 'MULTIPLY'})
    
    power_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_2, 1: 2.0},
        attrs={'operation': 'POWER'})
    
    multiply_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: power_1, 1: group_input.outputs["InnerControl1"]},
        attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_1, 1: multiply_3})
    
    subtract_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: add, 1: 0.0},
        attrs={'operation': 'SUBTRACT'})
    
    reroute_1 = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': group_input.outputs["OuterControl"]})
    
    subtract_4 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_3, 1: reroute_1},
        attrs={'operation': 'SUBTRACT'})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': subtract_3, 'Y': subtract_4})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': capture_attribute.outputs["Geometry"], 'Offset': combine_xyz})
    
    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': set_position, 'Scale': (1.5, 1.5, 1.5)})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': transform, 'Attribute': capture_attribute.outputs[2]})

@node_utils.to_nodegroup('nodegroup_eyelid_circle', singleton=False, type='GeometryNodeTree')
def nodegroup_eyelid_circle(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'ShapeW', 0.0),
            ('NodeSocketFloat', 'ShapeH', 0.0),
            ('NodeSocketInt', 'Resolution', 32)])
    
    reroute_3 = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': group_input.outputs["ShapeW"]})
    
    circle = nw.new_node(nodegroup_circle().name,
        input_kwargs={'R': reroute_3, 'Resolution': group_input.outputs["Resolution"]})
    
    spline_parameter = nw.new_node(Nodes.SplineParameter)
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': circle, 2: spline_parameter.outputs["Factor"]})
    
    position_1 = nw.new_node(Nodes.InputPosition)
    
    capture_attribute_1 = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': capture_attribute.outputs["Geometry"], 1: position_1},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    position = nw.new_node(Nodes.InputPosition)
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': position})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: -0.5},
        attrs={'operation': 'SUBTRACT'})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract, 1: subtract},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply, 1: -0.02},
        attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: multiply_1})
    
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["ShapeH"], 1: group_input.outputs["ShapeW"]},
        attrs={'operation': 'MULTIPLY'})
    
    reroute_1 = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': multiply_2})
    
    multiply_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: reroute_1, 1: reroute_1},
        attrs={'operation': 'MULTIPLY'})
    
    greater_than = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: 0.0},
        attrs={'operation': 'GREATER_THAN'})
    
    multiply_4 = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: 1.0},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_5 = nw.new_node(Nodes.Math,
        input_kwargs={0: greater_than, 1: multiply_4},
        attrs={'operation': 'MULTIPLY'})
    
    less_than = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: 0.0},
        attrs={'operation': 'LESS_THAN'})
    
    multiply_6 = nw.new_node(Nodes.Math,
        input_kwargs={0: less_than, 1: separate_xyz.outputs["X"]},
        attrs={'operation': 'MULTIPLY'})
    
    add_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_5, 1: multiply_6})
    
    reroute = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': add_1})
    
    multiply_7 = nw.new_node(Nodes.Math,
        input_kwargs={0: reroute, 1: reroute},
        attrs={'operation': 'MULTIPLY'})
    
    subtract_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_3, 1: multiply_7},
        attrs={'operation': 'SUBTRACT'})
    
    sqrt = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract_1},
        attrs={'operation': 'SQRT'})
    
    multiply_8 = nw.new_node(Nodes.Math,
        input_kwargs={0: reroute_3, 1: reroute_3},
        attrs={'operation': 'MULTIPLY'})
    
    subtract_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_3, 1: multiply_8},
        attrs={'operation': 'SUBTRACT'})
    
    sqrt_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract_2},
        attrs={'operation': 'SQRT'})
    
    reroute_2 = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': sqrt_1})
    
    subtract_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: sqrt, 1: reroute_2},
        attrs={'operation': 'SUBTRACT'})
    
    sign = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: 0.0},
        attrs={'operation': 'SIGN'})
    
    multiply_9 = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract_3, 1: sign},
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': separate_xyz.outputs["X"], 'Y': add, 'Z': multiply_9})
    
    set_position_1 = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': capture_attribute_1.outputs["Geometry"], 'Position': combine_xyz_1})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'W': 50.0, 'Scale': 0.5},
        attrs={'noise_dimensions': '4D'})
    
    subtract_4 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: noise_texture.outputs["Color"], 1: (0.5, 0.5, 0.5)},
        attrs={'operation': 'SUBTRACT'})
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.5
    
    multiply_10 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: subtract_4.outputs["Vector"], 1: value},
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Z': multiply_10.outputs["Vector"]})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': set_position_1, 'Offset': combine_xyz})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_position, "Attribute": capture_attribute.outputs[2], "Attribute1": capture_attribute_1.outputs["Attribute"]})

@node_utils.to_nodegroup('nodegroup_eye_ball', singleton=False, type='GeometryNodeTree')
def nodegroup_eye_ball(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'CorneaScaleX', 0.52),
            ('NodeSocketFloat', 'Height', 1.2),
            ('NodeSocketFloatFactor', 'CorneaScaleZ', 0.8),
            ('NodeSocketFloat', 'Y', 20.0),
            ('NodeSocketInt', 'EyeballResolution', 32),
            ('NodeSocketInt', 'CorneaResolution', 128)])
    
    cornea_008 = nw.new_node(nodegroup_cornea().name,
        input_kwargs={'ScaleX': group_input.outputs["CorneaScaleX"], 'Height': group_input.outputs["Height"], 'ScaleZ': group_input.outputs["CorneaScaleZ"], 'Y': group_input.outputs["Y"], 'Resolution': group_input.outputs["CorneaResolution"]})
    
    store_named_attribute = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': cornea_008, 'Name': 'tag_cornea', 5: True},
        attrs={'data_type': 'BOOLEAN'})
    
    eyeball_009 = nw.new_node(nodegroup_eyeball().name,
        input_kwargs={'Value': group_input.outputs["CorneaScaleX"], 'Resolution': group_input.outputs["EyeballResolution"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Cornea': store_named_attribute, 'Eyeball': eyeball_009})

@node_utils.to_nodegroup('nodegroup_raycast_append', singleton=False, type='GeometryNodeTree')
def nodegroup_raycast_append(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketGeometry', 'Target Geometry', None),
            ('NodeSocketVector', 'Ray Direction', (-1.0, 0.0, 0.0)),
            ('NodeSocketFloat', 'Default Offset', -0.005)])
    
    raycast = nw.new_node(Nodes.Raycast,
        input_kwargs={'Target Geometry': group_input.outputs["Target Geometry"], 'Ray Direction': group_input.outputs["Ray Direction"], 'Ray Length': 0.1})
    
    less_than = nw.new_node(Nodes.Math,
        input_kwargs={0: raycast.outputs["Hit Distance"], 1: 0.07},
        attrs={'operation': 'LESS_THAN'})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: raycast.outputs["Hit Distance"], 1: less_than},
        attrs={'operation': 'MULTIPLY'})
    
    named_attribute = nw.new_node(Nodes.NamedAttribute,
        input_kwargs={'Name': 'pos'},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    distance = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: named_attribute.outputs["Attribute"]},
        attrs={'operation': 'DISTANCE'})
    
    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = 1.2
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: distance.outputs["Value"], 1: value_1},
        attrs={'operation': 'SUBTRACT', 'use_clamp': True})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract, 1: 1.5},
        attrs={'operation': 'MULTIPLY', 'use_clamp': True})
    
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply, 1: multiply_1},
        attrs={'operation': 'MULTIPLY'})
    
    subtract_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: 1.0, 1: multiply_1},
        attrs={'operation': 'SUBTRACT', 'use_clamp': True})
    
    multiply_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract_1, 1: group_input.outputs["Default Offset"]},
        attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_2, 1: multiply_3})
    
    length = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Ray Direction"]},
        attrs={'operation': 'LENGTH'})
    
    divide = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Ray Direction"], 1: length.outputs["Value"]},
        attrs={'operation': 'DIVIDE'})
    
    multiply_4 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: add, 1: divide.outputs["Vector"]},
        attrs={'operation': 'MULTIPLY'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Offset': multiply_4.outputs["Vector"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_position})

@node_utils.to_nodegroup('nodegroup_vector_sum', singleton=False, type='GeometryNodeTree')
def nodegroup_vector_sum(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'Vector', (0.0, 0.0, 0.0))])
    
    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group_input.outputs["Vector"]})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["X"], 1: separate_xyz_1.outputs["Y"]})
    
    add_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: add, 1: separate_xyz_1.outputs["Z"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Sum': add_1})

def shader_material(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': (0.8, 0.0, 0.6028, 1.0)})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

@node_utils.to_nodegroup('nodegroup_part_surface_simple', singleton=False, type='GeometryNodeTree')
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

@node_utils.to_nodegroup('nodegroup_aspect_to_dim', singleton=False, type='GeometryNodeTree')
def nodegroup_aspect_to_dim(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'Aspect Ratio', 1.0)])
    
    greater_than = nw.new_node(Nodes.Compare,
        input_kwargs={0: group_input.outputs["Aspect Ratio"], 1: 1.0})
    
    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': group_input.outputs["Aspect Ratio"], 'Y': 1.0})
    
    divide = nw.new_node(Nodes.Math,
        input_kwargs={0: 1.0, 1: group_input.outputs["Aspect Ratio"]},
        attrs={'operation': 'DIVIDE'})
    
    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': 1.0, 'Y': divide})
    
    switch = nw.new_node(Nodes.Switch,
        input_kwargs={0: greater_than, 8: combine_xyz_1, 9: combine_xyz_2},
        attrs={'input_type': 'VECTOR'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'XY Scale': switch.outputs[3]})

@node_utils.to_nodegroup('nodegroup_polar_to_cart', singleton=False, type='GeometryNodeTree')
def nodegroup_polar_to_cart(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'Angle', 0.5),
            ('NodeSocketFloat', 'Length', 0.0),
            ('NodeSocketVector', 'Origin', (0.0, 0.0, 0.0))])
    
    cosine = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Angle"]},
        attrs={'operation': 'COSINE'})
    
    sine = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Angle"]},
        attrs={'operation': 'SINE'})
    
    construct_unit_vector = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': cosine, 'Z': sine},
        label='Construct Unit Vector')
    
    offset_polar = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Length"], 1: construct_unit_vector, 2: group_input.outputs["Origin"]},
        label='Offset Polar',
        attrs={'operation': 'MULTIPLY_ADD'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Vector': offset_polar.outputs["Vector"]})

@node_utils.to_nodegroup('nodegroup_switch4', singleton=False, type='GeometryNodeTree')
def nodegroup_switch4(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketInt', 'Arg', 0),
            ('NodeSocketVector', 'Arg == 0', (0.0, 0.0, 0.0)),
            ('NodeSocketVector', 'Arg == 1', (0.0, 0.0, 0.0)),
            ('NodeSocketVector', 'Arg == 2', (0.0, 0.0, 0.0)),
            ('NodeSocketVector', 'Arg == 3', (0.0, 0.0, 0.0))])
    
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
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Output': switch.outputs[3]})

def shader_eyeball_fish(nw: NodeWrangler, rand=True, **input_kwargs):
    # Code generated using version 2.6.3 of the node_transpiler

    attribute_2 = nw.new_node(Nodes.Attribute, attrs={'attribute_name': 'tag_cornea'})
    
    attribute_1 = nw.new_node(Nodes.Attribute, attrs={'attribute_name': 'EyeballPosition'})
    
    mapping = nw.new_node(Nodes.Mapping, input_kwargs={'Vector': attribute_1, 'Scale': (1.2000, 1.0000, 0.4000)})
    
    noise_texture_2 = nw.new_node(Nodes.NoiseTexture, input_kwargs={'Vector': mapping, 'Scale': 50.0000})
    
    mix_3 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.0200, 'Color1': mapping, 'Color2': noise_texture_2.outputs["Color"]})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': mix_3})
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.0000
    
    group = nw.new_node(nodegroup_rotate2_d().name,
        input_kwargs={0: separate_xyz.outputs["X"], 1: separate_xyz.outputs["Z"], 2: value})
    
    multiply = nw.new_node(Nodes.Math, input_kwargs={0: group.outputs[1], 1: 0.3000}, attrs={'operation': 'MULTIPLY'})
    
    multiply_1 = nw.new_node(Nodes.Math, input_kwargs={0: multiply, 1: multiply}, attrs={'operation': 'MULTIPLY'})
    
    multiply_2 = nw.new_node(Nodes.Math, input_kwargs={0: group.outputs["Value"], 1: 0.8000}, attrs={'operation': 'MULTIPLY'})
    
    multiply_3 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_2, 1: multiply_2}, attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.Math, input_kwargs={0: multiply_1, 1: multiply_3})
    
    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: add, 1: 0.6300})
    
    colorramp = nw.new_node(Nodes.ColorRamp, input_kwargs={'Fac': add_1})
    colorramp.color_ramp.elements[0].position = 0.6400
    colorramp.color_ramp.elements[0].color = [1.0000, 1.0000, 1.0000, 1.0000]
    colorramp.color_ramp.elements[1].position = 0.6591
    colorramp.color_ramp.elements[1].color = [0.0000, 0.0000, 0.0000, 1.0000]
    
    mapping_1 = nw.new_node(Nodes.Mapping, input_kwargs={'Vector': attribute_1, 'Scale': (1.0000, 100.0000, 1.0000)})
    
    mix_4 = nw.new_node(Nodes.MixRGB, input_kwargs={'Fac': 0.3000, 'Color1': mapping_1, 'Color2': attribute_1})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture, input_kwargs={'Vector': mix_4, 'Scale': 10.0000})
    
    mix = nw.new_node(Nodes.MixRGB, input_kwargs={'Fac': 0.7000, 'Color1': noise_texture.outputs["Fac"], 'Color2': mix_4})
    
    voronoi_texture = nw.new_node(Nodes.VoronoiTexture, input_kwargs={'Vector': mix, 'Scale': 20.0000})
    
    multiply_4 = nw.new_node(Nodes.Math,
        input_kwargs={0: voronoi_texture.outputs["Distance"], 1: voronoi_texture.outputs["Distance"], 2: 0.0000},
        attrs={'operation': 'MULTIPLY'})
    
    mapping_2 = nw.new_node(Nodes.Mapping, input_kwargs={'Vector': attribute_1, 'Scale': (1.0000, 20.0000, 1.0000)})
    
    mix_8 = nw.new_node(Nodes.MixRGB, input_kwargs={'Fac': 0.3000, 'Color1': mapping_2, 'Color2': attribute_1})
    
    noise_texture_3 = nw.new_node(Nodes.NoiseTexture, input_kwargs={'Vector': mix_8, 'Scale': 10.0000})
    
    mix_1 = nw.new_node(Nodes.MixRGB, input_kwargs={'Fac': 0.7000, 'Color1': noise_texture_3.outputs["Fac"], 'Color2': mix_8})
    
    voronoi_texture_1 = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': mix_1, 'W': 4.5000, 'Scale': 1.0000},
        attrs={'voronoi_dimensions': '4D'})
    
    multiply_5 = nw.new_node(Nodes.Math,
        input_kwargs={0: voronoi_texture_1.outputs["Distance"], 1: voronoi_texture_1.outputs["Distance"], 2: 0.0000},
        attrs={'operation': 'MULTIPLY'})
    
    mix_9 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 1.0000, 'Color1': multiply_4, 'Color2': multiply_5},
        attrs={'blend_type': 'OVERLAY'})
    
    bright_contrast = nw.new_node('ShaderNodeBrightContrast', input_kwargs={'Color': mix_9, 'Bright': 0.6000, 'Contrast': 1.5000})
    
    multiply_6 = nw.new_node(Nodes.Math, input_kwargs={0: group.outputs[1], 1: 0.6000}, attrs={'operation': 'MULTIPLY'})
    
    multiply_7 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_6, 1: multiply_6}, attrs={'operation': 'MULTIPLY'})
    
    multiply_8 = nw.new_node(Nodes.Math, input_kwargs={0: group.outputs["Value"], 1: 0.6000}, attrs={'operation': 'MULTIPLY'})
    
    multiply_9 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_8, 1: multiply_8}, attrs={'operation': 'MULTIPLY'})
    
    add_2 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_7, 1: multiply_9})
    
    add_3 = nw.new_node(Nodes.Math, input_kwargs={0: add_2})
    
    colorramp_1 = nw.new_node(Nodes.ColorRamp, input_kwargs={'Fac': add_3})
    colorramp_1.color_ramp.elements[0].position = 0.6159
    colorramp_1.color_ramp.elements[0].color = [1.0000, 1.0000, 1.0000, 1.0000]
    colorramp_1.color_ramp.elements[1].position = 0.6591
    colorramp_1.color_ramp.elements[1].color = [0.0000, 0.0000, 0.0000, 1.0000]
    
    colorramp_5 = nw.new_node(Nodes.ColorRamp, input_kwargs={'Fac': colorramp_1.outputs["Color"]})
    colorramp_5.color_ramp.elements[0].position = 0.0295
    colorramp_5.color_ramp.elements[0].color = [0.0000, 0.0000, 0.0000, 1.0000]
    colorramp_5.color_ramp.elements[1].position = 0.0523
    colorramp_5.color_ramp.elements[1].color = [1.0000, 1.0000, 1.0000, 1.0000]
    
    add_4 = nw.new_node(Nodes.Math,
        input_kwargs={0: bright_contrast, 1: colorramp_5.outputs["Color"]},
        attrs={'use_clamp': True})
    
    multiply_10 = nw.new_node(Nodes.Math, input_kwargs={0: group.outputs[1]}, attrs={'operation': 'MULTIPLY'})
    
    multiply_11 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_10, 1: multiply_10}, attrs={'operation': 'MULTIPLY'})
    
    multiply_12 = nw.new_node(Nodes.Math, input_kwargs={0: group.outputs["Value"], 1: 0.7000}, attrs={'operation': 'MULTIPLY'})
    
    multiply_13 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_12, 1: multiply_12}, attrs={'operation': 'MULTIPLY'})
    
    add_5 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_11, 1: multiply_13})
    
    add_6 = nw.new_node(Nodes.Math, input_kwargs={0: add_5, 1: 0.1800})
    
    colorramp_2 = nw.new_node(Nodes.ColorRamp, input_kwargs={'Fac': add_6})
    colorramp_2.color_ramp.elements[0].position = 0.4773
    colorramp_2.color_ramp.elements[0].color = [1.0000, 1.0000, 1.0000, 1.0000]
    colorramp_2.color_ramp.elements[1].position = 0.6659
    colorramp_2.color_ramp.elements[1].color = [0.0000, 0.0000, 0.0000, 1.0000]
    
    noise_texture_1 = nw.new_node(Nodes.NoiseTexture, input_kwargs={'W': 1.0000}, attrs={'noise_dimensions': '4D'})
    
    colorramp_4 = nw.new_node(Nodes.ColorRamp, input_kwargs={'Fac': noise_texture_1.outputs["Color"]})
    colorramp_4.color_ramp.interpolation = "CARDINAL"
    colorramp_4.color_ramp.elements[0].position = 0.2886
    colorramp_4.color_ramp.elements[0].color = [1.0000, 0.5767, 0.0000, 1.0000]
    colorramp_4.color_ramp.elements[1].position = 0.5455
    colorramp_4.color_ramp.elements[1].color = [1.0000, 0.0000, 0.0112, 1.0000]
    
    mix_7 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp_2.outputs["Color"], 'Color1': (0.7384, 0.5239, 0.2703, 1.0000), 'Color2': colorramp_4.outputs["Color"]})
    
    mix_6 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp_1.outputs["Color"], 'Color1': mix_7, 'Color2': (0.0000, 0.0000, 0.0000, 1.0000)})
    
    mix_5 = nw.new_node(Nodes.MixRGB, input_kwargs={'Fac': add_4, 'Color1': (0.0000, 0.0000, 0.0000, 1.0000), 'Color2': mix_6})
    
    mix_2 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp.outputs["Color"], 'Color1': mix_5, 'Color2': (0.0000, 0.0000, 0.0000, 1.0000)})
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF, input_kwargs={'Base Color': mix_2, 'Specular': 0.0000, 'Roughness': 0.0000})
    
    principled_bsdf_1 = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Specular': 1.0000, 'Roughness': 0.0000, 'IOR': 1.3500, 'Transmission': 1.0000})
    
    transparent_bsdf = nw.new_node(Nodes.TransparentBSDF)
    
    mix_shader_1 = nw.new_node(Nodes.MixShader, input_kwargs={'Fac': 0.1577, 1: principled_bsdf_1, 2: transparent_bsdf})
    
    mix_shader = nw.new_node(Nodes.MixShader,
        input_kwargs={'Fac': attribute_2.outputs["Color"], 1: principled_bsdf, 2: mix_shader_1})
    
    material_output = nw.new_node(Nodes.MaterialOutput, input_kwargs={'Surface': mix_shader}, attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_eyeball_eyelid_inner', singleton=False, type='GeometryNodeTree')
def nodegroup_eyeball_eyelid_inner(nw: NodeWrangler):
    # Code generated using version 2.6.3 of the node_transpiler

    group_input_2 = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'EyeRot', 0.5000),
            ('NodeSocketVector', 'EyelidCircleShape(W, H)', (0.0000, 0.0000, 0.0000)),
            ('NodeSocketVector', 'EyelidRadiusShape(Out, In1, In2)', (0.0000, 0.0000, 0.0000)),
            ('NodeSocketVector', 'EyelidResolution(Circle, Radius)', (0.0000, 0.0000, 0.0000)),
            ('NodeSocketVector', 'CorneaScale(W, H, Thickness)', (0.0000, 0.0000, 0.0000)),
            ('NodeSocketVector', 'EyeballResolution(White, Cornea)', (0.0000, 0.0000, 0.0000)),
            ('NodeSocketVectorXYZ', 'Scale', (1.0000, 1.0000, 1.0000))])
    
    separate_xyz_6 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': group_input_2.outputs["CorneaScale(W, H, Thickness)"]})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input_2.outputs["EyeRot"], 1: 0.0175},
        attrs={'operation': 'MULTIPLY'})
    
    separate_xyz_7 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': group_input_2.outputs["EyeballResolution(White, Cornea)"]})
    
    eyeball = nw.new_node(nodegroup_eye_ball().name,
        input_kwargs={'CorneaScaleX': separate_xyz_6.outputs["X"], 'Height': separate_xyz_6.outputs["Y"], 'CorneaScaleZ': separate_xyz_6.outputs["Z"], 'Y': multiply, 'EyeballResolution': separate_xyz_7.outputs["X"], 'CorneaResolution': separate_xyz_7.outputs["Y"]})
    
    join_geometry_2 = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [eyeball.outputs["Cornea"], eyeball.outputs["Eyeball"]]})
    
    set_material_1 = nw.new_node(Nodes.SetMaterial,
        input_kwargs={'Geometry': join_geometry_2, 'Material': surface.shaderfunc_to_material(shader_eyeball_tiger)})
    
    value_5 = nw.new_node(Nodes.Value)
    value_5.outputs[0].default_value = 1.5000
    
    transform_2 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': set_material_1, 'Translation': (0.0000, -1.3500, -0.0500), 'Scale': value_5})
    
    position_2 = nw.new_node(Nodes.InputPosition)
    
    store_named_attribute = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': transform_2, 'Name': 'EyeballPosition', 2: position_2},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    join_geometry_3 = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': store_named_attribute})
    
    separate_xyz_3 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': group_input_2.outputs["EyelidCircleShape(W, H)"]})
    
    separate_xyz_5 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': group_input_2.outputs["EyelidResolution(Circle, Radius)"]})
    
    eyelidcircle = nw.new_node(nodegroup_eyelid_circle().name,
        input_kwargs={'ShapeW': separate_xyz_3.outputs["X"], 'ShapeH': separate_xyz_3.outputs["Y"], 'Resolution': separate_xyz_5.outputs["X"]})
    
    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = 0.6000
    
    transform_1 = nw.new_node(Nodes.Transform, input_kwargs={'Geometry': eyelidcircle.outputs["Geometry"], 'Scale': value_1})
    
    separate_xyz_4 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': group_input_2.outputs["EyelidRadiusShape(Out, In1, In2)"]})
    
    eyelidradis = nw.new_node(nodegroup_eyelid_radius().name,
        input_kwargs={'OuterControl': separate_xyz_4.outputs["X"], 'InnerControl1': separate_xyz_4.outputs["Y"], 'InnerControl2': separate_xyz_4.outputs["Z"], 'Resolution': separate_xyz_5.outputs["Y"]})
    
    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
        input_kwargs={'Curve': transform_1, 'Profile Curve': eyelidradis.outputs["Geometry"]})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture, input_kwargs={'Scale': 0.7000})
    
    subtract = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: noise_texture.outputs["Fac"], 1: (0.5000, 0.5000, 0.5000)},
        attrs={'operation': 'SUBTRACT'})
    
    normal = nw.new_node(Nodes.InputNormal)
    
    multiply_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: subtract.outputs["Vector"], 1: normal},
        attrs={'operation': 'MULTIPLY'})
    
    value_2 = nw.new_node(Nodes.Value)
    value_2.outputs[0].default_value = 0.1000
    
    multiply_2 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: multiply_1.outputs["Vector"], 1: value_2},
        attrs={'operation': 'MULTIPLY'})
    
    set_position_1 = nw.new_node(Nodes.SetPosition, input_kwargs={'Geometry': curve_to_mesh, 'Offset': multiply_2.outputs["Vector"]})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': eyelidcircle})
    
    less_than = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz.outputs["Z"], 1: 0.0000}, attrs={'operation': 'LESS_THAN'})
    
    absolute = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz.outputs["X"], 1: 0.0000}, attrs={'operation': 'ABSOLUTE'})
    
    subtract_1 = nw.new_node(Nodes.Math, input_kwargs={0: -0.0000, 1: absolute}, attrs={'operation': 'SUBTRACT', 'use_clamp': True})
    
    multiply_3 = nw.new_node(Nodes.Math, input_kwargs={0: less_than, 1: subtract_1}, attrs={'operation': 'MULTIPLY'})
    
    greater_than = nw.new_node(Nodes.Math, input_kwargs={0: eyelidradis, 1: 0.6000}, attrs={'operation': 'GREATER_THAN'})
    
    multiply_4 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_3, 1: greater_than}, attrs={'operation': 'MULTIPLY'})
    
    multiply_5 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_4, 1: -1.2000}, attrs={'operation': 'MULTIPLY'})
    
    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Y': multiply_5})
    
    set_position_2 = nw.new_node(Nodes.SetPosition, input_kwargs={'Geometry': set_position_1, 'Offset': combine_xyz_2})
    
    transform_3 = nw.new_node(Nodes.Transform, input_kwargs={'Geometry': set_position_2, 'Scale': group_input_2.outputs["Scale"]})
    
    join_geometry_1 = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': transform_3})
    
    position = nw.new_node(Nodes.InputPosition)
    
    separate_xyz_2 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': position})
    
    cosine = nw.new_node(Nodes.Math, input_kwargs={0: multiply}, attrs={'operation': 'COSINE'})
    
    multiply_6 = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz_2.outputs["X"], 1: cosine}, attrs={'operation': 'MULTIPLY'})
    
    sine = nw.new_node(Nodes.Math, input_kwargs={0: multiply}, attrs={'operation': 'SINE'})
    
    multiply_7 = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz_2.outputs["Z"], 1: sine}, attrs={'operation': 'MULTIPLY'})
    
    subtract_2 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_6, 1: multiply_7}, attrs={'operation': 'SUBTRACT'})
    
    multiply_8 = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz_2.outputs["Z"], 1: cosine}, attrs={'operation': 'MULTIPLY'})
    
    multiply_9 = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz_2.outputs["X"], 1: sine}, attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.Math, input_kwargs={0: multiply_8, 1: multiply_9})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': subtract_2, 'Y': separate_xyz_2.outputs["Y"], 'Z': add})
    
    set_position = nw.new_node(Nodes.SetPosition, input_kwargs={'Geometry': join_geometry_1, 'Position': combine_xyz})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Eyeball': join_geometry_3, 'Eyelid': set_position},
        attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_append_eye', singleton=False, type='GeometryNodeTree')
def nodegroup_append_eye(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Target Geometry', None),
            ('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketVector', 'Translation', (0.0, 0.0, 0.0)),
            ('NodeSocketFloat', 'Scale', 0.0),
            ('NodeSocketVectorEuler', 'Rotation', (0.1745, 0.0, -1.3963)),
            ('NodeSocketVector', 'Ray Direction', (-1.0, 0.0, 0.0)),
            ('NodeSocketFloat', 'Default Offset', -0.002)])
    
    position = nw.new_node(Nodes.InputPosition)
    
    store_named_attribute = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Name': 'pos', 2: position},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': store_named_attribute, 'Translation': group_input.outputs["Translation"], 'Rotation': group_input.outputs["Rotation"], 'Scale': group_input.outputs["Scale"]})
    
    raycastappend = nw.new_node(nodegroup_raycast_append().name,
        input_kwargs={'Geometry': transform, 'Target Geometry': group_input.outputs["Target Geometry"], 'Ray Direction': group_input.outputs["Ray Direction"], 'Default Offset': group_input.outputs["Default Offset"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': raycastappend})

@node_utils.to_nodegroup('nodegroup_eye_sockets', singleton=False, type='GeometryNodeTree')
def nodegroup_eye_sockets(nw: NodeWrangler):
    # Code generated using version 2.6.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Skin Mesh', None),
            ('NodeSocketGeometry', 'Skeleton Curve', None),
            ('NodeSocketGeometry', 'Base Mesh', None),
            ('NodeSocketVector', 'Length/Yaw/Rad', (0.5000, 0.0000, 1.0000)),
            ('NodeSocketVector', 'Part Rot', (0.0000, 0.0000, 53.7000)),
            ('NodeSocketVectorXYZ', 'Scale', (2.0000, 2.0000, 2.0000))])
    
    eyehole_cutter = nw.new_node(nodegroup_simple_tube().name,
        input_kwargs={'Origin': (-0.1000, 0.0000, 0.0000), 'Angles Deg': (0.0000, 0.0000, 0.0000), 'Seg Lengths': (0.0500, 0.0500, 0.0900), 'Start Radius': 0.0200, 'Fullness': 0.3000},
        label='Eyehole Cutter')
    
    part_surface_simple = nw.new_node(nodegroup_part_surface_simple().name,
        input_kwargs={'Skeleton Curve': group_input.outputs["Skeleton Curve"], 'Skin Mesh': group_input.outputs["Base Mesh"], 'Length, Yaw, Rad': group_input.outputs["Length/Yaw/Rad"]})
    
    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': eyehole_cutter.outputs["Geometry"], 'Translation': part_surface_simple.outputs["Position"], 'Rotation': group_input.outputs["Part Rot"], 'Scale': group_input.outputs["Scale"]})
    
    symmetric_clone = nw.new_node(nodegroup_symmetric_clone().name, input_kwargs={'Geometry': transform})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': group_input.outputs["Skin Mesh"], 'Mesh': symmetric_clone.outputs["Both"], 'Position': part_surface_simple.outputs["Position"]},
        attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_simple_tube_v2', singleton=False, type='GeometryNodeTree')
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

@node_utils.to_nodegroup('nodegroup_surface_muscle', singleton=False, type='GeometryNodeTree')
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
    
    set_material = nw.new_node(Nodes.SetMaterial,
        input_kwargs={'Geometry': switch.outputs[6], 'Material': surface.shaderfunc_to_material(shader_material)})
    
    u_resolution = nw.new_node(Nodes.Integer,
        label='U Resolution',
        attrs={'integer': 16})
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
        label='V resolution',
        attrs={'integer': 10})
    v_resolution.integer = 10
    
    warped_circle_curve = nw.new_node(nodegroup_warped_circle_curve().name,
        input_kwargs={'Position': multiply.outputs["Vector"], 'Vertices': v_resolution})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group_input.outputs["StartRad, EndRad, Fullness"]})
    
    smoothtaper = nw.new_node(nodegroup_smooth_taper().name,
        input_kwargs={'start_rad': separate_xyz.outputs["X"], 'end_rad': separate_xyz.outputs["Y"], 'fullness': separate_xyz.outputs["Z"]})
    
    profilepart = nw.new_node(nodegroup_profile_part().name,
        input_kwargs={'Skeleton Curve': set_curve_tilt, 'Profile Curve': warped_circle_curve, 'Radius Func': smoothtaper})
    
    join_geometry_1 = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [set_material, profilepart]})
    
    switch_1 = nw.new_node(Nodes.Switch,
        input_kwargs={1: True, 15: join_geometry_1})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': switch_1.outputs[6]})

@node_utils.to_nodegroup('nodegroup_simple_tube', singleton=False, type='GeometryNodeTree')
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

@node_utils.to_nodegroup('nodegroup_smooth_taper', singleton=False, type='GeometryNodeTree')
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

@node_utils.to_nodegroup('nodegroup_warped_circle_curve', singleton=False, type='GeometryNodeTree')
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

@node_utils.to_nodegroup('nodegroup_profile_part', singleton=False, type='GeometryNodeTree')
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

@node_utils.to_nodegroup('nodegroup_polar_bezier', singleton=False, type='GeometryNodeTree')
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
    
    bezier_segment = nw.new_node(Nodes.CurveBezierSegment,
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

@node_utils.to_nodegroup('nodegroup_solidify', singleton=False, type='GeometryNodeTree')
def nodegroup_solidify(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Mesh', None),
            ('NodeSocketFloatDistance', 'Distance', 0.0)])
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Distance"]},
        attrs={'operation': 'MULTIPLY'})
    
    extrude_mesh = nw.new_node(Nodes.ExtrudeMesh,
        input_kwargs={'Mesh': group_input.outputs["Mesh"], 'Offset Scale': multiply, 'Individual': False})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Distance"], 1: -0.5},
        attrs={'operation': 'MULTIPLY'})
    
    extrude_mesh_1 = nw.new_node(Nodes.ExtrudeMesh,
        input_kwargs={'Mesh': group_input.outputs["Mesh"], 'Offset Scale': multiply_1, 'Individual': False})
    
    flip_faces = nw.new_node(Nodes.FlipFaces,
        input_kwargs={'Mesh': extrude_mesh_1.outputs["Mesh"]})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [extrude_mesh.outputs["Mesh"], flip_faces]})
    
    merge_by_distance = nw.new_node(Nodes.MergeByDistance,
        input_kwargs={'Geometry': join_geometry, 'Distance': 0.0})
    
    set_shade_smooth = nw.new_node(Nodes.SetShadeSmooth,
        input_kwargs={'Geometry': merge_by_distance, 'Shade Smooth': False})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_shade_smooth})

@node_utils.to_nodegroup('nodegroup_taper', singleton=False, type='GeometryNodeTree')
def nodegroup_taper(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketVector', 'Start', (1.0, 0.63, 0.72)),
            ('NodeSocketVector', 'End', (1.0, 1.0, 1.0))])
    
    position = nw.new_node(Nodes.InputPosition)
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': position})
    
    attribute_statistic = nw.new_node(Nodes.AttributeStatistic,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 2: separate_xyz.outputs["X"]})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Vector': separate_xyz.outputs["X"], 7: attribute_statistic.outputs["Min"], 8: attribute_statistic.outputs["Max"], 9: group_input.outputs["Start"], 10: group_input.outputs["End"]},
        attrs={'data_type': 'FLOAT_VECTOR', 'clamp': False})
    
    multiply = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: position, 1: map_range.outputs["Vector"]},
        attrs={'operation': 'MULTIPLY'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Position': multiply.outputs["Vector"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_position})

@node_utils.to_nodegroup('nodegroup_raycast_rotation', singleton=False, type='GeometryNodeTree')
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

@node_utils.to_nodegroup('nodegroup_part_surface', singleton=False, type='GeometryNodeTree')
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

@node_utils.to_nodegroup('nodegroup_eyeball_eyelid', singleton=False, type='GeometryNodeTree')
def nodegroup_eyeball_eyelid(nw: NodeWrangler):
    # Code generated using version 2.6.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Skin Mesh', None),
            ('NodeSocketGeometry', 'Base Mesh', None),
            ('NodeSocketGeometry', 'Skeleton Curve', None),
            ('NodeSocketVector', 'Length/Yaw/Rad', (0.5000, 0.0000, 1.0000)),
            ('NodeSocketGeometry', 'Target Geometry', None),
            ('NodeSocketFloat', 'EyeRot', -23.0000),
            ('NodeSocketVector', 'EyelidCircleShape(W, H)', (2.0000, 1.4500, 0.0000)),
            ('NodeSocketVector', 'EyelidRadiusShape(Out, In1, In2)', (0.4000, 5.3000, 0.4000)),
            ('NodeSocketVector', 'EyelidResolution(Circle, Radius)', (32.0000, 32.0000, 0.0000)),
            ('NodeSocketVector', 'CorneaScale(W, H, Thickness)', (0.8000, 0.8000, 0.5500)),
            ('NodeSocketVector', 'EyeballResolution(White, Cornea)', (32.0000, 128.0000, 0.0000)),
            ('NodeSocketVector', 'OffsetPreAppending', (0.0120, 0.0000, 0.0000)),
            ('NodeSocketFloat', 'Scale', 1.0),
            ('NodeSocketVectorEuler', 'Rotation', (0.1745, 0.0000, -1.3963)),
            ('NodeSocketVector', 'RayDirection', (-1.0000, 0.0000, 0.0000)),
            ('NodeSocketFloat', 'DefaultAppendDistance', -0.0020),
            ('NodeSocketVector', 'EyeSocketRot', (0.0000, 0.0000, 0.0000)),
            ('NodeSocketVectorXYZ', 'EyelidScale', (1.0000, 1.0000, 1.0000))])
    
    eyesockets = nw.new_node(nodegroup_eye_sockets().name,
        input_kwargs={'Skin Mesh': group_input.outputs["Skin Mesh"], 'Skeleton Curve': group_input.outputs["Skeleton Curve"], 'Base Mesh': group_input.outputs["Base Mesh"], 'Length/Yaw/Rad': group_input.outputs["Length/Yaw/Rad"], 'Part Rot': group_input.outputs["EyeSocketRot"], 'Scale': group_input.outputs["Scale"]})
    
    #transform = nw.new_node(Nodes.Transform,
    #    input_kwargs={'Geometry': eyesockets.outputs["Mesh"], 'Scale': group_input.outputs["Scale"]})
    
    tigereyeinner = nw.new_node(nodegroup_eyeball_eyelid_inner().name,
        input_kwargs={'EyeRot': group_input.outputs["EyeRot"], 'EyelidCircleShape(W, H)': group_input.outputs["EyelidCircleShape(W, H)"], 'EyelidRadiusShape(Out, In1, In2)': group_input.outputs["EyelidRadiusShape(Out, In1, In2)"], 'EyelidResolution(Circle, Radius)': group_input.outputs["EyelidResolution(Circle, Radius)"], 'CorneaScale(W, H, Thickness)': group_input.outputs["CorneaScale(W, H, Thickness)"], 'EyeballResolution(White, Cornea)': group_input.outputs["EyeballResolution(White, Cornea)"], 'Scale': group_input.outputs["EyelidScale"]})
    
    add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: eyesockets.outputs["Position"], 1: group_input.outputs["OffsetPreAppending"]})
    
    multiply = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Scale"], 1: 0.0170}, attrs={'operation': 'MULTIPLY'})
    
    appendeye = nw.new_node(nodegroup_append_eye().name,
        input_kwargs={'Target Geometry': group_input.outputs["Target Geometry"], 'Geometry': tigereyeinner.outputs["Eyeball"], 'Translation': add, 'Scale': multiply, 'Rotation': group_input.outputs['Rotation'], 'Ray Direction': group_input.outputs["RayDirection"], 'Default Offset': group_input.outputs["DefaultAppendDistance"]})
    
    appendeye_1 = nw.new_node(nodegroup_append_eye().name,
        input_kwargs={'Target Geometry': group_input.outputs["Target Geometry"], 'Geometry': tigereyeinner.outputs["Eyelid"], 'Translation': add, 'Scale': multiply, 'Rotation': group_input.outputs['Rotation'], 'Ray Direction': group_input.outputs["RayDirection"], 'Default Offset': group_input.outputs["DefaultAppendDistance"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': None, 'ParentCutter': eyesockets.outputs["Mesh"], 'Eyeballl': appendeye, 'BodyExtra_Lid': appendeye_1},
        attrs={'is_active_output': True})
    
@node_utils.to_nodegroup('nodegroup_carnivore__face_structure', singleton=False, type='GeometryNodeTree')
def nodegroup_carnivore__face_structure(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'Skull Length Width1 Width2', (0.0, 0.0, 0.0)),
            ('NodeSocketVector', 'Snout Length Width1 Width2', (0.0, 0.0, 0.0)),
            ('NodeSocketFloat', 'Snout Y Scale', 0.62),
            ('NodeSocketVectorXYZ', 'Nose Bridge Scale', (1.0, 0.35, 0.9)),
            ('NodeSocketVector', 'Jaw Muscle Middle Coord', (0.24, 0.41, 1.3)),
            ('NodeSocketVector', 'Jaw StartRad, EndRad, Fullness', (0.06, 0.11, 1.5)),
            ('NodeSocketVector', 'Jaw ProfileHeight, StartTilt, EndTilt', (0.8, 33.1, 0.0)),
            ('NodeSocketVector', 'Lip Muscle Middle Coord', (0.95, 0.0, 1.5)),
            ('NodeSocketVector', 'Lip StartRad, EndRad, Fullness', (0.05, 0.09, 1.48)),
            ('NodeSocketVector', 'Lip ProfileHeight, StartTilt, EndTilt', (0.8, 0.0, -17.2)),
            ('NodeSocketVector', 'Forehead Muscle Middle Coord', (0.7, -1.32, 1.31)),
            ('NodeSocketVector', 'Forehead StartRad, EndRad, Fullness', (0.06, 0.05, 2.5)),
            ('NodeSocketVector', 'Forehead ProfileHeight, StartTilt, EndTilt', (0.3, 60.6, 66.0)),
            ('NodeSocketFloat', 'aspect', 1.0)])
    
    vector = nw.new_node(Nodes.Vector)
    vector.vector = (-0.07, 0.0, 0.05)
    
    simple_tube_v2 = nw.new_node(nodegroup_simple_tube_v2().name,
        input_kwargs={'length_rad1_rad2': group_input.outputs["Skull Length Width1 Width2"], 'angles_deg': (-5.67, 0.0, 0.0), 'aspect': group_input.outputs["aspect"], 'fullness': 3.63, 'Origin': vector})
    
    snout_origin = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: simple_tube_v2.outputs["Endpoint"], 1: (-0.1, 0.0, 0.0)},
        label='Snout Origin')
    
    split_length_width1_width2 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group_input.outputs["Snout Length Width1 Width2"]},
        label='Split Length / Width1 / Width2')
    
    snout_seg_lengths = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: (0.33, 0.33, 0.33), 'Scale': split_length_width1_width2.outputs["X"]},
        label='Snout Seg Lengths',
        attrs={'operation': 'SCALE'})
    
    bridge = nw.new_node(nodegroup_simple_tube().name,
        input_kwargs={'Origin': snout_origin.outputs["Vector"], 'Angles Deg': (-4.0, -4.5, -5.61), 'Seg Lengths': snout_seg_lengths.outputs["Vector"], 'Start Radius': 0.17, 'End Radius': 0.1, 'Fullness': 5.44},
        label='Bridge')
    
    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': bridge.outputs["Geometry"], 'Translation': (0.0, 0.0, 0.03), 'Scale': group_input.outputs["Nose Bridge Scale"]})
    
    snout = nw.new_node(nodegroup_simple_tube().name,
        input_kwargs={'Origin': snout_origin.outputs["Vector"], 'Angles Deg': (-3.0, -4.5, -5.61), 'Seg Lengths': snout_seg_lengths.outputs["Vector"], 'Start Radius': split_length_width1_width2.outputs["Y"], 'End Radius': split_length_width1_width2.outputs["Z"], 'Fullness': 2.0},
        label='Snout')
    
    transform_1 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': snout.outputs["Geometry"], 'Translation': (0.0, 0.0, 0.03), 'Scale': (1.0, 0.7, 0.7)})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': 1.0, 'Y': group_input.outputs["Snout Y Scale"], 'Z': 1.0})
    
    transform_2 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': transform_1, 'Scale': combine_xyz})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [transform, transform_2]})
    
    union = nw.new_node(Nodes.MeshBoolean,
        input_kwargs={'Mesh 2': [join_geometry, simple_tube_v2.outputs["Geometry"]], 'Self Intersection': True},
        attrs={'operation': 'UNION'})
    
    curve_line_1 = nw.new_node(Nodes.CurveLine,
        input_kwargs={'Start': vector, 'End': snout.outputs["Endpoint"]})
    
    jaw_muscle = nw.new_node(nodegroup_surface_muscle().name,
        input_kwargs={'Skin Mesh': union.outputs["Mesh"], 'Skeleton Curve': curve_line_1, 'Coord 0': (0.19, -0.41, 0.78), 'Coord 1': group_input.outputs["Jaw Muscle Middle Coord"], 'Coord 2': (0.67, 1.26, 0.52), 'StartRad, EndRad, Fullness': group_input.outputs["Jaw StartRad, EndRad, Fullness"], 'ProfileHeight, StartTilt, EndTilt': group_input.outputs["Jaw ProfileHeight, StartTilt, EndTilt"]},
        label='Jaw Muscle')
    
    lip = nw.new_node(nodegroup_surface_muscle().name,
        input_kwargs={'Skin Mesh': union.outputs["Mesh"], 'Skeleton Curve': curve_line_1, 'Coord 0': (0.51, -0.13, 0.02), 'Coord 1': group_input.outputs["Lip Muscle Middle Coord"], 'Coord 2': (0.99, 10.57, 0.1), 'StartRad, EndRad, Fullness': group_input.outputs["Lip StartRad, EndRad, Fullness"], 'ProfileHeight, StartTilt, EndTilt': group_input.outputs["Lip ProfileHeight, StartTilt, EndTilt"]},
        label='Lip')
    
    forehead = nw.new_node(nodegroup_surface_muscle().name,
        input_kwargs={'Skin Mesh': simple_tube_v2.outputs["Geometry"], 'Skeleton Curve': simple_tube_v2.outputs["Skeleton Curve"], 'Coord 0': (0.31, -1.06, 0.97), 'Coord 1': group_input.outputs["Forehead Muscle Middle Coord"], 'Coord 2': (0.95, -1.52, 0.9), 'StartRad, EndRad, Fullness': group_input.outputs["Forehead StartRad, EndRad, Fullness"], 'ProfileHeight, StartTilt, EndTilt': group_input.outputs["Forehead ProfileHeight, StartTilt, EndTilt"]},
        label='Forehead')
    
    join_geometry_1 = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [jaw_muscle, lip, forehead]})
    
    symmetric_clone = nw.new_node(nodegroup_symmetric_clone().name,
        input_kwargs={'Geometry': join_geometry_1})
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: (0.33, 0.33, 0.33)},
        attrs={'operation': 'SCALE'})
    
    jaw_cutter = nw.new_node(nodegroup_simple_tube().name,
        input_kwargs={'Origin': (0.0, 0.0, 0.09), 'Angles Deg': (0.0, 0.0, 0.0), 'Seg Lengths': scale.outputs["Vector"], 'Start Radius': 0.13},
        label='Jaw Cutter')
    
    attach_part = nw.new_node(nodegroup_attach_part().name,
        input_kwargs={'Skin Mesh': union.outputs["Mesh"], 'Skeleton Curve': curve_line_1, 'Geometry': jaw_cutter.outputs["Geometry"], 'Length Fac': 0.2, 'Ray Rot': (0.0, 1.5708, 0.0), 'Rad': 1.25, 'Part Rot': (0.0, -8.5, 0.0), 'Do Tangent Rot': True})
    
    difference = nw.new_node(Nodes.MeshBoolean,
        input_kwargs={'Mesh 1': union.outputs["Mesh"], 'Mesh 2': attach_part.outputs["Geometry"], 'Self Intersection': True})
    
    join_geometry_2 = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [symmetric_clone.outputs["Both"], difference.outputs["Mesh"]]})
    
    subdivide_curve = nw.new_node(Nodes.SubdivideCurve,
        input_kwargs={'Curve': curve_line_1, 'Cuts': 10})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': join_geometry_2, 'Skeleton Curve': subdivide_curve, 'Base Mesh': union.outputs["Mesh"], 'Cranium Skeleton': simple_tube_v2.outputs["Skeleton Curve"]})

@node_utils.to_nodegroup('nodegroup_rotate2_d', singleton=False, type='ShaderNodeTree')
def nodegroup_rotate2_d(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'Value', 0.5),
            ('NodeSocketFloat', 'Value', 0.0175),
            ('NodeSocketFloat', 'Value2', 0.5)])
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs[2], 1: 0.0175},
        attrs={'operation': 'MULTIPLY'})
    
    reroute_3 = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': multiply})
    
    sine = nw.new_node(Nodes.Math,
        input_kwargs={0: reroute_3},
        attrs={'operation': 'SINE'})
    
    reroute_5 = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': group_input.outputs[1]})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: sine, 1: reroute_5},
        attrs={'operation': 'MULTIPLY'})
    
    reroute_4 = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': group_input.outputs["Value"]})
    
    cosine = nw.new_node(Nodes.Math,
        input_kwargs={0: reroute_3},
        attrs={'operation': 'COSINE'})
    
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: reroute_4, 1: cosine},
        attrs={'operation': 'MULTIPLY'})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_1, 1: multiply_2},
        attrs={'operation': 'SUBTRACT'})
    
    multiply_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: reroute_5, 1: cosine},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_4 = nw.new_node(Nodes.Math,
        input_kwargs={0: reroute_4, 1: sine},
        attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_3, 1: multiply_4})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={"Value": subtract, "Value1": add})

@node_utils.to_nodegroup('nodegroup_carnivore_jaw', singleton=False, type='GeometryNodeTree')
def nodegroup_carnivore_jaw(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'length_rad1_rad2', (0.0, 0.0, 0.0)),
            ('NodeSocketFloatFactor', 'Width Shaping', 0.6764),
            ('NodeSocketFloat', 'Canine Length', 0.05),
            ('NodeSocketFloat', 'Incisor Size', 0.01),
            ('NodeSocketFloat', 'Tooth Crookedness', 0.0),
            ('NodeSocketFloatFactor', 'Tongue Shaping', 1.0),
            ('NodeSocketFloat', 'Tongue X Scale', 0.9)])
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group_input.outputs["length_rad1_rad2"]})
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: (0.33, 0.33, 0.33), 'Scale': separate_xyz.outputs["X"]},
        attrs={'operation': 'SCALE'})
    
    polarbezier = nw.new_node(nodegroup_polar_bezier().name,
        input_kwargs={'angles_deg': (0.0, 0.0, 13.0), 'Seg Lengths': scale.outputs["Vector"]})
    
    position = nw.new_node(Nodes.InputPosition)
    
    vector_curves = nw.new_node(Nodes.VectorCurve,
        input_kwargs={'Vector': position})
    node_utils.assign_curve(vector_curves.mapping.curves[0], [(-1.0, -1.0), (0.0036, 0.0), (0.2436, 0.21), (1.0, 1.0)])
    node_utils.assign_curve(vector_curves.mapping.curves[1], [(-1.0, 0.12), (-0.7745, 0.06), (-0.6509, -0.44), (-0.3673, -0.4), (-0.0545, -0.01), (0.1055, 0.02), (0.5273, 0.5), (0.7964, 0.64), (1.0, 1.0)], handles=['AUTO', 'AUTO', 'AUTO', 'AUTO_CLAMPED', 'AUTO', 'AUTO', 'VECTOR', 'AUTO', 'AUTO'])
    node_utils.assign_curve(vector_curves.mapping.curves[2], [(-1.0, -1.0), (1.0, 1.0)])
    
    warped_circle_curve = nw.new_node(nodegroup_warped_circle_curve().name,
        input_kwargs={'Position': vector_curves})
    
    spline_parameter = nw.new_node(Nodes.SplineParameter)
    
    float_curve = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Factor': group_input.outputs["Width Shaping"], 'Value': spline_parameter.outputs["Factor"]})
    node_utils.assign_curve(float_curve.mapping.curves[0], [(0.0, 0.955), (0.4255, 0.785), (0.6545, 0.535), (0.9491, 0.75), (1.0, 0.595)], handles=['AUTO', 'AUTO', 'AUTO', 'AUTO_CLAMPED', 'AUTO'])
    
    smoothtaper = nw.new_node(nodegroup_smooth_taper().name,
        input_kwargs={'start_rad': separate_xyz.outputs["Y"], 'end_rad': separate_xyz.outputs["Z"], 'fullness': 2.6})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: float_curve, 1: smoothtaper},
        attrs={'operation': 'MULTIPLY'})
    
    profilepart = nw.new_node(nodegroup_profile_part().name,
        input_kwargs={'Skeleton Curve': polarbezier.outputs["Curve"], 'Profile Curve': warped_circle_curve, 'Radius Func': multiply})
    
    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': profilepart, 'Scale': (1.0, 1.7, 1.0)})
    
    scale_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: (0.33, 0.33, 0.33), 'Scale': group_input.outputs["Canine Length"]},
        attrs={'operation': 'SCALE'})
    
    canine_tooth = nw.new_node(nodegroup_simple_tube().name,
        input_kwargs={'Seg Lengths': scale_1.outputs["Vector"], 'Start Radius': 0.015, 'End Radius': 0.003},
        label='Canine Tooth')
    
    attach_part = nw.new_node(nodegroup_attach_part().name,
        input_kwargs={'Skin Mesh': transform, 'Skeleton Curve': polarbezier.outputs["Curve"], 'Geometry': canine_tooth.outputs["Geometry"], 'Length Fac': 0.9, 'Ray Rot': (1.5708, 0.1204, 1.5708), 'Rad': 1.0, 'Part Rot': (-17.6, -53.49, 0.0)})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': attach_part.outputs["Geometry"]})
    
    symmetric_clone = nw.new_node(nodegroup_symmetric_clone().name,
        input_kwargs={'Geometry': join_geometry})
    
    add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: attach_part.outputs["Position"], 1: (0.015, -0.05, 0.0)})
    
    multiply_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: add.outputs["Vector"], 1: (1.0, -1.0, 1.0)},
        attrs={'operation': 'MULTIPLY'})
    
    add_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: add.outputs["Vector"], 1: multiply_1.outputs["Vector"]})
    
    multiply_add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: add_1.outputs["Vector"], 1: (0.5, 0.5, 0.5), 2: (-0.02, 0.0, 0.0), 'Scale': 0.5},
        attrs={'operation': 'MULTIPLY_ADD'})
    
    quadratic_bezier = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 6, 'Start': add.outputs["Vector"], 'Middle': multiply_add.outputs["Vector"], 'End': multiply_1.outputs["Vector"]})
    
    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
        input_kwargs={'Curve': quadratic_bezier})
    
    transform_1 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': curve_to_mesh})
    
    scale_2 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: (3.0, 1.0, 0.6), 'Scale': group_input.outputs["Incisor Size"]},
        attrs={'operation': 'SCALE'})
    
    cube = nw.new_node(Nodes.MeshCube,
        input_kwargs={'Size': scale_2.outputs["Vector"]})
    
    subdivision_surface = nw.new_node(Nodes.SubdivisionSurface,
        input_kwargs={'Mesh': cube, 'Level': 3})
    
    transform_2 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': subdivision_surface})
    
    instance_on_points = nw.new_node(Nodes.InstanceOnPoints,
        input_kwargs={'Points': transform_1, 'Instance': transform_2, 'Rotation': (0.0, -1.5708, 0.0)})
    
    subtract = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: (2.0, 2.0, 2.0), 1: group_input.outputs["Tooth Crookedness"]},
        attrs={'operation': 'SUBTRACT'})
    
    random_value = nw.new_node(Nodes.RandomValue,
        input_kwargs={0: subtract.outputs["Vector"], 1: group_input.outputs["Tooth Crookedness"]},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    scale_instances = nw.new_node(Nodes.ScaleInstances,
        input_kwargs={'Instances': instance_on_points, 'Scale': random_value.outputs["Value"]})
    
    scale_3 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: (-3.0, -3.0, -3.0), 'Scale': group_input.outputs["Tooth Crookedness"]},
        attrs={'operation': 'SCALE'})
    
    scale_4 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: (3.0, 3.0, 3.0), 'Scale': group_input.outputs["Tooth Crookedness"]},
        attrs={'operation': 'SCALE'})
    
    random_value_1 = nw.new_node(Nodes.RandomValue,
        input_kwargs={0: scale_3.outputs["Vector"], 1: scale_4.outputs["Vector"]},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    deg2rad = nw.new_node(nodegroup_deg2_rad().name,
        input_kwargs={'Deg': random_value_1.outputs["Value"]})
    
    rotate_instances = nw.new_node(Nodes.RotateInstances,
        input_kwargs={'Instances': scale_instances, 'Rotation': deg2rad})
    
    realize_instances = nw.new_node(Nodes.RealizeInstances,
        input_kwargs={'Geometry': rotate_instances})
    
    join_geometry_1 = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [symmetric_clone.outputs["Both"], realize_instances]})
    
    resample_curve = nw.new_node(Nodes.ResampleCurve,
        input_kwargs={'Curve': polarbezier.outputs["Curve"]})
    
    spline_parameter_1 = nw.new_node(Nodes.SplineParameter)
    
    float_curve_1 = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Factor': group_input.outputs["Tongue Shaping"], 'Value': spline_parameter_1.outputs["Factor"]})
    node_utils.assign_curve(float_curve_1.mapping.curves[0], [(0.0, 1.0), (0.6982, 0.55), (0.9745, 0.35), (1.0, 0.175)])
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={3: separate_xyz.outputs["Y"], 4: separate_xyz.outputs["Z"]},
        attrs={'clamp': False})
    
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: float_curve_1, 1: map_range.outputs["Result"]},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_2, 1: 1.0},
        attrs={'operation': 'MULTIPLY'})
    
    set_curve_radius = nw.new_node(Nodes.SetCurveRadius,
        input_kwargs={'Curve': resample_curve, 'Radius': multiply_3})
    
    quadratic_bezier_1 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 3, 'Middle': (0.0, 0.7, 0.0)})
    
    curve_to_mesh_1 = nw.new_node(Nodes.CurveToMesh,
        input_kwargs={'Curve': set_curve_radius, 'Profile Curve': quadratic_bezier_1, 'Fill Caps': True})
    
    solidify = nw.new_node(nodegroup_solidify().name,
        input_kwargs={'Mesh': curve_to_mesh_1, 'Distance': 0.02})
    
    set_shade_smooth = nw.new_node(Nodes.SetShadeSmooth,
        input_kwargs={'Geometry': solidify, 'Shade Smooth': False})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': group_input.outputs["Tongue X Scale"], 'Y': 1.0, 'Z': 1.0})
    
    transform_3 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': set_shade_smooth, 'Rotation': (0.0, -0.0159, 0.0), 'Scale': combine_xyz})
    
    subdivision_surface_1 = nw.new_node(Nodes.SubdivisionSurface,
        input_kwargs={'Mesh': transform_3, 'Level': 2})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': transform, 'Skeleton Curve': polarbezier.outputs["Curve"], 'Teeth': join_geometry_1, 'Tongue': subdivision_surface_1})

@node_utils.to_nodegroup('nodegroup_deg2_rad', singleton=False, type='GeometryNodeTree')
def nodegroup_deg2_rad(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'Deg', (0.0, 0.0, 0.0))])
    
    multiply = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Deg"], 1: (0.0175, 0.0175, 0.0175)},
        attrs={'operation': 'MULTIPLY'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Rad': multiply.outputs["Vector"]})

@node_utils.to_nodegroup('nodegroup_cat_ear', singleton=False, type='GeometryNodeTree')
def nodegroup_cat_ear(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'length_rad1_rad2', (0.0, 0.0, 0.0)),
            ('NodeSocketFloat', 'Depth', 0.0),
            ('NodeSocketFloatDistance', 'Thickness', 0.0),
            ('NodeSocketFloatDistance', 'Curl Deg', 0.0)])
    
    multiply = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Curl Deg"], 1: (-1.0, 1.0, 1.0)},
        attrs={'operation': 'MULTIPLY'})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group_input.outputs["length_rad1_rad2"]})
    
    divide = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: 3.0},
        attrs={'operation': 'DIVIDE'})
    
    polarbezier = nw.new_node(nodegroup_polar_bezier().name,
        input_kwargs={'Origin': (-0.07, 0.0, 0.0), 'angles_deg': multiply.outputs["Vector"], 'Seg Lengths': divide})
    
    spline_parameter = nw.new_node(Nodes.SplineParameter)
    
    float_curve = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': spline_parameter.outputs["Factor"]})
    node_utils.assign_curve(float_curve.mapping.curves[0], [(0.0, 0.0), (0.3236, 0.98), (0.7462, 0.63), (1.0, 0.0)])
    
    set_curve_radius = nw.new_node(Nodes.SetCurveRadius,
        input_kwargs={'Curve': polarbezier.outputs["Curve"], 'Radius': float_curve})
    
    set_curve_tilt = nw.new_node(Nodes.SetCurveTilt,
        input_kwargs={'Curve': set_curve_radius})
    
    multiply_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: (-0.5, 0.0, 0.0)},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_2 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Depth"], 1: (0.0, -1.0, 0.0)},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_3 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: (0.5, 0.0, 0.0)},
        attrs={'operation': 'MULTIPLY'})
    
    quadratic_bezier = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Start': multiply_1.outputs["Vector"], 'Middle': multiply_2.outputs["Vector"], 'End': multiply_3.outputs["Vector"]})
    
    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
        input_kwargs={'Curve': set_curve_tilt, 'Profile Curve': quadratic_bezier})
    
    solidify = nw.new_node(nodegroup_solidify().name,
        input_kwargs={'Mesh': curve_to_mesh, 'Distance': group_input.outputs["Thickness"]})
    
    merge_by_distance = nw.new_node(Nodes.MergeByDistance,
        input_kwargs={'Geometry': solidify, 'Distance': 0.005})
    
    subdivision_surface = nw.new_node(Nodes.SubdivisionSurface,
        input_kwargs={'Mesh': merge_by_distance})
    
    set_shade_smooth = nw.new_node(Nodes.SetShadeSmooth,
        input_kwargs={'Geometry': subdivision_surface, 'Shade Smooth': False})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Mesh': set_shade_smooth})

@node_utils.to_nodegroup('nodegroup_symmetric_clone', singleton=False, type='GeometryNodeTree')
def nodegroup_symmetric_clone(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketVectorXYZ', 'Scale', (1.0, -1.0, 1.0))])
    
    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Scale': group_input.outputs["Scale"]})
    
    flip_faces = nw.new_node(Nodes.FlipFaces,
        input_kwargs={'Mesh': transform})
    
    join_geometry_2 = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [group_input.outputs["Geometry"], flip_faces]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Both': join_geometry_2, 'Orig': group_input.outputs["Geometry"], 'Inverted': flip_faces})

@node_utils.to_nodegroup('nodegroup_cat_nose', singleton=False, type='GeometryNodeTree')
def nodegroup_cat_nose(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloatDistance', 'Nose Radius', 0.06),
            ('NodeSocketFloatDistance', 'Nostril Size', 0.025),
            ('NodeSocketFloatFactor', 'Crease', 0.008),
            ('NodeSocketVectorXYZ', 'Scale', (1.2, 1.0, 1.0))])
    
    cube = nw.new_node(Nodes.MeshCube,
        input_kwargs={'Size': group_input.outputs["Nose Radius"]})
    
    subdivision_surface = nw.new_node(Nodes.SubdivisionSurface,
        input_kwargs={'Mesh': cube, 'Level': 3})
    
    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': subdivision_surface, 'Scale': group_input.outputs["Scale"]})
    
    uv_sphere = nw.new_node(Nodes.MeshUVSphere,
        input_kwargs={'Radius': group_input.outputs["Nostril Size"]})
    
    transform_1 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': uv_sphere, 'Translation': (0.04, 0.025, 0.015), 'Rotation': (0.5643, 0.0, 0.0), 'Scale': (1.0, 0.87, 0.31)})
    
    symmetric_clone = nw.new_node(nodegroup_symmetric_clone().name,
        input_kwargs={'Geometry': transform_1})
    
    difference = nw.new_node(Nodes.MeshBoolean,
        input_kwargs={'Mesh 1': transform, 'Mesh 2': symmetric_clone.outputs["Both"], 'Self Intersection': True})
    
    taper = nw.new_node(nodegroup_taper().name,
        input_kwargs={'Geometry': difference.outputs["Mesh"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': taper})

@node_utils.to_nodegroup('nodegroup_attach_part', singleton=False, type='GeometryNodeTree')
def nodegroup_attach_part(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

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
        input_kwargs={'Geometry': transform, 'Position': part_surface.outputs["Position"], 'Rotation': raycast_rotation})

@node_utils.to_nodegroup('nodegroup_carnivore_head', singleton=False, type='GeometryNodeTree')
def nodegroup_carnivore_head(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'length_rad1_rad2', (0.5, 0.0, 0.0)),
            ('NodeSocketVector', 'snout_length_rad1_rad2', (0.0, 0.0, 0.0)),
            ('NodeSocketFloat', 'Snout Y Scale', 0.62),
            ('NodeSocketVector', 'eye_coord', (0.96, -0.95, 0.79)),
            ('NodeSocketVectorXYZ', 'Nose Bridge Scale', (1.0, 0.35, 0.9)),
            ('NodeSocketVector', 'Jaw Muscle Middle Coord', (0.24, 0.41, 1.3)),
            ('NodeSocketVector', 'Jaw StartRad, EndRad, Fullness', (0.06, 0.11, 1.5)),
            ('NodeSocketVector', 'Jaw ProfileHeight, StartTilt, EndTilt', (0.8, 33.1, 0.0)),
            ('NodeSocketVector', 'Lip Muscle Middle Coord', (0.95, 0.0, 1.5)),
            ('NodeSocketVector', 'Lip StartRad, EndRad, Fullness', (0.05, 0.09, 1.48)),
            ('NodeSocketVector', 'Lip ProfileHeight, StartTilt, EndTilt', (0.8, 0.0, -17.2)),
            ('NodeSocketVector', 'Forehead Muscle Middle Coord', (0.7, -1.32, 1.31)),
            ('NodeSocketVector', 'Forehead StartRad, EndRad, Fullness', (0.06, 0.05, 2.5)),
            ('NodeSocketVector', 'Forehead ProfileHeight, StartTilt, EndTilt', (0.3, 60.6, 66.0)),
            ('NodeSocketFloat', 'aspect', 1.0)])
    
    carnivore_face_structure = nw.new_node(nodegroup_carnivore__face_structure().name,
        input_kwargs={'Skull Length Width1 Width2': group_input.outputs["length_rad1_rad2"], 'Snout Length Width1 Width2': group_input.outputs["snout_length_rad1_rad2"], 'Snout Y Scale': group_input.outputs["Snout Y Scale"], 'Nose Bridge Scale': group_input.outputs["Nose Bridge Scale"], 'Jaw Muscle Middle Coord': group_input.outputs["Jaw Muscle Middle Coord"], 'Jaw StartRad, EndRad, Fullness': group_input.outputs["Jaw StartRad, EndRad, Fullness"], 'Jaw ProfileHeight, StartTilt, EndTilt': group_input.outputs["Jaw ProfileHeight, StartTilt, EndTilt"], 'Lip Muscle Middle Coord': group_input.outputs["Lip Muscle Middle Coord"], 'Lip StartRad, EndRad, Fullness': group_input.outputs["Lip StartRad, EndRad, Fullness"], 'Lip ProfileHeight, StartTilt, EndTilt': group_input.outputs["Lip ProfileHeight, StartTilt, EndTilt"], 'Forehead Muscle Middle Coord': group_input.outputs["Forehead Muscle Middle Coord"], 'Forehead StartRad, EndRad, Fullness': group_input.outputs["Forehead StartRad, EndRad, Fullness"], 'Forehead ProfileHeight, StartTilt, EndTilt': group_input.outputs["Forehead ProfileHeight, StartTilt, EndTilt"], 'aspect': group_input.outputs["aspect"]})
    
    tigereye = nw.new_node(nodegroup_eyeball_eyelid().name,
        input_kwargs={
            'Skin Mesh': carnivore_face_structure.outputs["Geometry"], 
            'Base Mesh': carnivore_face_structure.outputs["Base Mesh"], 
            'Skeleton Curve': carnivore_face_structure.outputs["Cranium Skeleton"], 
            'Length/Yaw/Rad': group_input.outputs["eye_coord"], 
            'Target Geometry': carnivore_face_structure.outputs["Geometry"], 
            'EyelidCircleShape(W, H)': (2.0, 1.35, 0.0), 
            'CorneaScale(W, H, Thickness)': (0.8, 0.8, 0.7), 
            'DefaultAppendDistance': 0.002,
            'EyelidScale': (1.1, 1.6, 1.6),
            'Scale': 1.0,
            })
    
    difference = nw.new_node(Nodes.MeshBoolean,
        input_kwargs={'Mesh 1': carnivore_face_structure.outputs["Geometry"], 'Mesh 2': tigereye.outputs["ParentCutter"], 'Self Intersection': True})
    
    symmetric_clone = nw.new_node(nodegroup_symmetric_clone().name,
        input_kwargs={'Geometry': tigereye.outputs["Eyeballl"]})
    
    symmetric_clone_1 = nw.new_node(nodegroup_symmetric_clone().name,
        input_kwargs={'Geometry': tigereye.outputs["BodyExtra_Lid"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': difference.outputs["Mesh"], 'Skeleton Curve': carnivore_face_structure.outputs["Skeleton Curve"], 'Base Mesh': carnivore_face_structure.outputs["Base Mesh"], 'LeftEye': symmetric_clone.outputs["Orig"], 'RightEye': symmetric_clone.outputs["Inverted"], 'Eyelid': symmetric_clone_1.outputs["Both"]})

def shader_eyeball_tiger(nw: NodeWrangler, **input_kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    attribute_2 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'tag_cornea'})
    
    attribute_1 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'EyeballPosition'})
    
    reroute_8 = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': attribute_1.outputs["Color"]})
    
    reroute = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': reroute_8})
    
    noise_texture_2 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': reroute, 'Scale': 50.0})
    
    mix_3 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.02, 'Color1': reroute, 'Color2': noise_texture_2.outputs["Color"]})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': mix_3})
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 30.0
    
    group = nw.new_node(nodegroup_rotate2_d().name,
        input_kwargs={0: separate_xyz.outputs["X"], 1: separate_xyz.outputs["Z"], 2: value})
    
    w_offset = U(0, 0.2)
    iris_scale = U(0.4, 0.8)
    scale2 = iris_scale*1.7+N(0, 0.05)
    scale3 = iris_scale*1.75+N(0, 0.05)

    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: group.outputs[1], 1: iris_scale},
        attrs={'operation': 'MULTIPLY'})
    
    reroute_2 = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': multiply})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: reroute_2, 1: reroute_2},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: group.outputs["Value"], 1: iris_scale+w_offset},
        attrs={'operation': 'MULTIPLY'})
    
    reroute_1 = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': multiply_2})
    
    multiply_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: reroute_1, 1: reroute_1},
        attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_1, 1: multiply_3})
    
    add_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: add, 1: 0.63})
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': add_1})
    colorramp.color_ramp.elements[0].position = 0.64
    colorramp.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    colorramp.color_ramp.elements[1].position = 0.6591
    colorramp.color_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)
    
    mapping_1 = nw.new_node(Nodes.Mapping,
        input_kwargs={'Vector': reroute_8, 'Scale': (1.0, U(1, 200), 1.0)},
        attrs={'vector_type': 'NORMAL'})
    
    mix_4 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': U(0.2, 0.4), 'Color1': mapping_1, 'Color2': reroute_8})
    
    reroute_3 = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': mix_4})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': reroute_3, 'Scale': 10.0})
    
    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': U(0.5, 0.9), 'Color1': noise_texture.outputs["Fac"], 'Color2': reroute_3})
    
    voronoi_texture = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': mix, 'Scale': 10.0+N(0, 2)})
    
    multiply_4 = nw.new_node(Nodes.Math,
        input_kwargs={0: voronoi_texture.outputs["Distance"], 1: voronoi_texture.outputs["Distance"], 2: 0.0},
        attrs={'operation': 'MULTIPLY'})
    
    bright_contrast = nw.new_node('ShaderNodeBrightContrast',
        input_kwargs={'Color': multiply_4, 'Bright': 0.6, 'Contrast': U(0.8, 1.2)})
    
    multiply_5 = nw.new_node(Nodes.Math,
        input_kwargs={0: group.outputs[1], 1: scale3},
        attrs={'operation': 'MULTIPLY'})
    
    reroute_6 = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': multiply_5})
    
    multiply_6 = nw.new_node(Nodes.Math,
        input_kwargs={0: reroute_6, 1: reroute_6},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_7 = nw.new_node(Nodes.Math,
        input_kwargs={0: group.outputs["Value"], 1: scale3+w_offset},
        attrs={'operation': 'MULTIPLY'})
    
    reroute_7 = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': multiply_7})
    
    multiply_8 = nw.new_node(Nodes.Math,
        input_kwargs={0: reroute_7, 1: reroute_7},
        attrs={'operation': 'MULTIPLY'})
    
    add_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_6, 1: multiply_8})
    
    add_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: add_2, 1: 0.18})
    
    colorramp_3 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': add_3})
    colorramp_3.color_ramp.elements[0].position = 0.5955
    colorramp_3.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_3.color_ramp.elements[1].position = 1.0
    colorramp_3.color_ramp.elements[1].color = (0.7896, 0.7896, 0.7896, 1.0)
    
    add_4 = nw.new_node(Nodes.Math,
        input_kwargs={0: bright_contrast, 1: colorramp_3.outputs["Color"]})

    multiply_9 = nw.new_node(Nodes.Math,
        input_kwargs={0: group.outputs[1], 1: scale2},
        attrs={'operation': 'MULTIPLY'})
    
    reroute_4 = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': multiply_9})
    
    multiply_10 = nw.new_node(Nodes.Math,
        input_kwargs={0: reroute_4, 1: reroute_4},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_11 = nw.new_node(Nodes.Math,
        input_kwargs={0: group.outputs["Value"], 1: scale2+w_offset},
        attrs={'operation': 'MULTIPLY'})
    
    reroute_5 = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': multiply_11})
    
    multiply_12 = nw.new_node(Nodes.Math,
        input_kwargs={0: reroute_5, 1: reroute_5},
        attrs={'operation': 'MULTIPLY'})
    
    add_5 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_10, 1: multiply_12})
    
    add_6 = nw.new_node(Nodes.Math,
        input_kwargs={0: add_5})
    
    colorramp_1 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': add_6})
    colorramp_1.color_ramp.elements[0].position = 0.6159
    colorramp_1.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    colorramp_1.color_ramp.elements[1].position = 0.6591
    colorramp_1.color_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)
    
    colorramp_2 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': add_3})
    colorramp_2.color_ramp.elements[0].position = 0.4773
    colorramp_2.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    colorramp_2.color_ramp.elements[1].position = 0.6659
    colorramp_2.color_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)
    
    mix_7 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp_2.outputs["Color"], 'Color1': (U(0.5, 0.9), U(0.3, 0.8), U(0.3, 0.7), 1.0), 'Color2': (U(0.2, 0.6), U(0.15, 0.6), U(0.1, 0.4), 1.0)})
    
    mix_6 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp_1.outputs["Color"], 'Color1': mix_7, 'Color2': (U(0.1, 0.55), U(0.1, 0.55), U(0.1, 0.55), 1.0)})
    
    color1 = max(0, N(0.125, 0.05))
    mix_5 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': add_4, 'Color1': (color1, U(0, color1), U(0, color1), 1.0), 'Color2': mix_6})
    
    mix_2 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp.outputs["Color"], 'Color1': mix_5, 'Color2': (0.0, 0.0, 0.0, 1.0)})
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': mix_2, 'Specular': 0.0, 'Roughness': 0.0})
    
    principled_bsdf_1 = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Specular': 1.0, 'Roughness': 0.0, 'IOR': 1.35, 'Transmission': 1.0})
    
    transparent_bsdf = nw.new_node(Nodes.TransparentBSDF)
    
    mix_shader = nw.new_node(Nodes.MixShader,
        input_kwargs={'Fac': 0.1577, 1: principled_bsdf_1, 2: transparent_bsdf})
    
    mix_shader_1 = nw.new_node(Nodes.MixShader,
        input_kwargs={'Fac': attribute_2.outputs["Color"], 1: principled_bsdf, 2: mix_shader})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': mix_shader_1})

def geometry_tiger_head(nw: NodeWrangler, input_kwargs={}):
    # Code generated using version 2.4.3 of the node_transpiler

    carnivorehead = nw.new_node(nodegroup_carnivore_head().name,
        input_kwargs={'length_rad1_rad2': (0.36, 0.2, 0.18), 'snout_length_rad1_rad2': (0.25, 0.15, 0.15), 'eye_coord': (0.96, -0.85, 0.79), 'Lip Muscle Middle Coord': (0.95, -0.45, 2.03)})
    
    nose_radius = nw.new_node(nodegroup_cat_nose().name,
        input_kwargs={'Nose Radius': 0.11, 'Nostril Size': 0.03, 'Crease': 0.237},
        label='NoseRadius ~ N(0.1, 0.02)')
    
    attach_nose = nw.new_node(nodegroup_attach_part().name,
        input_kwargs={'Skin Mesh': carnivorehead.outputs["Base Mesh"], 'Skeleton Curve': carnivorehead.outputs["Skeleton Curve"], 'Geometry': nose_radius, 'Length Fac': 0.9017, 'Ray Rot': (0.0, -1.3277, 0.0), 'Rad': 0.56, 'Part Rot': (0.0, 26.86, 0.0), 'Do Normal Rot': True, 'Do Tangent Rot': True},
        label='Attach Nose')
    
    cat_ear = nw.new_node(nodegroup_cat_ear().name,
        input_kwargs={'length_rad1_rad2': (0.2, 0.1, 0.0), 'Depth': 0.06, 'Thickness': 0.01, 'Curl Deg': 49.0})
    
    deg2rad = nw.new_node(nodegroup_deg2_rad().name,
        input_kwargs={'Deg': (90.0, -44.0, 90.0)})
    
    attach_ear = nw.new_node(nodegroup_attach_part().name,
        input_kwargs={'Skin Mesh': carnivorehead.outputs["Base Mesh"], 'Skeleton Curve': carnivorehead.outputs["Skeleton Curve"], 'Geometry': cat_ear, 'Length Fac': 0.3328, 'Ray Rot': deg2rad, 'Rad': 1.0, 'Part Rot': (-43.3, -9.5, -29.6), 'Do Normal Rot': True},
        label='Attach Ear')
    
    symmetric_clone = nw.new_node(nodegroup_symmetric_clone().name,
        input_kwargs={'Geometry': attach_ear.outputs["Geometry"]})
    
    carnivore_jaw = nw.new_node(nodegroup_carnivore_jaw().name,
        input_kwargs={'length_rad1_rad2': (0.4, 0.12, 0.08), 'Width Shaping': 1.0, 'Tooth Crookedness': 1.2})
    
    join_geometry_3 = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [carnivore_jaw.outputs["Geometry"], carnivore_jaw.outputs["Teeth"], carnivore_jaw.outputs["Tongue"]]})
    
    attach_jaw = nw.new_node(nodegroup_attach_part().name,
        input_kwargs={'Skin Mesh': carnivorehead.outputs["Base Mesh"], 'Skeleton Curve': carnivorehead.outputs["Skeleton Curve"], 'Geometry': join_geometry_3, 'Length Fac': 0.2, 'Ray Rot': (0.0, 1.5708, 0.0), 'Rad': 0.36, 'Part Rot': (0.0, 21.1, 0.0), 'Do Normal Rot': True, 'Do Tangent Rot': True},
        label='Attach Jaw')
    
    join_geometry_2 = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [carnivorehead.outputs["Geometry"], attach_nose.outputs["Geometry"], carnivorehead.outputs["LeftEye"], symmetric_clone.outputs["Both"], attach_jaw.outputs["Geometry"], carnivorehead.outputs["RightEye"], carnivorehead.outputs["Eyelid"]]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': join_geometry_2})

class Eye(PartFactory):

    tags = ['head_detail', 'eye_socket']

    def sample_params(self):
        return {
            'Skin Mesh': None,
            'Base Mesh': None,
            'Skeleton Curve': None,
            'Length/Yaw/Rad': (0.5, 0.0, 1.0),
            'Target Geometry': None,
            'EyeRot': -U(15, 35),
            'EyelidCircleShape(W, H)': (2.0, U(1.3, 1.5), 0.0),
            'EyelidRadiusShape(Out, In1, In2)': (0.4, 5.3, 0.4),
            'EyelidResolution(Circle, Radius)': (32.0, 32.0, 0.0),
            'CorneaScale(W, H, Thickness)': (0.8, 0.8, 0.55),
            'EyeballResolution(White, Cornea)': (32.0, 128.0, 0.0),
            'OffsetPreAppending': (0.012, 0.0, 0.0),
            'Scale': (0.9, 1.1),
            'Rotation': (0.1745, 0.0, -1.3963),
            'RayDirection': (-1.0, 0.0, 0.0),
            'DefaultAppendDistance': -0.002,
        }
    
    def sample_params_fish(self):
        return {
            'Skin Mesh': None,
            'Base Mesh': None, 
            'Skeleton Curve': None, 
            'Length/Yaw/Rad': (0.8800, -0.6000, 1.0000), 
            'Target Geometry': None, 
            'EyeRot': 0.0000, 
            'EyelidCircleShape(W, H)': (2.0000, 1.0000, 0.0000), 
            'EyelidRadiusShape(Out, In1, In2)': (0.4000, 5.3000, 0.3000), 
            'CorneaScale(W, H, Thickness)': (0.8000, 0.8000, 0.8500), 
            'OffsetPreAppending': (0.0000, 0.0100, 0.0000), 
            'Scale': 1.5000, 
            'Rotation': (0.0873, 0.0000, -0.2618), 
            'RayDirection': (-0.3000, -0.8000, 0.0000), 
            'DefaultAppendDistance': 0.0050,
            'EyeSocketRot': (0.0000, 0.0000, 80.0000)
        }

    def make_part(self, params):
        part = part_util.nodegroup_to_part(nodegroup_eyeball_eyelid, params)
        return part

def apply(obj, geo_kwargs=None, shader_kwargs=None, **kwargs):
    surface.add_geomod(obj, geometry_tiger_head, apply=False, input_kwargs=geo_kwargs)
    
if __name__ == "__main__":
    mat = 'tigereye'
    for i in range(1):
        bpy.ops.wm.open_mainfile(filepath='test.blend')
        apply(bpy.data.objects['SolidModel'], geo_kwargs={}, shader_kwargs={})
        fn = os.path.join(os.path.abspath(os.curdir), 'tigereye_test.blend')
        bpy.ops.wm.save_as_mainfile(filepath=fn)
        #bpy.context.scene.render.filepath = os.path.join('surfaces/surface_thumbnails', '%s_%d.jpg'%(mat, i))
        #bpy.context.scene.render.image_settings.file_format='JPEG'
        #bpy.ops.render.render(write_still=True)