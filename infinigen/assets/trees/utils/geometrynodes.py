# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alejandro Newell


import imp
import bpy
import numpy as np

from . import helper, mesh
from .materials import new_link

from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils

C = bpy.context
D = bpy.data

def add_node_modifier(obj):
  # Add geometry node modifier
  helper.set_active_obj(obj)
  # bpy.ops.node.new_geometry_nodes_modifier() # Blender 3.2
  bpy.ops.object.modifier_add(type='NODES') # Blender 3.1
  return obj.modifiers[-1]


def setup_inps(ng, inp, nodes):
  for k_idx, (k, node, attr) in enumerate(nodes):
    new_link(ng, inp, k_idx, node, attr)
    ng.inputs[k_idx].name = k


@node_utils.to_nodegroup('CollectionDistribute', singleton=False)
def coll_distribute(nw, merge_dist=None):
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
                      ('NodeSocketBool', 'Selection', True),
                      ('NodeSocketCollection', 'Collection', None),
                      ('NodeSocketInt', 'Multi inst', 1),
                      ('NodeSocketFloat', 'Density', 0.5),
                      ('NodeSocketFloat', 'Min scale', 0.0),
                      ('NodeSocketFloat', 'Max scale', 1.0),
                      ('NodeSocketFloat', 'Pitch scaling', 0.2),
                      ('NodeSocketFloat', 'Pitch offset', 0.0),
                      ('NodeSocketFloat', 'Pitch variance', 0.4),
                      ('NodeSocketFloat', 'Yaw variance', 0.4),
                      ('NodeSocketBool', 'Realize Instance', False)])

    mesh_to_curve = nw.new_node('GeometryNodeMeshToCurve',
        input_kwargs={'Mesh': group_input.outputs["Geometry"], 'Selection': group_input.outputs["Selection"]})

    curve_to_points = nw.new_node('GeometryNodeCurveToPoints',
        input_kwargs={'Curve': mesh_to_curve, 'Count': group_input.outputs["Multi inst"]})

    mesh_to_points = nw.new_node('GeometryNodeMeshToPoints',
        input_kwargs={'Mesh': group_input.outputs["Geometry"], 'Selection': group_input.outputs["Selection"]})

    position = nw.new_node(Nodes.InputPosition)

    transfer_attribute_index = nw.new_node(Nodes.SampleNearest,
        input_kwargs={'Geometry': mesh_to_points})

    transfer_attribute = nw.new_node(Nodes.SampleIndex,
        input_kwargs={'Geometry': mesh_to_points, 'Value': position, 'Index': transfer_attribute_index},
        attrs={'data_type': 'FLOAT_VECTOR'})

    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': curve_to_points.outputs["Points"], 'Position': (transfer_attribute, "Value")})

    random_value = nw.new_node(Nodes.RandomValue)

    math = nw.new_node(Nodes.Math,
        input_kwargs={0: random_value.outputs[1], 1: group_input.outputs["Density"]},
        attrs={'operation': 'LESS_THAN'})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': curve_to_points.outputs["Rotation"]})

    math_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: 1.5708})

    math_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: math_1, 1: group_input.outputs["Pitch scaling"]},
        attrs={'operation': 'MULTIPLY'})

    math_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: math_2, 1: group_input.outputs["Pitch offset"]})

    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': math_3, 'Z': separate_xyz.outputs["Z"]})

    math_4 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Pitch variance"], 1: -1.0},
        attrs={'operation': 'MULTIPLY'})

    random_value_1 = nw.new_node(Nodes.RandomValue,
        input_kwargs={2: math_4, 3: group_input.outputs["Pitch variance"]})

    math_5 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Yaw variance"], 1: -1.0},
        attrs={'operation': 'MULTIPLY'})

    random_value_2 = nw.new_node(Nodes.RandomValue,
        input_kwargs={2: math_5, 3: group_input.outputs["Yaw variance"]})

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': random_value_1.outputs[1], 'Z': random_value_2.outputs[1]})

    vector_math = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: combine_xyz, 1: combine_xyz_1})

    random_value_3 = nw.new_node(Nodes.RandomValue,
        input_kwargs={2: group_input.outputs["Min scale"], 3: group_input.outputs["Max scale"]})

    geo = nw.new_node(Nodes.CollectionInfo,
        input_kwargs={'Collection': group_input.outputs["Collection"], 'Separate Children': True, 'Reset Children': True})

    if merge_dist is not None:
       geo = nw.new_node(Nodes.MergeByDistance, [geo, None, merge_dist])

    instance_on_points = nw.new_node(Nodes.InstanceOnPoints,
        input_kwargs={'Points': set_position, 'Selection': math, 'Instance': geo, 'Pick Instance': True, 'Rotation': vector_math.outputs["Vector"], 'Scale': random_value_3.outputs[1]})

    realize_instances = nw.new_node(Nodes.RealizeInstances,
        input_kwargs={'Geometry': instance_on_points})

    switch = nw.new_node(Nodes.Switch,
        input_kwargs={1: group_input.outputs["Realize Instance"], 14: instance_on_points, 15: realize_instances})

    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': switch.outputs[6]})


@node_utils.to_nodegroup('PhylloDist', singleton=False)
def phyllotaxis_distribute(nw):
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
                      ('NodeSocketInt', 'Count', 50),
                      ('NodeSocketFloat', 'Max radius', 2.0),
                      ('NodeSocketFloat', 'Radius exp', 0.5),
                      ('NodeSocketFloat', 'Inner pct', 0.0),
                      ('NodeSocketFloat', 'Min angle', -0.5236),
                      ('NodeSocketFloat', 'Max angle', 0.7854),
                      ('NodeSocketFloat', 'Min scale', 0.3),
                      ('NodeSocketFloat', 'Max scale', 0.3),
                      ('NodeSocketFloat', 'Min z', 0.0),
                      ('NodeSocketFloat', 'Max z', 1.0),
                      ('NodeSocketFloat', 'Clamp z', 1.0),
                      ('NodeSocketFloat', 'Yaw offset', -np.pi / 2)])

    mesh_line = nw.new_node('GeometryNodeMeshLine',
        input_kwargs={'Count': group_input.outputs["Count"]})

    mesh_to_points = nw.new_node('GeometryNodeMeshToPoints',
        input_kwargs={'Mesh': mesh_line})

    position = nw.new_node(Nodes.InputPosition)

    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': mesh_to_points, 1: position},
        attrs={'data_type': 'FLOAT_VECTOR'})

    index = nw.new_node('GeometryNodeInputIndex')

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 1.0

    math = nw.new_node(Nodes.Math,
        input_kwargs={0: index, 1: value},
        attrs={'operation': 'DIVIDE'})

    math_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: math},
        attrs={'operation': 'FLOOR'})

    math_6 = nw.new_node(Nodes.Math,
        input_kwargs={0: math_1, 1: 2.3998},
        attrs={'operation': 'MULTIPLY'})

    math_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: math},
        attrs={'operation': 'FRACT'})

    math_5 = nw.new_node(Nodes.Math,
        input_kwargs={0: math_2, 1: 6.2832},
        attrs={'operation': 'MULTIPLY'})

    math_7 = nw.new_node(Nodes.Math,
        input_kwargs={0: math_6, 1: math_5})

    math_8 = nw.new_node(Nodes.Math,
        input_kwargs={0: math_7},
        attrs={'operation': 'COSINE'})

    math_9 = nw.new_node(Nodes.Math,
        input_kwargs={0: math_7},
        attrs={'operation': 'SINE'})

    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': math_8, 'Y': math_9})

    math_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Count"], 1: value},
        attrs={'operation': 'DIVIDE'})

    math_4 = nw.new_node(Nodes.Math,
        input_kwargs={0: math_1, 1: math_3},
        attrs={'operation': 'DIVIDE'})

    math_10 = nw.new_node(Nodes.Math,
        input_kwargs={0: math_4, 1: group_input.outputs["Radius exp"]},
        attrs={'operation': 'POWER'})

    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': math_10, 3: group_input.outputs["Inner pct"]})

    math_11 = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range.outputs["Result"], 1: group_input.outputs["Max radius"]},
        attrs={'operation': 'MULTIPLY'})

    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': math_4, 3: 1.5708, 4: 1.5708})

    math_12 = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_1.outputs["Result"]},
        attrs={'operation': 'SINE'})

    math_13 = nw.new_node(Nodes.Math,
        input_kwargs={0: math_11, 1: math_12},
        attrs={'operation': 'MULTIPLY'})

    vector_math = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: combine_xyz, 1: math_13},
        attrs={'operation': 'MULTIPLY'})

    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': vector_math.outputs["Vector"]})

    map_range_2 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': math_4, 2: group_input.outputs["Clamp z"], 3: group_input.outputs["Min z"], 4: group_input.outputs["Max z"]})

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': separate_xyz.outputs["X"], 'Y': separate_xyz.outputs["Y"], 'Z': map_range_2.outputs["Result"]})

    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': capture_attribute.outputs["Geometry"], 'Position': combine_xyz_1})

    attribute_statistic = nw.new_node(Nodes.AttributeStatistic,
        input_kwargs={'Geometry': capture_attribute.outputs["Geometry"], 2: map_range.outputs["Result"]})

    map_range_3 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': map_range.outputs["Result"], 1: attribute_statistic.outputs["Max"], 2: attribute_statistic.outputs["Min"], 3: group_input.outputs["Min angle"], 4: group_input.outputs["Max angle"]})

    random_value_1 = nw.new_node(Nodes.RandomValue,
        input_kwargs={2: -0.1, 3: 0.1})

    math_14 = nw.new_node(Nodes.Math,
        input_kwargs={0: math_7, 1: group_input.outputs["Yaw offset"]})

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': map_range_3.outputs["Result"], 'Y': random_value_1.outputs[1], 'Z': math_14})

    random_value = nw.new_node(Nodes.RandomValue,
        input_kwargs={2: group_input.outputs["Min scale"], 3: group_input.outputs["Max scale"]})

    instance_on_points = nw.new_node(Nodes.InstanceOnPoints,
        input_kwargs={'Points': set_position, 'Instance': group_input.outputs["Geometry"], 'Rotation': combine_xyz_2, 'Scale': random_value.outputs[1]})

    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Instances': instance_on_points})


@node_utils.to_nodegroup('FollowCurve', singleton=False)
def follow_curve(nw):
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
                      ('NodeSocketGeometry', 'Curve', None),
                      ('NodeSocketFloat', 'Offset', 0.5)])

    position = nw.new_node(Nodes.InputPosition)

    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 1: position},
        attrs={'data_type': 'FLOAT_VECTOR'})

    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': capture_attribute.outputs["Attribute"]})

    math = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: group_input.outputs["Offset"]})

    sample_curve = nw.new_node('GeometryNodeSampleCurve',
        input_kwargs={'Curve': group_input.outputs["Curve"], 'Length': math})

    vector_math = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: sample_curve.outputs["Tangent"], 1: sample_curve.outputs["Normal"]},
        attrs={'operation': 'CROSS_PRODUCT'})

    vector_math_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: vector_math.outputs["Vector"], 'Scale': separate_xyz.outputs["X"]},
        attrs={'operation': 'SCALE'})

    vector_math_2 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: sample_curve.outputs["Normal"], 'Scale': separate_xyz.outputs["Y"]},
        attrs={'operation': 'SCALE'})

    vector_math_3 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: vector_math_1.outputs["Vector"], 1: vector_math_2.outputs["Vector"]})

    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': capture_attribute.outputs["Geometry"], 'Position': sample_curve.outputs["Position"], 'Offset': vector_math_3.outputs["Vector"]})

    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_position})


@node_utils.to_nodegroup('SetTreeRadius', singleton=False, type='GeometryNodeTree')
def set_tree_radius(nw):
    # Code generated using version 2.3.1 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketBool', 'Selection', True),
            ('NodeSocketFloat', 'Reverse depth', 0.5),
            ('NodeSocketFloat', 'Scaling', 0.2),
            ('NodeSocketFloat', 'Exponent', 1.5),
            ('NodeSocketFloat', 'Min radius', 0.02),
            ('NodeSocketFloat', 'Max radius', 5.0),
            ('NodeSocketInt', 'Profile res', 20),
            ('NodeSocketFloatDistance', 'Merge dist', 0.001)])
    
    mesh_to_curve = nw.new_node(Nodes.MeshToCurve,
        input_kwargs={'Mesh': group_input.outputs["Geometry"], 'Selection': group_input.outputs["Selection"]})
    
    set_spline_type = nw.new_node(Nodes.CurveSplineType,
        input_kwargs={'Curve': mesh_to_curve},
        attrs={'spline_type': 'BEZIER'})
    
    set_handle_type = nw.new_node(Nodes.SetHandleType,
                                  input_kwargs={'Curve': set_spline_type})
    
    position = nw.new_node(Nodes.InputPosition)
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': position, 'Scale': 1.0})
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: noise_texture.outputs["Color"], 'Scale': 0.02},
        attrs={'operation': 'SCALE'})
    
    set_handle_positions = nw.new_node(Nodes.SetHandlePositions,
                                       input_kwargs={'Curve': set_handle_type, 'Offset': scale.outputs["Vector"]})
    
    switch = nw.new_node(Nodes.Switch,
        input_kwargs={1: True, 14: mesh_to_curve, 15: set_handle_positions})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Reverse depth"], 1: group_input.outputs["Scaling"]},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply, 1: 0.1},
        attrs={'operation': 'MULTIPLY'})
    
    power = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_1, 1: group_input.outputs["Exponent"]},
        attrs={'operation': 'POWER'})
    
    maximum = nw.new_node(Nodes.Math,
        input_kwargs={0: power, 1: group_input.outputs["Min radius"]},
        attrs={'operation': 'MAXIMUM'})
    
    minimum = nw.new_node(Nodes.Math,
        input_kwargs={0: maximum, 1: group_input.outputs["Max radius"]},
        attrs={'operation': 'MINIMUM'})
    
    set_curve_radius = nw.new_node(Nodes.SetCurveRadius,
        input_kwargs={'Curve': switch.outputs[6], 'Radius': minimum})
    
    curve_circle = nw.new_node(Nodes.CurveCircle,
        input_kwargs={'Resolution': group_input.outputs["Profile res"]})
    
    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
        input_kwargs={'Curve': set_curve_radius, 'Profile Curve': curve_circle.outputs["Curve"], 'Fill Caps': True})
    
    set_shade_smooth = nw.new_node(Nodes.SetShadeSmooth,
        input_kwargs={'Geometry': curve_to_mesh, 'Shade Smooth': False})
    
    merge_by_distance = nw.new_node(Nodes.MergeByDistance,
        input_kwargs={'Geometry': set_shade_smooth, 'Distance': group_input.outputs["Merge dist"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': merge_by_distance})
    

@node_utils.to_material('BarkMat2', singleton=False)
def bark_shader_2(nw):
  attribute = nw.new_node(Nodes.Attribute,
      attrs={'attribute_name': 'offset_barkgeo2'})
  
  reroute = nw.new_node(Nodes.Reroute,
      input_kwargs={'Input': attribute.outputs["Color"]})
  
  math = nw.new_node(Nodes.Math,
      input_kwargs={0: reroute, 1: 0.0})
  
  colorramp_1 = nw.new_node(Nodes.ColorRamp,
      input_kwargs={'Fac': math})
  for i in range(2):
    colorramp_1.color_ramp.elements.new(0)
  colorramp_1.color_ramp.elements[0].position = 0.0
  # colorramp_1.color_ramp.elements[0].color = (0.0025, 0.0019, 0.0017, 1.0)
  colorramp_1.color_ramp.elements[0].color = (0.1004, 0.049, 0.0344, 1.0)
  colorramp_1.color_ramp.elements[1].position = 0.163
  colorramp_1.color_ramp.elements[1].color = (0.1004, 0.049, 0.0344, 1.0)
  colorramp_1.color_ramp.elements[2].position = 0.4529
  colorramp_1.color_ramp.elements[2].color = (0.1094, 0.0656, 0.054, 1.0)
  colorramp_1.color_ramp.elements[3].position = 0.6268
  colorramp_1.color_ramp.elements[3].color = (0.0712, 0.0477, 0.0477, 1.0)
  
  math_1 = nw.new_node(Nodes.Math,
      input_kwargs={0: 1.0, 1: reroute},
      attrs={'operation': 'SUBTRACT'})
  
  principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
      input_kwargs={'Base Color': colorramp_1.outputs["Color"], 'Roughness': math_1},
      attrs={'subsurface_method': 'BURLEY'})
  
  material_output = nw.new_node(Nodes.MaterialOutput,
      input_kwargs={'Surface': principled_bsdf})


@node_utils.to_material('BarkMat1', singleton=False)
def bark_shader_1(nw):
  
  texture_coordinate = nw.new_node(Nodes.TextureCoord)
  
  mapping = nw.new_node(Nodes.Mapping,
      input_kwargs={'Vector': texture_coordinate.outputs["Object"]})
  
  noise_texture = nw.new_node(Nodes.NoiseTexture,
      input_kwargs={'Vector': mapping, 'Detail': 16.0, 'Roughness': 0.62})
  
  attribute = nw.new_node(Nodes.Attribute,
      attrs={'attribute_name': 'offset_barkgeo1'})
  
  mix = nw.new_node(Nodes.MixRGB,
      input_kwargs={'Color1': noise_texture.outputs["Fac"], 'Color2': attribute.outputs["Color"]},
      attrs={'blend_type': 'MULTIPLY'})
  
  colorramp = nw.new_node(Nodes.ColorRamp,
      input_kwargs={'Fac': mix})
  colorramp.color_ramp.elements.new(1)
  colorramp.color_ramp.elements[0].position = 0.0
  colorramp.color_ramp.elements[0].color = (0.0171, 0.005, 0.0, 1.0)
  colorramp.color_ramp.elements[1].position = 0.4636
  colorramp.color_ramp.elements[1].color = (0.1132, 0.0653, 0.0471, 1.0)
  colorramp.color_ramp.elements[2].position = 1.0
  colorramp.color_ramp.elements[2].color = (0.2243, 0.1341, 0.1001, 1.0)
  
  colorramp_2 = nw.new_node(Nodes.ColorRamp,
      input_kwargs={'Fac': noise_texture.outputs["Fac"]})
  colorramp_2.color_ramp.elements[0].position = 0.0
  colorramp_2.color_ramp.elements[0].color = (0.5173, 0.5173, 0.5173, 1.0)
  colorramp_2.color_ramp.elements[1].position = 1.0
  colorramp_2.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
  
  principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
      input_kwargs={'Base Color': colorramp.outputs["Color"], 'Roughness': colorramp_2.outputs["Color"]},
      attrs={'subsurface_method': 'BURLEY'})
  
  material_output = nw.new_node(Nodes.MaterialOutput,
      input_kwargs={'Surface': principled_bsdf})


@node_utils.to_nodegroup('BarkGeo2', singleton=False)
def bark_geo_2(nw):
    
    group_input = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
    
    position = nw.new_node(Nodes.InputPosition)
    
    vector = nw.new_node(Nodes.Vector)
    vector.vector = (0.1, 0.1, 0.1)
    
    vector_math_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: position, 1: vector},
        attrs={'operation': 'MULTIPLY'})
    
    value_2 = nw.new_node(Nodes.Value)
    value_2.outputs[0].default_value = 0.38
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 5.0
    
    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = 2.0
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': vector_math_1.outputs["Vector"], 'Scale': value, 'Detail': value_1})
    
    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': value_2, 'Color1': noise_texture.outputs["Color"], 'Color2': (0.0, 0.0, 0.0, 1.0)})
    
    vector_math_2 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: vector_math_1.outputs["Vector"], 1: mix})
    
    value_4 = nw.new_node(Nodes.Value)
    value_4.outputs[0].default_value = 0.0
    
    value_3 = nw.new_node(Nodes.Value)
    value_3.outputs[0].default_value = 20.0
    
    voronoi_texture = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': vector_math_2.outputs["Vector"], 'W': value_4, 'Scale': value_3},
        attrs={'voronoi_dimensions': '4D', 'feature': 'F2'})
    
    math_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: voronoi_texture.outputs["Distance"], 1: voronoi_texture.outputs["Distance"]},
        attrs={'operation': 'MULTIPLY'})
    
    voronoi_texture_1 = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': vector_math_2.outputs["Vector"], 'W': value_4, 'Scale': value_3},
        attrs={'voronoi_dimensions': '4D'})
    
    math_4 = nw.new_node(Nodes.Math,
        input_kwargs={0: voronoi_texture_1.outputs["Distance"], 1: voronoi_texture_1.outputs["Distance"]},
        attrs={'operation': 'MULTIPLY'})
    
    math_5 = nw.new_node(Nodes.Math,
        input_kwargs={0: math_3, 1: math_4},
        attrs={'operation': 'SUBTRACT'})
    
    value_5 = nw.new_node(Nodes.Value)
    value_5.outputs[0].default_value = 0.6
    
    math_7 = nw.new_node(Nodes.Math,
        input_kwargs={0: math_5, 1: value_5},
        attrs={'operation': 'MINIMUM'})
    
    math_6 = nw.new_node(Nodes.Math,
        input_kwargs={0: math_5, 1: value_5},
        attrs={'operation': 'MAXIMUM'})
    
    value_6 = nw.new_node(Nodes.Value)
    value_6.outputs[0].default_value = 0.1
    
    math_8 = nw.new_node(Nodes.Math,
        input_kwargs={0: math_6, 1: value_6},
        attrs={'operation': 'MULTIPLY'})
    
    math_9 = nw.new_node(Nodes.Math,
        input_kwargs={0: math_7, 1: math_8},
        attrs={'operation': 'SUBTRACT'})
    
    normal = nw.new_node(Nodes.InputNormal)
    
    vector_math_3 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: math_9, 1: normal},
        attrs={'operation': 'MULTIPLY'})
    
    face_area = nw.new_node('GeometryNodeInputMeshFaceArea')
    
    math_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: face_area},
        attrs={'operation': 'SQRT'})
    
    value_7 = nw.new_node(Nodes.Value)
    value_7.outputs[0].default_value = 2.0
    
    math = nw.new_node(Nodes.Math,
        input_kwargs={0: math_1, 1: value_7},
        attrs={'operation': 'MULTIPLY'})
    
    vector_math_4 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: vector_math_3.outputs["Vector"], 1: math},
        attrs={'operation': 'MULTIPLY'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Offset': vector_math_4.outputs["Vector"]})
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': set_position, 1: math_7},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': capture_attribute.outputs["Geometry"], 'offset_barkgeo2': capture_attribute.outputs["Attribute"]})


@node_utils.to_nodegroup('BarkGeo1', singleton=False)
def bark_geo_1(nw):
    
    group_input = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
    
    position = nw.new_node(Nodes.InputPosition)
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.2
    
    vector_math = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: position, 1: value},
        attrs={'operation': 'MULTIPLY'})
    
    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = 10.0
    
    value_2 = nw.new_node(Nodes.Value)
    value_2.outputs[0].default_value = 15.0
    
    wave_texture = nw.new_node(Nodes.WaveTexture,
        input_kwargs={'Vector': vector_math.outputs["Vector"], 'Scale': value_1, 'Distortion': value_2})
    
    normal = nw.new_node(Nodes.InputNormal)
    
    vector_math_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: wave_texture.outputs["Color"], 1: normal},
        attrs={'operation': 'MULTIPLY'})
    
    face_area = nw.new_node('GeometryNodeInputMeshFaceArea')
    
    math_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: face_area},
        attrs={'operation': 'SQRT'})
    
    value_3 = nw.new_node(Nodes.Value)
    value_3.outputs[0].default_value = 1.0
    
    math = nw.new_node(Nodes.Math,
        input_kwargs={0: math_1, 1: value_3},
        attrs={'operation': 'MULTIPLY'})
    
    vector_math_2 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: vector_math_1.outputs["Vector"], 1: math},
        attrs={'operation': 'MULTIPLY'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Offset': vector_math_2.outputs["Vector"]})
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': set_position, 1: wave_texture.outputs["Color"]},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': capture_attribute.outputs["Geometry"], 'offset_barkgeo1': capture_attribute.outputs["Attribute"]})


'''
def create_berry(sphere):
  # Create a sphere
  phyllotaxis_distribute('berry', sphere,
                         min_radius_pct=0, max_radius=1,
                         sin_max=2.5, sin_clamp_max=.8,
                         z_max=.8, z_clamp=.7)
'''


def sample_points_and_normals(obj, max_density=3,
                              surface_dist=1, max_points=10000):
  # Need to instantiate point distribute
  m = add_node_modifier(obj)
  ng = m.node_group
  inp = ng.nodes.get('Group Input')
  out = ng.nodes.get('Group Output')
  dist = ng.nodes.new(type='GeometryNodeDistributePointsOnFaces')
  pos = ng.nodes.new('GeometryNodeInputPosition')
  scale_factor = ng.nodes.new('ShaderNodeValue')
  mult_normal = ng.nodes.new('ShaderNodeVectorMath')
  add_pos = ng.nodes.new('ShaderNodeVectorMath')
  set_pos = ng.nodes.new('GeometryNodeSetPosition')
  to_vtx = ng.nodes.new('GeometryNodePointsToVertices')

  new_link(ng, inp, 'Geometry', dist, 'Mesh')
  new_link(ng, dist, 'Normal', mult_normal, 0)
  new_link(ng, scale_factor, 0, mult_normal, 1)
  new_link(ng, pos, 0, add_pos, 0)
  new_link(ng, mult_normal, 0, add_pos, 1)
  new_link(ng, dist, 'Points', set_pos, 'Geometry')
  new_link(ng, add_pos, 0, set_pos, 'Position')
  new_link(ng, set_pos, 'Geometry', to_vtx, 'Points')
  new_link(ng, to_vtx, 'Mesh', out, 'Geometry')

  mult_normal.operation = 'MULTIPLY'
  scale_factor.outputs[0].default_value = surface_dist
  dist.distribute_method = 'POISSON'
  dist.inputs.get('Density Max').default_value = max_density

  # Get point coordinates
  dgraph = C.evaluated_depsgraph_get()
  obj_eval = obj.evaluated_get(dgraph)
  vtx = mesh.vtx2cds(obj_eval.data.vertices, obj_eval.matrix_world)

  # Get normals
  scale_factor.outputs[0].default_value = 1
  for l in ng.links:
    if l.from_node == pos:
      ng.links.remove(l)

  dgraph = C.evaluated_depsgraph_get()
  obj_eval = obj.evaluated_get(dgraph)
  normals = mesh.vtx2cds(obj_eval.data.vertices, np.eye(4))

  obj.modifiers.remove(obj.modifiers[-1])
  D.node_groups.remove(ng)

  idxs = mesh.subsample_vertices(vtx, max_num=max_points)
  return vtx[idxs], normals[idxs]

