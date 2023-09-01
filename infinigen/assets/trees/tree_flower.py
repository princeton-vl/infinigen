# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Yiming Zuo - modifications
# - Alexander Raistrick - authored original flower.py


# Code generated using version v2.0.1 of the node_transpiler
import bpy
import mathutils
from numpy.random import uniform, normal
import numpy as np

from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core import surface

from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil, color
from infinigen.core.util.math import FixedSeed, dict_lerp
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

@node_utils.to_nodegroup('nodegroup_polar_to_cart_old', singleton=True)
def nodegroup_polar_to_cart_old(nw):
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'Addend', (0.0, 0.0, 0.0)),
        ('NodeSocketFloat', 'Value', 0.5),
        ('NodeSocketVector', 'Vector', (0.0, 0.0, 0.0))])
    
    cosine = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Value"]},
        attrs={'operation': 'COSINE'})
    
    sine = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Value"]},
        attrs={'operation': 'SINE'})
    
    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Y': cosine, 'Z': sine})
    
    multiply_add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Vector"], 1: combine_xyz_4, 2: group_input.outputs["Addend"]},
        attrs={'operation': 'MULTIPLY_ADD'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Vector': multiply_add.outputs["Vector"]})

@node_utils.to_nodegroup('nodegroup_follow_curve', singleton=True)
def nodegroup_follow_curve(nw):
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
        ('NodeSocketGeometry', 'Curve', None),
        ('NodeSocketFloat', 'Curve Min', 0.5),
        ('NodeSocketFloat', 'Curve Max', 1.0)])
    
    position = nw.new_node(Nodes.InputPosition)
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 1: position},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': capture_attribute.outputs["Attribute"]})
    
    attribute_statistic = nw.new_node(Nodes.AttributeStatistic,
        input_kwargs={'Geometry': capture_attribute.outputs["Geometry"], 2: separate_xyz.outputs["Z"]})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_xyz.outputs["Z"], 1: attribute_statistic.outputs["Min"], 2: attribute_statistic.outputs["Max"], 3: group_input.outputs["Curve Min"], 4: group_input.outputs["Curve Max"]})
    
    curve_length = nw.new_node(Nodes.CurveLength,
        input_kwargs={'Curve': group_input.outputs["Curve"]})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range.outputs["Result"], 1: curve_length},
        attrs={'operation': 'MULTIPLY'})
    
    sample_curve = nw.new_node(Nodes.SampleCurve,
        input_kwargs={'Curve': group_input.outputs["Curve"], 'Length': multiply},
        attrs={'mode': 'LENGTH'})
    
    cross_product = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: sample_curve.outputs["Tangent"], 1: sample_curve.outputs["Normal"]},
        attrs={'operation': 'CROSS_PRODUCT'})
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: cross_product.outputs["Vector"], 'Scale': separate_xyz.outputs["X"]},
        attrs={'operation': 'SCALE'})
    
    scale_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: sample_curve.outputs["Normal"], 'Scale': separate_xyz.outputs["Y"]},
        attrs={'operation': 'SCALE'})
    
    add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: scale.outputs["Vector"], 1: scale_1.outputs["Vector"]})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': capture_attribute.outputs["Geometry"], 'Position': sample_curve.outputs["Position"], 'Offset': add.outputs["Vector"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_position})

@node_utils.to_nodegroup('nodegroup_norm_index', singleton=True)
def nodegroup_norm_index(nw):
    index = nw.new_node(Nodes.Index)
    
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketInt', 'Count', 0)])
    
    divide = nw.new_node(Nodes.Math,
        input_kwargs={0: index, 1: group_input.outputs["Count"]},
        attrs={'operation': 'DIVIDE'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'T': divide})

@node_utils.to_nodegroup('nodegroup_flower_petal', singleton=True)
def nodegroup_flower_petal(nw):
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
        ('NodeSocketFloat', 'Length', 0.2),
        ('NodeSocketFloat', 'Point', 1.0),
        ('NodeSocketFloat', 'Point height', 0.5),
        ('NodeSocketFloat', 'Bevel', 6.8),
        ('NodeSocketFloat', 'Base width', 0.2),
        ('NodeSocketFloat', 'Upper width', 0.3),
        ('NodeSocketInt', 'Resolution H', 8),
        ('NodeSocketInt', 'Resolution V', 4),
        ('NodeSocketFloat', 'Wrinkle', 0.1),
        ('NodeSocketFloat', 'Curl', 0.0)])
    
    multiply_add = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Resolution H"], 1: 2.0, 2: 1.0},
        attrs={'operation': 'MULTIPLY_ADD'})
    
    grid = nw.new_node(Nodes.MeshGrid,
        input_kwargs={'Vertices X': group_input.outputs["Resolution V"], 'Vertices Y': multiply_add})
    
    position = nw.new_node(Nodes.InputPosition)
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': grid, 1: position},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': capture_attribute.outputs["Attribute"]})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: 0.05},
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': multiply, 'Y': separate_xyz.outputs["Y"]})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': combine_xyz, 'Scale': 7.9, 'Detail': 0.0, 'Distortion': 0.2},
        attrs={'noise_dimensions': '2D'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: noise_texture.outputs["Fac"], 1: -0.5})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: add, 1: group_input.outputs["Wrinkle"]},
        attrs={'operation': 'MULTIPLY'})
    
    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': capture_attribute.outputs["Attribute"]})
    
    add_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["X"]})
    
    absolute = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["Y"]},
        attrs={'operation': 'ABSOLUTE'})
    
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: absolute, 1: 2.0},
        attrs={'operation': 'MULTIPLY'})
    
    power = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_2, 1: group_input.outputs["Bevel"]},
        attrs={'operation': 'POWER'})
    
    multiply_add_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: power, 1: -1.0, 2: 1.0},
        attrs={'operation': 'MULTIPLY_ADD'})
    
    multiply_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: add_1, 1: multiply_add_1},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_add_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_3, 1: group_input.outputs["Upper width"], 2: group_input.outputs["Base width"]},
        attrs={'operation': 'MULTIPLY_ADD'})
    
    multiply_4 = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["Y"], 1: multiply_add_2},
        attrs={'operation': 'MULTIPLY'})
    
    power_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: absolute, 1: group_input.outputs["Point"]},
        attrs={'operation': 'POWER'})
    
    multiply_add_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: power_1, 1: -1.0, 2: 1.0},
        attrs={'operation': 'MULTIPLY_ADD'})
    
    multiply_5 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_add_3, 1: group_input.outputs["Point height"]},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_add_4 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Point height"], 1: -1.0, 2: 1.0},
        attrs={'operation': 'MULTIPLY_ADD'})
    
    add_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_5, 1: multiply_add_4})
    
    multiply_6 = nw.new_node(Nodes.Math,
        input_kwargs={0: add_2, 1: multiply_add_1},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_7 = nw.new_node(Nodes.Math,
        input_kwargs={0: add_1, 1: multiply_6},
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': multiply_1, 'Y': multiply_4, 'Z': multiply_7})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': capture_attribute.outputs["Geometry"], 'Position': combine_xyz_1})
    
    multiply_8 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Length"]},
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Y': multiply_8})
    
    reroute = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': group_input.outputs["Curl"]})
    
    group_1 = nw.new_node(nodegroup_polar_to_cart_old().name,
        input_kwargs={'Addend': combine_xyz_3, 'Value': reroute, 'Vector': multiply_8})
    
    quadratic_bezier = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 8, 'Start': (0.0, 0.0, 0.0), 'Middle': combine_xyz_3, 'End': group_1})
    
    group = nw.new_node(nodegroup_follow_curve().name,
        input_kwargs={'Geometry': set_position, 'Curve': quadratic_bezier, 'Curve Min': 0.0})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': tag_nodegroup(nw, group, 'petal')})

@node_utils.to_nodegroup('nodegroup_phyllo_points', singleton=True)
def nodegroup_phyllo_points(nw):
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketInt', 'Count', 50),
        ('NodeSocketFloat', 'Min Radius', 0.0),
        ('NodeSocketFloat', 'Max Radius', 2.0),
        ('NodeSocketFloat', 'Radius exp', 0.5),
        ('NodeSocketFloat', 'Min angle', -0.5236),
        ('NodeSocketFloat', 'Max angle', 0.7854),
        ('NodeSocketFloat', 'Min z', 0.0),
        ('NodeSocketFloat', 'Max z', 1.0),
        ('NodeSocketFloat', 'Clamp z', 1.0),
        ('NodeSocketFloat', 'Yaw offset', -1.5708)])
    
    mesh_line = nw.new_node(Nodes.MeshLine,
        input_kwargs={'Count': group_input.outputs["Count"]})
    
    mesh_to_points = nw.new_node(Nodes.MeshToPoints,
        input_kwargs={'Mesh': mesh_line})
    
    position = nw.new_node(Nodes.InputPosition)
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': mesh_to_points, 1: position},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    index = nw.new_node(Nodes.Index)
    
    cosine = nw.new_node(Nodes.Math,
        input_kwargs={0: index},
        attrs={'operation': 'COSINE'})
    
    sine = nw.new_node(Nodes.Math,
        input_kwargs={0: index},
        attrs={'operation': 'SINE'})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': cosine, 'Y': sine})
    
    divide = nw.new_node(Nodes.Math,
        input_kwargs={0: index, 1: group_input.outputs["Count"]},
        attrs={'operation': 'DIVIDE'})
    
    power = nw.new_node(Nodes.Math,
        input_kwargs={0: divide, 1: group_input.outputs["Radius exp"]},
        attrs={'operation': 'POWER'})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': power, 3: group_input.outputs["Min Radius"], 4: group_input.outputs["Max Radius"]})
    
    multiply = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: combine_xyz, 1: map_range.outputs["Result"]},
        attrs={'operation': 'MULTIPLY'})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': multiply.outputs["Vector"]})
    
    map_range_2 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': divide, 2: group_input.outputs["Clamp z"], 3: group_input.outputs["Min z"], 4: group_input.outputs["Max z"]})
    
    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': separate_xyz.outputs["X"], 'Y': separate_xyz.outputs["Y"], 'Z': map_range_2.outputs["Result"]})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': capture_attribute.outputs["Geometry"], 'Position': combine_xyz_1})
    
    map_range_3 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': divide, 3: group_input.outputs["Min angle"], 4: group_input.outputs["Max angle"]})
    
    random_value = nw.new_node(Nodes.RandomValue,
        input_kwargs={2: -0.1, 3: 0.1})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: index, 1: group_input.outputs["Yaw offset"]})
    
    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': map_range_3.outputs["Result"], 'Y': random_value.outputs[1], 'Z': add})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Points': set_position, 'Rotation': combine_xyz_2})

@node_utils.to_nodegroup('nodegroup_plant_seed', singleton=True)
def nodegroup_plant_seed(nw):
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'Dimensions', (0.0, 0.0, 0.0)),
        ('NodeSocketIntUnsigned', 'U', 4),
        ('NodeSocketInt', 'V', 8)])
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group_input.outputs["Dimensions"]})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': separate_xyz.outputs["X"]})
    
    multiply_add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: combine_xyz, 1: (0.5, 0.5, 0.5)},
        attrs={'operation': 'MULTIPLY_ADD'})
    
    quadratic_bezier_1 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': group_input.outputs["U"], 'Start': (0.0, 0.0, 0.0), 'Middle': multiply_add.outputs["Vector"], 'End': combine_xyz})
    
    group = nw.new_node(nodegroup_norm_index().name,
        input_kwargs={'Count': group_input.outputs["U"]})
    
    float_curve = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': group})
    node_utils.assign_curve(float_curve.mapping.curves[0], [(0.0, 0.0), (0.3159, 0.4469), (1.0, 0.0156)])
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': float_curve, 4: 3.0})
    
    set_curve_radius = nw.new_node(Nodes.SetCurveRadius,
        input_kwargs={'Curve': quadratic_bezier_1, 'Radius': map_range.outputs["Result"]})
    
    curve_circle = nw.new_node(Nodes.CurveCircle,
        input_kwargs={'Resolution': group_input.outputs["V"], 'Radius': separate_xyz.outputs["Y"]})
    
    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
        input_kwargs={'Curve': set_curve_radius, 'Profile Curve': curve_circle.outputs["Curve"], 'Fill Caps': True})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Mesh': tag_nodegroup(nw, curve_to_mesh, 'seed')})

def shader_flower_center(nw):
    ambient_occlusion = nw.new_node(Nodes.AmbientOcclusion)
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': ambient_occlusion.outputs["Color"]})
    colorramp.color_ramp.elements.new(1)
    colorramp.color_ramp.elements[0].position = 0.4841
    colorramp.color_ramp.elements[0].color = (0.0127, 0.0075, 0.0026, 1.0)
    colorramp.color_ramp.elements[1].position = 0.8591
    colorramp.color_ramp.elements[1].color = (0.0848, 0.0066, 0.0007, 1.0)
    colorramp.color_ramp.elements[2].position = 1.0
    colorramp.color_ramp.elements[2].color = (1.0, 0.6228, 0.1069, 1.0)
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': colorramp.outputs["Color"]})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

def shader_petal(nw, petal_color_name):

    translucent_color_change = uniform(0.1, 0.6)
    specular = normal(0.6, 0.1)
    roughness = normal(0.4, 0.05)
    translucent_amt = normal(0.3, 0.05)

    petal_color = nw.new_node(Nodes.RGB)
    petal_color.outputs[0].default_value = color.color_category(petal_color_name)

    translucent_color = nw.new_node(Nodes.MixRGB, [translucent_color_change, petal_color, color.color_category(petal_color_name)])

    translucent_bsdf = nw.new_node(Nodes.TranslucentBSDF,
        input_kwargs={'Color': translucent_color})
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': petal_color, 'Specular': specular, 'Roughness': roughness })
    
    mix_shader = nw.new_node(Nodes.MixShader,
        input_kwargs={'Fac': translucent_amt, 1: principled_bsdf, 2: translucent_bsdf})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': mix_shader})

def geo_flower(nw, petal_material, center_material):
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
        ('NodeSocketFloat', 'Center Rad', 0.0),
        ('NodeSocketVector', 'Petal Dims', (0.0, 0.0, 0.0)),
        ('NodeSocketFloat', 'Seed Size', 0.0),
        ('NodeSocketFloat', 'Min Petal Angle', 0.1),
        ('NodeSocketFloat', 'Max Petal Angle', 1.36),
        ('NodeSocketFloat', 'Wrinkle', 0.01),
        ('NodeSocketFloat', 'Curl', 13.89)])
    
    uv_sphere = nw.new_node(Nodes.MeshUVSphere,
        input_kwargs={'Segments': 8, 'Rings': 8, 'Radius': group_input.outputs["Center Rad"]})
    
    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': uv_sphere, 'Scale': (1.0, 1.0, 0.05)})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Seed Size"], 1: 1.5},
        attrs={'operation': 'MULTIPLY'})
    
    distribute_points_on_faces = nw.new_node(Nodes.DistributePointsOnFaces,
        input_kwargs={'Mesh': transform, 'Distance Min': multiply, 'Density Max': 50000.0},
        attrs={'distribute_method': 'POISSON'})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Seed Size"], 1: 10.0},
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': multiply_1, 'Y': group_input.outputs["Seed Size"]})
    
    group_3 = nw.new_node(nodegroup_plant_seed().name,
        input_kwargs={'Dimensions': combine_xyz, 'U': 6, 'V': 6})
    
    musgrave_texture = nw.new_node(Nodes.MusgraveTexture,
        input_kwargs={'W': 13.8, 'Scale': 2.41},
        attrs={'musgrave_dimensions': '4D'})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': musgrave_texture, 3: 0.34, 4: 1.21})
    
    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': map_range.outputs["Result"], 'Y': 1.0, 'Z': 1.0})
    
    instance_on_points_1 = nw.new_node(Nodes.InstanceOnPoints,
        input_kwargs={'Points': distribute_points_on_faces.outputs["Points"], 'Instance': group_3, 'Rotation': (0.0, -1.5708, 0.0541), 'Scale': combine_xyz_1})
    
    realize_instances = nw.new_node(Nodes.RealizeInstances,
        input_kwargs={'Geometry': instance_on_points_1})
    
    join_geometry_1 = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [realize_instances, transform]})
    
    set_material_1 = nw.new_node(Nodes.SetMaterial,
        input_kwargs={'Geometry': join_geometry_1, 'Material': center_material})
    
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Center Rad"], 1: 6.2832},
        attrs={'operation': 'MULTIPLY'})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group_input.outputs["Petal Dims"]})
    
    divide = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_2, 1: separate_xyz.outputs["Y"]},
        attrs={'operation': 'DIVIDE'})
    
    multiply_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: divide, 1: 1.2},
        attrs={'operation': 'MULTIPLY'})
    
    reroute_3 = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': group_input.outputs["Center Rad"]})
    
    reroute_1 = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': group_input.outputs["Min Petal Angle"]})
    
    reroute = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': group_input.outputs["Max Petal Angle"]})
    
    group_1 = nw.new_node(nodegroup_phyllo_points().name,
        input_kwargs={'Count': multiply_3, 'Min Radius': reroute_3, 'Max Radius': reroute_3, 'Radius exp': 0.0, 'Min angle': reroute_1, 'Max angle': reroute, 'Max z': 0.0})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: separate_xyz.outputs["Y"]},
        attrs={'operation': 'SUBTRACT', 'use_clamp': True})
    
    reroute_2 = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': group_input.outputs["Wrinkle"]})
    
    reroute_4 = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': group_input.outputs["Curl"]})
    
    group = nw.new_node(nodegroup_flower_petal().name,
        input_kwargs={'Length': separate_xyz.outputs["X"], 'Point': 0.56, 'Point height': -0.1, 'Bevel': 1.83, 'Base width': separate_xyz.outputs["Y"], 'Upper width': subtract, 
        'Resolution H': 8, 'Resolution V': 16, 'Wrinkle': reroute_2, 'Curl': reroute_4})
    
    instance_on_points = nw.new_node(Nodes.InstanceOnPoints,
        input_kwargs={'Points': group_1.outputs["Points"], 'Instance': group, 'Rotation': group_1.outputs["Rotation"]})
    
    realize_instances_1 = nw.new_node(Nodes.RealizeInstances,
        input_kwargs={'Geometry': instance_on_points})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Scale': 3.73, 'Detail': 5.41, 'Distortion': -1.0})
    
    subtract_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: noise_texture.outputs["Color"], 1: (0.5, 0.5, 0.5)},
        attrs={'operation': 'SUBTRACT'})
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.025
    
    multiply_4 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: subtract_1.outputs["Vector"], 1: value},
        attrs={'operation': 'MULTIPLY'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': realize_instances_1, 'Offset': multiply_4.outputs["Vector"]})
    
    set_material = nw.new_node(Nodes.SetMaterial,
        input_kwargs={'Geometry': set_position, 'Material': petal_material})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [set_material_1, set_material]})
    
    set_shade_smooth = nw.new_node(Nodes.SetShadeSmooth,
        input_kwargs={'Geometry': join_geometry, 'Shade Smooth': False})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_shade_smooth})

class TreeFlowerFactory(AssetFactory):

    def __init__(self, factory_seed, rad=uniform(0.15, 0.25), diversity_fac=0.25):
        super(TreeFlowerFactory, self).__init__(factory_seed=factory_seed)

        self.rad = rad
        self.diversity_fac = diversity_fac

        self.petal_color = np.random.choice(['pink', 'white', 'red', 'yellowish'], p=[0.4, 0.2, 0.2, 0.2])

        with FixedSeed(factory_seed):
            self.petal_material = surface.shaderfunc_to_material(shader_petal, self.petal_color)
            self.center_material = surface.shaderfunc_to_material(shader_flower_center)
            self.species_params = self.get_flower_params(self.rad)

    @staticmethod
    def get_flower_params(overall_rad=0.05):
        pct_inner = uniform(0.05, 0.4) 
        base_width = 2 * np.pi * overall_rad * pct_inner / normal(20, 5)
        top_width = overall_rad * np.clip(normal(0.7, 0.3), base_width * 1.2, 100)

        min_angle, max_angle = np.deg2rad(np.sort(uniform(-20, 100, 2)))

        return {
            'Center Rad': overall_rad * pct_inner,
            'Petal Dims': np.array([overall_rad * (1 - pct_inner), base_width, top_width], dtype=np.float32),
            'Seed Size': uniform(0.005, 0.01),
            'Min Petal Angle': min_angle,
            'Max Petal Angle': max_angle,
            'Wrinkle': uniform(0.003, 0.02),
            'Curl': np.deg2rad(normal(30, 50))
        }

    def create_asset(self, **kwargs) -> bpy.types.Object:
        
        vert = butil.spawn_vert('flower')
        mod = surface.add_geomod(vert, geo_flower, 
            input_kwargs={'petal_material': self.petal_material, 'center_material': self.center_material})

        inst_params = self.get_flower_params(self.rad * normal(1, 0.05))
        params = dict_lerp(self.species_params, inst_params, 0.25)
        surface.set_geomod_inputs(mod, params)

        butil.apply_modifiers(vert, mod)

        vert.rotation_euler.z = uniform(0, 360)
        tag_object(vert, 'flower')
        return vert