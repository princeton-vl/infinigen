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

from .shader_utils import nodegroup_add_noise, nodegroup_color_noise

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

@node_utils.to_nodegroup('nodegroup_add_hair', singleton=False, type='GeometryNodeTree')
def nodegroup_add_hair(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Mesh', None),
            ('NodeSocketGeometry', 'Hair', None),
            ('NodeSocketFloat', 'Density', 100.0),
            ('NodeSocketVector', 'rot mean', (1.18, 0.0, 0.0)),
            ('NodeSocketFloat', 'scale mean', 0.05)])
    
    distribute_points_on_faces = nw.new_node(Nodes.DistributePointsOnFaces,
        input_kwargs={'Mesh': group_input.outputs["Mesh"], 'Density': group_input.outputs["Density"]})
    
    randomrotationscale = nw.new_node(nodegroup_random_rotation_scale().name,
        input_kwargs={'random seed': -2.4, 'rot mean': group_input.outputs["rot mean"], 'scale mean': group_input.outputs["scale mean"]})
    
    instanceonpoints = nw.new_node(nodegroup_instance_on_points().name,
        input_kwargs={'rotation base': distribute_points_on_faces.outputs["Rotation"], 'rotation delta': randomrotationscale.outputs["Vector"], 'scale': randomrotationscale.outputs["Value"], 'Points': distribute_points_on_faces.outputs["Points"], 'Instance': group_input.outputs["Hair"]})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [instanceonpoints, group_input.outputs["Mesh"]]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Instances': join_geometry})

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


@node_utils.to_nodegroup('nodegroup_random_rotation_scale', singleton=False, type='GeometryNodeTree')
def nodegroup_random_rotation_scale(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'random seed', 0.0),
            ('NodeSocketFloat', 'noise scale', 10.0),
            ('NodeSocketVector', 'rot mean', (0.0, 0.0, 0.0)),
            ('NodeSocketFloat', 'rot std z', 1.0),
            ('NodeSocketFloat', 'scale mean', 0.35),
            ('NodeSocketFloat', 'scale std', 0.1)])
    
    position_3 = nw.new_node(Nodes.InputPosition)
    
    add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: position_3, 1: group_input.outputs["random seed"]})
    
    noise_texture_2 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': add.outputs["Vector"], 'Scale': group_input.outputs["noise scale"]})
    
    value_2 = nw.new_node(Nodes.Value)
    value_2.outputs[0].default_value = 0.5
    
    subtract = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: noise_texture_2.outputs["Color"], 1: value_2},
        attrs={'operation': 'SUBTRACT'})
    
    separate_xyz_2 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': subtract.outputs["Vector"]})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["X"], 1: group_input.outputs["rot std z"]},
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Z': multiply})
    
    add_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["rot mean"], 1: combine_xyz})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Y"], 1: group_input.outputs["scale std"]},
        attrs={'operation': 'MULTIPLY'})
    
    add_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_1, 1: group_input.outputs["scale mean"]},
        attrs={'use_clamp': True})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Vector': add_1.outputs["Vector"], 'Value': add_2})

@node_utils.to_nodegroup('nodegroup_instance_on_points', singleton=False, type='GeometryNodeTree')
def nodegroup_instance_on_points(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVectorEuler', 'rotation base', (0.0, 0.0, 0.0)),
            ('NodeSocketVectorEuler', 'rotation delta', (-1.5708, 0.0, 0.0)),
            ('NodeSocketVectorTranslation', 'translation', (0.0, -0.5, 0.0)),
            ('NodeSocketFloat', 'scale', 0.0),
            ('NodeSocketGeometry', 'Points', None),
            ('NodeSocketGeometry', 'Instance', None)])
    
    rotate_euler_1 = nw.new_node(Nodes.RotateEuler,
        input_kwargs={'Rotation': group_input.outputs["rotation base"], 'Rotate By': group_input.outputs["rotation delta"]},
        attrs={'space': 'LOCAL'})
    
    instance_on_points_1 = nw.new_node(Nodes.InstanceOnPoints,
        input_kwargs={'Points': group_input.outputs["Points"], 'Instance': group_input.outputs["Instance"], 'Rotation': rotate_euler_1, 'Scale': group_input.outputs["scale"]})
    
    translate_instances = nw.new_node(Nodes.TranslateInstances,
        input_kwargs={'Instances': instance_on_points_1, 'Translation': group_input.outputs["translation"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Instances': translate_instances})

def shader_dragonfly_body_shader(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    attribute_1 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'pos'})
    
    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': attribute_1.outputs["Vector"]})
    
    absolute = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["X"]},
        attrs={'operation': 'ABSOLUTE'})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["Z"], 1: 3.0},
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': absolute, 'Y': separate_xyz_1.outputs["Y"], 'Z': multiply})
    
    attribute_2 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'body seed'})
    
    musgrave_texture = nw.new_node(Nodes.MusgraveTexture,
        input_kwargs={'Vector': combine_xyz_1, 'W': attribute_2.outputs["Fac"], 'Scale': 0.5, 'Dimension': 1.0, 'Lacunarity': 1.0},
        attrs={'musgrave_dimensions': '4D'})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': musgrave_texture, 1: -0.26, 2: 0.06})
    
    attribute = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'spline parameter'})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': attribute.outputs["Fac"]})
    
    group = nw.new_node(nodegroup_add_noise().name,
        input_kwargs={'Vector': combine_xyz, 'Scale': 0.5, 'amount': (0.16, 0.26, 0.0), 'Noise Eval Position': combine_xyz_1})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group})
    
    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': separate_xyz.outputs["X"], 'Y': attribute_2.outputs["Fac"]})
    
    voronoi_texture = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': combine_xyz_2, 'Scale': 10.0},
        attrs={'voronoi_dimensions': '2D'})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': voronoi_texture.outputs["Distance"], 1: 0.14, 2: 0.82})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_1.outputs["Result"], 1: map_range.outputs["Result"]})
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': add})
    colorramp.color_ramp.elements[0].position = 0.7386
    colorramp.color_ramp.elements[0].color = (0.4397, 0.5841, 0.011, 1.0)
    colorramp.color_ramp.elements[1].position = 1.0
    colorramp.color_ramp.elements[1].color = (0.008, 0.0065, 0.0116, 1.0)
    
    group_1 = nw.new_node(nodegroup_color_noise().name,
        input_kwargs={'Color': colorramp.outputs["Color"], 'Value To Min': 0.4})
    
    principled_bsdf_1 = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': group_1, 'Metallic': 0.2182, 'Specular': 0.8318, 'Roughness': 0.1545})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf_1})


@node_utils.to_nodegroup('nodegroup_surface_bump', singleton=False, type='GeometryNodeTree')
def nodegroup_surface_bump(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketFloat', 'Displacement', 0.02),
            ('NodeSocketFloat', 'Scale', 50.0),
            ('NodeSocketFloat', 'seed', 0.0)])
    
    normal = nw.new_node(Nodes.InputNormal)
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'W': group_input.outputs["seed"], 'Scale': group_input.outputs["Scale"]},
        attrs={'noise_dimensions': '4D'})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: noise_texture.outputs["Fac"]},
        attrs={'operation': 'SUBTRACT'})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract, 1: group_input.outputs["Displacement"]},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: normal, 1: multiply},
        attrs={'operation': 'MULTIPLY'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Offset': multiply_1.outputs["Vector"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_position})

@node_utils.to_nodegroup('nodegroup_circle_cross_section', singleton=False, type='GeometryNodeTree')
def nodegroup_circle_cross_section(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'random seed', 0.0),
            ('NodeSocketFloat', 'noise scale', 0.5),
            ('NodeSocketFloat', 'noise amount', 0.0),
            ('NodeSocketInt', 'Resolution', 256),
            ('NodeSocketFloat', 'radius', 1.0),
            ('NodeSocketBool', 'symmetric noise', False)])
    
    curve_circle = nw.new_node(Nodes.CurveCircle,
        input_kwargs={'Resolution': group_input.outputs["Resolution"]})
    
    normal = nw.new_node(Nodes.InputNormal)
    
    position = nw.new_node(Nodes.InputPosition)
    
    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': position})
    
    absolute = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["Y"]},
        attrs={'operation': 'ABSOLUTE'})
    
    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': separate_xyz_1.outputs["X"], 'Y': absolute, 'Z': separate_xyz_1.outputs["Z"]})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': combine_xyz_1, 'W': group_input.outputs["random seed"], 'Scale': group_input.outputs["noise scale"]},
        attrs={'noise_dimensions': '4D'})
    
    subtract = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: noise_texture.outputs["Color"], 1: (0.5, 0.5, 0.5)},
        attrs={'operation': 'SUBTRACT'})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': subtract.outputs["Vector"]})
    
    absolute_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"]},
        attrs={'operation': 'ABSOLUTE'})
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: normal, 'Scale': absolute_1},
        attrs={'operation': 'SCALE'})
    
    scale_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: scale.outputs["Vector"], 'Scale': group_input.outputs["noise amount"]},
        attrs={'operation': 'SCALE'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': curve_circle.outputs["Curve"], 'Offset': scale_1.outputs["Vector"]})
    
    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': set_position, 'Scale': group_input.outputs["radius"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': transform})

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

@node_utils.to_nodegroup('nodegroup_shape_quadratic', singleton=False, type='GeometryNodeTree')
def nodegroup_shape_quadratic(nw: NodeWrangler, radius_control_points=[(0.0, 0.5), (1.0, 0.5)]):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Profile Curve', None),
            ('NodeSocketFloat', 'random seed tilt', 0.5),
            ('NodeSocketFloat', 'noise scale tilt', 0.5),
            ('NodeSocketFloat', 'noise amount tilt', 5.0),
            ('NodeSocketFloat', 'random seed pos', 0.0),
            ('NodeSocketFloat', 'noise scale pos', 0.0),
            ('NodeSocketFloat', 'noise amount pos', 0.0),
            ('NodeSocketIntUnsigned', 'Resolution', 256),
            ('NodeSocketVectorTranslation', 'Start', (0.0, 0.15, -1.5)),
            ('NodeSocketVectorTranslation', 'Middle', (0.0, 0.0, 0.0)),
            ('NodeSocketVectorTranslation', 'End', (0.0, 0.0, 1.5))])
    
    quadratic_bezier = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': group_input.outputs["Resolution"], 'Start': group_input.outputs["Start"], 'Middle': group_input.outputs["Middle"], 'End': group_input.outputs["End"]})
    
    spline_parameter_2 = nw.new_node(Nodes.SplineParameter)
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': quadratic_bezier, 2: spline_parameter_2.outputs["Factor"]})
    
    curve_tangent = nw.new_node(Nodes.CurveTangent)
    
    capture_attribute_1 = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': capture_attribute.outputs["Geometry"], 1: curve_tangent},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    position = nw.new_node(Nodes.InputPosition)
    
    add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: position, 1: group_input.outputs["random seed pos"]})
    
    noise_texture_3 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': add.outputs["Vector"], 'Scale': group_input.outputs["noise scale pos"]})
    
    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = 0.5
    
    subtract = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: noise_texture_3.outputs["Color"], 1: value_1},
        attrs={'operation': 'SUBTRACT'})
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: subtract.outputs["Vector"], 'Scale': spline_parameter_2.outputs["Factor"]},
        attrs={'operation': 'SCALE'})
    
    scale_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: scale.outputs["Vector"], 'Scale': group_input.outputs["noise amount pos"]},
        attrs={'operation': 'SCALE'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': capture_attribute_1.outputs["Geometry"], 'Offset': scale_1.outputs["Vector"]})
    
    spline_parameter = nw.new_node(Nodes.SplineParameter)
    
    add_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: spline_parameter.outputs["Factor"], 1: group_input.outputs["random seed tilt"]})
    
    noise_texture_1 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'W': add_1, 'Scale': group_input.outputs["noise scale tilt"]},
        attrs={'noise_dimensions': '1D'})
    
    subtract_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: noise_texture_1.outputs["Fac"]},
        attrs={'operation': 'SUBTRACT'})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract_1, 1: group_input.outputs["noise amount tilt"]},
        attrs={'operation': 'MULTIPLY'})
    
    set_curve_tilt = nw.new_node(Nodes.SetCurveTilt,
        input_kwargs={'Curve': set_position, 'Tilt': multiply})
    
    spline_parameter_1 = nw.new_node(Nodes.SplineParameter)
    
    float_curve = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': spline_parameter_1.outputs["Factor"]})
    node_utils.assign_curve(float_curve.mapping.curves[0], radius_control_points)
    
    set_curve_radius = nw.new_node(Nodes.SetCurveRadius,
        input_kwargs={'Curve': set_curve_tilt, 'Radius': float_curve})
    
    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
        input_kwargs={'Curve': set_curve_radius, 'Profile Curve': group_input.outputs["Profile Curve"], 'Fill Caps': True})
    
    curve_to_points = nw.new_node(Nodes.CurveToPoints,
        input_kwargs={'Curve': set_position},
        attrs={'mode': 'EVALUATED'})
    
    geometry_proximity = nw.new_node(Nodes.Proximity,
        input_kwargs={'Target': curve_to_points.outputs["Points"]},
        attrs={'target_element': 'POINTS'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Mesh': curve_to_mesh, 'spline parameter': capture_attribute.outputs[2], 'spline tangent': capture_attribute_1.outputs["Attribute"], 'radius to center': geometry_proximity.outputs["Distance"]})

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