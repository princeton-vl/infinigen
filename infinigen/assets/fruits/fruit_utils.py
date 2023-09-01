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
        
@node_utils.to_nodegroup('nodegroup_random_rotation_scale', singleton=False, type='GeometryNodeTree')
def nodegroup_random_rotation_scale(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'random seed', 0.0),
            ('NodeSocketFloat', 'noise scale', 10.0),
            ('NodeSocketVector', 'rot mean', (0.0, 0.0, 0.0)),
            ('NodeSocketFloat', 'rot std', 1.0),
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
        input_kwargs={0: separate_xyz_2.outputs["X"], 1: group_input.outputs["rot std"]},
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

@node_utils.to_nodegroup('nodegroup_surface_bump', singleton=False, type='GeometryNodeTree')
def nodegroup_surface_bump(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketFloat', 'Displacement', 0.02),
            ('NodeSocketFloat', 'Scale', 50.0)])
    
    normal = nw.new_node(Nodes.InputNormal)
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Scale': group_input.outputs["Scale"]})
    
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

@node_utils.to_nodegroup('nodegroup_point_on_mesh', singleton=False, type='GeometryNodeTree')
def nodegroup_point_on_mesh(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Mesh', None),
            ('NodeSocketFloat', 'spline parameter', 0.0),
            ('NodeSocketFloatDistance', 'Distance Min', 0.2),
            ('NodeSocketFloat', 'parameter max', 1.0),
            ('NodeSocketFloat', 'parameter min', 0.0),
            ('NodeSocketFloat', 'noise amount', 1.0),
            ('NodeSocketFloat', 'noise scale', 5.0)])
    
    distribute_points_on_faces = nw.new_node(Nodes.DistributePointsOnFaces,
        input_kwargs={'Mesh': group_input.outputs["Mesh"], 'Distance Min': group_input.outputs["Distance Min"], 'Density Max': 10000.0},
        attrs={'distribute_method': 'POISSON'})
    
    greater_than = nw.new_node(Nodes.Compare,
        input_kwargs={0: group_input.outputs["spline parameter"], 1: group_input.outputs["parameter min"]})
    
    less_than = nw.new_node(Nodes.Compare,
        input_kwargs={0: group_input.outputs["spline parameter"], 1: group_input.outputs["parameter max"]},
        attrs={'operation': 'LESS_THAN'})
    
    nand = nw.new_node(Nodes.BooleanMath,
        input_kwargs={0: greater_than, 1: less_than},
        attrs={'operation': 'NAND'})
    
    delete_geometry = nw.new_node(Nodes.DeleteGeometry,
        input_kwargs={'Geometry': distribute_points_on_faces.outputs["Points"], 'Selection': nand})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Scale': group_input.outputs["noise scale"]})
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.5
    
    subtract = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: noise_texture.outputs["Color"], 1: value},
        attrs={'operation': 'SUBTRACT'})
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: subtract.outputs["Vector"], 'Scale': group_input.outputs["noise amount"]},
        attrs={'operation': 'SCALE'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': delete_geometry, 'Offset': scale.outputs["Vector"]})
    
    geometry_proximity = nw.new_node(Nodes.Proximity,
        input_kwargs={'Target': group_input.outputs["Mesh"]})
    
    set_position_1 = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': set_position, 'Position': geometry_proximity.outputs["Position"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_position_1, 'Rotation': distribute_points_on_faces.outputs["Rotation"]})

@node_utils.to_nodegroup('nodegroup_instance_on_points', singleton=False, type='GeometryNodeTree')
def nodegroup_instance_on_points(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVectorEuler', 'rotation base', (0.0, 0.0, 0.0)),
            ('NodeSocketVectorEuler', 'rotation delta', (0.0, 0.0, 0.0)),
            ('NodeSocketVectorTranslation', 'translation', (0.0, 0.0, 0.0)),
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

@node_utils.to_nodegroup('nodegroup_shape_quadratic', singleton=False, type='GeometryNodeTree')
def nodegroup_shape_quadratic(nw: NodeWrangler, radius_control_points=[(0.0, 0.5), (1.0, 0.5)]):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Profile Curve', None),
            ('NodeSocketFloat', 'random seed tilt', 0.5),
            ('NodeSocketFloat', 'noise scale tilt', 0.5),
            ('NodeSocketFloat', 'noise amount tilt', 0.0),
            ('NodeSocketFloat', 'random seed pos', 0.0),
            ('NodeSocketFloat', 'noise scale pos', 0.0),
            ('NodeSocketFloat', 'noise amount pos', 0.0),
            ('NodeSocketIntUnsigned', 'Resolution', 256),
            ('NodeSocketVectorTranslation', 'Start', (0.0, 0.0, -1.5)),
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
    
    float_curve_2 = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': spline_parameter_1.outputs["Factor"]})
    node_utils.assign_curve(float_curve_2.mapping.curves[0], radius_control_points)
    
    set_curve_radius = nw.new_node(Nodes.SetCurveRadius,
        input_kwargs={'Curve': set_curve_tilt, 'Radius': float_curve_2})
    
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

@node_utils.to_nodegroup('nodegroup_add_dent', singleton=False, type='GeometryNodeTree')
def nodegroup_add_dent(nw: NodeWrangler, dent_control_points):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketFloat', 'spline parameter', 0.0),
            ('NodeSocketVector', 'spline tangent', (0.0, 0.0, 0.0)),
            ('NodeSocketFloat', 'distance to center', 0.0),
            ('NodeSocketBool', 'bottom', False),
            ('NodeSocketFloat', 'intensity', 1.0),
            ('NodeSocketFloat', 'max radius', 1.0)])
    
    greater_than = nw.new_node(Nodes.Compare,
        input_kwargs={0: group_input.outputs["spline parameter"], 1: 0.5})
    
    op_not = nw.new_node(Nodes.BooleanMath,
        input_kwargs={0: greater_than},
        attrs={'operation': 'NOT'})
    
    switch = nw.new_node(Nodes.Switch,
        input_kwargs={0: group_input.outputs["bottom"], 6: greater_than, 7: op_not},
        attrs={'input_type': 'BOOLEAN'})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': group_input.outputs["distance to center"], 2: group_input.outputs["max radius"]})
    
    float_curve_3 = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': map_range.outputs["Result"]})
    node_utils.assign_curve(float_curve_3.mapping.curves[0], dent_control_points)
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': float_curve_3, 3: -1.0})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_1.outputs["Result"], 1: group_input.outputs["intensity"]},
        attrs={'operation': 'MULTIPLY'})
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["spline tangent"], 'Scale': multiply},
        attrs={'operation': 'SCALE'})
    
    set_position_2 = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Selection': switch.outputs[2], 'Offset': scale.outputs["Vector"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_position_2})

@node_utils.to_nodegroup('nodegroup_add_crater', singleton=False, type='GeometryNodeTree')
def nodegroup_add_crater(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketGeometry', 'Points', None),
            ('NodeSocketFloat', 'Strength', 1.5)])
    
    normal = nw.new_node(Nodes.InputNormal)
    
    geometry_proximity = nw.new_node(Nodes.Proximity,
        input_kwargs={'Target': group_input.outputs["Points"]},
        attrs={'target_element': 'POINTS'})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': geometry_proximity.outputs["Distance"], 2: 0.08, 3: -0.04, 4: 0.0})
    
    smooth_min = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_1.outputs["Result"], 1: 0.0, 2: 0.05},
        attrs={'operation': 'SMOOTH_MIN'})
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: normal, 'Scale': smooth_min},
        attrs={'operation': 'SCALE'})
    
    scale_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: scale.outputs["Vector"], 'Scale': group_input.outputs["Strength"]},
        attrs={'operation': 'SCALE'})
    
    set_position_1 = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Offset': scale_1.outputs["Vector"]})
    
    subdivision_surface = nw.new_node(Nodes.SubdivisionSurface,
        input_kwargs={'Mesh': set_position_1})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': subdivision_surface})


@node_utils.to_nodegroup('nodegroup_mix_vector', singleton=False, type='GeometryNodeTree')
def nodegroup_mix_vector(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'Vector 1', (0.0, 0.0, 0.0)),
            ('NodeSocketVector', 'Vector 2', (0.0, 0.0, 0.0)),
            ('NodeSocketFloat', 'alpha', 0.5)])
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: 1.0, 1: group_input.outputs["alpha"]},
        attrs={'operation': 'SUBTRACT'})
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Vector 1"], 'Scale': subtract},
        attrs={'operation': 'SCALE'})
    
    scale_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Vector 2"], 'Scale': group_input.outputs["alpha"]},
        attrs={'operation': 'SCALE'})
    
    add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: scale.outputs["Vector"], 1: scale_1.outputs["Vector"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Vector': add.outputs["Vector"]})

@node_utils.to_nodegroup('nodegroup_add_noise_scalar', singleton=False, type='GeometryNodeTree')
def nodegroup_add_noise_scalar(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    position = nw.new_node(Nodes.InputPosition)
    
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'value', 0.5),
            ('NodeSocketFloat', 'noise random seed', 0.0),
            ('NodeSocketFloat', 'noise scale', 5.0),
            ('NodeSocketFloat', 'noise amount', 0.5)])
    
    add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: position, 1: group_input.outputs["noise random seed"]})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': add.outputs["Vector"], 'Scale': group_input.outputs["noise scale"]})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: noise_texture.outputs["Fac"]},
        attrs={'operation': 'SUBTRACT'})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract, 1: group_input.outputs["noise amount"]},
        attrs={'operation': 'MULTIPLY'})
    
    add_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply, 1: group_input.outputs["value"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'value': add_1})

@node_utils.to_nodegroup('nodegroup_attach_to_nearest', singleton=False, type='GeometryNodeTree')
def nodegroup_attach_to_nearest(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketGeometry', 'Target', None),
            ('NodeSocketFloat', 'threshold', 0.0),
            ('NodeSocketFloat', 'multiplier', 0.5),
            ('NodeSocketVectorTranslation', 'Offset', (0.0, 0.0, 0.0))])
    
    position = nw.new_node(Nodes.InputPosition)
    
    geometry_proximity = nw.new_node(Nodes.Proximity,
        input_kwargs={'Target': group_input.outputs["Target"]})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["threshold"], 1: geometry_proximity.outputs["Distance"]},
        attrs={'operation': 'SUBTRACT'})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract, 1: group_input.outputs["multiplier"]},
        attrs={'operation': 'MULTIPLY'})
    
    exponent = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply},
        attrs={'operation': 'EXPONENT'})
    
    clamp = nw.new_node(Nodes.Clamp,
        input_kwargs={'Value': exponent})
    
    mixvector = nw.new_node(nodegroup_mix_vector().name,
        input_kwargs={'Vector 1': position, 'Vector 2': geometry_proximity.outputs["Position"], 'alpha': clamp})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Position': mixvector, 'Offset': group_input.outputs["Offset"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_position})

@node_utils.to_nodegroup('nodegroup_manhattan', singleton=False, type='GeometryNodeTree')
def nodegroup_manhattan(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'v1', (0.0, 0.0, 0.0)),
            ('NodeSocketVector', 'v2', (0.0, 0.0, 0.0))])
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group_input.outputs["v1"]})
    
    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group_input.outputs["v2"]})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: separate_xyz_1.outputs["X"]},
        attrs={'operation': 'SUBTRACT'})
    
    absolute = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract},
        attrs={'operation': 'ABSOLUTE'})
    
    subtract_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: separate_xyz_1.outputs["Y"]},
        attrs={'operation': 'SUBTRACT'})
    
    absolute_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract_1},
        attrs={'operation': 'ABSOLUTE'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: absolute, 1: absolute_1})
    
    subtract_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: separate_xyz_1.outputs["Z"]},
        attrs={'operation': 'SUBTRACT'})
    
    absolute_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract_2},
        attrs={'operation': 'ABSOLUTE'})
    
    add_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: add, 1: absolute_2})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Value': add_1})

@node_utils.to_nodegroup('nodegroup_rot_semmetry', singleton=False, type='GeometryNodeTree')
def nodegroup_rot_semmetry(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketInt', 'N', 0),
            ('NodeSocketFloat', 'spline parameter', 0.5)])
    
    divide = nw.new_node(Nodes.Math,
        input_kwargs={1: group_input.outputs["N"]},
        attrs={'operation': 'DIVIDE'})
    
    pingpong = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["spline parameter"], 1: divide},
        attrs={'operation': 'PINGPONG'})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': pingpong, 2: divide})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Result': map_range.outputs["Result"]})

@node_utils.to_nodegroup('nodegroup_scale_mesh', singleton=False, type='GeometryNodeTree')
def nodegroup_scale_mesh(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketFloat', 'Scale', 1.0)])
    
    position = nw.new_node(Nodes.InputPosition)
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: position, 'Scale': group_input.outputs["Scale"]},
        attrs={'operation': 'SCALE'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Position': scale.outputs["Vector"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_position})

@node_utils.to_nodegroup('nodegroup_hair', singleton=False, type='GeometryNodeTree')
def nodegroup_hair(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'shape noise random seed', 0.0),
            ('NodeSocketFloat', 'shape noise amount', 1.0),
            ('NodeSocketIntUnsigned', 'length resolution', 8),
            ('NodeSocketInt', 'cross section resolution', 4),
            ('NodeSocketFloat', 'scale', 0.0),
            ('NodeSocketFloatDistance', 'Radius', 0.01),
            ('NodeSocketMaterial', 'Material', None),
            ('NodeSocketVectorTranslation', 'Start', (0.0, 0.0, 0.0)),
            ('NodeSocketVectorTranslation', 'Middle', (0.0, 0.3, 1.0)),
            ('NodeSocketVectorTranslation', 'End', (0.0, -1.4, 2.0))])
    
    quadratic_bezier_1 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': group_input.outputs["length resolution"], 'Start': group_input.outputs["Start"], 'Middle': group_input.outputs["Middle"], 'End': group_input.outputs["End"]})
    
    subdivide_curve = nw.new_node(Nodes.SubdivideCurve,
        input_kwargs={'Curve': quadratic_bezier_1})
    
    position = nw.new_node(Nodes.InputPosition)
    
    add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: position, 1: group_input.outputs["shape noise random seed"]})
    
    noise_texture_3 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': add.outputs["Vector"], 'Scale': 1.0})
    
    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = 0.5
    
    subtract = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: noise_texture_3.outputs["Color"], 1: value_1},
        attrs={'operation': 'SUBTRACT'})
    
    spline_parameter_2 = nw.new_node(Nodes.SplineParameter)
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: subtract.outputs["Vector"], 'Scale': spline_parameter_2.outputs["Factor"]},
        attrs={'operation': 'SCALE'})
    
    scale_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: scale.outputs["Vector"], 'Scale': group_input.outputs["shape noise amount"]},
        attrs={'operation': 'SCALE'})
    
    set_position_1 = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': subdivide_curve, 'Offset': scale_1.outputs["Vector"]})
    
    curve_circle_1 = nw.new_node(Nodes.CurveCircle,
        input_kwargs={'Resolution': group_input.outputs["cross section resolution"], 'Radius': group_input.outputs["Radius"]})
    
    curve_to_mesh_1 = nw.new_node(Nodes.CurveToMesh,
        input_kwargs={'Curve': set_position_1, 'Profile Curve': curve_circle_1.outputs["Curve"], 'Fill Caps': True})
    
    transform_1 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': curve_to_mesh_1, 'Scale': group_input.outputs["scale"]})
    
    set_material_1 = nw.new_node(Nodes.SetMaterial,
        input_kwargs={'Geometry': transform_1, 'Material': group_input.outputs["Material"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_material_1})

@node_utils.to_nodegroup('nodegroup_random_rotation_scale', singleton=False, type='GeometryNodeTree')
def nodegroup_random_rotation_scale(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'random seed', 0.0),
            ('NodeSocketFloat', 'noise scale', 10.0),
            ('NodeSocketVector', 'rot mean', (0.0, 0.0, 0.0)),
            ('NodeSocketFloat', ' rot std z', 1.0),
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
        input_kwargs={0: separate_xyz_2.outputs["X"], 1: group_input.outputs[" rot std z"]},
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

@node_utils.to_nodegroup('nodegroup_align_top_to_horizon', singleton=False, type='GeometryNodeTree')
def nodegroup_align_top_to_horizon(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None)])
    
    bounding_box = nw.new_node(Nodes.BoundingBox,
        input_kwargs={'Geometry': group_input.outputs["Geometry"]})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': bounding_box.outputs["Max"]})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: -1.0},
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Z': multiply})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Offset': combine_xyz})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_position})
