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

from infinigen.assets.creatures.insects.utils.shader_utils import nodegroup_color_noise, nodegroup_add_noise
from infinigen.assets.creatures.insects.utils.geom_utils import nodegroup_random_rotation_scale, nodegroup_circle_cross_section, nodegroup_shape_quadratic, nodegroup_surface_bump, nodegroup_instance_on_points

def shader_dragonfly_tail_shader(nw: NodeWrangler, base_color, v, ring_length):
    # Code generated using version 2.4.3 of the node_transpiler

    attribute = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'cross section parameter'})
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': attribute.outputs["Fac"]})
    colorramp.color_ramp.elements.new(0)
    colorramp.color_ramp.elements.new(0)
    colorramp.color_ramp.elements[0].position = 0.0
    colorramp.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    colorramp.color_ramp.elements[1].position = 0.4455
    colorramp.color_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)
    colorramp.color_ramp.elements[2].position = 0.5045
    colorramp.color_ramp.elements[2].color = (0.0, 0.0, 0.0, 1.0)
    colorramp.color_ramp.elements[3].position = 1.0
    colorramp.color_ramp.elements[3].color = (1.0, 1.0, 1.0, 1.0)
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': colorramp.outputs["Color"], 1: 0.02, 2: 0.38})
    
    attribute_1 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'spline parameter'})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': attribute_1.outputs["Fac"], 1: 0.18, 2: 0.42})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': map_range.outputs["Result"], 'Y': map_range_1.outputs["Result"]})
    
    texture_coordinate = nw.new_node(Nodes.TextureCoord)
    
    group = nw.new_node(nodegroup_add_noise().name,
        input_kwargs={'Vector': combine_xyz, 'amount': (1.0, 1.0, 0.0), 'Noise Eval Position': texture_coordinate.outputs["Object"]})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: separate_xyz.outputs["Y"]})
    
    voronoi_texture = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'W': attribute_1.outputs["Fac"], 'Scale': 5.34, 'Randomness': 0.0},
        attrs={'voronoi_dimensions': '1D'})
    
    map_range_2 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': voronoi_texture.outputs["Distance"], 2: 0.1, 3: 1.0, 4: 0.0})
    
    maximum = nw.new_node(Nodes.Math,
        input_kwargs={0: add, 1: map_range_2.outputs["Result"]},
        attrs={'operation': 'MAXIMUM'})
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = ring_length
    
    add_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: value, 1: 0.05})
    
    map_range_4 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': attribute_1.outputs["Fac"], 1: value, 2: add_1})
    
    minimum = nw.new_node(Nodes.Math,
        input_kwargs={0: maximum, 1: map_range_4.outputs["Result"]},
        attrs={'operation': 'MINIMUM'})
    
    rgb = nw.new_node(Nodes.RGB)
    rgb.outputs[0].default_value = base_color
    
    hue_saturation_value = nw.new_node('ShaderNodeHueSaturation',
        input_kwargs={'Value': v, 'Color': rgb})
    
    group_2 = nw.new_node(nodegroup_color_noise().name,
        input_kwargs={'Scale': 1.34, 'Color': rgb, 'Value From Max': 0.7, 'Value To Min': 0.18})
    
    mix_1 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': minimum, 'Color1': hue_saturation_value, 'Color2': group_2})
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': mix_1, 'Metallic': 0.5, 'Specular': 0.5114, 'Roughness': 0.2568})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})


@node_utils.to_nodegroup('nodegroup_dragonfly_tail', singleton=False, type='GeometryNodeTree')
def nodegroup_dragonfly_tail(nw: NodeWrangler,
        base_color=(0.2789, 0.3864, 0.0319, 1.0), 
        v=0.3,
        ring_length=0.3
        ):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVectorTranslation', 'Start', (0.0, 0.0, 0.0)),
            ('NodeSocketVectorTranslation', 'Middle', (1.84, 0.0, 0.14)),
            ('NodeSocketVectorTranslation', 'End', (3.14, 0.0, -0.32)),
            ('NodeSocketFloatDistance', 'Segment Length', 0.44),
            ('NodeSocketFloat', 'Segment Scale', 0.25),
            ('NodeSocketFloat', 'Random Seed', 3.2),
            ('NodeSocketFloat', 'Radius', 0.9)])
    
    quadratic_bezier = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Start': group_input.outputs["Start"], 'Middle': group_input.outputs["Middle"], 'End': group_input.outputs["End"]})
    
    spline_parameter = nw.new_node(Nodes.SplineParameter)
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': quadratic_bezier, 2: spline_parameter.outputs["Factor"]})
    
    curve_to_points = nw.new_node(Nodes.CurveToPoints,
        input_kwargs={'Curve': capture_attribute.outputs["Geometry"], 'Length': group_input.outputs["Segment Length"]},
        attrs={'mode': 'LENGTH'})
    
    reroute_1 = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': group_input.outputs["Segment Scale"]})
    
    randomrotationscale = nw.new_node(nodegroup_random_rotation_scale().name,
        input_kwargs={'rot std z': 0.0, 'scale mean': reroute_1, 'scale std': 0.05})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': capture_attribute.outputs[2], 3: 1.0, 4: 0.8})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: randomrotationscale.outputs["Value"], 1: map_range.outputs["Result"]},
        attrs={'operation': 'MULTIPLY'})
    
    droplast = nw.new_node(nodegroup_droplast().name,
        input_kwargs={'Geometry': curve_to_points.outputs["Points"]})
    
    integer = nw.new_node(Nodes.Integer,
        attrs={'integer': 128})
    integer.integer = 128
    
    circlecrosssection = nw.new_node(nodegroup_circle_cross_section().name,
        input_kwargs={'random seed': 23.4, 'noise amount': 0.9, 'Resolution': integer, 'radius': group_input.outputs["Radius"]})
    
    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': circlecrosssection, 'Rotation': (0.0, 0.0, 1.5708)})
    
    spline_parameter_1 = nw.new_node(Nodes.SplineParameter)
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: spline_parameter_1.outputs["Factor"]},
        attrs={'operation': 'SUBTRACT'})
    
    absolute = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract},
        attrs={'operation': 'ABSOLUTE'})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: absolute, 1: 2.0},
        attrs={'operation': 'MULTIPLY'})
    
    store_named_attribute_2 = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': transform, 'Name': 'cross section parameter', 'Value': multiply_1})
    
    reroute = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': store_named_attribute_2})
    
    shapequadratic_001 = nw.new_node(nodegroup_shape_quadratic(radius_control_points=[(0.0, 0.3906), (0.1795, 0.4656), (0.5, 0.4563), (0.8795, 0.45), (1.0, 0.4344)]).name,
        input_kwargs={'Profile Curve': reroute, 'noise amount tilt': 0.0, 'Resolution': integer, 'Start': (0.0, 0.0, -1.5), 'End': (0.0, 0.0, 0.68)})
    
    store_named_attribute_1 = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': shapequadratic_001.outputs["Mesh"], 'Name': 'spline parameter', 'Value': shapequadratic_001.outputs["spline parameter"]})
    
    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = 0.02
    
    surfacebump = nw.new_node(nodegroup_surface_bump().name,
        input_kwargs={'Geometry': store_named_attribute_1, 'Displacement': value_1, 'Scale': 20.0, 'seed': group_input.outputs["Random Seed"]})
    
    addverticalstripes = nw.new_node(nodegroup_add_vertical_stripes().name,
        input_kwargs={'Geometry': surfacebump, 'Seed': group_input.outputs["Random Seed"]})
    
    instanceonpoints = nw.new_node(nodegroup_instance_on_points().name,
        input_kwargs={'rotation base': curve_to_points.outputs["Rotation"], 'rotation delta': randomrotationscale.outputs["Vector"], 'translation': (0.0, 0.0, 0.0), 'scale': multiply, 'Points': droplast.outputs["Others"], 'Instance': addverticalstripes})
    
    shapequadratic_003 = nw.new_node(nodegroup_shape_quadratic(radius_control_points=[(0.0, 0.3312), (0.1773, 0.4281), (0.4318, 0.5031), (0.5886, 0.3562), (0.7864, 0.2687), (1.0, 0.0)]).name,
        input_kwargs={'Profile Curve': reroute, 'noise amount tilt': 0.0, 'Resolution': integer, 'Start': (0.26, 0.0, -1.5), 'Middle': (0.32, 0.0, 0.0), 'End': (-0.04, 0.0, 1.5)})
    
    transform_1 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': shapequadratic_003.outputs["Mesh"], 'Translation': (0.0, 0.28, 0.0), 'Rotation': (0.0, 0.0, -1.5708)})
    
    store_named_attribute = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': transform_1, 'Name': 'spline parameter', 'Value': shapequadratic_003.outputs["spline parameter"]})
    
    surfacebump_1 = nw.new_node(nodegroup_surface_bump().name,
        input_kwargs={'Geometry': store_named_attribute, 'Displacement': value_1, 'Scale': 20.0})
    
    addverticalstripes_1 = nw.new_node(nodegroup_add_vertical_stripes().name,
        input_kwargs={'Geometry': surfacebump_1, 'Seed': group_input.outputs["Random Seed"]})
    
    instance_on_points = nw.new_node(Nodes.InstanceOnPoints,
        input_kwargs={'Points': droplast.outputs["Last"], 'Instance': addverticalstripes_1, 'Rotation': curve_to_points.outputs["Rotation"], 'Scale': reroute_1})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [instanceonpoints, instance_on_points]})
    
    set_material = nw.new_node(Nodes.SetMaterial,
        input_kwargs={'Geometry': join_geometry, 'Material': surface.shaderfunc_to_material(shader_dragonfly_tail_shader, base_color, v, ring_length)})
    
    realize_instances = nw.new_node(Nodes.RealizeInstances,
        input_kwargs={'Geometry': set_material})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': realize_instances})

@node_utils.to_nodegroup('nodegroup_droplast', singleton=False, type='GeometryNodeTree')
def nodegroup_droplast(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None)])
    
    index = nw.new_node(Nodes.Index)
    
    domain_size = nw.new_node(Nodes.DomainSize,
        input_kwargs={'Geometry': group_input.outputs["Geometry"]},
        attrs={'component': 'POINTCLOUD'}
        )
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: domain_size.outputs["Point Count"], 1: 1.0},
        attrs={'operation': 'SUBTRACT'})
    
    equal = nw.new_node(Nodes.Compare,
        input_kwargs={2: index, 3: subtract},
        attrs={'data_type': 'INT', 'operation': 'EQUAL'})
    
    op_not = nw.new_node(Nodes.BooleanMath,
        input_kwargs={0: equal},
        attrs={'operation': 'NOT'})
    
    delete_geometry_1 = nw.new_node(Nodes.DeleteGeometry,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Selection': op_not})
    
    delete_geometry = nw.new_node(Nodes.DeleteGeometry,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Selection': equal})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Last': delete_geometry_1, 'Others': delete_geometry})

@node_utils.to_nodegroup('nodegroup_add_vertical_stripes', singleton=False, type='GeometryNodeTree')
def nodegroup_add_vertical_stripes(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketFloat', 'Scale', 5.0),
            ('NodeSocketFloat', 'Seed', 0.0)])
    
    position = nw.new_node(Nodes.InputPosition)
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': position})
    
    absolute = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"]},
        attrs={'operation': 'ABSOLUTE'})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: 0.05},
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': absolute, 'Y': separate_xyz.outputs["Y"], 'Z': multiply})
    
    voronoi_texture = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': combine_xyz, 'W': group_input.outputs["Seed"], 'Scale': group_input.outputs["Scale"]},
        attrs={'voronoi_dimensions': '4D', 'feature': 'DISTANCE_TO_EDGE'})
    
    reroute_1 = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': voronoi_texture.outputs["Distance"]})
    
    store_named_attribute_3 = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Name': 'tail vertical strips', 'Value': reroute_1})
    
    normal = nw.new_node(Nodes.InputNormal)
        
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: reroute_1, 1: 0.1},
        attrs={'operation': 'MULTIPLY'})
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: normal, 'Scale': multiply_1},
        attrs={'operation': 'SCALE'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': store_named_attribute_3, 'Offset': scale.outputs["Vector"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_position})
## old version
# def shader_dragonfly_tail_shader(nw: NodeWrangler):
#     # Code generated using version 2.4.3 of the node_transpiler

#     texture_coordinate = nw.new_node(Nodes.TextureCoord)

#     attribute_1 = nw.new_node(Nodes.Attribute,
#         attrs={'attribute_name': 'cross section parameter'})
    
#     colorramp_1 = nw.new_node(Nodes.ColorRamp,
#         input_kwargs={'Fac': attribute_1.outputs["Fac"]})
#     colorramp_1.color_ramp.elements.new(0)
#     colorramp_1.color_ramp.elements.new(0)
#     colorramp_1.color_ramp.elements[0].position = 0.0
#     colorramp_1.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
#     colorramp_1.color_ramp.elements[1].position = 0.4455
#     colorramp_1.color_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)
#     colorramp_1.color_ramp.elements[2].position = 0.5045
#     colorramp_1.color_ramp.elements[2].color = (0.0, 0.0, 0.0, 1.0)
#     colorramp_1.color_ramp.elements[3].position = 1.0
#     colorramp_1.color_ramp.elements[3].color = (1.0, 1.0, 1.0, 1.0)
    
#     map_range_1 = nw.new_node(Nodes.MapRange,
#         input_kwargs={'Value': colorramp_1.outputs["Color"], 1: 0.02, 2: 0.38})
    
#     attribute = nw.new_node(Nodes.Attribute,
#         attrs={'attribute_name': 'spline parameter'})
    
#     map_range = nw.new_node(Nodes.MapRange,
#         input_kwargs={'Value': attribute.outputs["Fac"], 1: 0.18, 2: 0.42})
    
#     combine_xyz = nw.new_node(Nodes.CombineXYZ,
#         input_kwargs={'X': map_range_1.outputs["Result"], 'Y': map_range.outputs["Result"]})
    
#     group_2 = nw.new_node(nodegroup_add_noise().name,
#         input_kwargs={'Vector': combine_xyz, 'amount': (1.0, 1.0, 0.0), 'Noise Eval Position': texture_coordinate.outputs["Object"],})
    
#     separate_xyz = nw.new_node(Nodes.SeparateXYZ,
#         input_kwargs={'Vector': group_2})
    
#     add = nw.new_node(Nodes.Math,
#         input_kwargs={0: separate_xyz.outputs["X"], 1: separate_xyz.outputs["Y"]})
    
#     voronoi_texture = nw.new_node(Nodes.VoronoiTexture,
#         input_kwargs={'W': attribute.outputs["Fac"], 'Scale': 5.34, 'Randomness': 0.0},
#         attrs={'voronoi_dimensions': '1D'})
    
#     map_range_2 = nw.new_node(Nodes.MapRange,
#         input_kwargs={'Value': voronoi_texture.outputs["Distance"], 2: 0.1, 3: 1.0, 4: 0.0})
    
#     maximum = nw.new_node(Nodes.Math,
#         input_kwargs={0: add, 1: map_range_2.outputs["Result"]},
#         attrs={'operation': 'MAXIMUM'})
    
#     group_1 = nw.new_node(nodegroup_color_noise().name,
#         input_kwargs={'Scale': 6.4, 'Color': (0.1582, 0.291, 1.0, 1.0), 'Value To Min': 0.4})
    
#     attribute_2 = nw.new_node(Nodes.Attribute,
#         attrs={'attribute_name': 'tail vertical strips'})
    
#     map_range_3 = nw.new_node(Nodes.MapRange,
#         input_kwargs={'Value': attribute_2.outputs["Fac"], 1: 0.16, 2: 0.34})
    
#     mix_1 = nw.new_node(Nodes.MixRGB,
#         input_kwargs={'Fac': 0.0, 'Color1': (0.0144, 0.016, 0.0152, 1.0), 'Color2': (0.544, 0.5299, 0.5841, 1.0)})
    
#     mix = nw.new_node(Nodes.MixRGB,
#         input_kwargs={'Fac': maximum, 'Color1': group_1, 'Color2': mix_1})
    
#     principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
#         input_kwargs={'Base Color': mix, 'Metallic': 0.9, 'Specular': 0.5114, 'Roughness': 0.2568})
    
#     material_output = nw.new_node(Nodes.MaterialOutput,
#         input_kwargs={'Surface': principled_bsdf})