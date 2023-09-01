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

from infinigen.assets.creatures.insects.utils.geom_utils import nodegroup_circle_cross_section, nodegroup_surface_bump, nodegroup_random_rotation_scale, nodegroup_instance_on_points
from infinigen.assets.creatures.insects.parts.hair.principled_hair import nodegroup_principled_hair
from infinigen.assets.creatures.insects.utils.shader_utils import shader_black_w_noise_shader, nodegroup_add_noise, nodegroup_color_noise

def shader_dragonfly_body_shader(nw: NodeWrangler, base_color, v):
    # Code generated using version 2.4.3 of the node_transpiler

    attribute = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'pos'})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': attribute.outputs["Vector"]})
    
    absolute = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"]},
        attrs={'operation': 'ABSOLUTE'})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: 3.0},
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': absolute, 'Y': separate_xyz.outputs["Y"], 'Z': multiply})
    
    attribute_1 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'body seed'})
    
    musgrave_texture = nw.new_node(Nodes.MusgraveTexture,
        input_kwargs={'Vector': combine_xyz, 'W': attribute_1.outputs["Fac"], 'Scale': 0.5, 'Dimension': 1.0, 'Lacunarity': 1.0},
        attrs={'musgrave_dimensions': '4D'})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': musgrave_texture, 1: -0.26, 2: 0.06})
    
    attribute_2 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'spline parameter'})
    
    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': attribute_2.outputs["Fac"]})
    
    group = nw.new_node(nodegroup_add_noise().name,
        input_kwargs={'Vector': combine_xyz_1, 'Scale': 0.5, 'amount': (0.16, 0.26, 0.0), 'Noise Eval Position': combine_xyz})
    
    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group})
    
    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': separate_xyz_1.outputs["X"], 'Y': attribute_1.outputs["Fac"]})
    
    voronoi_texture = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': combine_xyz_2, 'Scale': 10.0},
        attrs={'voronoi_dimensions': '2D'})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': voronoi_texture.outputs["Distance"], 1: 0.14, 2: 0.82})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range.outputs["Result"], 1: map_range_1.outputs["Result"]})
    
    map_range_2 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': add, 1: 0.7, 3: 1.0, 4: 0.0})
    
    rgb_1 = nw.new_node(Nodes.RGB)
    rgb_1.outputs[0].default_value = base_color
    
    group_2 = nw.new_node(nodegroup_color_noise().name,
        input_kwargs={'Scale': 1.34, 'Color': rgb_1, 'Value From Max': 0.7, 'Value To Min': 0.18})
    
    hue_saturation_value = nw.new_node('ShaderNodeHueSaturation',
        input_kwargs={'Value': v, 'Color': rgb_1})
    
    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': map_range_2.outputs["Result"], 'Color1': group_2, 'Color2': hue_saturation_value})
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': mix, 'Metallic': 0.2182, 'Specular': 0.8318, 'Roughness': 0.1545})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

@node_utils.to_nodegroup('nodegroup_dragonfly_body', singleton=False, type='GeometryNodeTree')
def nodegroup_dragonfly_body(nw: NodeWrangler, 
    curve_control_points=[(0.0, 0.15), (0.1586, 0.4688), (0.36, 0.66), (0.7427, 0.4606), (0.9977, 0.2562)],
    base_color=(0.2789, 0.3864, 0.0319, 1.0), 
    v=0.3,
    ):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'Body Length', 10.0),
            ('NodeSocketFloat', 'Random Seed', 0.0),
            ('NodeSocketFloat', 'Hair Density', 200.0)])
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Z': group_input.outputs["Body Length"]})
    
    curve_line = nw.new_node(Nodes.CurveLine,
        input_kwargs={'End': combine_xyz})
    
    resample_curve = nw.new_node(Nodes.ResampleCurve,
        input_kwargs={'Curve': curve_line, 'Count': 128})
    
    spline_parameter = nw.new_node(Nodes.SplineParameter)
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': resample_curve, 2: spline_parameter.outputs["Factor"]})
    
    float_curve = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': spline_parameter.outputs["Factor"]})
    node_utils.assign_curve(float_curve.mapping.curves[0], curve_control_points)
    
    set_curve_radius = nw.new_node(Nodes.SetCurveRadius,
        input_kwargs={'Curve': capture_attribute.outputs["Geometry"], 'Radius': float_curve})
    
    circlecrosssection = nw.new_node(nodegroup_circle_cross_section().name,
        input_kwargs={'random seed': group_input.outputs["Random Seed"], 'noise amount': 1.26, 'radius': 4.0})
    
    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': circlecrosssection, 'Rotation': (0.0, 0.0, 1.5708)})
    
    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
        input_kwargs={'Curve': set_curve_radius, 'Profile Curve': transform, 'Fill Caps': True})
    
    normal = nw.new_node(Nodes.InputNormal)
    
    position_2 = nw.new_node(Nodes.InputPosition)
    
    multiply = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: position_2, 1: (1.0, 0.2, 0.8)},
        attrs={'operation': 'MULTIPLY'})
    
    voronoi_texture = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': multiply.outputs["Vector"], 'W': group_input.outputs["Random Seed"], 'Scale': 0.5},
        attrs={'voronoi_dimensions': '4D'})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': voronoi_texture.outputs["Distance"], 4: 0.4},
        attrs={'clamp': False})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range.outputs["Result"], 1: -1.0},
        attrs={'operation': 'MULTIPLY'})
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: normal, 'Scale': multiply_1},
        attrs={'operation': 'SCALE'})
    
    add = nw.new_node(Nodes.VectorMath,
        input_kwargs={1: scale.outputs["Vector"]})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': curve_to_mesh, 'Offset': add.outputs["Vector"]})
    
    set_material = nw.new_node(Nodes.SetMaterial,
        input_kwargs={'Geometry': set_position, 'Material': surface.shaderfunc_to_material(shader_dragonfly_body_shader, base_color, v)})
    
    surfacebump = nw.new_node(nodegroup_surface_bump().name,
        input_kwargs={'Geometry': set_material, 'Displacement': -0.12, 'Scale': 75.8, 'seed': group_input.outputs["Random Seed"]})
    
    position = nw.new_node(Nodes.InputPosition)
    
    store_named_attribute = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': surfacebump, 'Name': 'pos', 2: position},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': position})
    
    greater_than = nw.new_node(Nodes.Compare,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: 0.5})
    
    reroute = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': capture_attribute.outputs[2]})
    
    less_than = nw.new_node(Nodes.Compare,
        input_kwargs={0: reroute, 1: 0.4},
        attrs={'operation': 'LESS_THAN'})
    
    op_and = nw.new_node(Nodes.BooleanMath,
        input_kwargs={0: greater_than, 1: less_than})
    
    reroute_1 = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': group_input.outputs["Hair Density"]})
    
    distribute_points_on_faces = nw.new_node(Nodes.DistributePointsOnFaces,
        input_kwargs={'Mesh': store_named_attribute, 'Selection': op_and, 'Density': reroute_1})
    
    randomrotationscale = nw.new_node(nodegroup_random_rotation_scale().name,
        input_kwargs={'random seed': -2.4, 'rot mean': (-1.0, 0.0, 0.0), 'rot std z': -10.2, 'scale mean': 0.03})
    
    leghair = nw.new_node(nodegroup_principled_hair().name,
        input_kwargs={'Resolution': 2})
    
    transform_3 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': leghair, 'Scale': (1.0, 1.0, 5.0)})
    
    set_material_2 = nw.new_node(Nodes.SetMaterial,
        input_kwargs={'Geometry': transform_3, 'Material': surface.shaderfunc_to_material(shader_black_w_noise_shader)})
    
    instanceonpoints = nw.new_node(nodegroup_instance_on_points().name,
        input_kwargs={'rotation base': distribute_points_on_faces.outputs["Rotation"], 'rotation delta': randomrotationscale.outputs["Vector"], 'translation': (0.0, 0.0, 0.0), 'scale': randomrotationscale.outputs["Value"], 'Points': distribute_points_on_faces.outputs["Points"], 'Instance': set_material_2})
    
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: reroute_1, 1: 0.3},
        attrs={'operation': 'MULTIPLY'})
    
    distribute_points_on_faces_1 = nw.new_node(Nodes.DistributePointsOnFaces,
        input_kwargs={'Mesh': store_named_attribute, 'Density': multiply_2, 'Seed': 1})
    
    instanceonpoints_1 = nw.new_node(nodegroup_instance_on_points().name,
        input_kwargs={'rotation base': distribute_points_on_faces_1.outputs["Rotation"], 'rotation delta': randomrotationscale.outputs["Vector"], 'translation': (0.0, 0.0, 0.0), 'scale': randomrotationscale.outputs["Value"], 'Points': distribute_points_on_faces_1.outputs["Points"], 'Instance': set_material_2})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [store_named_attribute, instanceonpoints, instanceonpoints_1]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': join_geometry, 'Skeleton Curve': resample_curve, 'spline parameter': reroute})