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

from infinigen.assets.fruits.fruit_utils import nodegroup_surface_bump, nodegroup_add_noise_scalar, nodegroup_attach_to_nearest, nodegroup_scale_mesh
from infinigen.assets.fruits.cross_section_lib import nodegroup_cylax_cross_section

@node_utils.to_nodegroup('nodegroup_empty_stem', singleton=False, type='GeometryNodeTree')
def nodegroup_empty_stem(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    points = nw.new_node('GeometryNodePoints',
        input_kwargs={'Count': 0})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': points})

def shader_basic_stem_shader(nw: NodeWrangler, stem_color):
    # Code generated using version 2.4.3 of the node_transpiler

    texture_coordinate = nw.new_node(Nodes.TextureCoord)
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': texture_coordinate.outputs["Object"], 'Scale': 0.8, 'Detail': 10.0, 'Roughness': 0.7})
    
    separate_rgb = nw.new_node(Nodes.SeparateColor,
        input_kwargs={'Color': noise_texture.outputs["Color"]})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_rgb.outputs["Green"], 1: 0.4, 2: 0.7, 3: 0.48, 4: 0.55},
        attrs={'interpolation_type': 'SMOOTHSTEP'})
    
    map_range_2 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_rgb.outputs["Blue"], 1: 0.4, 2: 0.7, 3: 0.4},
        attrs={'interpolation_type': 'SMOOTHSTEP'})
    
    hue_saturation_value = nw.new_node('ShaderNodeHueSaturation',
        input_kwargs={'Hue': map_range_1.outputs["Result"], 'Value': map_range_2.outputs["Result"], 'Color': stem_color})
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': hue_saturation_value, 'Specular': 0.1205, 'Roughness': 0.5068})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

@node_utils.to_nodegroup('nodegroup_basic_stem', singleton=False, type='GeometryNodeTree')
def nodegroup_basic_stem(nw: NodeWrangler, stem_color=(0.179, 0.836, 0.318, 1.0)):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVectorTranslation', 'quad_start', (0.0, 0.0, 0.0)),
            ('NodeSocketVectorTranslation', 'quad_mid', (0.0, -0.05, 0.2)),
            ('NodeSocketVectorTranslation', 'quad_end', (-0.1, 0.0, 0.4)),
            ('NodeSocketIntUnsigned', 'quad_res', 128),
            ('NodeSocketFloatDistance', 'cross_radius', 0.08),
            ('NodeSocketInt', 'cross_res', 128),
            ('NodeSocketVectorTranslation', 'Translation', (0.0, 0.0, 1.0)),
            ('NodeSocketVectorXYZ', 'Scale', (1.0, 1.0, 2.0))])
    
    quadratic_bezier_2 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': group_input.outputs["quad_res"], 'Start': group_input.outputs["quad_start"], 'Middle': group_input.outputs["quad_mid"], 'End': group_input.outputs["quad_end"]})
    
    curve_circle_2 = nw.new_node(Nodes.CurveCircle,
        input_kwargs={'Resolution': group_input.outputs["cross_res"], 'Radius': group_input.outputs["cross_radius"]})
    
    curve_to_mesh_2 = nw.new_node(Nodes.CurveToMesh,
        input_kwargs={'Curve': quadratic_bezier_2, 'Profile Curve': curve_circle_2.outputs["Curve"], 'Fill Caps': True})
    
    surfacebump = nw.new_node(nodegroup_surface_bump().name,
        input_kwargs={'Geometry': curve_to_mesh_2, 'Displacement': 0.01, 'Scale': 2.9})
    
    surfacebump_1 = nw.new_node(nodegroup_surface_bump().name,
        input_kwargs={'Geometry': surfacebump, 'Scale': 20.0})
    
    transform_3 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': surfacebump_1, 'Translation': group_input.outputs["Translation"], 'Scale': group_input.outputs["Scale"]})

    set_material = nw.new_node(Nodes.SetMaterial,
        input_kwargs={'Geometry': transform_3, 'Material': surface.shaderfunc_to_material(shader_basic_stem_shader, stem_color)})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_material})

def shader_calyx_shader(nw: NodeWrangler, stem_color):
    # Code generated using version 2.4.3 of the node_transpiler

    noise_texture_1 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Scale': 2.8, 'Detail': 10.0, 'Roughness': 0.7})
    
    separate_rgb = nw.new_node(Nodes.SeparateColor,
        input_kwargs={'Color': noise_texture_1.outputs["Color"]})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_rgb.outputs["Green"], 1: 0.4, 2: 0.7, 3: 0.48, 4: 0.55},
        attrs={'interpolation_type': 'SMOOTHSTEP'})
    
    map_range_2 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_rgb.outputs["Blue"], 1: 0.4, 2: 0.7, 3: 0.4},
        attrs={'interpolation_type': 'SMOOTHSTEP'})
    
    hue_saturation_value = nw.new_node('ShaderNodeHueSaturation',
        input_kwargs={'Hue': map_range_1.outputs["Result"], 'Value': map_range_2.outputs["Result"], 'Color': stem_color})
    
    translucent_bsdf = nw.new_node(Nodes.TranslucentBSDF,
        input_kwargs={'Color': hue_saturation_value})
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': hue_saturation_value, 'Specular': 0.5136, 'Roughness': 0.7614})
    
    mix_shader = nw.new_node(Nodes.MixShader,
        input_kwargs={'Fac': 0.5083, 1: translucent_bsdf, 2: principled_bsdf})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': mix_shader})

### straberry calyx ###
@node_utils.to_nodegroup('nodegroup_calyx_stem', singleton=False, type='GeometryNodeTree')
def nodegroup_calyx_stem(nw: NodeWrangler, stem_color=(0.1678, 0.4541, 0.0397, 1.0)):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[
            ('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketInt', 'fork number', 10),
            ('NodeSocketFloatDistance', 'outer radius', 1.0),
            ('NodeSocketFloat', 'inner radius', 0.2),
            ('NodeSocketFloat', 'cross section noise amount', 0.4),
            ('NodeSocketFloat', 'z noise amount', 1.0),
            ('NodeSocketFloatDistance', 'noise random seed', 0.0),
            ('NodeSocketVectorTranslation', 'quad_start', (0.0, 0.0, 0.0)),
            ('NodeSocketVectorTranslation', 'quad_mid', (0.0, -0.05, 0.2)),
            ('NodeSocketVectorTranslation', 'quad_end', (-0.1, 0.0, 0.4)),
            ('NodeSocketVectorTranslation', 'Translation', (0.0, 0.0, 1.0)),
            ('NodeSocketFloatDistance', 'cross_radius', 0.04)])
    
    cylaxcrosssection = nw.new_node(nodegroup_cylax_cross_section().name,
        input_kwargs={'fork number': group_input.outputs["fork number"], 'bottom radius': group_input.outputs["inner radius"], 'noise random seed': group_input.outputs["noise random seed"], 'noise amount': group_input.outputs["cross section noise amount"], 'radius': group_input.outputs["outer radius"]})
    
    fill_curve = nw.new_node(Nodes.FillCurve,
        input_kwargs={'Curve': cylaxcrosssection})
    
    triangulate = nw.new_node('GeometryNodeTriangulate',
        input_kwargs={'Mesh': fill_curve})
    
    subdivide_mesh = nw.new_node(Nodes.SubdivideMesh,
        input_kwargs={'Mesh': triangulate, 'Level': 3})
    
    position = nw.new_node(Nodes.InputPosition)
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': position})
    
    addnoisescalar = nw.new_node(nodegroup_add_noise_scalar().name,
        input_kwargs={'value': separate_xyz.outputs["Z"], 'noise random seed': group_input.outputs["noise random seed"], 'noise scale': 1.0, 'noise amount': group_input.outputs["z noise amount"]})
    
    length = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: position},
        attrs={'operation': 'LENGTH'})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: addnoisescalar, 1: length.outputs["Value"]},
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': separate_xyz.outputs["X"], 'Y': separate_xyz.outputs["Y"], 'Z': multiply})
    
    set_position_1 = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': subdivide_mesh, 'Position': combine_xyz})
    
    basicstem = nw.new_node(nodegroup_basic_stem().name,
        input_kwargs={'quad_start': group_input.outputs["quad_start"], 'quad_mid': group_input.outputs["quad_mid"], 'quad_end': group_input.outputs["quad_end"], 'quad_res': 16, 'cross_radius': group_input.outputs["cross_radius"], 'cross_res': 16, 'Translation': (0.0, 0.0, 0.0)})
    
    join_geometry_2 = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [set_position_1, basicstem]})

    set_material = nw.new_node(Nodes.SetMaterial,
        input_kwargs={'Geometry': join_geometry_2, 'Material': surface.shaderfunc_to_material(shader_calyx_shader, stem_color)})

    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': set_material, 'Translation': group_input.outputs["Translation"], 'Scale': (1.0, 1.0, 1.0)})

    attachtonearest = nw.new_node(nodegroup_attach_to_nearest().name,
        input_kwargs={'Geometry': transform, 'Target': group_input.outputs["Geometry"], 'threshold': 0.1, 'multiplier': 10.0, 'Offset': (0.0, 0.0, 0.05)})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': attachtonearest})

### coconutgreen ###
@node_utils.to_nodegroup('nodegroup_jigsaw', singleton=False, type='GeometryNodeTree')
def nodegroup_jigsaw(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    spline_parameter = nw.new_node(Nodes.SplineParameter)
    
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'Value', 0.5),
            ('NodeSocketFloat', 'noise scale', 30.0),
            ('NodeSocketFloatFactor', 'noise randomness', 0.7),
            ('NodeSocketFloat', 'From Max', 0.15),
            ('NodeSocketFloat', 'To Min', 0.9)])
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={1: group_input.outputs["Value"]},
        attrs={'operation': 'SUBTRACT'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Value"]})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': spline_parameter.outputs["Factor"], 1: subtract, 2: add})
    
    voronoi_texture = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'W': map_range_1.outputs["Result"], 'Scale': group_input.outputs["noise scale"], 'Randomness': group_input.outputs["noise randomness"]},
        attrs={'voronoi_dimensions': '1D', 'feature': 'DISTANCE_TO_EDGE'})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': voronoi_texture.outputs["Distance"], 2: group_input.outputs["From Max"], 3: group_input.outputs["To Min"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Result': map_range.outputs["Result"]})

def shader_coconut_calyx_shader(nw: NodeWrangler, basic_color, edge_color):
    # Code generated using version 2.4.3 of the node_transpiler

    texture_coordinate = nw.new_node(Nodes.TextureCoord)
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': texture_coordinate.outputs["Object"], 'Scale': 10.0, 'Detail': 10.0, 'Roughness': 0.7})
    
    separate_rgb = nw.new_node(Nodes.SeparateColor,
        input_kwargs={'Color': noise_texture.outputs["Color"]})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_rgb.outputs["Green"], 1: 0.4, 2: 0.7, 3: 0.45, 4: 0.52},
        attrs={'interpolation_type': 'SMOOTHSTEP'})
    
    map_range_2 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_rgb.outputs["Blue"], 1: 0.4, 2: 0.7, 3: 0.6},
        attrs={'interpolation_type': 'SMOOTHSTEP'})
    
    attribute = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'distance to edge'})
    
    noise_texture_1 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Scale': 3.0})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: noise_texture_1.outputs["Fac"]},
        attrs={'operation': 'SUBTRACT'})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract, 1: 0.1},
        attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: attribute.outputs["Fac"], 1: multiply})
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': add})
    colorramp.color_ramp.elements.new(0)
    colorramp.color_ramp.elements[0].position = 0.0159
    colorramp.color_ramp.elements[0].color = edge_color # (0.0369, 0.0086, 0.0, 1.0)
    colorramp.color_ramp.elements[1].position = 0.0716
    colorramp.color_ramp.elements[1].color = basic_color # (0.1119, 0.2122, 0.008, 1.0)
    colorramp.color_ramp.elements[2].position = 1.0
    colorramp.color_ramp.elements[2].color = basic_color # (0.1119, 0.2122, 0.008, 1.0)
    
    hue_saturation_value = nw.new_node('ShaderNodeHueSaturation',
        input_kwargs={'Hue': map_range_1.outputs["Result"], 'Value': map_range_2.outputs["Result"], 'Color': colorramp.outputs["Color"]})
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': hue_saturation_value, 'Roughness': 0.90})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

@node_utils.to_nodegroup('nodegroup_coconut_calyx', singleton=False, type='GeometryNodeTree')
def nodegroup_coconut_calyx(nw: NodeWrangler, basic_color, edge_color):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'width', 0.5),
            ('NodeSocketInt', 'resolution', 128),
            ('NodeSocketFloatDistance', 'radius', 1.0),
            ('NodeSocketInt', 'subdivision', 5),
            ('NodeSocketFloat', 'bump displacement', 0.16),
            ('NodeSocketFloat', 'bump scale', 3.22),])
    
    curve_circle = nw.new_node(Nodes.CurveCircle,
        input_kwargs={'Resolution': group_input.outputs["resolution"], 'Radius': group_input.outputs["radius"]})
    
    jigsaw = nw.new_node(nodegroup_jigsaw().name,
        input_kwargs={'Value': group_input.outputs["width"], 'noise scale': 30.22})
    
    scale_mesh = nw.new_node(nodegroup_scale_mesh().name,
        input_kwargs={'Geometry': curve_circle.outputs["Curve"], 'Scale': jigsaw},
        label='ScaleMesh')
    
    spline_parameter_1 = nw.new_node(Nodes.SplineParameter)
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.5
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: spline_parameter_1.outputs["Factor"], 1: value},
        attrs={'operation': 'SUBTRACT'})
    
    absolute = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract},
        attrs={'operation': 'ABSOLUTE'})
    
    map_range_2 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': absolute, 1: value, 2: group_input.outputs["width"]})
    
    float_curve = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': map_range_2.outputs["Result"]})
    node_utils.assign_curve(float_curve.mapping.curves[0], [(0.0, 0.0), (0.2409, 0.0), (0.7068, 0.275), (1.0, 0.9781)])
    
    scale_mesh_1 = nw.new_node(nodegroup_scale_mesh().name,
        input_kwargs={'Geometry': scale_mesh, 'Scale': float_curve},
        label='ScaleMesh')
    
    fill_curve = nw.new_node(Nodes.FillCurve,
        input_kwargs={'Curve': scale_mesh_1},
        attrs={'mode': 'NGONS'})
    
    subdivide_mesh = nw.new_node(Nodes.SubdivideMesh,
        input_kwargs={'Mesh': fill_curve, 'Level': group_input.outputs["subdivision"]})
    
    surfacebump = nw.new_node(nodegroup_surface_bump().name,
        input_kwargs={'Geometry': subdivide_mesh, 'Displacement': group_input.outputs["bump displacement"], 'Scale': group_input.outputs["bump scale"]})
    
    set_material = nw.new_node(Nodes.SetMaterial,
        input_kwargs={'Geometry': surfacebump, 'Material': surface.shaderfunc_to_material(shader_coconut_calyx_shader, basic_color, edge_color)})
    
    geometry_proximity = nw.new_node(Nodes.Proximity,
        input_kwargs={'Target': fill_curve},
        attrs={'target_element': 'EDGES'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_material, 'distance to edge': geometry_proximity.outputs["Distance"]})

@node_utils.to_nodegroup('nodegroup_coconut_stem', singleton=False, type='GeometryNodeTree')
def nodegroup_coconut_stem(nw: NodeWrangler, basic_color=(0.1119, 0.2122, 0.008, 1.0), edge_color=(0.0369, 0.0086, 0.0, 1.0)):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[
            ('NodeSocketGeometry', 'Target', None),
            ('NodeSocketFloat', 'radius', 0.0),
            ('NodeSocketVectorTranslation', 'Translation', (0.0, 0.0, 1.08)),
            ('NodeSocketInt', 'Count', 6),
            ('NodeSocketFloat', 'base scale', 0.3),
            ('NodeSocketFloat', 'top scale', 0.24),
            ('NodeSocketFloat', 'attach threshold', 0.1),
            ('NodeSocketFloat', 'attach multiplier', 10.0),
            ('NodeSocketFloat', 'calyx width', 0.5),
            ('NodeSocketVectorTranslation', 'stem_mid', (0.0, 0.0, 1.0)),
            ('NodeSocketVectorTranslation', 'stem_end', (0.0, 0.0, 1.0)),
            ('NodeSocketFloat', 'stem_radius', 0.5),
            ])

    coconutcalyx = nw.new_node(nodegroup_coconut_calyx(basic_color=basic_color, 
                                                        edge_color=edge_color).name,
        input_kwargs={'width': group_input.outputs['calyx width']})
    
    capture_attribute_1 = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': coconutcalyx.outputs["Geometry"], 2: coconutcalyx.outputs["distance to edge"]})
    
    spiral = nw.new_node('GeometryNodeCurveSpiral',
        input_kwargs={'Rotations': 1.0, 'Start Radius': group_input.outputs["radius"], 'End Radius': group_input.outputs["radius"], 'Height': 0.0})
    
    spline_parameter = nw.new_node(Nodes.SplineParameter)
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': spiral, 2: spline_parameter.outputs["Factor"]})
    
    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': capture_attribute.outputs["Geometry"], 'Translation': group_input.outputs["Translation"]})
    
    curve_to_points = nw.new_node(Nodes.CurveToPoints,
        input_kwargs={'Curve': transform, 'Count': group_input.outputs["Count"]})
    
    align_euler_to_vector = nw.new_node(Nodes.AlignEulerToVector,
        input_kwargs={'Rotation': curve_to_points.outputs["Rotation"]},
        attrs={'axis': 'Z'})
    
    map_range_2 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': capture_attribute.outputs[2], 3: group_input.outputs["base scale"], 4: group_input.outputs["top scale"]},
        attrs={'interpolation_type': 'SMOOTHERSTEP'})
    
    instance_on_points = nw.new_node(Nodes.InstanceOnPoints,
        input_kwargs={'Points': curve_to_points.outputs["Points"], 'Instance': capture_attribute_1.outputs["Geometry"], 'Rotation': align_euler_to_vector, 'Scale': map_range_2.outputs["Result"]})
    
    realize_instances = nw.new_node(Nodes.RealizeInstances,
        input_kwargs={'Geometry': instance_on_points})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': capture_attribute.outputs[2], 4: 0.01})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Z': map_range_1.outputs["Result"]})
    
    attachtonearest = nw.new_node(nodegroup_attach_to_nearest().name,
        input_kwargs={'Geometry': realize_instances, 'Target': group_input.outputs["Target"], 'threshold': group_input.outputs["attach threshold"], 'multiplier': group_input.outputs["attach multiplier"], 'Offset': combine_xyz})

    basicstem = nw.new_node(nodegroup_basic_stem(basic_color).name,
        input_kwargs={'cross_radius': group_input.outputs['stem_radius'], 
            'quad_mid': group_input.outputs['stem_mid'], 
            'quad_end': group_input.outputs['stem_end'], 
            'Translation': (0.0, 0.0, 0.98), 
            'Scale': (1.0, 1.0, 1.0)})

    join_geometry = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [basicstem, attachtonearest]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': join_geometry, 'distance to edge': capture_attribute_1.outputs[2]})

### pineapple ###
def shader_leaf(nw: NodeWrangler, basic_color):
    # Code generated using version 2.4.3 of the node_transpiler

    texture_coordinate_1 = nw.new_node(Nodes.TextureCoord)
    
    noise_texture_1 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': texture_coordinate_1.outputs["Object"], 'Scale': 3.48, 'Detail': 10.0, 'Roughness': 0.7})
    
    separate_rgb = nw.new_node(Nodes.SeparateColor,
        input_kwargs={'Color': noise_texture_1.outputs["Color"]})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_rgb.outputs["Green"], 1: 0.4, 2: 0.7, 3: 0.48, 4: 0.55},
        attrs={'interpolation_type': 'SMOOTHSTEP'})
    
    map_range_3 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_rgb.outputs["Red"], 1: 0.52, 2: 0.48, 3: 0.32, 4: 0.74},
        attrs={'interpolation_type': 'SMOOTHSTEP'})
    
    map_range_2 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_rgb.outputs["Blue"], 1: 0.4, 2: 0.7, 3: 0.94, 4: 1.1},
        attrs={'interpolation_type': 'SMOOTHSTEP'})
    
    hue_saturation_value = nw.new_node('ShaderNodeHueSaturation',
        input_kwargs={'Hue': map_range_1.outputs["Result"], 
        'Saturation': map_range_3.outputs["Result"], 
        'Value': map_range_2.outputs["Result"], 
        'Color': basic_color}) # (0.0545, 0.1981, 0.0409, 1.0)
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': hue_saturation_value, 'Specular': 0.5955, 'Roughness': 1.0})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

@node_utils.to_nodegroup('nodegroup_pineapple_leaf', singleton=False, type='GeometryNodeTree')
def nodegroup_pineapple_leaf(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketIntUnsigned', 'Resolution', 8),
            ('NodeSocketVectorTranslation', 'Start', (0.0, 0.0, 0.0)),
            ('NodeSocketVectorTranslation', 'Middle', (0.0, -0.32, 3.72)),
            ('NodeSocketVectorTranslation', 'End', (0.0, 0.92, 4.32))])
    
    quadratic_bezier_1 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': group_input.outputs["Resolution"], 'Start': group_input.outputs["Start"], 'Middle': group_input.outputs["Middle"], 'End': group_input.outputs["End"]})
    
    spline_parameter_1 = nw.new_node(Nodes.SplineParameter)
    
    float_curve_1 = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': spline_parameter_1.outputs["Factor"]})
    node_utils.assign_curve(float_curve_1.mapping.curves[0], [(0.0, 1.0), (0.6818, 0.5063), (1.0, 0.0)])
    
    set_curve_radius_1 = nw.new_node(Nodes.SetCurveRadius,
        input_kwargs={'Curve': quadratic_bezier_1, 'Radius': float_curve_1})
    
    curve_circle_1 = nw.new_node(Nodes.CurveCircle,
        input_kwargs={'Resolution': group_input.outputs["Resolution"]})
    
    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': curve_circle_1.outputs["Curve"], 'Scale': (0.5, 0.1, 1.0)})
    
    position = nw.new_node(Nodes.InputPosition)
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': position})
    
    absolute = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"]},
        attrs={'operation': 'ABSOLUTE'})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: absolute},
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Y': multiply})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': transform, 'Offset': combine_xyz})
    
    curve_to_mesh_1 = nw.new_node(Nodes.CurveToMesh,
        input_kwargs={'Curve': set_curve_radius_1, 'Profile Curve': set_position, 'Fill Caps': True})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': curve_to_mesh_1})

@node_utils.to_nodegroup('nodegroup_pineapple_crown', singleton=False, type='GeometryNodeTree')
def nodegroup_pineapple_crown(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    spiral_1 = nw.new_node('GeometryNodeCurveSpiral',
        input_kwargs={'Resolution': 10, 'Rotations': 5.0, 'Start Radius': 0.01, 'End Radius': 0.01, 'Height': 0.0})
    
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Leaf', None),
            ('NodeSocketVectorTranslation', 'translation', (0.0, 0.0, 0.7)),
            ('NodeSocketVectorEuler', 'rotation base', (-0.4363, 0.0, 0.0)),
            ('NodeSocketInt', 'number of leaves', 75),
            ('NodeSocketFloat', 'noise amount', 0.1),
            ('NodeSocketFloat', 'noise scale', 50.0),
            ('NodeSocketFloat', 'scale base', 0.4),
            ('NodeSocketFloat', 'scale z base', 0.12),
            ('NodeSocketFloat', 'scale z top', 0.68),
            ('NodeSocketFloat', 'rot z base', -0.64),
            ('NodeSocketFloat', 'rot z top', 0.38)])
    
    transform_4 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': spiral_1, 'Translation': group_input.outputs["translation"]})
    
    resample_curve_1 = nw.new_node(Nodes.ResampleCurve,
        input_kwargs={'Curve': transform_4, 'Count': group_input.outputs["number of leaves"]})
    
    surfacebump = nw.new_node(nodegroup_surface_bump().name,
        input_kwargs={'Geometry': resample_curve_1, 'Displacement': group_input.outputs["noise amount"], 'Scale': group_input.outputs["noise scale"]})
    
    curve_tangent_1 = nw.new_node(Nodes.CurveTangent)
    
    align_euler_to_vector_1 = nw.new_node(Nodes.AlignEulerToVector,
        input_kwargs={'Vector': curve_tangent_1})
    
    rotate_euler_3 = nw.new_node(Nodes.RotateEuler,
        input_kwargs={'Rotation': align_euler_to_vector_1, 'Rotate By': group_input.outputs["rotation base"]},
        attrs={'space': 'LOCAL'})
    
    spline_parameter_2 = nw.new_node(Nodes.SplineParameter)
    
    random_value = nw.new_node(Nodes.RandomValue,
        input_kwargs={2: -0.1, 3: 0.1})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: spline_parameter_2.outputs["Factor"], 1: random_value.outputs[1]})
    
    map_range_2 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': add, 3: 0.2})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': map_range_2.outputs["Result"], 3: group_input.outputs["rot z base"], 4: group_input.outputs["rot z top"]})
    
    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': map_range_1.outputs["Result"]})
    
    rotate_euler_2 = nw.new_node(Nodes.RotateEuler,
        input_kwargs={'Rotation': rotate_euler_3, 'Rotate By': combine_xyz_1},
        attrs={'space': 'LOCAL'})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': map_range_2.outputs["Result"], 3: group_input.outputs["scale z base"], 4: group_input.outputs["scale z top"]},
        attrs={'interpolation_type': 'SMOOTHERSTEP'})
    
    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': group_input.outputs["scale base"], 'Y': map_range.outputs["Result"], 'Z': map_range.outputs["Result"]})
    
    instance_on_points_2 = nw.new_node(Nodes.InstanceOnPoints,
        input_kwargs={'Points': surfacebump, 'Instance': group_input.outputs["Leaf"], 'Rotation': rotate_euler_2, 'Scale': combine_xyz_3})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': instance_on_points_2})

@node_utils.to_nodegroup('nodegroup_pineapple_stem', singleton=False, type='GeometryNodeTree')
def nodegroup_pineapple_stem(nw: NodeWrangler, basic_color):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketIntUnsigned', 'Resolution', 16),
            ('NodeSocketVectorTranslation', 'Start', (0.0, 0.0, 0.0)),
            ('NodeSocketVectorTranslation', 'Middle', (0.0, -0.32, 3.72)),
            ('NodeSocketVectorTranslation', 'End', (0.0, 0.92, 4.32)),
            ('NodeSocketVectorTranslation', 'translation', (0.0, 0.0, 0.7)),
            ('NodeSocketVectorEuler', 'rotation base', (-0.5236, 0.0, 0.0)),
            ('NodeSocketInt', 'number of leaves', 75),
            ('NodeSocketFloat', 'noise amount', 0.1),
            ('NodeSocketFloat', 'noise scale', 20.0),
            ('NodeSocketFloat', 'scale base', 0.5),
            ('NodeSocketFloat', 'scale z base', 0.15),
            ('NodeSocketFloat', 'scale z top', 0.62),
            ('NodeSocketFloat', 'rot z base', -0.62),
            ('NodeSocketFloat', 'rot z top', 0.54)])
    
    pineappleleaf = nw.new_node(nodegroup_pineapple_leaf().name,
        input_kwargs={'Resolution': group_input.outputs["Resolution"], 'Start': group_input.outputs["Start"], 'Middle': group_input.outputs["Middle"], 'End': group_input.outputs["End"]})
    
    set_material_2 = nw.new_node(Nodes.SetMaterial,
        input_kwargs={'Geometry': pineappleleaf, 'Material': surface.shaderfunc_to_material(shader_leaf, basic_color)})
    
    pineapplecrown = nw.new_node(nodegroup_pineapple_crown().name,
        input_kwargs={'Leaf': set_material_2, 'translation': group_input.outputs["translation"], 'rotation base': group_input.outputs["rotation base"], 'noise amount': group_input.outputs["noise amount"], 'noise scale': group_input.outputs["noise scale"], 'scale base': group_input.outputs["scale base"], 'scale z base': group_input.outputs["scale z base"], 'scale z top': group_input.outputs["scale z top"], 'rot z base': group_input.outputs["rot z base"], 'rot z top': group_input.outputs["rot z top"], 'number of leaves': group_input.outputs['number of leaves']})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': pineapplecrown})


