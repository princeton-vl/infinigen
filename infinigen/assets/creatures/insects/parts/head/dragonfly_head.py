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

from infinigen.assets.creatures.insects.utils.geom_utils import nodegroup_add_hair, nodegroup_attach_part, nodegroup_symmetric_clone, nodegroup_surface_bump
from infinigen.assets.creatures.insects.parts.mouth.dragonfly_mouth import nodegroup_dragonfly_mouth
from infinigen.assets.creatures.insects.parts.eye.dragonfly_eye import nodegroup_dragonfly_eye
from infinigen.assets.creatures.insects.parts.antenna.dragonfly_antenna import nodegroup_dragonfly_antenna
from infinigen.assets.creatures.insects.parts.hair.principled_hair import nodegroup_principled_hair 
from infinigen.assets.creatures.insects.utils.shader_utils import nodegroup_color_noise, shader_black_w_noise_shader

def shader_dragonfly_head_shader(nw: NodeWrangler, base_color, v):
    # Code generated using version 2.4.3 of the node_transpiler

    attribute = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'pos'})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': attribute.outputs["Vector"]})
    
    absolute = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"]},
        attrs={'operation': 'ABSOLUTE'})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': separate_xyz.outputs["X"], 'Y': absolute, 'Z': separate_xyz.outputs["Z"]})
    
    musgrave_texture = nw.new_node(Nodes.MusgraveTexture,
        input_kwargs={'Vector': combine_xyz, 'W': 28.0, 'Scale': 2.0, 'Detail': 1.0},
        attrs={'musgrave_dimensions': '4D'})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': musgrave_texture, 1: -0.28, 2: 0.48})
    
    rgb = nw.new_node(Nodes.RGB)
    rgb.outputs[0].default_value = base_color
    
    hue_saturation_value = nw.new_node('ShaderNodeHueSaturation',
        input_kwargs={'Value': v, 'Color': rgb})
    
    group = nw.new_node(nodegroup_color_noise().name,
        input_kwargs={'Scale': 1.34, 'Color': rgb, 'Value From Max': 0.7, 'Value To Min': 0.18})
    
    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': map_range.outputs["Result"], 'Color1': hue_saturation_value, 'Color2': group})
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': mix, 'Specular': 0.7545, 'Roughness': 0.0636})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

@node_utils.to_nodegroup('nodegroup_dragon_fly_head', singleton=False, type='GeometryNodeTree')
def nodegroup_dragon_fly_head(nw: NodeWrangler,
        base_color=(0.2789, 0.3864, 0.0319, 1.0), 
        eye_color=(0.2789, 0.3864, 0.0319, 1.0),
        v=0.3):
    # Code generated using version 2.4.3 of the node_transpiler

    curve_line = nw.new_node(Nodes.CurveLine,
        input_kwargs={'End': (1.8, 0.0, 0.0)})
    
    resample_curve = nw.new_node(Nodes.ResampleCurve,
        input_kwargs={'Curve': curve_line, 'Count': 32})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': resample_curve})
    
    spline_parameter_1 = nw.new_node(Nodes.SplineParameter)
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': set_position, 2: spline_parameter_1.outputs["Factor"]})
    
    float_curve_1 = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': capture_attribute.outputs[2]})
    node_utils.assign_curve(float_curve_1.mapping.curves[0], [(0.0, 0.14), (0.3055, 0.93), (0.7018, 0.79), (0.9236, 0.455), (1.0, 0.0)])
    
    set_curve_radius = nw.new_node(Nodes.SetCurveRadius,
        input_kwargs={'Curve': capture_attribute.outputs["Geometry"], 'Radius': float_curve_1})
    
    curve_circle = nw.new_node(Nodes.CurveCircle,
        input_kwargs={'Resolution': 200, 'Radius': 1.1})
    
    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
        input_kwargs={'Curve': set_curve_radius, 'Profile Curve': curve_circle.outputs["Curve"], 'Fill Caps': True})
    
    set_material_1 = nw.new_node(Nodes.SetMaterial,
        input_kwargs={'Geometry': curve_to_mesh, 'Material': surface.shaderfunc_to_material(shader_dragonfly_head_shader, base_color, v)})
    
    leghair = nw.new_node(nodegroup_principled_hair().name,
        input_kwargs={'Resolution': 2})
    
    transform_3 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': leghair, 'Scale': (1.0, 1.0, 5.0)})
    
    set_material_2 = nw.new_node(Nodes.SetMaterial,
        input_kwargs={'Geometry': transform_3, 'Material': surface.shaderfunc_to_material(shader_black_w_noise_shader)})
    
    addhair = nw.new_node(nodegroup_add_hair().name,
        input_kwargs={'Mesh': set_material_1, 'Hair': set_material_2, 'Density': 500.0, 'rot mean': (0.36, 0.0, 0.0), 'scale mean': 0.01})
    
    reroute = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': addhair})
    
    dragonflyeye = nw.new_node(nodegroup_dragonfly_eye(base_color=eye_color, v=0.0).name,
        input_kwargs={'Rings': 128})
    
    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = 0.6
    
    transform_1 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': dragonflyeye, 'Scale': value_1})
    
    attach_part = nw.new_node(nodegroup_attach_part().name,
        input_kwargs={'Skin Mesh': reroute, 'Skeleton Curve': set_position, 'Geometry': transform_1, 'Length Fac': 0.5625, 'Ray Rot': (1.5474, -0.3944, 1.4556), 'Rad': 0.64, 'Part Rot': (27.1, 0.0, 0.0)})
    
    symmetric_clone = nw.new_node(nodegroup_symmetric_clone().name,
        input_kwargs={'Geometry': attach_part.outputs["Geometry"]})
    
    dragonflymouth = nw.new_node(nodegroup_dragonfly_mouth().name)

    set_material_3 = nw.new_node(Nodes.SetMaterial,
        input_kwargs={'Geometry': dragonflymouth, 'Material': surface.shaderfunc_to_material(shader_dragonfly_head_shader, base_color, v)})
    
    addhair_1 = nw.new_node(nodegroup_add_hair().name,
        input_kwargs={'Mesh': set_material_3, 'Hair': set_material_2, 'Density': 5.0, 'rot mean': (-0.04, 0.0, 0.0), 'scale mean': 0.1})
    
    surfacebump = nw.new_node(nodegroup_surface_bump().name,
        input_kwargs={'Geometry': addhair_1, 'Displacement': 0.05, 'Scale': 5.0})
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.07
    
    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': surfacebump, 'Scale': value})
    
    attach_part_1 = nw.new_node(nodegroup_attach_part().name,
        input_kwargs={'Skin Mesh': reroute, 'Skeleton Curve': resample_curve, 'Geometry': transform, 'Length Fac': 0.9667, 'Part Rot': (0.0, 31.5, 0.0), 'Do Normal Rot': True})
    
    antenna = nw.new_node(nodegroup_dragonfly_antenna().name,
        input_kwargs={'length_rad1_rad2': (1.24, 0.05, 0.04), 'angles_deg': (0.0, -31.0, 0.0)})
    
    surfacebump_1 = nw.new_node(nodegroup_surface_bump().name,
        input_kwargs={'Geometry': antenna.outputs["Geometry"], 'Scale': 5.0})
    
    set_material = nw.new_node(Nodes.SetMaterial,
        input_kwargs={'Geometry': surfacebump_1, 'Material': surface.shaderfunc_to_material(shader_black_w_noise_shader)})
    
    value_2 = nw.new_node(Nodes.Value)
    value_2.outputs[0].default_value = 0.48
    
    transform_2 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': set_material, 'Translation': (-0.02, 0.0, 0.0), 'Scale': value_2})
    
    attach_part_2 = nw.new_node(nodegroup_attach_part().name,
        input_kwargs={'Skin Mesh': reroute, 'Skeleton Curve': resample_curve, 'Geometry': transform_2, 'Length Fac': 0.6408, 'Ray Rot': (1.9722, -1.4364, 1.5708), 'Rad': 0.9, 'Part Rot': (108.1, -49.8, 26.7)})
    
    symmetric_clone_1 = nw.new_node(nodegroup_symmetric_clone().name,
        input_kwargs={'Geometry': attach_part_2.outputs["Geometry"]})
    
    join_geometry_1 = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [symmetric_clone.outputs["Both"], reroute, attach_part_1.outputs["Geometry"], symmetric_clone_1.outputs["Both"]]})
    
    position = nw.new_node(Nodes.InputPosition)
    
    store_named_attribute = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': join_geometry_1, 'Name': 'pos', "Value": position},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': store_named_attribute})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': join_geometry})