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

from infinigen.assets.fruits.fruit_utils import nodegroup_point_on_mesh, nodegroup_random_rotation_scale, nodegroup_surface_bump, nodegroup_instance_on_points
from infinigen.assets.fruits.cross_section_lib import nodegroup_circle_cross_section
from infinigen.assets.fruits.stem_lib import nodegroup_pineapple_leaf

@node_utils.to_nodegroup('nodegroup_pineapple_surface', singleton=False, type='GeometryNodeTree')
def nodegroup_pineapple_surface(nw: NodeWrangler, 
        color_bottom=(0.0823, 0.0953, 0.0097, 1.0), 
        color_mid=(0.552, 0.1845, 0.0222, 1.0), 
        color_top= (0.4508, 0.0999, 0.0003, 1.0), 
        color_center=(0.8388, 0.5395, 0.314, 1.0)):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketFloat', 'spline parameter', 0.0),
            ('NodeSocketFloatDistance', 'point distance', 0.22),
            ('NodeSocketFloat', 'cell scale', 0.2),
            ('NodeSocketFloat', 'random seed', 0.0)])
    
    pointonmesh = nw.new_node(nodegroup_point_on_mesh().name,
        input_kwargs={'Mesh': group_input.outputs["Geometry"], 'spline parameter': group_input.outputs["spline parameter"], 'Distance Min': group_input.outputs["point distance"], 'parameter max': 0.999, 'noise amount': 0.05})
    
    randomrotationscale = nw.new_node(nodegroup_random_rotation_scale().name,
        input_kwargs={'random seed': group_input.outputs["random seed"], ' rot std z': 0.3, 'scale mean': group_input.outputs["cell scale"]})
    
    pineapplecellbody = nw.new_node(nodegroup_pineapple_cell_body().name,
        input_kwargs={'resolution': 16, 'scale diff': -0.3})
    
    set_material = nw.new_node(Nodes.SetMaterial,
        input_kwargs={'Geometry': pineapplecellbody.outputs["Geometry"], 
        'Material': surface.shaderfunc_to_material(shader_cell, color_bottom, color_mid, color_top, color_center)})
    
    pineappleleaf = nw.new_node(nodegroup_pineapple_leaf().name,
        input_kwargs={'Middle': (0.0, -0.1, 1.0), 'End': (0.0, 0.9, 2.5)})
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.3
    
    transform_2 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': pineappleleaf, 'Translation': (0.0, -0.1, 0.3), 'Rotation': (-1.0315, 0.0, 0.0), 'Scale': value})
    
    set_material_3 = nw.new_node(Nodes.SetMaterial,
        input_kwargs={'Geometry': transform_2, 'Material': surface.shaderfunc_to_material(shader_needle, color_center, color_top)})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [set_material, set_material_3]})
    
    surfacebump = nw.new_node(nodegroup_surface_bump().name,
        input_kwargs={'Geometry': join_geometry, 'Displacement': 0.2, 'Scale': 10.0})
    
    instanceonpoints = nw.new_node(nodegroup_instance_on_points().name,
        input_kwargs={'rotation base': pointonmesh.outputs["Rotation"], 'rotation delta': randomrotationscale.outputs["Vector"], 'translation': (0.0, 0.0, 0.0), 'scale': randomrotationscale.outputs["Value"], 'Points': pointonmesh.outputs["Geometry"], 'Instance': surfacebump})
    
    set_material_1 = nw.new_node(Nodes.SetMaterial,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 
        'Material': surface.shaderfunc_to_material(shader_cell, color_bottom, color_mid, color_top, color_center)})
    
    join_geometry_1 = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [instanceonpoints, set_material_1]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': join_geometry_1, 'spline parameter': pineapplecellbody.outputs["spline parameter"]})


def shader_needle(nw: NodeWrangler, color1, color2):
    # Code generated using version 2.4.3 of the node_transpiler

    texture_coordinate = nw.new_node(Nodes.TextureCoord)
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': texture_coordinate.outputs["Object"], 'Scale': 8.0, 'Detail': 0.0})
    
    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': noise_texture.outputs["Fac"], 
        'Color1': color1,  # (0.7758, 0.4678, 0.2346, 1.0)
        'Color2': color2}) # (0.3467, 0.0595, 0.0, 1.0)
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': mix})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})
        

def shader_cell(nw: NodeWrangler, color_bottom, color_mid, color_top, color_center):
    # Code generated using version 2.4.3 of the node_transpiler

    texture_coordinate = nw.new_node(Nodes.TextureCoord)
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': texture_coordinate.outputs["Object"], 'Scale': 4.6})
    
    attribute = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'radius'})
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': attribute.outputs["Fac"]})
    colorramp.color_ramp.elements.new(0)
    colorramp.color_ramp.elements.new(0)
    colorramp.color_ramp.elements[0].position = 0.0
    colorramp.color_ramp.elements[0].color = color_bottom # (0.0823, 0.0953, 0.0097, 1.0)
    colorramp.color_ramp.elements[1].position = 0.67
    colorramp.color_ramp.elements[1].color = color_mid # (0.552, 0.1845, 0.0222, 1.0)
    colorramp.color_ramp.elements[2].position = 0.93
    colorramp.color_ramp.elements[2].color = color_top # (0.4508, 0.0999, 0.0003, 1.0)
    colorramp.color_ramp.elements[3].position = 1.0
    colorramp.color_ramp.elements[3].color = color_center # (0.8388, 0.5395, 0.314, 1.0)
    
    hue_saturation_value = nw.new_node('ShaderNodeHueSaturation',
        input_kwargs={'Hue': 0.55, 'Color': colorramp.outputs["Color"]})
    
    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': noise_texture.outputs["Fac"], 'Color1': hue_saturation_value, 'Color2': colorramp.outputs["Color"]})
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': mix, 'Roughness': 0.2})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

@node_utils.to_nodegroup('nodegroup_pineapple_cell_body', singleton=False, type='GeometryNodeTree')
def nodegroup_pineapple_cell_body(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketInt', 'resolution', 0),
            ('NodeSocketFloat', 'scale diff', 0.0)])
    
    quadratic_bezier = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': group_input.outputs["resolution"], 'Start': (0.0, 0.0, 0.0), 'Middle': (0.0, 0.0, 0.2), 'End': (0.0, 0.0, 0.4)})
    
    spline_parameter = nw.new_node(Nodes.SplineParameter)
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': quadratic_bezier, 2: spline_parameter.outputs["Factor"]})
    
    float_curve = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': spline_parameter.outputs["Factor"]})
    node_utils.assign_curve(float_curve.mapping.curves[0], [(0.0, 1.0), (0.1568, 0.875), (0.8045, 0.5313), (1.0, 0.0)])
    
    set_curve_radius = nw.new_node(Nodes.SetCurveRadius,
        input_kwargs={'Curve': capture_attribute.outputs["Geometry"], 'Radius': float_curve})
    
    circlecrosssection = nw.new_node(nodegroup_circle_cross_section().name,
        input_kwargs={'noise scale': 8.0, 'noise amount': 0.4, 'Resolution': 64, 'radius': 1.0})
    
    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
        input_kwargs={'Curve': set_curve_radius, 'Profile Curve': circlecrosssection})
    
    position_1 = nw.new_node(Nodes.InputPosition)
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': position_1})
    
    greater_than = nw.new_node(Nodes.Compare,
        input_kwargs={0: separate_xyz.outputs["Y"]})
    
    multiply = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: position_1, 1: group_input.outputs["scale diff"]},
        attrs={'operation': 'MULTIPLY'})
    
    set_position_1 = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': curve_to_mesh, 'Selection': greater_than, 'Offset': multiply.outputs["Vector"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_position_1, 'spline parameter': capture_attribute.outputs[2]})