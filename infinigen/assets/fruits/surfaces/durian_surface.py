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

from infinigen.assets.fruits.fruit_utils import nodegroup_manhattan, nodegroup_point_on_mesh, nodegroup_surface_bump

def shader_durian_shader(nw: NodeWrangler, peak_color, base_color):
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
        input_kwargs={'Value': separate_rgb.outputs["Blue"], 1: 0.4, 2: 0.7, 3: 0.6},
        attrs={'interpolation_type': 'SMOOTHSTEP'})
    
    attribute = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'durian thorn coordiante'})
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': attribute.outputs["Fac"]})
    colorramp.color_ramp.elements[0].position = 0.0
    colorramp.color_ramp.elements[0].color = peak_color
    colorramp.color_ramp.elements[1].position = 0.2705
    colorramp.color_ramp.elements[1].color = base_color
    
    hue_saturation_value = nw.new_node('ShaderNodeHueSaturation',
        input_kwargs={'Hue': map_range_1.outputs["Result"], 'Value': map_range_2.outputs["Result"], 'Color': colorramp.outputs["Color"]})
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': hue_saturation_value, 'Specular': 0.1205, 'Roughness': 0.5068})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

@node_utils.to_nodegroup('nodegroup_durian_surface', singleton=False, type='GeometryNodeTree')
def nodegroup_durian_surface(nw: NodeWrangler, thorn_control_points=[(0.0, 0.0), (0.7318, 0.4344), (1.0, 1.0)], 
    peak_color=(0.2401, 0.1455, 0.0313, 1.0), base_color=(0.3278, 0.3005, 0.0704, 1.0)):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketFloat', 'displacement', 1.0),
            ('NodeSocketFloat', 'spline parameter', 0.0),
            ('NodeSocketFloatDistance', 'distance Min', 0.1),
            ('NodeSocketFloat', 'noise amount', 0.3),
            ('NodeSocketFloat', 'noise scale', 5.0)])

    surfacebump = nw.new_node(nodegroup_surface_bump().name,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Displacement': 0.5, 'Scale': 0.5})
    
    normal = nw.new_node(Nodes.InputNormal)
    
    pointonmesh = nw.new_node(nodegroup_point_on_mesh().name,
        input_kwargs={'Mesh': surfacebump, 'spline parameter': group_input.outputs["spline parameter"], 'Distance Min': group_input.outputs["distance Min"], 'noise amount': group_input.outputs["noise amount"], 'noise scale': group_input.outputs["noise scale"]})
    
    position_1 = nw.new_node(Nodes.InputPosition)
    
    geometry_proximity = nw.new_node(Nodes.Proximity,
        input_kwargs={'Target': pointonmesh.outputs["Geometry"], 'Source Position': position_1},
        attrs={'target_element': 'POINTS'})
    
    manhattan = nw.new_node(nodegroup_manhattan().name,
        input_kwargs={'v1': geometry_proximity.outputs["Position"], 'v2': position_1},
        label='manhattan')
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["distance Min"], 1: 2.0},
        attrs={'operation': 'MULTIPLY'})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': manhattan, 2: multiply, 3: 1.0, 4: 0.0})
    
    float_curve = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': map_range.outputs["Result"]})
    node_utils.assign_curve(float_curve.mapping.curves[0], thorn_control_points)
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: normal, 'Scale': float_curve},
        attrs={'operation': 'SCALE'})
    
    scale_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: scale.outputs["Vector"], 'Scale': group_input.outputs["displacement"]},
        attrs={'operation': 'SCALE'})
    
    set_position_1 = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': surfacebump, 'Offset': scale_1.outputs["Vector"]})
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': set_position_1, 2: map_range.outputs["Result"]})
    
    set_material = nw.new_node(Nodes.SetMaterial,
        input_kwargs={'Geometry': capture_attribute.outputs["Geometry"], 
        'Material': surface.shaderfunc_to_material(shader_durian_shader, peak_color, base_color)})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_material, 'distance to center': capture_attribute.outputs[2]})