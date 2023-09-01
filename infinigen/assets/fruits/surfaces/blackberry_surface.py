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

from infinigen.assets.fruits.fruit_utils import nodegroup_shape_quadratic, nodegroup_random_rotation_scale, nodegroup_surface_bump, nodegroup_point_on_mesh, nodegroup_instance_on_points
from infinigen.assets.fruits.cross_section_lib import nodegroup_circle_cross_section

def shader_berry_shader(nw: NodeWrangler, berry_color):
    # Code generated using version 2.4.3 of the node_transpiler

    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': berry_color, 'Specular': 0.5705, 'Roughness': 0.2})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

def shader_hair_shader(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    texture_coordinate = nw.new_node(Nodes.TextureCoord)
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': texture_coordinate.outputs["Object"], 'Scale': 0.8, 'Detail': 10.0, 'Roughness': 0.7})
    
    separate_rgb = nw.new_node(Nodes.SeparateColor,
        input_kwargs={'Color': noise_texture.outputs["Color"]},
        attrs={'mode': 'HSV'})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_rgb.outputs["Green"], 1: 0.4, 2: 0.7, 3: 0.48, 4: 0.55},
        attrs={'interpolation_type': 'SMOOTHSTEP'})
    
    map_range_2 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_rgb.outputs["Blue"], 1: 0.4, 2: 0.7, 3: 0.4},
        attrs={'interpolation_type': 'SMOOTHSTEP'})
    
    hue_saturation_value = nw.new_node('ShaderNodeHueSaturation',
        input_kwargs={'Hue': map_range_1.outputs["Result"], 'Value': map_range_2.outputs["Result"], 'Color': (0.6939, 0.2307, 0.0529, 1.0)})
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': hue_saturation_value})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

@node_utils.to_nodegroup('nodegroup_blackberry_surface', singleton=False, type='GeometryNodeTree')
def nodegroup_blackberry_surface(nw: NodeWrangler, berry_color=(0.0212, 0.0212, 0.0284, 1.0)):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None), 
                    ('NodeSocketFloat', 'spline parameter', 0.5)])

    surfacebump = nw.new_node(nodegroup_surface_bump().name,
        input_kwargs={'Geometry': group_input.outputs['Geometry'], 'Displacement': 0.5, 'Scale': 0.5})

    pointonmesh = nw.new_node(nodegroup_point_on_mesh().name,
        input_kwargs={'Mesh': surfacebump, 'Distance Min': 0.4, 'spline parameter': group_input.outputs['spline parameter'], 'noise amount': 0.5, 'noise scale': 2.0})
    
    randomrotationscale = nw.new_node(nodegroup_random_rotation_scale().name,
        input_kwargs={'rot mean': (3.89, 0.0, 0.0)})
    
    uv_sphere_2 = nw.new_node(Nodes.MeshUVSphere,
        input_kwargs={'Segments': 32, 'Rings': 16})
    
    surfacebump_1 = nw.new_node(nodegroup_surface_bump().name,
        input_kwargs={'Geometry': uv_sphere_2, 'Displacement': 0.5, 'Scale': 0.3})
    
    subdivision_surface = nw.new_node(Nodes.SubdivisionSurface,
        input_kwargs={'Mesh': surfacebump_1})
    
    set_material = nw.new_node(Nodes.SetMaterial,
        input_kwargs={'Geometry': subdivision_surface, 'Material': surface.shaderfunc_to_material(shader_berry_shader, berry_color)})
    
    circlecrosssection_1 = nw.new_node(nodegroup_circle_cross_section().name,
        input_kwargs={'noise amount': 0.0, 'Resolution': 8, 'radius': 0.15})
    
    shapequadratic_1 = nw.new_node(nodegroup_shape_quadratic().name,
        input_kwargs={'Profile Curve': circlecrosssection_1, 'random seed tilt': 0.0, 'noise scale tilt': 0.0, 'noise amount tilt': 0.0, 'noise scale pos': 1.0, 
        'noise amount pos': 2.0, 'Resolution': 8, 'Start': (0.0, 0.0, 0.0),  
        'Middle': (0.0, 0.0, -1.0), 'End': (0.0, 0.0, -2.0)})
    
    value_4 = nw.new_node(Nodes.Value)
    value_4.outputs[0].default_value = 0.2
    
    transform_3 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': shapequadratic_1, 'Translation': (0.0, 0.0, -1.0), 'Scale': value_4})
    
    set_material_3 = nw.new_node(Nodes.SetMaterial,
        input_kwargs={'Geometry': transform_3, 'Material': surface.shaderfunc_to_material(shader_hair_shader)})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [set_material, set_material_3]})
    
    instanceonpoints = nw.new_node(nodegroup_instance_on_points().name,
        input_kwargs={'rotation base': pointonmesh.outputs["Rotation"], 'rotation delta': randomrotationscale.outputs["Vector"], 'translation': (0.0, -0.5, 0.0),
        'scale': randomrotationscale.outputs["Value"], 'Points': pointonmesh.outputs["Geometry"], 'Instance': join_geometry})

    realize_instances = nw.new_node(Nodes.RealizeInstances,
        input_kwargs={'Geometry': instanceonpoints})

    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': realize_instances})
