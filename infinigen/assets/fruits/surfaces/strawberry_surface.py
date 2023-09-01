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

from infinigen.assets.fruits.seed_lib import nodegroup_strawberry_seed
from infinigen.assets.fruits.fruit_utils import nodegroup_point_on_mesh, nodegroup_add_crater, nodegroup_surface_bump, nodegroup_random_rotation_scale, nodegroup_instance_on_points, nodegroup_add_noise_scalar

def shader_strawberry_shader(nw: NodeWrangler, top_pos, main_color, top_color):
    # Code generated using version 2.4.3 of the node_transpiler

    texture_coordinate = nw.new_node(Nodes.TextureCoord)
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': texture_coordinate.outputs["Object"], 'Scale': 0.5})
    
    attribute = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'strawberry seed height'})
    
    colorramp_1 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': attribute.outputs["Color"]})
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements[0].position = 0.0
    colorramp_1.color_ramp.elements[0].color = main_color
    colorramp_1.color_ramp.elements[1].position = top_pos
    colorramp_1.color_ramp.elements[1].color = main_color
    colorramp_1.color_ramp.elements[2].position = 1.0
    colorramp_1.color_ramp.elements[2].color = top_color
    
    hue_saturation_value = nw.new_node('ShaderNodeHueSaturation',
        input_kwargs={'Hue': 0.55, 'Saturation': 1.5, 'Value': 0.2, 'Fac': 0.3, 'Color': colorramp_1.outputs["Color"]})
    
    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': noise_texture.outputs["Fac"], 'Color1': colorramp_1.outputs["Color"], 'Color2': hue_saturation_value})
    
    translucent_bsdf = nw.new_node(Nodes.TranslucentBSDF,
        input_kwargs={'Color': mix})
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': mix, 'Specular': 1.0, 'Roughness': 0.15})
    
    mix_shader = nw.new_node(Nodes.MixShader,
        input_kwargs={'Fac': 0.8, 1: translucent_bsdf, 2: principled_bsdf})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': mix_shader})

@node_utils.to_nodegroup('nodegroup_strawberry_surface', singleton=False, type='GeometryNodeTree')
def nodegroup_strawberry_surface(nw: NodeWrangler, top_pos=0.9, main_color=(0.8879, 0.0097, 0.0319, 1.0), top_color=(0.8148, 0.6105, 0.1746, 1.0)):
    # Code generated using version 2.4.3 of the node_transpiler
    strawberryseed = nw.new_node(nodegroup_strawberry_seed().name)
    
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketFloat', 'spline parameter', 0.0),
            ('NodeSocketFloatDistance', 'Distance Min', 0.12),
            ('NodeSocketFloat', 'Strength', 0.74),
            ('NodeSocketFloat', 'noise random seed', 0.0)])

    surfacebump = nw.new_node(nodegroup_surface_bump().name,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Displacement': 0.4, 'Scale': 0.5})
    
    addnoisescalar = nw.new_node(nodegroup_add_noise_scalar().name,
        input_kwargs={'noise random seed': group_input.outputs["noise random seed"], 
            'value': group_input.outputs["spline parameter"], 
            'noise amount': 0.2})
    
    pointonmesh = nw.new_node(nodegroup_point_on_mesh().name,
        input_kwargs={'Mesh': surfacebump, 'spline parameter': addnoisescalar, 'Distance Min': group_input.outputs["Distance Min"], 'parameter max': top_pos, 'noise amount': 0.1, 'noise scale': 2.0})
    
    addcrater = nw.new_node(nodegroup_add_crater().name,
        input_kwargs={'Geometry': surfacebump, 'Points': pointonmesh.outputs["Geometry"], 'Strength': group_input.outputs["Strength"]})
    
    surfacebump_1 = nw.new_node(nodegroup_surface_bump().name,
        input_kwargs={'Geometry': addcrater, 'Displacement': 0.03, 'Scale': 20.0})
    
    set_material = nw.new_node(Nodes.SetMaterial,
        input_kwargs={'Geometry': surfacebump_1, 'Material': surface.shaderfunc_to_material(shader_strawberry_shader, top_pos, main_color, top_color)})
    
    randomrotationscale = nw.new_node(nodegroup_random_rotation_scale().name,
        input_kwargs={'rot mean': (-1.571, 0.0, 0.0), 'scale mean': 0.08})
    
    instanceonpoints = nw.new_node(nodegroup_instance_on_points().name,
        input_kwargs={'rotation base': pointonmesh.outputs["Rotation"], 'rotation delta': randomrotationscale.outputs["Vector"], 'translation': (0.0, 0.3, 0.0), 'scale': randomrotationscale.outputs["Value"], 'Points': pointonmesh.outputs["Geometry"], 'Instance': strawberryseed})
    
    join_geometry_1 = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [set_material, instanceonpoints]})

    realize_instances = nw.new_node(Nodes.RealizeInstances,
        input_kwargs={'Geometry': join_geometry_1})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': realize_instances, 'curve parameters': addnoisescalar})