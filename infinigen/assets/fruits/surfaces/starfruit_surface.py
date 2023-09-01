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

from infinigen.assets.fruits.fruit_utils import nodegroup_surface_bump, nodegroup_add_dent

def shader_starfruit_shader(nw: NodeWrangler, base_color, ridge_color):
    # Code generated using version 2.4.3 of the node_transpiler

    attribute = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'star parameters'})
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': attribute.outputs["Color"]})
    colorramp.color_ramp.elements.new(0)
    colorramp.color_ramp.elements.new(0)
    colorramp.color_ramp.elements[0].position = 0.0
    colorramp.color_ramp.elements[0].color = base_color
    colorramp.color_ramp.elements[1].position = 0.9
    colorramp.color_ramp.elements[1].color = base_color
    colorramp.color_ramp.elements[2].position = 0.95
    colorramp.color_ramp.elements[2].color = ridge_color
    colorramp.color_ramp.elements[3].position = 1.0
    colorramp.color_ramp.elements[3].color = base_color
    
    translucent_bsdf = nw.new_node(Nodes.TranslucentBSDF,
        input_kwargs={'Color': colorramp.outputs["Color"]})
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': colorramp.outputs["Color"], 'Specular': 0.775, 'Roughness': 0.2})
    
    mix_shader = nw.new_node(Nodes.MixShader,
        input_kwargs={'Fac': 0.7, 1: translucent_bsdf, 2: principled_bsdf})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': mix_shader})


@node_utils.to_nodegroup('nodegroup_starfruit_surface', singleton=False, type='GeometryNodeTree')
def nodegroup_starfruit_surface(nw: NodeWrangler, 
    dent_control_points=[(0.0, 0.4219), (0.0977, 0.4469), (0.2273, 0.4844), (0.5568, 0.5125), (1.0, 0.5)], 
    base_color=(0.7991, 0.6038, 0.0009, 1.0), 
    ridge_color=(0.3712, 0.4179, 0.0006, 1.0)):

    # Code generated using version 2.4.3 of the node_transpiler
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketFloat', 'spline parameter', 0.0),
            ('NodeSocketVector', 'spline tangent', (0.0, 0.0, 0.0)),
            ('NodeSocketFloat', 'distance to center', 0.0),
            ('NodeSocketFloat', 'dent intensity', 1.0)
        ])

    adddent = nw.new_node(nodegroup_add_dent(dent_control_points=dent_control_points).name,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 
        'spline parameter': group_input.outputs["spline parameter"], 
        'spline tangent': group_input.outputs["spline tangent"], 
        'distance to center': group_input.outputs["distance to center"],
        'intensity': group_input.outputs["dent intensity"]
        })
    
    surfacebump_002 = nw.new_node(nodegroup_surface_bump().name,
        input_kwargs={'Geometry': adddent, 'Displacement': 0.03, 'Scale': 10.0})
    
    set_material = nw.new_node(Nodes.SetMaterial,
        input_kwargs={'Geometry': surfacebump_002, 
        'Material': surface.shaderfunc_to_material(shader_starfruit_shader, base_color, ridge_color)})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_material})

    