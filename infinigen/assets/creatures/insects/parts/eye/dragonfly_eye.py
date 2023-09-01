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

from infinigen.assets.creatures.insects.utils.shader_utils import nodegroup_color_noise

def shader_dragonfly_eye_shader(nw: NodeWrangler, base_color, v):
    # Code generated using version 2.4.3 of the node_transpiler

    musgrave_texture = nw.new_node(Nodes.MusgraveTexture,
        input_kwargs={'Scale': 2.0, 'Detail': 1.0})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': musgrave_texture, 1: -1.0, 2: 0.2})
    
    rgb = nw.new_node(Nodes.RGB)
    rgb.outputs[0].default_value = base_color
    
    hue_saturation_value = nw.new_node('ShaderNodeHueSaturation',
        input_kwargs={'Value': v, 'Color': rgb})
    
    group_1 = nw.new_node(nodegroup_color_noise().name,
        input_kwargs={'Scale': 1.34, 'Color': rgb, 'Value From Max': 0.7, 'Value To Min': 0.18})
    
    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': map_range.outputs["Result"], 'Color1': hue_saturation_value, 'Color2': group_1})
    
    voronoi_texture = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Scale': 1000.0},
        attrs={'feature': 'DISTANCE_TO_EDGE'})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': voronoi_texture.outputs["Distance"], 1: 0.03, 2: 0.2, 3: 1.0, 4: -0.78})
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': mix, 'Specular': map_range_1.outputs["Result"], 'Roughness': 0.0})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

@node_utils.to_nodegroup('nodegroup_dragonfly_eye', singleton=False, type='GeometryNodeTree')
def nodegroup_dragonfly_eye(nw: NodeWrangler,
    base_color=(0.2789, 0.3864, 0.0319, 1.0), 
    v=0.3,
    ):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketInt', 'Rings', 16)])
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Rings"], 1: 2.0},
        attrs={'operation': 'MULTIPLY'})
    
    uv_sphere = nw.new_node(Nodes.MeshUVSphere,
        input_kwargs={'Segments': multiply, 'Rings': group_input.outputs["Rings"]})
    
    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': uv_sphere, 'Scale': (1.0, 1.0, 1.3)})
    
    set_material = nw.new_node(Nodes.SetMaterial,
        input_kwargs={'Geometry': transform, 'Material': surface.shaderfunc_to_material(shader_dragonfly_eye_shader, base_color, v)})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_material})