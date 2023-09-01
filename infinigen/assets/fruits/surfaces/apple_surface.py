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

from infinigen.assets.fruits.fruit_utils import nodegroup_add_dent

def shader_apple_shader(nw: NodeWrangler, color1, color2, random_seed):
    # Code generated using version 2.4.3 of the node_transpiler

    texture_coordinate = nw.new_node(Nodes.TextureCoord)
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = random_seed
    
    add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: texture_coordinate.outputs["Object"], 1: value})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': add.outputs["Vector"]})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: 0.2},
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': separate_xyz.outputs["X"], 'Y': separate_xyz.outputs["Y"], 'Z': multiply})
    
    musgrave_texture_2 = nw.new_node(Nodes.MusgraveTexture,
        input_kwargs={'Vector': combine_xyz, 'Scale': 10.0, 'Detail': 10.0, 'Dimension': 0.3, 'Lacunarity': 3.0})
    
    musgrave_texture = nw.new_node(Nodes.MusgraveTexture,
        input_kwargs={'Vector': add.outputs["Vector"], 'Scale': 0.6, 'Lacunarity': 1.0})
    
    rgb = nw.new_node(Nodes.RGB)
    rgb.outputs[0].default_value = color1 # 
    
    rgb_1 = nw.new_node(Nodes.RGB)
    rgb_1.outputs[0].default_value = color2 # 
    
    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': musgrave_texture, 'Color1': rgb, 'Color2': rgb_1})
    
    hue_saturation_value = nw.new_node('ShaderNodeHueSaturation',
        input_kwargs={'Hue': 0.55, 'Color': mix})
    
    mix_3 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': musgrave_texture_2, 'Color1': mix, 'Color2': hue_saturation_value})
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': mix_3})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

@node_utils.to_nodegroup('nodegroup_apple_surface', singleton=False, type='GeometryNodeTree')
def nodegroup_apple_surface(nw: NodeWrangler, 
        color1=(0.2881, 0.6105, 0.0709, 1.0), 
        color2=(0.7454, 0.6172, 0.0296, 1.0), 
        random_seed=0.0, 
        dent_control_points=[(0.0045, 0.3719), (0.0727, 0.4532), (0.2273, 0.4844), (0.5568, 0.5125), (1.0, 0.5)]):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketFloat', 'spline parameter', 0.0),
            ('NodeSocketVector', 'spline tangent', (0.0, 0.0, 0.0)),
            ('NodeSocketFloat', 'distance to center', 0.0)])
    
    adddent = nw.new_node(nodegroup_add_dent(dent_control_points=dent_control_points).name,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'spline parameter': group_input.outputs["spline parameter"], 'spline tangent': group_input.outputs["spline tangent"], 'distance to center': group_input.outputs["distance to center"], 'intensity': 1.5, 'max radius': 1.5})
    
    adddent_1 = nw.new_node(nodegroup_add_dent(dent_control_points=dent_control_points).name,
        input_kwargs={'Geometry': adddent, 'spline parameter': group_input.outputs["spline parameter"], 'spline tangent': group_input.outputs["spline tangent"], 'distance to center': group_input.outputs["distance to center"], 'bottom': True, 'intensity': -1.0, 'max radius': 1.5})
    
    set_material = nw.new_node(Nodes.SetMaterial,
        input_kwargs={'Geometry': adddent_1, 'Material': surface.shaderfunc_to_material(shader_apple_shader, color1, color2, random_seed)})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_material})


