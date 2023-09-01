# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import bpy
import mathutils
from numpy.random import uniform, normal, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface
from infinigen.assets.materials.utils.surface_utils import clip, sample_range, sample_ratio, sample_color

def shader_eyeball(nw: NodeWrangler, rand=True, coord="X", **input_kwargs):
    # Code generated using version 2.4.3 of the node_transpiler
    texture_coordinate = nw.new_node(Nodes.TextureCoord)
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': texture_coordinate.outputs["Generated"]})
    if coord == "Y":
        math = nw.new_node(Nodes.Math,
            input_kwargs={0: 1.0, 1: separate_xyz.outputs["Y"]},
            attrs={'operation': 'SUBTRACT'})
        val = math
    else:
        val = separate_xyz.outputs["X"]

    colorramp_1 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': val})
    colorramp_1.color_ramp.interpolation = "CONSTANT"
    colorramp_1.color_ramp.elements[0].position = 0.0045
    colorramp_1.color_ramp.elements[0].color = (0.5921, 0.5921, 0.5921, 1.0)
    colorramp_1.color_ramp.elements[1].position = uniform(0.84, 0.88) if rand else 0.854
    colorramp_1.color_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)

    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': val})
    colorramp.color_ramp.interpolation = "CONSTANT"
    colorramp.color_ramp.elements.new(0)
    colorramp.color_ramp.elements[0].position = 0.6618
    colorramp.color_ramp.elements[0].color = (0.2403, 0.217, 0.1528, 1.0)
    colorramp.color_ramp.elements[1].position = colorramp_1.color_ramp.elements[1].position
    colorramp.color_ramp.elements[1].color = (0.4961, 0.8862, 0.1703, 1.0)
    colorramp.color_ramp.elements[2].position = colorramp_1.color_ramp.elements[1].position+0.01
    colorramp.color_ramp.elements[2].color = (0.0, 0.0, 0.0, 1.0)
    if rand:
        sample_color(colorramp.color_ramp.elements[1].color)

    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': colorramp.outputs["Color"], 'Metallic': 0.0, 'Roughness': 0.03})
    
    transparent_bsdf = nw.new_node(Nodes.TransparentBSDF,
        input_kwargs={'Color': (0.757, 0.757, 0.757, 1.0)})
    
    translucent_bsdf = nw.new_node(Nodes.TranslucentBSDF,
        input_kwargs={'Color': (1.0, 1.0, 1.0, 1.0)})
    
    mix_shader_1 = nw.new_node(Nodes.MixShader,
        input_kwargs={'Fac': 0.1, 1: transparent_bsdf, 2: translucent_bsdf})
    
    mix_shader = nw.new_node(Nodes.MixShader,
        input_kwargs={'Fac': colorramp_1.outputs["Color"], 1: principled_bsdf, 2: mix_shader_1})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': mix_shader})

def shader_eyeball_old(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    texture_coordinate = nw.new_node(Nodes.TextureCoord)
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': texture_coordinate.outputs["Generated"]})
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': separate_xyz.outputs["X"]})
    colorramp.color_ramp.interpolation = "CONSTANT"
    colorramp.color_ramp.elements.new(0)
    colorramp.color_ramp.elements[0].position = 0.8982
    colorramp.color_ramp.elements[0].color = color_category('eye_schlera')
    colorramp.color_ramp.elements[1].position = 0.9473
    colorramp.color_ramp.elements[1].color = color_category('eye_pupil')
    colorramp.color_ramp.elements[2].position = 0.9636
    colorramp.color_ramp.elements[2].color = (0.0, 0.0, 0.0, 1.0)
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': colorramp.outputs["Color"], 'Roughness': 0.0})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

def apply(obj, shader_kwargs={}, **kwargs):
    surface.add_material(obj, shader_eyeball, input_kwargs=shader_kwargs)