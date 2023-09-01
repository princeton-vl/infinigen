# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Ankit Goyal, Zeyu Ma, Hongyu Wen

import gin
from numpy.random import uniform

from infinigen.core.nodes.node_wrangler import Nodes
from infinigen.core import surface
from infinigen.core.util.organization import SurfaceTypes
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import random_color_neighbour

type = SurfaceTypes.SDFPerturb
mod_name = "geo_ice"
name = "ice"

import bpy
import mathutils
from numpy.random import uniform, normal, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface

def shader_ice(nw: NodeWrangler):
    geometry = nw.new_node(Nodes.NewGeometry)
    
    noise_value = nw.new_node(Nodes.Value)
    noise_value.outputs[0].default_value = 6.5000
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': geometry.outputs["Position"], 'W': noise_value, 'Scale': 4.0000, 'Detail': 15.0000},
        attrs={'noise_dimensions': '4D'})
    
    color_ramp = nw.new_node(Nodes.ColorRamp, input_kwargs={'Fac': noise_texture.outputs["Fac"]})
    color_ramp.color_ramp.elements[0].position = 0.5000
    color_ramp.color_ramp.elements[0].color = [0.0844, 0.0844, 0.0844, 1.0000]
    color_ramp.color_ramp.elements[1].position = 0.7500
    color_ramp.color_ramp.elements[1].color = [1.0000, 1.0000, 1.0000, 1.0000]
    
    col_ice = random_color_neighbour((0.6469, 0.6947, 0.9522, 1.0000), 0.05, 0.1, 0.1)
    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            'Subsurface': 1.0000, 
            'Subsurface Radius': (0.1000, 0.1000, 0.2000), 
            'Subsurface Color': tuple(col_ice),
            'Roughness': color_ramp.outputs["Color"], 
            'IOR': 1.3100
        },
    )
    
    material_output = nw.new_node(Nodes.MaterialOutput, input_kwargs={'Surface': principled_bsdf}, attrs={'is_active_output': True})
    return principled_bsdf

@gin.configurable
def geo_ice(nw: NodeWrangler, random_seed=0, selection=None):
    # Code generated using version 2.6.4 of the node_transpiler
    with FixedSeed(random_seed):
        group_input = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
        
        normal_1 = nw.new_node(Nodes.InputNormal)
        
        position_1 = nw.new_node(Nodes.InputPosition)
        
        noise_texture_2 = nw.new_node(Nodes.NoiseTexture,
            input_kwargs={'Vector': position_1, 'W': nw.new_value(uniform(0, 10), "W1"), 'Scale': nw.new_value(uniform(7, 9), "Scale1"), 'Detail': 20.0000, 'Roughness': 1.0000},
            attrs={'noise_dimensions': '4D'})
        
        colorramp = nw.new_node(Nodes.ColorRamp, input_kwargs={'Fac': noise_texture_2.outputs["Fac"]})
        colorramp.color_ramp.elements[0].position = 0.5000
        colorramp.color_ramp.elements[0].color = [0.0000, 0.0000, 0.0000, 1.0000]
        colorramp.color_ramp.elements[1].position = 1.0000
        colorramp.color_ramp.elements[1].color = [1.0000, 1.0000, 1.0000, 1.0000]
        
        scale = nw.new_node(Nodes.VectorMath,
            input_kwargs={0: colorramp.outputs["Color"], 'Scale': 0.0300},
            attrs={'operation': 'SCALE'})
        
        normal_2 = nw.new_node(Nodes.InputNormal)
        
        position_2 = nw.new_node(Nodes.InputPosition)
        
        noise_texture_3 = nw.new_node(Nodes.NoiseTexture,
            input_kwargs={'Vector': position_2, 'W': nw.new_value(uniform(0, 10), "W2"), 'Scale': nw.new_value(uniform(1.3, 1.7), "Scale2"), 'Detail': 15.0000, 'Roughness': 0.7000, 'Distortion': 1.5000},
            attrs={'noise_dimensions': '4D'})
        
        multiply = nw.new_node(Nodes.VectorMath,
            input_kwargs={0: normal_2, 1: noise_texture_3.outputs["Fac"]},
            attrs={'operation': 'MULTIPLY'})
        
        scale_1 = nw.new_node(Nodes.VectorMath,
            input_kwargs={0: multiply.outputs["Vector"], 'Scale': 0.0800},
            attrs={'operation': 'SCALE'})
        
        multiply_add = nw.new_node(Nodes.VectorMath,
            input_kwargs={0: normal_1, 1: scale.outputs["Vector"], 2: scale_1.outputs["Vector"]},
            attrs={'operation': 'MULTIPLY_ADD'})
        
        offset = multiply_add.outputs["Vector"]
        if selection is not None:
            offset = nw.multiply(offset, surface.eval_argument(nw, selection))
        
        set_position_1 = nw.new_node(Nodes.SetPosition,
            input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Offset': offset})
        
        group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': set_position_1}, attrs={'is_active_output': True})


def apply(obj, selection=None, **kwargs):
    surface.add_geomod(obj, geo_ice, selection=selection)
    surface.add_material(obj, shader_ice, selection=selection)

