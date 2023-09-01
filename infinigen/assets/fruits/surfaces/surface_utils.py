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

@node_utils.to_nodegroup('nodegroup_stripe_pattern', singleton=False, type='ShaderNodeTree')
def nodegroup_stripe_pattern(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    texture_coordinate = nw.new_node(Nodes.TextureCoord)
    
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketColor', 'Color', (0.8, 0.8, 0.8, 1.0)),
            ('NodeSocketFloat', 'attribute', 0.0),
            ('NodeSocketFloat', 'voronoi scale', 50.0),
            ('NodeSocketFloatFactor', 'voronoi randomness', 1.0),
            ('NodeSocketFloat', 'seed', 0.0),
            ('NodeSocketFloat', 'noise scale', 10.0),
            ('NodeSocketFloat', 'noise amount', 1.4),
            ('NodeSocketFloat', 'hue min', 0.6),
            ('NodeSocketFloat', 'hue max', 1.085)])
    
    add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: texture_coordinate.outputs["Object"], 1: group_input.outputs["seed"]})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': add.outputs["Vector"], 'Scale': group_input.outputs["noise scale"], 'Detail': 1.0})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: noise_texture.outputs["Fac"], 1: group_input.outputs["noise amount"]},
        attrs={'operation': 'MULTIPLY'})
    
    voronoi_texture = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'W': group_input.outputs["attribute"], 'Scale': group_input.outputs["voronoi scale"], 'Randomness': group_input.outputs["voronoi randomness"]},
        attrs={'voronoi_dimensions': '1D'})
    
    add_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply, 1: voronoi_texture.outputs["Distance"]})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': add_1, 3: group_input.outputs["hue min"], 4: group_input.outputs["hue max"]})
    
    hue_saturation_value = nw.new_node('ShaderNodeHueSaturation',
        input_kwargs={'Value': map_range.outputs["Result"], 'Color': group_input.outputs["Color"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Color': hue_saturation_value})
