# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick, Lingjie Mei
# Acknowledgment: This file draws inspiration from https://www.youtube.com/watch?v=X9YmJ0zGWHw by Polyfjord


import bpy
from mathutils import Vector

from numpy.random import uniform as U, normal as N, randint, uniform
import numpy as np

from infinigen.assets.utils.misc import log_uniform
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.placement import placement
from infinigen.core.placement.placement import placeholder_locs
from infinigen.core.util.math import FixedSeed
from infinigen.core.placement.factory import AssetFactory


@node_utils.to_nodegroup('nodegroup_caustics', singleton=False, type='ShaderNodeTree')
def nodegroup_caustics(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketVector', 'Vector', (0.0, 0.0, 0.0)),
        ('NodeSocketFloat', 'Prewarp', 0.15), ('NodeSocketFloat', 'Scale', 0.0),
        ('NodeSocketFloat', 'Smoothness', 0.0), ('NodeSocketFloat', 'AnimSpeed', .02)])

    w = nw.new_node(Nodes.Value, label='W')
    w.outputs[0].default_value = 0.0

    multiply = nw.new_node(Nodes.Math, input_kwargs={1: group_input.outputs["AnimSpeed"]},
                           attrs={'operation': 'MULTIPLY'})
    driver = multiply.inputs[0].driver_add('default_value').driver
    driver.expression = f"frame / {log_uniform(100, 200)}"

    noise_texture = nw.new_node(Nodes.NoiseTexture, input_kwargs={
        'Vector': group_input.outputs["Vector"],
        'W': multiply,
        'Scale': log_uniform(2, 8),
        'Roughness': N(0.5, 0.05),
        'Distortion': N(0.5, 0.02)
    }, attrs={'noise_dimensions': '4D'})

    scale = nw.new_node(Nodes.VectorMath,
                        input_kwargs={0: noise_texture.outputs["Color"], 'Scale': group_input.outputs["Prewarp"]
                        }, attrs={'operation': 'SCALE'})

    add = nw.new_node(Nodes.VectorMath,
                      input_kwargs={0: group_input.outputs["Vector"], 1: scale.outputs["Vector"]})

    voronoi_texture_1 = nw.new_node(Nodes.VoronoiTexture, input_kwargs={
        'Vector': add.outputs["Vector"],
        'W': multiply,
        'Scale': group_input.outputs["Scale"],
        'Smoothness': group_input.outputs["Smoothness"]
    }, attrs={'voronoi_dimensions': '4D', 'feature': 'SMOOTH_F1'})

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Smoothness"], 1: U(.04, .08)})

    voronoi_texture = nw.new_node(Nodes.VoronoiTexture, input_kwargs={
        'Vector': add.outputs["Vector"],
        'W': multiply,
        'Scale': group_input.outputs["Scale"],
        'Smoothness': add_1
    }, attrs={'voronoi_dimensions': '4D', 'feature': 'SMOOTH_F1'})

    difference = nw.scalar_multiply(nw.math('ABSOLUTE', nw.scalar_sub(voronoi_texture, voronoi_texture_1)),
                                    20.0)

    noise = nw.math('ABSOLUTE',
                    nw.scalar_sub(nw.new_node(Nodes.NoiseTexture, input_kwargs={'Scale': uniform(2, 5)}), .5))
    noise = nw.new_node(Nodes.MapRange, [noise, 0, 1, .6, 1.])
    ramp = nw.new_node(Nodes.FloatCurve, input_kwargs={'Value': nw.scalar_multiply(difference, noise)})
    node_utils.assign_curve(ramp.mapping.curves[0], 
        [(0.0, 0.0), (0.19, 0.08), (0.34, 1.0), (1.0, 1.0)], 
        handles=['AUTO', 'AUTO', 'VECTOR', 'VECTOR'])
    

    nw.new_node(Nodes.GroupOutput, input_kwargs={'Color': ramp})


def shader_caustic_lamp(nw: NodeWrangler, params: dict):
    coord = nw.new_node(Nodes.TextureCoord)
    caustics = nw.new_node(nodegroup_caustics().name,
                           input_kwargs={'Vector': coord.outputs['Normal'], **params})
    emission = nw.new_node(Nodes.Emission, input_kwargs={'Strength': caustics})
    nw.new_node(Nodes.LightOutput, [emission])


class CausticsLampFactory(AssetFactory):

    def __init__(self, factory_seed):
        super(CausticsLampFactory, self).__init__(factory_seed)
        with FixedSeed(factory_seed):
            self.params = {
                'Prewarp': U(0.1, 0.5), 
                'Scale': U(30, 100), 
                'Smoothness': 0.2 * N(1, 0.1), 
                'AnimSpeed': 0.1 * N(1, 0.1)
            }

    def create_asset(self, **params) -> bpy.types.Object:
        bpy.ops.object.light_add(type='SPOT')
        lamp = bpy.context.active_object
        lamp.data.shadow_soft_size = 0
        lamp.data.spot_blend = 1
        lamp.data.spot_size = np.pi * .4
        lamp.rotation_euler = 0, 0, uniform(0, np.pi * 2)
        lamp.data.use_nodes = True

        nw = NodeWrangler(lamp.data.node_tree)
        shader_caustic_lamp(nw, params=self.params)
        return lamp


def add_caustics(obj, zoff=200):

    fac = CausticsLampFactory(randint(1e7))
    loc = Vector(np.array(obj.bound_box).mean(axis=0)) + Vector((0, 0, zoff))
    lamp = fac.spawn_asset(0, loc=loc)
    lamp.scale = (50, 50, 50) # only affects UI
    lamp.data.energy = U(100e6, 200e6)
