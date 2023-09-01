# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick
# Acknowledgment: This file draws inspiration from https://www.youtube.com/watch?v=YDrbyITWMGU by Mr. Cheebs


import pdb
import logging

from numpy.random import normal, uniform

import bpy

from infinigen.core.surface import attribute_to_vertex_group

from infinigen.core.util import blender as butil
from infinigen.core.util.math import dict_convex_comb
from infinigen.core.util.logging import Timer
from infinigen.core.nodes.node_wrangler import NodeWrangler, Nodes

logger = logging.getLogger(__name__)

def local_pos_rigity_mask(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketFloat', 'To Min', 0.4),
            ('NodeSocketFloat', 'To Max', 0.9)])
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': nw.expose_input('Local Pos', attribute='local_pos')})
    
    clamp = nw.new_node(Nodes.Clamp,
        input_kwargs={'Value': nw.expose_input("Radius", attribute='skeleton_rad'), 'Min': 0.03, 'Max': 0.49})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: clamp, 1: -1.0},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: clamp, 1: 1.5},
        attrs={'operation': 'MULTIPLY'})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_xyz.outputs["Z"], 1: multiply, 2: multiply_1})
    
    musgrave_texture = nw.new_node(Nodes.MusgraveTexture,
        input_kwargs={'W': uniform(1e3), 'Scale': normal(10, 1)},
        attrs={'musgrave_dimensions': '4D'})
    
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: musgrave_texture, 1: normal(0.07, 0.007)},
        attrs={'operation': 'MULTIPLY'})
    
    musgrave_texture_1 = nw.new_node(Nodes.MusgraveTexture,
        input_kwargs={'Scale': normal(5, 0.5), 'W': uniform(1e3)},
        attrs={'musgrave_dimensions': '4D'})
    
    multiply_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: musgrave_texture_1, 1: normal(0.12, 0.01)},
        attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_2, 1: multiply_3})
    
    add_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range.outputs["Result"], 1: add})
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': add_1})
    colorramp.color_ramp.elements.new(1)
    colorramp.color_ramp.elements[0].position = normal(0.23, 0.05)
    colorramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp.color_ramp.elements[1].position = normal(0.6, 0.05)
    colorramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    colorramp.color_ramp.elements[2].position = 1.0
    colorramp.color_ramp.elements[2].color = (0.0, 0.0, 0.0, 1.0)
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': colorramp.outputs["Color"], 3: group_input.outputs["To Min"], 4: group_input.outputs["To Max"]})
    
    musgrave_texture_2 = nw.new_node(Nodes.MusgraveTexture)
    
    multiply_4 = nw.new_node(Nodes.Math,
        input_kwargs={0: musgrave_texture_2, 1: normal(0.1, 0.02)},
        attrs={'operation': 'MULTIPLY'})
    
    return nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_1.outputs["Result"], 1: multiply_4})

def bake_cloth(obj, settings, attributes, frame_start=None, frame_end=None):

    if frame_start is None:
        frame_start = bpy.context.scene.frame_start
    if frame_end is None:
        frame_end = bpy.context.scene.frame_end

    mod = obj.modifiers.new('bake_cloth', 'CLOTH')

    mod.settings.effector_weights.gravity = settings.pop('gravity', 1)

    for k, v in settings.items():
        setattr(mod.settings, k, v)

    with butil.DisableModifiers(obj):
        for name, attr in attributes.items():
            vgroup = attribute_to_vertex_group(obj, attr, name=f'skin_sim.{name}')
            setattr(mod.settings, name, vgroup.name)

    mod.point_cache.frame_start = frame_start
    mod.point_cache.frame_end = frame_end

    with butil.ViewportMode(obj, mode='OBJECT'), butil.SelectObjects(obj), Timer('Baking fish cloth'):
        override = {'scene': bpy.context.scene, 'active_object': obj, 'point_cache': mod.point_cache}
        bpy.ops.ptcache.bake(override, bake=True)

    return mod



