# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: 
# - Alexander Raistrick: primary author 
# - Lahav Lipson: resample nodegroup


from numpy.random import uniform, normal
import numpy as np
from tqdm import trange

import bpy

from infinigen.core import surface
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.util.blender import group_in_collection
from infinigen.core.util.color import random_color_mapping


def to_material(name, singleton):
    """Wrapper for initializing and registering materials."""

    if singleton:
        name += ' (no gc)'

    def registration_fn(fn):
        def init_fn(*args, **kwargs):
            if singleton and name in bpy.data.materials:
                return bpy.data.materials[name]
            else:
                return surface.shaderfunc_to_material(fn, *args, name=name, *kwargs)

        return init_fn

    return registration_fn


def to_nodegroup(name, singleton, type='GeometryNodeTree'):
    """Wrapper for initializing and registering new nodegroups."""

    if singleton:
        name += ' (no gc)'

    def registration_fn(fn):
        def init_fn(*args, **kwargs):
            if singleton and name in bpy.data.node_groups:
                return bpy.data.node_groups[name]
            else:
                ng = bpy.data.node_groups.new(name, type)
                nw = NodeWrangler(ng)
                fn(nw, *args, **kwargs)
                return ng

        return init_fn

    return registration_fn


def assign_curve(c, points, handles=None):
    for i, p in enumerate(points):
        if i < 2:
            c.points[i].location = p
        else:
            c.points.new(*p)

        if handles is not None:
            c.points[i].handle_type = handles[i]


def facing_mask(nw, dir, thresh=0.5):
    normal = nw.new_node(Nodes.InputNormal)
    up_mask = nw.new_node(Nodes.VectorMath, input_kwargs={0: normal, 1: dir},
                          attrs={'operation': 'DOT_PRODUCT'})
    up_mask = nw.new_node(Nodes.Math, input_args=[up_mask, thresh], attrs={'operation': 'GREATER_THAN'})

    return up_mask


def noise(nw, scale, **kwargs):
    return nw.new_node(Nodes.NoiseTexture, input_kwargs={'Scale': scale, 'W': uniform(1e3),
        # Making this as big as 1e6 seems to cause bugs
        'Detail': kwargs.get('detail', uniform(0, 10)), 'Roughness': kwargs.get('roughness', uniform(0, 1)),
        'Distortion': kwargs.get('distortion', normal(0.7, 0.4))}, attrs={'noise_dimensions': '4D'})

def resample_node_group(nw: NodeWrangler, scene_seed: int):
    for node in nw.nodes:
        # Randomize 'W' in noise nodes
        if node.bl_idname in {Nodes.NoiseTexture, Nodes.WhiteNoiseTexture}:
            node.noise_dimensions = '4D'
            node.inputs['W'].default_value = np.random.uniform(1000)

        if node.bl_idname == Nodes.ColorRamp:
            for element in node.color_ramp.elements:
                element.color = random_color_mapping(element.color, scene_seed)

        if node.bl_idname == Nodes.RGB:
            node.outputs['Color'].default_value = random_color_mapping(node.outputs['Color'].default_value, scene_seed)

        # Randomized fixed color input
        for input_socket in node.inputs:
            if input_socket.type == 'RGBA':
                # print(f"Mapping", input_socket)
                input_socket.default_value = random_color_mapping(input_socket.default_value, scene_seed)

            if input_socket.name == "Seed":
                input_socket.default_value = np.random.randint(1000)
