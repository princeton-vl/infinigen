from numpy.random import uniform, normal
import numpy as np
from tqdm import trange

import bpy

from surfaces import surface
from nodes.node_wrangler import Nodes, NodeWrangler
from util.blender import group_in_collection

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
            if singleton and name in bpy.data.node_groups:
                return bpy.data.node_groups[name]
            else:
                nw = NodeWrangler(ng)
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

    normal = nw.new_node(Nodes.InputNormal)

    return up_mask


def noise(nw, scale, **kwargs):
