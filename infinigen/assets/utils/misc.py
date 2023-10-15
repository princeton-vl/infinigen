# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import bpy
import numpy as np
from numpy.random import normal, uniform

from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler


class CountInstance:

    def __init__(self, name):
        self.name = name

    @staticmethod
    def count_instance():
        depsgraph = bpy.context.evaluated_depsgraph_get()
        return len([inst for inst in depsgraph.object_instances if inst.is_instance])

    def __enter__(self):
        self.count = self.count_instance()

    def __exit__(self, *args):
        count = self.count_instance()
        print(f"{count - self.count} {self.name} instances created.")


def log_uniform(low, high, size=None):
    return np.exp(uniform(np.log(low), np.log(high), size))


def sample_direction(min_z):
    for _ in range(100):
        x = normal(size=3)
        y = x / np.linalg.norm(x)
        if y[-1] > min_z:
            return y
    return 0, 0, 1


def build_color_ramp(nw: NodeWrangler, x, positions, colors, mode='HSV'):
    cr = nw.new_node(Nodes.ColorRamp, input_kwargs={'Fac': x})
    cr.color_ramp.color_mode = mode
    elements = cr.color_ramp.elements
    size = len(positions)
    assert len(colors) == size
    if size > 2:
        for _ in range(size - 2):
            elements.new(0)
    for i, (p, c) in enumerate(zip(positions, colors)):
        elements[i].position = p
        elements[i].color = c
    return cr


def make_circular_angle(xs):
    return np.array([xs[-1] - np.pi * 2, *xs, xs[0] + np.pi * 2])


def make_circular(xs):
    return np.array([xs[-1], *xs, xs[0]])
