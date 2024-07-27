# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.
import string
from functools import update_wrapper, wraps

import bpy
import numpy as np
from numpy.random import normal, uniform

from infinigen.assets.utils.object import origin2lowest
from infinigen.core.nodes import Nodes, NodeWrangler
from infinigen.core.util import blender as butil
from infinigen.core.util.math import clip_gaussian

# Authors: Lingjie Mei


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


def sample_direction(min_z):
    for _ in range(100):
        x = normal(size=3)
        y = x / np.linalg.norm(x)
        if y[-1] > min_z:
            return y
    return 0, 0, 1


def subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in subclasses(c)]
    )


def make_normalized_factory(cls):
    @wraps(cls, updated=())
    class CLS(cls):
        def __init__(self, *args, **kwargs):
            super(CLS, self).__init__(*args, **kwargs)
            update_wrapper(self, *args, **kwargs)

        def create_asset(self, **params):
            obj = super(CLS, self).create_asset(**params)
            obj.rotation_euler = uniform(-np.pi, np.pi, 3)
            butil.apply_transform(obj)
            origin2lowest(obj)
            return obj

    return CLS


def build_color_ramp(nw: NodeWrangler, x, positions, colors, mode="HSV"):
    cr = nw.new_node(Nodes.ColorRamp, input_kwargs={"Fac": x})
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


def toggle_hide(obj, recursive=True):
    if obj.name in bpy.data.collections:
        obj.hide_viewport = True
        obj.hide_render = True
        for o in obj.objects:
            toggle_hide(o, recursive)
    else:
        obj.hide_set(True)
        obj.hide_render = True
        if recursive:
            for c in obj.children:
                toggle_hide(c)


def toggle_show(obj, recursive=True):
    if obj.name in bpy.data.collections:
        obj.hide_viewport = False
        obj.hide_render = False
        for o in obj.objects:
            toggle_show(o, recursive)
    else:
        obj.hide_set(False)
        obj.hide_render = False
        if recursive:
            for c in obj.children:
                toggle_hide(c)


def assign_material(obj, material):
    if not isinstance(obj, list):
        obj = [obj]
    for o in obj:
        with butil.SelectObjects(o):
            while len(o.data.materials):
                bpy.ops.object.material_slot_remove()
        if not isinstance(material, list):
            material = [material]
        for m in material:
            o.data.materials.append(m)


character_set = list(string.ascii_lowercase + string.ascii_uppercase + string.digits)
character_set_weights = np.concatenate(
    [
        1.5 * np.ones(len(string.ascii_lowercase)),
        0.5 * np.ones(len(string.ascii_uppercase)),
        0.5 * np.ones(len(string.digits)),
    ]
)
character_set_weights /= character_set_weights.sum()


def generate_text():
    return "".join(
        np.random.choice(
            character_set,
            size=int(clip_gaussian(3, 7, 2, 15)),
            replace=True,
            p=character_set_weights,
        )
    )
