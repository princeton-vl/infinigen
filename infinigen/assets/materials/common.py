# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import functools
from collections.abc import Callable, Iterable

import numpy as np

from infinigen.assets.utils.decorate import read_material_index, write_material_index
from infinigen.core import surface
from infinigen.core.surface import read_attr_data

from infinigen.core.util.math import FixedSeed


def apply(obj, shader_func, selection=None, *args, **kwargs):
    if not isinstance(obj, Iterable):
        obj = [obj]
    if isinstance(shader_func, Callable):
        material = surface.shaderfunc_to_material(shader_func, *args, **kwargs)
    else:
        material = shader_func
    for o in obj:
        index = len(o.data.materials)
        o.data.materials.append(material)
        material_index = read_material_index(o)
        full_like = np.full_like(material_index, index)
        if selection is None:
            material_index = full_like
        elif isinstance(selection, t.Tag):
            material_index = np.where(sel, index, material_index)
        elif isinstance(selection, str):
            try:
                sel = read_attr_data(o, selection.lstrip('!'), 'FACE')
                material_index = np.where(1 - sel if selection.startswith('!') else sel, index, material_index)
            except:
                material_index = np.zeros(len(material_index), dtype=int)
        else:
            material_index = np.where(selection, index, material_index)
        write_material_index(o, material_index)


def get_selection(obj, selection):
    if selection is None:
        return np.ones(len(obj.data.polygons))
    elif isinstance(selection, t.Tag):
    elif isinstance(selection, str):
        return read_attr_data(obj, selection.lstrip('!'), 'FACE')
    else:
        return selection


def unique_surface(surface, seed=None):
    if seed is None:
        seed = np.random.randint(1e7)

    class Surface:

        @classmethod
        def apply(cls, *args, **kwargs):
            with FixedSeed(seed):
                return surface.apply(*args, **kwargs)

    return Surface
