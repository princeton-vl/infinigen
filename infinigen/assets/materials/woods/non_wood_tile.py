# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import numpy as np


def apply(
    obj,
    selection=None,
    vertical=False,
    shader_func=None,
    scale=None,
    alternating=None,
    shape=None,
    **kwargs,
):
    from infinigen.assets.materials import tile

    from .wood import shader_wood

    shader_funcs = tile.get_shader_funcs()
    shader_funcs = [(f, w) for f, w in shader_funcs if f != shader_wood]
    funcs, weights = zip(*shader_funcs)
    weights = np.array(weights) / sum(weights)
    if shader_func is None:
        shader_func = np.random.choice(funcs, p=weights)
    if shape is None:
        shape = np.random.choice(["square", "hexagon", "rectangle"])
    tile.apply(
        obj, selection, vertical, shader_func, scale, alternating, shape, **kwargs
    )
