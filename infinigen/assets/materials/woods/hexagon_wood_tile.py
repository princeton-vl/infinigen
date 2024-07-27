# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.


# Authors: Lingjie Mei
from infinigen.assets.materials import tile

from .wood import shader_wood


def apply(
    obj,
    selection=None,
    vertical=False,
    scale=None,
    alternating=None,
    shape=None,
    **kwargs,
):
    shader_func = shader_wood
    tile.apply(
        obj, selection, vertical, shader_func, scale, alternating, "hexagon", **kwargs
    )
