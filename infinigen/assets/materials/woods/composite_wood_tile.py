# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from functools import wraps

from numpy.random import uniform

from infinigen.core.nodes import Nodes
from infinigen.core.util.random import log_uniform

from .. import shader_wood, tile
from ..tile import shader_staggered_tile
from .tiled_wood import shader_wood_tiled


def apply(obj, selection=None, vertical=False, scale=None, alternating=None, shape=None, **kwargs):
    shader_func = shader_wood
    tile.apply(obj, selection, vertical, shader_func, scale, alternating, "composite", **kwargs)
