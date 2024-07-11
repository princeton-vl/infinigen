# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import numpy as np

from infinigen.assets.materials.ceramic import tile
from infinigen.assets.materials.wood.wood import shader_wood


class NonWoodTile:
    def generate(self, shader_func=None, shape=None):
        shader_funcs = tile.get_shader_funcs()
        shader_funcs = [(f, w) for f, w in shader_funcs if f != shader_wood]
        funcs, weights = zip(*shader_funcs)
        weights = np.array(weights) / sum(weights)
        if shader_func is None:
            shader_func = np.random.choice(funcs, p=weights)
        if shape is None:
            shape = np.random.choice(["square", "hexagon", "rectangle"])

        return tile.Tile().generate(shader_func,shape)
    
    __call__ = generate