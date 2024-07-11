# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.


# Authors: Lingjie Mei

from infinigen.assets.materials.ceramic import tile

from .wood import shader_wood


class SquareWoodTile:
    def generate(self):
        shader_func = shader_wood
        return tile.Tile().generate(shader_func, "square")
    
    __call__ = generate


# def make_sphere():e
