# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import bpy

from infinigen.core.placement.factory import AssetFactory


class BaseCoralFactory(AssetFactory):
    tentacle_distance = 0.05
    default_scale = [0.8] * 3
    noise_strength = 0.02
    tentacle_prob = 0.5
    bump_prob = 0.3
    density = 500

    def __init__(self, factory_seed, coarse=False):
        super(BaseCoralFactory, self).__init__(factory_seed, coarse)
        self.points_fn = lambda nw, points: points

    def create_asset(self, **params) -> bpy.types.Object:
        raise NotImplementedError
