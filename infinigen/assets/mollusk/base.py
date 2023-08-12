# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import bpy

from infinigen.core.placement.factory import AssetFactory


class BaseMolluskFactory(AssetFactory):
    max_expected_radius = .5
    noise_strength = .02
    ratio = 1
    x_scale = 2
    z_scale = 1
    distortion = 1

    def __init__(self, factory_seed, coarse=False):
        super(BaseMolluskFactory, self).__init__(factory_seed, coarse)

    def create_asset(self, **params) -> bpy.types.Object:
        raise NotImplemented
