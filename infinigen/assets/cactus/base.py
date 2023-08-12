# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import bpy

from infinigen.assets.cactus.spike import make_default_selections
from infinigen.assets.utils.decorate import write_attribute
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.placement.factory import AssetFactory

class BaseCactusFactory(AssetFactory):
    spike_distance = .025
    cap_percentage = .1
    noise_strength = .02
    base_radius = .002
    density = 5e4

    def __init__(self, factory_seed, coarse=False):
        super(BaseCactusFactory, self).__init__(factory_seed, coarse)
        self.points_fn = make_default_selections(self.spike_distance, self.cap_percentage, self.density)

    def create_asset(self, **params) -> bpy.types.Object:
        raise NotImplemented
