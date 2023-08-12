# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import numpy as np
from numpy.random import uniform as U

from infinigen.assets.mushroom import MushroomFactory
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.placement.factory import AssetFactory, make_asset_collection
from infinigen.core.placement.instance_scatter import scatter_instances

class Mushrooms:
    
    def __init__(self, n=10):

        self.n_species = np.random.randint(2, 3)
        self.factories = [MushroomFactory(np.random.randint(1e5)) for i in range(self.n_species)]
        self.col = make_asset_collection(
            self.factories, name='mushroom', n=n, verbose=True,
            weights=np.random.uniform(0.5, 1, len(self.factories)))

    def apply(self, obj, scale=0.3, density=1., selection=None):

        scatter_obj = scatter_instances(
            base_obj=obj, collection=self.col,
            density=density, min_spacing=scale,
            scale=scale, scale_rand=U(0.5, 0.9),
            selection=selection, taper_scale=True)

        return scatter_obj
