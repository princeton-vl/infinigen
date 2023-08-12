# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import numpy as np
from numpy.random import uniform as U

from infinigen.assets.underwater.urchin import UrchinFactory
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.placement.factory import AssetFactory, make_asset_collection
from infinigen.core.placement.instance_scatter import scatter_instances


def apply(obj, n=5, selection=None):
    n_species = np.random.randint(2, 3)
    factories = list(UrchinFactory(np.random.randint(1e5)) for i in range(n_species))
    urchin = make_asset_collection(factories, name='urchin',
                                              weights=np.random.uniform(0.5, 1, len(factories)), n=n,
                                              verbose=True)
    
    scale = U(0.1, 0.8)

    def ground_offset(nw: NodeWrangler):
        return nw.uniform(.4 * scale, .8 * scale)

    scatter_obj = scatter_instances(
        base_obj=obj, collection=urchin,
        vol_density=U(0.5, 2), ground_offset=ground_offset,
        scale=scale, scale_rand=U(0.2, 0.4),
        selection=selection)

    return scatter_obj, urchin
