# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import numpy as np
from numpy.random import uniform as U

from infinigen.assets.underwater.seaweed import SeaweedFactory
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.placement.instance_scatter import scatter_instances
from infinigen.core.placement.factory import AssetFactory, make_asset_collection


def apply(obj, scale=1, density=1., n=5, selection=None, **kwargs):
    n_species = np.random.randint(2, 5)
    factories = [SeaweedFactory(np.random.randint(1e5)) for i in range(n_species)]
    seaweeds = make_asset_collection(factories, name='seaweed',
                                                weights=np.random.uniform(0.5, 1, len(factories)), n=n,
                                                verbose=True, **kwargs)

    scatter_obj = scatter_instances(
        base_obj=obj, collection=seaweeds, 
        vol_density=U(2, 10), min_spacing=0.02, 
        scale=U(0.2, 1), scale_rand=U(0.1, 0.9), scale_rand_axi=U(0, 0.2),
        normal_fac=0.3,
        selection=selection)
    
    return scatter_obj, seaweeds
