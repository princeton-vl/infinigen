# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import numpy as np
from numpy.random import uniform as U

from infinigen.assets.monocot.pinecone import PineconeFactory
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.placement.factory import AssetFactory, make_asset_collection
from infinigen.core.placement.instance_scatter import scatter_instances

from infinigen.assets.scatters.chopped_trees import approx_settle_transform

def apply(obj, n=5, selection=None):
    n_species = np.random.randint(2, 3)
    factories = [PineconeFactory(np.random.randint(1e5)) for i in range(n_species)]
    pinecones = make_asset_collection(
        factories, n=n, verbose=True,
        weights=np.random.uniform(0.5, 1, len(factories)))
    
    for o in pinecones.objects:
        approx_settle_transform(o, samples=30)

    d = np.deg2rad(90)
    ar = np.deg2rad(20)
    scatter_obj = scatter_instances(
        base_obj=obj, collection=pinecones,
        vol_density=U(0.05, 0.25), min_spacing=0.05,
        rotation_offset=lambda nw: nw.uniform((d-ar, -ar, -ar), (d+ar, ar, ar)),
        scale=U(0.05, 0.8), scale_rand=U(0.2, 0.8), scale_rand_axi=U(0, 0.1),
        selection=selection, taper_density=True)

    return scatter_obj, pinecones
