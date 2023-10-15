# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import numpy as np
from numpy.random import uniform as U, uniform

from infinigen.assets.mollusk import MolluskFactory
from infinigen.core.placement.factory import AssetFactory, make_asset_collection
from infinigen.core.placement.instance_scatter import scatter_instances
from infinigen.assets.scatters.chopped_trees import approx_settle_transform

from infinigen.core.util.random import random_general as rg

def apply(obj, density=('uniform', 0.2, 1.), n=10, selection=None):
    n_species = np.random.randint(4, 6)
    factories = list(MolluskFactory(np.random.randint(1e5)) for _ in range(n_species))
    mollusk = make_asset_collection(
        factories, name='mollusk', verbose=True,
        weights=np.random.uniform(0.5, 1, len(factories)), n=n, face_size=.02)

    #for o in mollusk.objects:
    #    approx_settle_transform(o, samples=30)

    scale = uniform(.3, .5)
    scatter_obj = scatter_instances(
        base_obj=obj, collection=mollusk,
        vol_density=rg(density),
        scale=scale, scale_rand=U(0.5, 0.9), scale_rand_axi=U(0.1, 0.5),
        selection=selection, taper_density=True,
        ground_offset=lambda nw: nw.uniform(0, scale)
    )

    return scatter_obj, mollusk
