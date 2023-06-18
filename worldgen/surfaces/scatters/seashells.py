# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
# Date Signed: Jun 14 2023

import numpy as np
from numpy.random import uniform as U

from assets.mollusk import MolluskFactory
from placement.factory import AssetFactory, make_asset_collection
from placement.instance_scatter import scatter_instances
from surfaces.scatters.chopped_trees import approx_settle_transform

from util.random import random_general as rg

def apply(obj, density=('uniform', 0.2, 1.7), n=10, selection=None):
    n_species = np.random.randint(4, 6)
    factories = list(MolluskFactory(np.random.randint(1e5)) for _ in range(n_species))
    mollusk = make_asset_collection(
        factories, name='mollusk', verbose=True,
        weights=np.random.uniform(0.5, 1, len(factories)), n=n)
    
    #for o in mollusk.objects:
    #    approx_settle_transform(o, samples=30)

    scatter_obj = scatter_instances(
        base_obj=obj, collection=mollusk,
        vol_density=rg(density),
        scale=U(0.1, 0.4), scale_rand=U(0.5, 0.9), scale_rand_axi=U(0.1, 0.5),
        selection=selection, taper_density=True,
        ground_offset=lambda nw: nw.uniform(0.07, .2)
    )

    return scatter_obj, mollusk
