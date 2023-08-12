# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import numpy as np
from numpy.random import uniform as U

from infinigen.core.placement.factory import AssetFactory, make_asset_collection
from infinigen.assets.corals.generate import CoralFactory, TableCoralFactory

from infinigen.core.placement.instance_scatter import scatter_instances


def apply(obj, scale=1, density=5., n=12, selection=None, horizontal=False, **kwargs):
    if horizontal:
        return apply_horizontal(obj, scale, density, n, selection)
    else:
        return apply_all(obj, scale, density, n, selection)


def apply_all(obj, scale=1, density=5., n=12, selection=None):
    n_species = np.random.randint(5, 10)
    factories = [CoralFactory(np.random.randint(1e7)) for i in range(n_species)]
    corals = make_asset_collection(factories, name='coral', weights=U(0.8, 1, len(factories)), n=n)

    scatter_obj = scatter_instances(
        base_obj=obj, collection=corals, 
        density=density, min_spacing=scale*0.7,
        scale=scale, scale_rand=0.5, scale_rand_axi=U(0, 0.2),
        selection=selection)

    return scatter_obj, corals


def apply_horizontal(obj, scale=1, density=5., n=4, selection=None):
    n_species = np.random.randint(2, 3)
    factories = [TableCoralFactory(np.random.randint(1e5)) for _ in range(n_species)]
    corals = make_asset_collection(factories, name='coral',
                                              weights=np.random.uniform(0.8, 1, len(factories)), n=n,
                                              verbose=True)
    r = np.deg2rad(10)
    scatter_obj = scatter_instances(
        base_obj=obj, collection=corals, 
        density=density, min_spacing=scale * 0.5, 
        scale=1.5, scale_rand=U(0.2, 0.8), scale_rand_axi=U(0, 0.3),
        normal=(0, 0, 1), 
        rotation_offset=lambda nw: nw.uniform(3*(-r,), 3*(r,)), 
        selection=selection
    )

    return scatter_obj, corals
