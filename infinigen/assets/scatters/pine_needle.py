# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import colorsys

import bpy
import mathutils
import numpy as np
from numpy.random import uniform as U

from infinigen.core.placement.factory import make_asset_collection
from infinigen.core.placement.instance_scatter import scatter_instances

from infinigen.assets.debris import PineNeedleFactory

def apply(obj, scale=1, density=2e3, n=3, selection=None):
    n_species = np.random.randint(2, 3)
    factories = [PineNeedleFactory(np.random.randint(1e5)) for i in range(n_species)]
    pine_needle = make_asset_collection(factories,
                                                   weights=U(0.5, 1, len(factories)), n=n,
                                                   verbose=True)
    
    d = np.deg2rad(U(5, 15))
    scatter_obj = scatter_instances(
        base_obj=obj, collection=pine_needle,
        vol_density=U(0.01, 0.03),        rotation_offset=lambda nw: nw.uniform((-d,)*3, (d,)*3),
        ground_offset=lambda nw: nw.uniform(0, 0.015),
        scale=U(2, 3), scale_rand=U(0.4, 0.8), scale_rand_axi=U(0.3, 0.7),
        selection=selection, taper_density=True
    )

    return scatter_obj, pine_needle
