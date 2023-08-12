# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Beining Han


import numpy as np
from infinigen.core.placement.instance_scatter import scatter_instances
from infinigen.core.placement.factory import AssetFactory, make_asset_collection

from infinigen.assets.small_plants.fern import FernFactory

from infinigen.core.util.random import random_general as rg
from infinigen.assets.scatters.utils.wind import wind

def apply(obj, selection=None, density=('uniform', 1, 6), **kwargs):


    fern_col = make_asset_collection(FernFactory(np.random.randint(1e5)), n=2, verbose=True)
    scatter_obj = scatter_instances(
        base_obj=obj, collection=fern_col,
        scale=0.7, scale_rand=0.7, scale_rand_axi=0.3,
        vol_density=rg(density),
        normal_fac=0.3, 
        rotation_offset=wind(strength=10),
        selection=selection
    )
    return scatter_obj, fern_col