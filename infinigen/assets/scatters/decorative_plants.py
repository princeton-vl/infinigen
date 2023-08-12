# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import numpy as np
from numpy.random import uniform as U


from infinigen.core.placement.factory import AssetFactory, make_asset_collection
from infinigen.core.placement.instance_scatter import scatter_instances
from infinigen.core.placement import detail
from infinigen.core.nodes import node_utils

from infinigen.assets.small_plants import succulent

from infinigen.assets.scatters.utils.wind import wind

def apply(obj, n=4, selection=None, **kwargs):
 
    fac_class = np.random.choice([
        succulent.SucculentFactory
    ])

    monocots = make_asset_collection(
        fac_class(np.random.randint(1e5)),  
        n=n, verbose=True, **kwargs)

    scatter_obj = scatter_instances(
        base_obj=obj, collection=monocots,
        vol_density=U(0.05, 2), min_spacing=0.1, 
        normal_fac=0.5,
        scale=U(0.3, 1), scale_rand=U(0.5, 0.95), 
        rotation_offset=wind(strength=10),
        taper_density=True,
        selection=selection)
    return scatter_obj, monocots
