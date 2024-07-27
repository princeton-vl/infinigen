# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Beining Han


import numpy as np

from infinigen.assets.objects.grassland.flowerplant import FlowerPlantFactory
from infinigen.assets.scatters.utils.wind import wind
from infinigen.core.placement.factory import make_asset_collection
from infinigen.core.placement.instance_scatter import scatter_instances


def apply(obj, selection=None, density=1.0):
    flowerplant_col = make_asset_collection(
        FlowerPlantFactory(np.random.randint(1e5)), n=12, verbose=True
    )

    avg_vol = np.mean([np.prod(list(o.dimensions)) for o in flowerplant_col.objects])
    density = np.clip(density / avg_vol, 0, 200)
    scatter_obj = scatter_instances(
        base_obj=obj,
        collection=flowerplant_col,
        scale=1.5,
        scale_rand=0.7,
        scale_rand_axi=0.2,
        density=float(density),
        ground_offset=0,
        normal_fac=0.3,
        rotation_offset=wind(strength=20),
        selection=selection,
        taper_scale=True,
    )

    return scatter_obj, flowerplant_col
