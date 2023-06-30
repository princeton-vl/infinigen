# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
# Date Signed: April 13 2023 

import numpy as np
from numpy.random import uniform as U

from assets.monocot.generate import MonocotFactory
from nodes.node_wrangler import NodeWrangler
from placement.camera import ng_dist2camera
from placement.instance_scatter import scatter_instances
from placement.factory import AssetFactory, make_asset_collection
from surfaces.scatters.utils.wind import wind

def apply(obj, n=4, grass=None, selection=None, **kwargs):
 
    monocots = make_asset_collection(
        MonocotFactory(np.random.randint(1e5), grass=grass),  
        n=n, verbose=True, **kwargs)

    scatter_obj = scatter_instances(
        base_obj=obj, collection=monocots,
        vol_density=U(0.2, 4), min_spacing=0.1, 
        ground_offset=(0, 0, -0.05),
        scale=U(0.05, 0.4), scale_rand=U(0.5, 0.95), 
        rotation_offset=wind(strength=20),
        normal_fac=0.3,
        selection=selection)
    return scatter_obj, monocots
