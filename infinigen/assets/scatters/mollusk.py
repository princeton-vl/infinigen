# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import numpy as np

from infinigen.assets.mollusk import MolluskFactory
from infinigen.assets.utils.misc import CountInstance
from infinigen.core.placement.factory import AssetFactory, make_asset_collection
from infinigen.assets.utils.decorate import toggle_hide
from infinigen.core.util import blender as butil
from infinigen.core.nodes import node_utils
from infinigen.core.placement.instance_scatter import scatter_instances
from infinigen.core import surface


def apply(obj, scale=0.4, density=1., n=10, selection=None):
    with CountInstance('mollusk'):
        n_species = np.random.randint(4, 6)
        factories = list(MolluskFactory(np.random.randint(1e5)) for _ in range(n_species))
        mollusk = make_asset_collection(factories, name='mollusk',
                                                    weights=np.random.uniform(0.5, 1, len(factories)), n=n,
                                                    verbose=True)

        def scaling(nw):
            return nw.uniform([.4 * scale] * 3, [.8 * scale] * 3, data_type='FLOAT_VECTOR')

        scatter_obj = scatter_instances('mollusk',
            base_obj=obj, collection=mollusk,
            density=density, scaling=scaling, 
            min_spacing=scale, normal=(0,0,1),
            selection=selection, taper_density=True)

        return scatter_obj, mollusk
