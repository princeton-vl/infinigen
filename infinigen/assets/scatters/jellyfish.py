# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import numpy as np
from numpy.random import uniform as U

from infinigen.assets.creatures.jellyfish import JellyfishFactory
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.placement.factory import AssetFactory, make_asset_collection
from infinigen.core.placement.instance_scatter import scatter_instances


def apply(obj, scale=1, density=1., n=6, selection=None):
    n_species = np.random.randint(2, 3)
    factories = list(JellyfishFactory(np.random.randint(1e5)) for i in range(n_species))
    jellyfish = make_asset_collection(factories, name='jellyfish',
                                                 weights=np.random.uniform(0.5, 1, len(factories)), n=n,
                                                 verbose=True)

    def ground_offset(nw: NodeWrangler):
        return nw.uniform(4 * scale, 8 * scale)

    r = np.pi / 3
    scatter_obj = scatter_instances(
        base_obj=obj, collection=jellyfish,
        density=density, min_spacing=scale * 4,
        scale=scale, scale_rand=U(0.2, 0.9),
        ground_offset=ground_offset, selection=selection,
        normal_fac=0.0,
        rotation_offset=lambda nw: nw.uniform((-r, 0, 0), (r, 0, 0)), reset_children=False,
    )
    return scatter_obj, jellyfish
