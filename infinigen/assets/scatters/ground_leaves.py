# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


from numpy.random import uniform as U
from mathutils import Vector

from infinigen.core.placement.instance_scatter import scatter_instances
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler

from infinigen.assets.trees.generate import random_leaf_collection

def apply(obj, selection=None, density=70, season=None, **kwargs):
    leaf_col=random_leaf_collection(season=season)
    return scatter_instances(
        base_obj=obj,
        collection=leaf_col,
        scale=0.3, scale_rand=U(0, 0.9),
        density=density, 
        ground_offset=0.05,
        selection=selection,
        taper_density=True)
