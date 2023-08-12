# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


from functools import reduce

import bpy
import colorsys
import numpy as np
from numpy.random import uniform, normal as N

from infinigen.assets.utils.decorate import assign_material
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory, make_asset_collection
from infinigen.core.placement.instance_scatter import scatter_instances
from infinigen.core import surface
from infinigen.core.placement.factory import AssetFactory
from infinigen.infinigen_gpl.extras.diff_growth import build_diff_growth
from infinigen.assets.utils.object import data2mesh
from infinigen.assets.utils.mesh import polygon_angles
from infinigen.core.util import blender as butil
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

from infinigen.assets.debris import LichenFactory

class Lichen:

    def __init__(self):
        self.fac = LichenFactory(np.random.randint(1e5))
        self.col = make_asset_collection(self.fac, name='lichen', n=5)

    def apply(self, obj, selection=None):

        scatter_obj = scatter_instances(
            base_obj=obj, collection=self.col, 
            density=5e3,  min_spacing=.08, 
            scale=1, scale_rand=N(0.5, 0.07),
            selection=selection
        )
        return scatter_obj


def apply(obj, selection=None):
    fac = LichenFactory(np.random.randint(1e5))
    col = make_asset_collection(fac, name='lichen', n=5)
    scatter_obj = scatter_instances(
        base_obj=obj, collection=col, 
        density=5e3,  min_spacing=.08, 
        scale=1, scale_rand=N(0.5, 0.07),
        selection=selection
    )
    return scatter_obj
