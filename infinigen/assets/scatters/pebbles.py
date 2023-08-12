# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import bpy
from mathutils import Vector
import numpy as np
from numpy.random import uniform as U

from infinigen.core.nodes.node_wrangler import Nodes
from infinigen.core.placement.instance_scatter import scatter_instances
from infinigen.core.placement.factory import AssetFactory, make_asset_collection

from infinigen.core import surface
from infinigen.core.util import blender as butil
from infinigen.assets.rocks.blender_rock import BlenderRockFactory

def apply(obj, n=5, detail=3, selection=None, **kwargs):

    fac = BlenderRockFactory(np.random.randint(1e5), detail=detail)
    rocks = make_asset_collection(fac, n=n)

    surface.registry('rock_collection').apply(list(rocks.objects))

    scatter_obj = scatter_instances(
        base_obj=obj, collection=rocks,
        vol_density=U(0.05, 0.4), ground_offset=0.03, 
        scale=U(0.05, 1), scale_rand=U(0.75, 0.95), scale_rand_axi=U(0.4, 0.6),
        selection=selection, taper_density=True)

    return scatter_obj, rocks
