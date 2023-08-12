# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alex Raistrick


from math import prod

import bpy

import numpy as np
from numpy.random import uniform as U
from mathutils import Vector

from infinigen.core.util import blender as butil


from infinigen.core.placement.factory import AssetFactory, make_asset_collection
from infinigen.core.placement.instance_scatter import scatter_instances
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core import surface


from infinigen.assets.materials import simple_greenery, grass_blade_texture

from infinigen.assets.grassland.grass_tuft import GrassTuftFactory

from infinigen.assets.scatters.utils.wind import wind

def scale_grass(nw: NodeWrangler):
    random_scaling = nw.new_node(Nodes.RandomValue, input_kwargs={0: Vector((1.,1.,1.)), 1: Vector((1.2,1.2,2.))}, attrs={"data_type":'FLOAT_VECTOR'})
    return nw.multiply(random_scaling, Vector((2.5,2.5,2.5)))

def apply(obj, selection=None, **kwargs):

    n_fac = 1
    facs = [GrassTuftFactory(np.random.randint(1e7)) for _ in range(n_fac)]
    grass_col = make_asset_collection(facs, n=10)

    scatter_obj = scatter_instances(
        base_obj=obj,
        collection=grass_col,
        scale=U(1, 3), scale_rand=U(0.7, 1), scale_rand_axi=0.1,
        vol_density=U(0.5, 5),
        ground_offset=0, normal_fac=U(0, 0.5),
        rotation_offset=wind(strength=10),
        selection=selection,
        taper_scale=True                                
    )

    return scatter_obj, grass_col
