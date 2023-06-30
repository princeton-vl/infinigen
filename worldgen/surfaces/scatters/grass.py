from math import prod

import bpy

import numpy as np
from numpy.random import uniform as U

from util import blender as butil


from placement.factory import AssetFactory, make_asset_collection
from placement.instance_scatter import scatter_instances
from nodes.node_wrangler import Nodes, NodeWrangler
from surfaces import surface


from surfaces.templates import simple_greenery, grass_blade_texture

from assets.grassland.grass_tuft import GrassTuftFactory

from surfaces.scatters.utils.wind import wind

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
