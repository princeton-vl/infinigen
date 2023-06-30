from numpy.random import uniform as U
from mathutils import Vector

from placement.instance_scatter import scatter_instances
from nodes.node_wrangler import Nodes, NodeWrangler

from assets.trees.generate import random_leaf_collection

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
