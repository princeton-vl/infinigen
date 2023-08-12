# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


from random import random
import bpy

import numpy as np
from numpy.random import uniform as U
from mathutils import Vector

from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj

from infinigen.core.nodes import node_utils
from infinigen.core.placement.instance_scatter import scatter_instances
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core import surface

from infinigen.assets.trees.generate import make_twig_collection, random_species
from .chopped_trees import approx_settle_transform

def apply(obj, selection=None, n_leaf=0, n_twig=10, **kwargs):

    (_, twig_params, leaf_params), _ = random_species(season='winter')
    twigs = make_twig_collection(np.random.randint(1e5), twig_params, leaf_params, 
        n_leaf=n_leaf, n_twig=n_twig, leaf_types=None, trunk_surface=surface.registry('bark'))

    for o in twigs.objects:
        approx_settle_transform(o, samples=40)

    scatter_obj = scatter_instances(
        base_obj=obj, collection=twigs,
        scale=U(0.15, 0.3), scale_rand=U(0, 0.3), scale_rand_axi=U(0, 0.2),  
        density=10, ground_offset=0.05,
        selection=selection, taper_density=True)
    
    return scatter_obj, twigs
