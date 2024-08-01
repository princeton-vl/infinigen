# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import numpy as np
from numpy.random import uniform as U

from infinigen.assets.objects.trees.generate import make_twig_collection, random_species
from infinigen.assets.utils.misc import toggle_hide, toggle_show
from infinigen.core import surface
from infinigen.core.placement.instance_scatter import scatter_instances

from .chopped_trees import approx_settle_transform


def apply(obj, selection=None, n_leaf=0, n_twig=10, **kwargs):
    (_, twig_params, leaf_params), _ = random_species(season="winter")
    twigs = make_twig_collection(
        np.random.randint(1e5),
        twig_params,
        leaf_params,
        n_leaf=n_leaf,
        n_twig=n_twig,
        leaf_types=None,
        trunk_surface=surface.registry("bark"),
    )

    toggle_show(twigs)
    for o in twigs.objects:
        approx_settle_transform(o, samples=40)
    toggle_hide(twigs)

    scatter_obj = scatter_instances(
        base_obj=obj,
        collection=twigs,
        scale=U(0.15, 0.3),
        scale_rand=U(0, 0.3),
        scale_rand_axi=U(0, 0.2),
        density=10,
        ground_offset=0.05,
        selection=selection,
        taper_density=True,
    )

    return scatter_obj, twigs
