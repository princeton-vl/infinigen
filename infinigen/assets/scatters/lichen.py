# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import numpy as np
from numpy.random import normal as N

from infinigen.assets.objects.particles import LichenFactory
from infinigen.core.placement.factory import make_asset_collection
from infinigen.core.placement.instance_scatter import scatter_instances


class Lichen:
    def __init__(self):
        self.fac = LichenFactory(np.random.randint(1e5))
        self.col = make_asset_collection(self.fac, name="lichen", n=5)

    def apply(self, obj, selection=None):
        scatter_obj = scatter_instances(
            base_obj=obj,
            collection=self.col,
            density=5e3,
            min_spacing=0.08,
            scale=1,
            scale_rand=N(0.5, 0.07),
            selection=selection,
        )
        return scatter_obj


def apply(obj, selection=None):
    fac = LichenFactory(np.random.randint(1e5))
    col = make_asset_collection(fac, name="lichen", n=5)
    scatter_obj = scatter_instances(
        base_obj=obj,
        collection=col,
        density=5e3,
        min_spacing=0.08,
        scale=1,
        scale_rand=N(0.5, 0.07),
        selection=selection,
    )
    return scatter_obj
