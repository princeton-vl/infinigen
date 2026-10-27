# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Yiming Zuo: original Infinigen wrapped leaf (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/assets/objects/leaves/leaf_wrapped.py)
# - David Yan, Jia Deng: optimized implementation

import numpy as np
from numpy.random import randint, uniform

from infinigen.assets.objects.leaves.leaf_wrapped import (
    LeafFactoryWrapped as _DefaultLeafFactoryWrapped,
)
from infinigen.assets.objects.leaves.leaf_wrapped import nodegroup_nodegroup_apply_wrap
from infinigen.core import surface
from infinigen.perf.assets.utils.heightfield import mesh_route


class LeafFactoryWrapped(_DefaultLeafFactoryWrapped):
    def create_asset(self, **params):
        fac_id = randint(len(self.factory_list))
        fac = self.factory_list[fac_id]

        wrap_params = {
            "angle": uniform(-70, 70),
            "rotation": uniform(0.2, 2.0),
            "end_radius": np.exp(uniform(-2.0, 2.0)),
            "noise_level": uniform(0.0, 0.5),
        }

        obj = fac.create_asset()
        surface.add_geomod(
            obj, nodegroup_nodegroup_apply_wrap, apply=False, input_kwargs=wrap_params
        )

        return mesh_route.convert_active_to_mesh(obj)
