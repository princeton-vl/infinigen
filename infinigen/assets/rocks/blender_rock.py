# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import bpy
from mathutils import Vector
import numpy as np
from numpy.random import uniform as U, normal as N

from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.math import FixedSeed
from infinigen.core.util import blender as butil
from infinigen.core.placement.factory import AssetFactory
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

bpy.ops.preferences.addon_enable(module='add_mesh_extra_objects')

class BlenderRockFactory(AssetFactory):

    def __init__(self, factory_seed, detail=1):
        super(BlenderRockFactory, self).__init__(factory_seed)
        self.detail = detail

    __repr__ = AssetFactory.__repr__

    def create_asset(self, **params):
        seed = np.random.randint(0, 99999)

        zscale = U(0.2, 0.8)
        zrand = U(0, 0.7)

        while True:
            try:
                kwargs = dict(
                    use_random_seed=False,
                    user_seed=seed,
                    display_detail=self.detail, detail=self.detail,
                    scale_Z=(zrand*zscale, zscale), scale_fac=(1, 1, 1),
                    scale_X=(1.00, 1.01), scale_Y=(1.00, 1.01),  # Bug occurs otherwise, I think
                    deform=U(2, 10),    
                    rough=U(0.5, 1.0)  # Higher than 1.0 can cause self-intersection
                )
                # The rock generator is poorly built.
                # It uses a weibull distribution to sample from a list, which will fail w/ 1.111% probability.
                bpy.ops.mesh.add_mesh_rock(**kwargs)
                break
            except IndexError:
                pass
            except RuntimeError:
                pass
        obj = bpy.context.active_object
        bpy.ops.object.shade_flat()
        tag_object(obj, 'blender_rock')

        return obj