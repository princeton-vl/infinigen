# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


import gin
import numpy as np
from numpy import ascontiguousarray as AC
from infinigen.terrain.utils import random_int
from infinigen.core.util.organization import Materials, ElementNames, Transparency, ElementTag, Tags
from .core import Element


@gin.configurable
class WarpedRocks(Element):
    name = ElementNames.WarpedRocks
    def __init__(
        self,
        device,
        caves,
        material=Materials.MountainCollection,
        transparency=Transparency.Opaque,
        slope_is_3d=False,
        supressing_param=3,
        content_min_freq=0.06, content_max_freq=0.1, content_octaves=15, content_scale=40,
        warp_min_freq=0.1, warp_max_freq=0.15, warp_octaves=3, warp_scale=5,
        slope_freq=0.02, slope_octaves=5, slope_scale=20, slope_shift=0
    ):
        self.device = device
        seed = random_int()
        self.aux_names = []
        if caves is None:
            self.aux_names.append(None)
        else:
            self.aux_names.append(Tags.Cave)
            self.int_params2 = caves.int_params
            self.float_params2 = caves.float_params
        
        self.int_params = AC(np.array([seed, slope_is_3d], dtype=np.int32))
        self.float_params = AC(np.array([
            supressing_param,
            content_min_freq, content_max_freq, content_octaves, content_scale,
            warp_min_freq, warp_max_freq, warp_octaves, warp_scale,
            slope_freq, slope_octaves, slope_scale, slope_shift,
        ], dtype=np.float32))

        self.meta_params = [caves is not None]
        Element.__init__(self, "warped_rocks", material, transparency)
        self.tag = ElementTag.WarpedRocks