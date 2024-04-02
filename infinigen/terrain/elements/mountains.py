# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


import gin
import numpy as np
from numpy import ascontiguousarray as AC
from infinigen.terrain.utils import random_int
from infinigen.core.util.organization import Materials, Transparency, ElementNames, ElementTag
from infinigen.core.util.random import random_general as rg

from .core import Element

# this element is used as auxiliary element
@gin.configurable
class Mountains(Element):
    name = ElementNames.Mountains
    def __init__(
        self,
        device,
        min_freq, max_freq,
        height,  # i.e. scale (not base height)
        coverage,
        slope_height, # i.e. slope_scale
        n_groups=3,
        is_3d=False,
        spherical_radius=-1,
        octaves=15,
        mask_freq_ratio=1,
        mask_octaves=2,
        slope_freq=0.01,
        slope_octaves=9,
        material=Materials.MountainCollection,
        transparency=Transparency.Opaque,
    ):
        self.device = device
        min_freq = rg(min_freq)
        max_freq = rg(max_freq)
        octaves = rg(octaves)
        height = rg(height)
        mask_freq = min_freq * mask_freq_ratio
        mask_octaves = rg(mask_octaves)
        coverage = rg(coverage)
        mask_ramp_min, mask_ramp_max = -1.1 - coverage*2, -0.9 - coverage*2
        slope_freq = rg(slope_freq)
        slope_octaves = rg(slope_octaves)
        slope_height = rg(slope_height)
        
        self.int_params = AC(np.array([random_int(), n_groups, is_3d], dtype=np.int32))
        self.float_params = AC(np.array([
            spherical_radius,
            min_freq, max_freq, octaves, height * 2,
            mask_freq, mask_octaves,
            mask_ramp_min, mask_ramp_max,
            slope_freq, slope_octaves, slope_height,
        ], dtype=np.float32))

        Element.__init__(self, "mountains", material, transparency)
        self.tag = ElementTag.Terrain