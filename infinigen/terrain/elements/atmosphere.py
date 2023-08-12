# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


import gin
import numpy as np
from numpy import ascontiguousarray as AC
from infinigen.core.util.organization import Materials, Transparency, ElementNames

from .core import Element

@gin.configurable
class Atmosphere(Element):
    name = ElementNames.Atmosphere
    def __init__(
        self,
        device,
        waterbody,
        height=130,
        spherical_radius=-1,
        hacky_offset=0,
        material=Materials.Atmosphere,
        transparency=Transparency.IndividualTransparent,
    ):
        self.device = device
        self.int_params = AC(np.array([], dtype=np.int32))
        self.float_params = AC(np.array([height, spherical_radius, hacky_offset], dtype=np.float32))
        if waterbody is not None:
            self.int_params2 = waterbody.int_params
            self.float_params2 = waterbody.float_params

        self.meta_params = [waterbody is not None]
        Element.__init__(self, "atmosphere", material, transparency)