# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


import gin
import numpy as np
from numpy import ascontiguousarray as AC
from infinigen.core.util.organization import Materials, Transparency, ElementNames, ElementTag, Attributes

from .core import Element


@gin.configurable
class Waterbody(Element):
    name = ElementNames.Liquid
    def __init__(
        self,
        device,
        landtiles,
        height=0,
        spherical_radius=-1,
        material=Materials.LiquidCollection,
        transparency=Transparency.IndividualTransparent,
    ):
        self.device = device
        self.height = height

        self.int_params = AC(np.zeros(0, dtype=np.int32))
        self.float_params = AC(np.array([height, spherical_radius], dtype=np.float32))
        self.meta_params = [landtiles is not None]
        if landtiles is not None:
            self.int_params2 = landtiles.int_params
            self.float_params2 = landtiles.float_params
            self.int_params3 = landtiles.int_params2
            self.float_params3 = landtiles.float_params2
            self.meta_params.append(landtiles.meta_params[0])
            self.aux_names = [Attributes.BoundarySDF]
        else:
            self.aux_names = [None]

        Element.__init__(self, "waterbody", material, transparency)
        self.tag = ElementTag.Liquid