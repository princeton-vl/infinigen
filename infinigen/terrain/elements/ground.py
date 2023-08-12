# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


import gin
import numpy as np
from numpy import ascontiguousarray as AC
from infinigen.terrain.utils import random_int
from infinigen.core.util.organization import Materials, Transparency, Tags, ElementNames, ElementTag
from infinigen.core.util.random import random_general as rg

from .core import Element


@gin.configurable
class Ground(Element):
    name = ElementNames.Ground
    def __init__(
        self,
        device,
        caves,
        material=Materials.GroundCollection,
        transparency=Transparency.Opaque,
        is_3d=False,
        spherical_radius=-1,
        freq=0.01,
        octaves=9,
        scale=5,
        height=0,
        with_sand_dunes=0,
        sand_dunes_warping_freq=("log_uniform", 0.01, 0.1),
        sand_dunes_warping_octaves=2,
        sand_dunes_warping_scale=("uniform", 3, 6),
        sand_dunes_freq=("log_uniform", 0.01, 0.1),
        sand_dunes_scale=0.1,
    ):
        self.device = device
        seed = random_int()
        sand_dunes_warping_freq = rg(sand_dunes_warping_freq)
        sand_dunes_warping_scale = rg(sand_dunes_warping_scale)
        sand_dunes_freq = rg(sand_dunes_freq)
        self.aux_names = []
        if caves is None:
            self.aux_names.append(None)
        else:
            self.aux_names.append(Tags.Cave)
            self.int_params2 = caves.int_params
            self.float_params2 = caves.float_params

        self.int_params = AC(np.array([seed, is_3d, with_sand_dunes], dtype=np.int32))
        self.float_params = AC(np.array([
            spherical_radius,
            freq, octaves, scale, height,
            sand_dunes_warping_freq, sand_dunes_warping_octaves, sand_dunes_warping_scale,
            sand_dunes_freq, sand_dunes_scale
        ], dtype=np.float32))

        self.meta_params = [caves is not None]
        Element.__init__(self, "ground", material, transparency)
        self.tag = ElementTag.Terrain