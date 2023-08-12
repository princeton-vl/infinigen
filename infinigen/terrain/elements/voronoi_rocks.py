# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


import gin
import numpy as np
from numpy import ascontiguousarray as AC
from infinigen.terrain.utils import random_int
from infinigen.core.util.organization import Materials, ElementNames, Transparency, ElementTag, Tags

from .core import Element
from .landtiles import LandTiles

def none_to_0(x):
    if x is None: return 0
    return x

@gin.configurable
class VoronoiRocks(Element):
    name = ElementNames.VoronoiRocks
    def __init__(
        self,
        device,
        attachment,
        caves,
        material=Materials.MountainCollection,
        variable_material=False,
        transparency=Transparency.Opaque,
        n_lattice=3,
        min_freq=1, max_freq=10,
        gap_min_freq=0.003, gap_max_freq=0.03, gap_scale=0.1, gap_octaves=2, gap_base=10,
        warp_min_freq=0.1, warp_max_freq=0.5, warp_octaves=3, warp_prob=0.5,
        warp_modu_sigmoidscale=3, warp_modu_scale=0.4, warp_modu_octaves=2, warp_modu_freq=0.01,
        mask_octaves=11, mask_freq=0.05, mask_shift=-0.2,
    ):
        self.device = device
        seed = random_int()

        height_modification = hasattr(attachment, "attribute_modification_start_height") and attachment.attribute_modification_start_height is not None
        attribute_modification_start_height = attachment.attribute_modification_start_height if height_modification else None
        attribute_modification_end_height = attachment.attribute_modification_end_height if height_modification else None
        if height_modification and variable_material:
            self.aux_names = [Materials.Beach]
        else:
            self.aux_names = [None]
        if caves is None:
            self.aux_names.append(None)
        else:
            self.aux_names.append(Tags.Cave)

        self.int_params = AC(np.array([seed, n_lattice, height_modification], dtype=np.int32))
        self.float_params = AC(np.array([
            min_freq, max_freq,
            gap_min_freq, gap_max_freq, gap_scale, gap_octaves, gap_base,
            warp_min_freq, warp_max_freq, warp_octaves, warp_prob,
            warp_modu_sigmoidscale, warp_modu_scale, warp_modu_octaves, warp_modu_freq,
            mask_octaves, mask_freq, mask_shift,
            none_to_0(attribute_modification_start_height), none_to_0(attribute_modification_end_height)
        ], dtype=np.float32))

        self.int_params2 = attachment.int_params
        self.float_params2 = attachment.float_params
    
        if caves is not None:
            self.int_params3 = caves.int_params
            self.float_params3 = caves.float_params
        
        self.meta_params = [not isinstance(attachment, LandTiles), caves is not None]
        Element.__init__(self, "voronoi_rocks", material, transparency)
        self.tag = ElementTag.VoronoiRocks

class VoronoiGrains(VoronoiRocks):
    name = ElementNames.VoronoiGrains
    def __init__(
        self,
        device,
        attachment,
        caves,
        min_freq=30, max_freq=300,
    ):
        VoronoiRocks.__init__(self, device, attachment, caves, min_freq=min_freq, max_freq=max_freq, mask_shift=9, warp_prob=0, variable_material=1)
        self.tag = ElementTag.VoronoiGrains
