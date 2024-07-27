# Copyright (C) 2023, Princeton University.

# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

from infinigen.assets.materials import common
from infinigen.assets.utils.uv import unwrap_faces
from infinigen.core.util.random import random_general as rg

from .coarse_knit_fabric import shader_coarse_knit_fabric
from .fine_knit_fabric import shader_fine_knit_fabric
from .leather import shader_leather
from .sofa_fabric import shader_sofa_fabric

fabric_shader_list = (
    "weighted_choice",
    (1, shader_coarse_knit_fabric),
    (1, shader_fine_knit_fabric),
    (2, shader_leather),
    (1, shader_sofa_fabric),
)


def apply(obj, selection=None, **kwargs):
    unwrap_faces(obj, selection)
    common.apply(obj, rg(fabric_shader_list), selection=selection, **kwargs)
