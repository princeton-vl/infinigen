# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

from infinigen.core.util.random import random_general as rg

from ...utils.uv import unwrap_faces
from .. import common
from .coarse_knit_fabric import shader_fabric_random as shader_coarse_fabric_random
from .fine_knit_fabric import shader_fabric_random as shader_fine_fabric_random

# Authors: Lingjie Mei
from .general_fabric import shader_fabric
from .leather import shader_leather
from .lined_fabric import shader_lined_fur_base
from .sofa_fabric import shader_sofa_fabric

fabric_shader_list = (
    "weighted_choice",
    (1, shader_coarse_fabric_random),
    (1, shader_fine_fabric_random),
    (2, shader_leather),
    (1, shader_sofa_fabric),
)  # (1, shader_fabric),


def apply(obj, selection=None, **kwargs):
    unwrap_faces(obj, selection)
    common.apply(obj, rg(fabric_shader_list), selection=selection, **kwargs)
