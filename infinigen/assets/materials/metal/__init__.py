# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei





import numpy as np

from .aluminum import Aluminum
from .appliance import BlackGlass, BrushedBlackMetal, WhiteMetal
from .brushed_metal import BrushedMetal
from .galvanized_metal import GalvanizedMetal
from .grained_and_polished_metal import GrainedMetal
from .hammered_metal import HammeredMetal
from .metal_basic import MetalBasic
from .mirror import Mirror


def get_shader():
    return np.random.choice(
        [
            brushed_metal.shader_brushed_metal,
            galvanized_metal.shader_galvanized_metal,
            grained_and_polished_metal.shader_grained_metal,
            hammered_metal.shader_hammered_metal,
        ]
    )
