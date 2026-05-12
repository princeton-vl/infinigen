# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import numpy as np

from .cantilever import CantileverStaircaseFactory
from .curved import CurvedStaircaseFactory
from .l_shaped import LShapedStaircaseFactory
from .spiral import SpiralStaircaseFactory
from .straight import StraightStaircaseFactory
from .u_shaped import UShapedStaircaseFactory


def random_staircase_factory():
    door_factories = [
        StraightStaircaseFactory,
        LShapedStaircaseFactory,
        UShapedStaircaseFactory,
        SpiralStaircaseFactory,
        CurvedStaircaseFactory,
        CantileverStaircaseFactory,
    ]
    door_probs = np.array([2, 2, 2, 0.5, 2, 2])
    return np.random.choice(door_factories, p=door_probs / door_probs.sum())
