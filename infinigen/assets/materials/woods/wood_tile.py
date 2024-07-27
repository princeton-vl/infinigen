# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import numpy as np


def get_wood_tiles():
    from . import (
        composite_wood_tile,
        crossed_wood_tile,
        hexagon_wood_tile,
        square_wood_tile,
        staggered_wood_tile,
    )

    return [
        square_wood_tile,
        staggered_wood_tile,
        crossed_wood_tile,
        composite_wood_tile,
        hexagon_wood_tile,
    ]


def apply(obj, selection=None, vertical=False, scale=None, alternating=None, **kwargs):
    func = np.random.choice(get_wood_tiles())
    func.apply(obj, selection, vertical, scale, alternating, **kwargs)
