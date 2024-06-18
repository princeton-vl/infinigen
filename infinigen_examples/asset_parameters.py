# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import numpy as np

from infinigen.assets.clothes import blanket
from infinigen.assets.materials import metal, fabrics, ceramic
from infinigen.assets.materials.woods import wood
from infinigen.assets.scatters.clothes import ClothesCover
from infinigen.assets.seating import ChairFactory
from infinigen.assets.tableware import PotFactory, PanFactory, FruitContainerFactory
from infinigen.core.surface import NoApply

parameters = {
    'ChairFactory': {
        'factories': [ChairFactory] * 16,
        'globals': {
        },
        'individuals': [{}, {'arm_mid': [-.03, -.03, .09], 'leg_height': .5, 'leg_x_offset': 0},
                           {'arm_mid': [0, 0, 0], 'leg_height': .6, 'leg_x_offset': .02},
                           {'arm_mid': [.03, .09, -.03], 'leg_height': .7, 'leg_x_offset': .05}, {},
                           {'leg_offset_bar': (.2, .4), 'seat_front': 1., 'back_vertical_cuts': 1},
                           {'leg_offset_bar': (.4, .6), 'seat_front': 1.1, 'back_vertical_cuts': 2},
                           {'leg_offset_bar': (.6, .8), 'seat_front': 1.2, 'back_vertical_cuts': 3}, {}] + [{}] * 7,
        'repeat': 12,
        'indices': [0] * 9 + list(range(1, 8)),
        'scene_idx': 4,
    },

    'PanFactory': {
        'factories': [PanFactory] * 8 + [PanFactory] * 2 + [PotFactory] * 3 + [FruitContainerFactory] * 3,
        'globals': {
        },
        'individuals': [{}, {'scale': .1, 'depth': .3, 'x_handle': 2, }, {'scale': .12, 'depth': .5, 'x_handle': 1.5},
                           {'scale': .15, 'depth': .8, 'x_handle': 1.2}, {},
                           {'s_handle': .8, 'r_expand': 1, 'x_guard': 1, },
                           {'s_handle': 1., 'r_expand': 1.15, 'x_guard': 1.3},
                           {'s_handle': 1.2, 'r_expand': 1.3, 'x_guard': 1.6}, {}] + [{}] * 7,
        'repeat': 12,
        'indices': [0] * 9 + list(range(1, 8)),
        'scene_idx': 2,
    },

}
