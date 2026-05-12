# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei

from infinigen.assets.objects.seating import ChairFactory
from infinigen.assets.objects.tableware import (
    FruitContainerFactory,
    PanFactory,
    PotFactory,
)

parameters = {
    "ChairFactory": {
        "factories": [ChairFactory] * 16,
        "globals": {},
        "individuals": [
            {},
            {"arm_mid": [-0.03, -0.03, 0.09], "leg_height": 0.5, "leg_x_offset": 0},
            {"arm_mid": [0, 0, 0], "leg_height": 0.6, "leg_x_offset": 0.02},
            {"arm_mid": [0.03, 0.09, -0.03], "leg_height": 0.7, "leg_x_offset": 0.05},
            {},
            {"leg_offset_bar": (0.2, 0.4), "seat_front": 1.0, "back_vertical_cuts": 1},
            {"leg_offset_bar": (0.4, 0.6), "seat_front": 1.1, "back_vertical_cuts": 2},
            {"leg_offset_bar": (0.6, 0.8), "seat_front": 1.2, "back_vertical_cuts": 3},
            {},
        ]
        + [{}] * 7,
        "repeat": 12,
        "indices": [0] * 9 + list(range(1, 8)),
        "scene_idx": 4,
    },
    "PanFactory": {
        "factories": [PanFactory] * 8
        + [PanFactory] * 2
        + [PotFactory] * 3
        + [FruitContainerFactory] * 3,
        "globals": {},
        "individuals": [
            {},
            {
                "scale": 0.1,
                "depth": 0.3,
                "x_handle": 2,
            },
            {"scale": 0.12, "depth": 0.5, "x_handle": 1.5},
            {"scale": 0.15, "depth": 0.8, "x_handle": 1.2},
            {},
            {
                "s_handle": 0.8,
                "r_expand": 1,
                "x_guard": 1,
            },
            {"s_handle": 1.0, "r_expand": 1.15, "x_guard": 1.3},
            {"s_handle": 1.2, "r_expand": 1.3, "x_guard": 1.6},
            {},
        ]
        + [{}] * 7,
        "repeat": 12,
        "indices": [0] * 9 + list(range(1, 8)),
        "scene_idx": 2,
    },
}
