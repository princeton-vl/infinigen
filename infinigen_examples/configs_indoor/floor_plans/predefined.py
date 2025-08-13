# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lingjie Mei: Predefined floor plan solver

import shapely


def example(factory_seed):
    return {
        "rooms": {
            "dining-room_0/0": {
                "shape": shapely.Polygon([(0, 0), (7, 0), (7, 2), (2, 7), (0, 7)])
            },
            "kitchen_0/0": {"shape": shapely.box(-2.5, 0, 0, 6)},
        },
        "doors": {
            "door": {"shape": shapely.LineString([(0, 4), (0, 6)])},
            "door.001": {"shape": shapely.LineString([(1, 0), (2.5, 0)])},
        },
        "opens": {
            "open": {"shape": shapely.LineString([(7, 2), (2, 7), (1, 7)])},
            "open.001": {"shape": shapely.LineString([(0, 0), (0, 2)])},
            "open.002": {"shape": shapely.LineString([(-2.5, 0), (-2.5, 5)])},
        },
        "interiors": {"interior": {"shape": shapely.LineString([(0, 2), (0, 4)])}},
        "windows": {
            "window": {"shape": shapely.LineString([(7, 0.5), (7, 1.5)])},
            "window.001": {
                "shape": shapely.LineString([(-2, 0), (-0.5, 0)]),
                "is_panoramic": 1,
            },
        },
    }
