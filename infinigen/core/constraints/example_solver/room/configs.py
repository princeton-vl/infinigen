# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from collections import defaultdict

from infinigen.assets.materials import brick, plaster, rug, tile
from infinigen.assets.materials.stone_and_concrete import concrete
from infinigen.assets.materials.woods import tiled_wood
from infinigen.core.constraints.example_solver.room.types import RoomType
from infinigen.core.util.color import hsv2rgba
from infinigen.core.util.random import log_uniform

EXTERIOR_CONNECTED_ROOM_TYPES = [
    RoomType.Bedroom,
    RoomType.Garage,
    RoomType.Balcony,
    RoomType.DiningRoom,
    RoomType.Kitchen,
    RoomType.LivingRoom,
]
SQUARE_ROOM_TYPES = [
    RoomType.Kitchen,
    RoomType.Bedroom,
    RoomType.LivingRoom,
    RoomType.Closet,
    RoomType.Bathroom,
    RoomType.Garage,
    RoomType.Balcony,
    RoomType.DiningRoom,
    RoomType.Utility,
]
TYPICAL_AREA_ROOM_TYPES = {
    RoomType.Kitchen: 20,
    RoomType.Bedroom: 25,
    RoomType.LivingRoom: 30,
    RoomType.DiningRoom: 20,
    RoomType.Closet: 4,
    RoomType.Bathroom: 8,
    RoomType.Utility: 4,
    RoomType.Garage: 30,
    RoomType.Balcony: 8,
    RoomType.Hallway: 8,
    RoomType.Staircase: 20,
}
ROOM_NUMBERS = {RoomType.Bathroom: (1, 10), RoomType.LivingRoom: (1, 10)}
COMBINED_ROOM_TYPES = [
    [RoomType.Hallway, RoomType.LivingRoom, RoomType.DiningRoom],
    [RoomType.Garage],
]
PANORAMIC_ROOM_TYPES = {
    RoomType.Hallway: 0.3,
    RoomType.LivingRoom: 0.5,
    RoomType.DiningRoom: 0.5,
    RoomType.Balcony: 1,
}
FUNCTIONAL_ROOM_TYPES = [
    RoomType.Kitchen,
    RoomType.Bedroom,
    RoomType.LivingRoom,
    RoomType.Bathroom,
    RoomType.DiningRoom,
]
WINDOW_ROOM_TYPES = defaultdict(
    lambda: 1,
    {
        RoomType.Utility: 0.3,
        RoomType.Closet: 0.0,
        RoomType.Bathroom: 0.5,
        RoomType.Garage: 0.5,
    },
)


def make_room_colors():
    bedroom_color = hsv2rgba(0.0, 0.8, log_uniform(0.02, 0.1))
    hallway_color = hsv2rgba(0.4, 0.8, log_uniform(0.02, 0.1))
    utility_color = hsv2rgba(0.8, 0.8, log_uniform(0.02, 0.1))
    return {
        RoomType.Kitchen: hallway_color,
        RoomType.Bedroom: bedroom_color,
        RoomType.LivingRoom: hallway_color,
        RoomType.Closet: bedroom_color,
        RoomType.Hallway: hallway_color,
        RoomType.Bathroom: bedroom_color,
        RoomType.Garage: utility_color,
        RoomType.Balcony: utility_color,
        RoomType.DiningRoom: hallway_color,
        RoomType.Utility: utility_color,
        RoomType.Staircase: hallway_color,
    }


ROOM_COLORS = make_room_colors()
ROOM_CHILDREN = defaultdict(
    dict,
    {
        RoomType.LivingRoom: {
            RoomType.LivingRoom: ("bool", 0.1),
            RoomType.Bedroom: ("categorical", 0.0, 0.45, 0.4, 0.1, 0.05),
            RoomType.Closet: ("bool", 0.1),
            RoomType.Bathroom: ("bool", 0.2),
            RoomType.Garage: ("bool", 0.2),
            RoomType.Balcony: ("bool", 0.2),
            RoomType.DiningRoom: ("bool", 1.0),
            RoomType.Utility: ("bool", 0.2),
            RoomType.Hallway: ("categorical", 0.5, 0.4, 0.1),
        },
        RoomType.Kitchen: {
            RoomType.Garage: ("bool", 0.5),
            RoomType.Utility: ("bool", 0.1),
        },
        RoomType.Bedroom: {
            RoomType.Bathroom: ("bool", 0.3),
            RoomType.Closet: ("bool", 0.5),
        },
        RoomType.Bathroom: {RoomType.Closet: ("bool", 0.2)},
        RoomType.DiningRoom: {
            RoomType.Kitchen: ("bool", 1.0),
            RoomType.Hallway: ("bool", 0.2),
        },
    },
)

STUDIO_ROOM_CHILDREN = defaultdict(
    dict,
    {
        RoomType.LivingRoom: {
            RoomType.Bedroom: ("categorical", 0.0, 1.0),
            RoomType.DiningRoom: ("bool", 1.0),
        },
        RoomType.Bedroom: {RoomType.Bathroom: ("bool", 1.0)},
        RoomType.DiningRoom: {RoomType.Kitchen: ("bool", 1.0)},
    },
)
UPSTAIRS_ROOM_CHILDREN = defaultdict(
    dict,
    {
        RoomType.LivingRoom: {
            RoomType.Bedroom: ("categorical", 0.0, 0.4, 0.5, 0.2),
            RoomType.Closet: ("bool", 0.2),
            RoomType.Bathroom: ("bool", 0.4),
            RoomType.Balcony: ("bool", 0.4),
            RoomType.Utility: ("bool", 0.2),
            RoomType.Hallway: ("categorical", 0.0, 0.5, 0.5),
        },
        RoomType.Bedroom: {
            RoomType.Bathroom: ("bool", 0.3),
            RoomType.Closet: ("bool", 0.5),
        },
        RoomType.Bathroom: {RoomType.Closet: ("bool", 0.2)},
        RoomType.Balcony: {
            RoomType.Utility: ("bool", 0.4),
            RoomType.Hallway: ("bool", 0.1),
        },
    },
)
LOOP_ROOM_TYPES = {
    RoomType.LivingRoom: {
        RoomType.Garage: 0.2,
        RoomType.Balcony: 0.2,
        RoomType.Kitchen: 0.1,
    },
    RoomType.Bedroom: {RoomType.Balcony: 0.1},
}

ROOM_WALLS = defaultdict(
    lambda: plaster,
    {
        RoomType.Kitchen: ("weighted_choice", (2, tile), (5, plaster)),
        RoomType.Garage: ("weighted_choice", (5, concrete), (1, brick), (3, plaster)),
        RoomType.Utility: (
            "weighted_choice",
            (1, concrete),
            (1, brick),
            (1, brick),
            (5, plaster),
        ),
        RoomType.Balcony: ("weighted_choice", (1, brick), (5, plaster)),
        RoomType.Bathroom: tile,
    },
)

ROOM_FLOORS = defaultdict(
    lambda: ("weighted_choice", (3, tiled_wood), (1, tile), (1, rug)),
    {
        RoomType.Garage: concrete,
        RoomType.Utility: ("weighted_choice", (1, concrete), (1, plaster), (1, tile)),
        RoomType.Bathroom: tile,
        RoomType.Balcony: tile,
    },
)

PILLAR_ROOM_TYPES = [
    RoomType.Hallway,
    RoomType.LivingRoom,
    RoomType.Staircase,
    RoomType.DiningRoom,
]
