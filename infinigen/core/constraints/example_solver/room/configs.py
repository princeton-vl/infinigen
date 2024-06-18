# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from collections import defaultdict

from infinigen.assets.materials import brick, hardwood_floor, plaster, rug, tile
from infinigen.assets.materials.woods import tiled_wood
from infinigen.assets.materials.stone_and_concrete import concrete
from infinigen.core.util.color import hsv2rgba
from infinigen.core.util.random import log_uniform
from infinigen.core.util.random import random_general as rg

EXTERIOR_CONNECTED_ROOM_TYPES = [RoomType.Bedroom, RoomType.Garage, RoomType.Balcony, RoomType.DiningRoom,
    RoomType.Kitchen, RoomType.LivingRoom]
SQUARE_ROOM_TYPES = [RoomType.Kitchen, RoomType.Bedroom, RoomType.LivingRoom, RoomType.Closet,
    RoomType.Bathroom, RoomType.Garage, RoomType.Balcony, RoomType.DiningRoom, RoomType.Utility]
TYPICAL_AREA_ROOM_TYPES = {
    RoomType.Kitchen: 20,
    RoomType.Bedroom: 25,
    RoomType.LivingRoom: 30,
    RoomType.Closet: 4,
    RoomType.Utility: 4,
    RoomType.Garage: 30,
    RoomType.Balcony: 8,
    RoomType.Hallway: 8,
    RoomType.Staircase: 20,
}
ROOM_NUMBERS = {RoomType.Bathroom: (1, 10), RoomType.LivingRoom: (1, 10)}
COMBINED_ROOM_TYPES = [[RoomType.Hallway, RoomType.LivingRoom, RoomType.DiningRoom], [RoomType.Garage]]
FUNCTIONAL_ROOM_TYPES = [RoomType.Kitchen, RoomType.Bedroom, RoomType.LivingRoom, RoomType.Bathroom,
    RoomType.DiningRoom]
WINDOW_ROOM_TYPES = defaultdict(lambda: 1, {
    RoomType.Utility: .3,
    RoomType.Closet: 0.,
    RoomType.Bathroom: .5,
    RoomType.Garage: .5,
})


def make_room_colors():
    bedroom_color = hsv2rgba(0., .8, log_uniform(.02, .1))
    hallway_color = hsv2rgba(.4, .8, log_uniform(.02, .1))
    utility_color = hsv2rgba(.8, .8, log_uniform(.02, .1))
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
ROOM_CHILDREN = defaultdict(dict, {
    RoomType.LivingRoom: {
        RoomType.LivingRoom: ('bool', .1),
        RoomType.Closet: ('bool', .1),
        RoomType.Balcony: ('bool', .2),
        RoomType.Utility: ('bool', .2),
    },
    },
    RoomType.Bedroom: {RoomType.Bathroom: ('bool', .3), RoomType.Closet: ('bool', .5)},
    RoomType.Bathroom: {RoomType.Closet: ('bool', .2)},
    RoomType.DiningRoom: {RoomType.Kitchen: ('bool', 1.), RoomType.Hallway: ('bool', .2)
    }
})

STUDIO_ROOM_CHILDREN = defaultdict(dict, {
    RoomType.LivingRoom: {
        RoomType.Bedroom: ('categorical', .0, 1.),
        RoomType.DiningRoom: ('bool', 1.),
    },
    RoomType.Bedroom: {RoomType.Bathroom: ('bool', 1.)},
    RoomType.DiningRoom: {RoomType.Kitchen: ('bool', 1.)
    }
})
UPSTAIRS_ROOM_CHILDREN = defaultdict(dict, {
    RoomType.LivingRoom: {
        RoomType.Bedroom: ('categorical', .0, .4, .5, .2),
        RoomType.Closet: ('bool', .2),
        RoomType.Bathroom: ('bool', .4),
        RoomType.Balcony: ('bool', .4),
        RoomType.Utility: ('bool', .2),
        RoomType.Hallway: ('categorical', .0, .5, .5)
    },
    RoomType.Bedroom: {RoomType.Bathroom: ('bool', .3), RoomType.Closet: ('bool', .5)},
    RoomType.Bathroom: {RoomType.Closet: ('bool', .2)},
    RoomType.Balcony: {RoomType.Utility: ('bool', .4), RoomType.Hallway: ('bool', .1)},
})
LOOP_ROOM_TYPES = {
    RoomType.LivingRoom: {RoomType.Garage: .2, RoomType.Balcony: .2, RoomType.Kitchen: .1},
    RoomType.Bedroom: {RoomType.Balcony: .1},
}

ROOM_WALLS = defaultdict(lambda: plaster, {
    RoomType.Kitchen: ('weighted_choice', (2, tile), (5, plaster)),
    RoomType.Garage: ('weighted_choice', (5, concrete), (1, brick), (3, plaster)),
    RoomType.Utility: ('weighted_choice', (1, concrete), (1, brick), (1, brick), (5, plaster)),
    RoomType.Balcony: ('weighted_choice', (1, brick), (5, plaster)),
    RoomType.Bathroom: tile
})

    RoomType.Garage: concrete,
    RoomType.Utility: ('weighted_choice', (1, concrete), (1, plaster), (1, tile)),
    RoomType.Bathroom: tile,
    RoomType.Balcony: tile
})

PILLAR_ROOM_TYPES = [RoomType.Hallway, RoomType.LivingRoom, RoomType.Staircase, RoomType.DiningRoom]
