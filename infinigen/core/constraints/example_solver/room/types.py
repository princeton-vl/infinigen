# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from typing import List

import networkx as nx


class RoomType:
    Kitchen = "kitchen"
    Bedroom = "bedroom"
    LivingRoom = "living-room"
    Closet = "closet"
    Hallway = "hallway"
    Bathroom = "bathroom"
    Garage = "garage"
    Balcony = "balcony"
    DiningRoom = "dining-room"
    Utility = "utility"
    Staircase = "staircase"


def get_room_type(name):
    return name.split("_")[0]


def get_room_level(name):
    return int(name.split("-")[-1])


class RoomGraph:
    def __init__(self, children: List[List[int]], rooms, entrance=None):
        self.neighbours = [[] for _ in children]
        for i, cs in enumerate(children):
            for c in cs:
                self.neighbours[i].append(c)
                self.neighbours[c].append(i)
        self.rooms = rooms
        self.entrance = entrance

    @property
    def is_planar(self):
        try:
            nx.planar_layout(self.to_nx)
            return True
        except nx.NetworkXException:
            return False

    @property
    def to_nx(self):
        g = nx.Graph()
        g.add_nodes_from(self.rooms)
        for k in range(len(self.rooms)):
            for l in self.neighbours[k]:
                g.add_edge(self.rooms[k], self.rooms[l])
        return g

    @property
    def cycle_basis(self):
        return nx.cycle_basis(self.to_nx)

    def __getitem__(self, item):
        return [i for i, r in enumerate(self.rooms) if get_room_type(r) == item]

    def __len__(self):
        return len(self.rooms)

    def __str__(self):
        return {
            "neighbours": self.neighbours,
            "rooms": self.rooms,
            "entrance": self.entrance,
        }


def make_demo_tree():
    children = [
        [1, 2],
        [],
        [3, 4],
        [5, 6],
        [7],
        [8, 9],
        [10, 11],
        [],
        [],
        [12],
        [],
        [13],
        [],
        [14],
        [],
    ]
    rooms = [
        "hallway_0",
        "closet_0",
        "kitchen_0",
        "dining-room_0",
        "utility_0",
        "hallway_1",
        "living-room_0",
        "utility_1",
        "bathroom_0",
        "bedroom_0",
        "balcony_0",
        "bedroom_1",
        "closet_1",
        "bathroom_1",
        "closet_2",
    ]
    return RoomGraph(children, rooms, 0)


DEMO_GRAPH = make_demo_tree()
