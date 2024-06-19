# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from collections import defaultdict, deque
from collections.abc import Sequence

import gin
import networkx as nx
import numpy as np
from numpy.random import uniform

from infinigen.core.constraints.example_solver.room.configs import (
    LOOP_ROOM_TYPES,
    ROOM_CHILDREN,
    ROOM_NUMBERS,
    STUDIO_ROOM_CHILDREN,
    TYPICAL_AREA_ROOM_TYPES,
    UPSTAIRS_ROOM_CHILDREN,
)
from infinigen.core.constraints.example_solver.room.types import (
    RoomGraph,
    RoomType,
    get_room_type,
)
from infinigen.core.constraints.example_solver.room.utils import unit_cast
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform
from infinigen.core.util.random import random_general as rg


@gin.configurable(denylist=["factory_seed", "level"])
class GraphMaker:
    def __init__(
        self,
        factory_seed,
        level=0,
        requires_staircase=False,
        room_children="home",
        typical_area_room_types=TYPICAL_AREA_ROOM_TYPES,
        loop_room_types=LOOP_ROOM_TYPES,
        room_numbers=ROOM_NUMBERS,
        max_cycle_basis=1,
        requires_bathroom_privacy=True,
        entrance_type=("weighted_choice", (0.5, "porch"), (0.5, "hallway")),
        hallway_alpha=1,
        no_hallway_children_prob=0.4,
    ):
        self.factory_seed = factory_seed
        with FixedSeed(factory_seed):
            self.requires_staircase = requires_staircase
            match room_children:
                case "home":
                    self.room_children = (
                        ROOM_CHILDREN if level == 0 else UPSTAIRS_ROOM_CHILDREN
                    )
                case _:
                    self.room_children = STUDIO_ROOM_CHILDREN
            self.hallway_room_types = [
                r for r, m in self.room_children.items() if RoomType.Hallway in m
            ]
            self.typical_area_room_types = typical_area_room_types
            self.loop_room_types = loop_room_types
            self.room_numbers = room_numbers
            self.max_samples = 1000
            self.slackness = log_uniform(1.5, 1.8)
            self.max_cycle_basis = max_cycle_basis
            self.requires_bathroom_privacy = requires_bathroom_privacy
            self.entrance_type = rg(entrance_type)
            self.hallway_prob = lambda x: 1 / (x + hallway_alpha)
            self.no_hallway_children_prob = no_hallway_children_prob
            self.skewness_min = 0.7

    def make_graph(self, i):
        with FixedSeed(i):
            for _ in range(self.max_samples):
                room_type_counts = defaultdict(int)
                rooms = []
                children = defaultdict(list)
                queue = deque()

                def add_room(t, p):
                    i = len(rooms)
                    name = f"{t}_{room_type_counts[t]}"
                    room_type_counts[t] += 1
                    if p is not None:
                        children[p].append(i)
                    rooms.append(name)
                    queue.append(i)

                add_room(RoomType.LivingRoom, None)
                while len(queue) > 0:
                    i = queue.popleft()
                    for rt, spec in self.room_children[get_room_type(rooms[i])].items():
                        for _ in range(rg(spec)):
                            add_room(rt, i)

                if self.requires_bathroom_privacy and not self.has_bathroom_privacy:
                    continue

                for i, r in enumerate(rooms):
                    for j, s in enumerate(rooms):
                        if (rt := get_room_type(r)) in self.loop_room_types:
                            if (rt_ := get_room_type(s)) in self.loop_room_types[rt]:
                                if (
                                    uniform() < self.loop_room_types[rt][rt_]
                                    and j not in children[i]
                                ):
                                    children[i].append(j)

                for i, r in enumerate(rooms):
                    if get_room_type(r) in self.hallway_room_types:
                        hallways = [
                            j
                            for j in children[i]
                            if get_room_type(rooms[j]) == RoomType.Hallway
                        ]
                        other_rooms = [
                            j
                            for j in children[i]
                            if get_room_type(rooms[j]) != RoomType.Hallway
                        ]
                        children[i] = hallways.copy()
                        for k, o in enumerate(other_rooms):
                            if (
                                uniform() < self.no_hallway_children_prob
                                or len(hallways) == 0
                            ):
                                children[i].append(o)
                            else:
                                children[
                                    hallways[np.random.randint(len(hallways))]
                                ].append(o)

                hallways = [
                    i
                    for i, r in enumerate(rooms)
                    if get_room_type(r) == RoomType.Hallway
                ]
                if len(hallways) == 0:
                    entrance = 0
                else:
                    if self.requires_staircase:
                        prob = np.array(
                            [self.hallway_prob(len(children[h])) for h in hallways]
                        )
                        add_room(
                            RoomType.Staircase,
                            np.random.choice(hallways, p=prob / prob.sum()),
                        )
                    prob = np.array(
                        [self.hallway_prob(len(children[h])) for h in hallways]
                    )
                    entrance = np.random.choice(hallways, p=prob / prob.sum())
                    if self.entrance_type == "porch":
                        add_room(RoomType.Balcony, entrance)
                        entrance = queue.pop()
                    elif self.entrance_type == "none":
                        entrance = None

                children_ = [children[i] for i in range(len(rooms))]
                room_graph = RoomGraph(children_, rooms, entrance)
                if self.satisfies_constraint(room_graph):
                    return room_graph

    __call__ = make_graph

    def satisfies_constraint(self, graph):
        if not graph.is_planar or len(graph.cycle_basis) > self.max_cycle_basis:
            return False
        for room_type, constraint in self.room_numbers.items():
            if isinstance(constraint, Sequence):
                n_min, n_max = constraint
            else:
                n_min, n_max = constraint, constraint
            if not n_min <= len(graph[room_type]) <= n_max:
                return False
        return True

    def has_bathroom_privacy(self, rooms, children):
        for i, r in rooms:
            if get_room_type(r) == RoomType.LivingRoom:
                has_public_bathroom = any(
                    get_room_type(rooms[j]) == RoomType.Bathroom for j in children[i]
                )
                if not has_public_bathroom:
                    for j in children[i]:
                        if get_room_type(rooms[j] == RoomType.Bedroom):
                            if not any(get_room_type(rooms[k]) for k in children[j]):
                                return False
        return True

    def suggest_dimensions(self, graph, width=None, height=None):
        area = (
            sum([self.typical_area_room_types[get_room_type(r)] for r in graph.rooms])
            * self.slackness
        )
        if width is None and height is None:
            skewness = uniform(self.skewness_min, 1 / self.skewness_min)
            width = unit_cast(np.sqrt(area * skewness).item())
            height = unit_cast(np.sqrt(area / skewness).item())
        elif uniform(0, 1) < 0.5:
            height_ = unit_cast(area / width)
            height = (
                None
                if height_ > height
                and self.skewness_min < height_ / width < 1 / self.skewness_min
                else height_
            )
        else:
            width_ = unit_cast(area / height)
            width = (
                None
                if width_ > width
                and self.skewness_min < width_ / height < 1 / self.skewness_min
                else width_
            )

        return width, height

    def draw(self, graph):
        g = nx.Graph()
        shortnames = [r[:3].upper() + r.split("_")[-1] for r in graph.rooms]
        g.add_nodes_from(shortnames)
        for k in range(len(shortnames)):
            for l in graph.neighbours[k]:
                g.add_edge(shortnames[k], shortnames[l])
        nx.draw_planar(g, with_labels=True)
