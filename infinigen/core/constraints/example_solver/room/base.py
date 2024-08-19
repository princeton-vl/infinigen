# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lingjie Mei
from typing import List

import networkx as nx
from matplotlib import pyplot as plt

from infinigen.core.tags import Semantics
from infinigen.core.util.math import int_hash


class RoomGraph:
    def __init__(self, children: List[List[int]], names, entrance=None):
        self.ns = [[] for _ in children]
        for i, cs in enumerate(children):
            for c in cs:
                if c not in self.ns[i]:
                    self.ns[i].append(c)
                if i not in self.ns[c]:
                    self.ns[c].append(i)
        self.names = names
        self._entrance = entrance
        self.invalid_indices = {
            i
            for i, n in enumerate(self.names)
            if room_type(n) in {Semantics.Exterior, Semantics.Entrance}
        }

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
        g.add_nodes_from(self.names)
        for k in range(len(self.names)):
            for l in self.ns[k]:
                g.add_edge(self.names[k], self.names[l])
        return g

    @property
    def cycle_basis(self):
        return nx.cycle_basis(self.to_nx)

    def __getitem__(self, item):
        return [i for i, r in enumerate(self.names) if room_type(r) == item]

    def __len__(self):
        return len(self.names) - 1

    def __str__(self):
        return str(
            {"neighbours": self.ns, "rooms": self.names, "entrance": self._entrance}
        )

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return int_hash(str(self))

    @property
    def neighbours(self):
        return {
            self.names[i]: set(self.names[n_] for n_ in n)
            for i, n in enumerate(self.ns)
        }

    @property
    def valid_neighbours(self):
        return {
            self.names[i]: set(
                self.names[n_] for n_ in n if n_ not in self.invalid_indices
            )
            for i, n in enumerate(self.ns)
            if i not in self.invalid_indices
        }

    @property
    def valid_ns(self):
        return {
            i: set(n_ for n_ in n if n_ not in self.invalid_indices)
            for i, n in enumerate(self.ns)
            if i not in self.invalid_indices
        }

    @property
    def entrance(self):
        return None if self._entrance is None else self.names[self._entrance]

    @property
    def root(self):
        if self.entrance is None:
            return self.names[self[Semantics.StaircaseRoom][0]]
        return self.names[self._entrance]

    def draw(self):
        g = nx.Graph()
        shortnames = [r[:3].upper() + r.split("_")[-1] for r in self.names]
        g.add_nodes_from(shortnames)
        for k in range(len(shortnames)):
            for l in self.ns[k]:
                g.add_edge(shortnames[k], shortnames[l])
        nx.draw(g, pos=nx.spring_layout(g), with_labels=True)
        plt.show()


def room_type(name):
    return Semantics(name.split("_")[0])


def room_level(name):
    return int(name.split("/")[0].split("_")[1])


def room_name(t, level, n=0):
    return f"{t.value}_{level}/{n}"


def valid_rooms(state):
    for name, obj_st in state.objs.items():
        if room_type(name) not in [Semantics.Exterior, Semantics.Staircase]:
            yield name, obj_st
