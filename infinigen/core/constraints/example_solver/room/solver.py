# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Lingjie Mei: primary author
# - Karhan Kayan: fix constants

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import shapely.affinity
from numpy.random import uniform

from infinigen.core.constraints.constraint_language.constants import RoomConstants
from infinigen.core.constraints.constraint_language.util import plot_geometry
from infinigen.core.constraints.example_solver import state_def
from infinigen.core.tags import Semantics

from .base import room_level, room_name, room_type, valid_rooms
from .utils import update_contour, update_exterior, update_shared

_eps = 1e-3


class FloorPlanMoves:
    def __init__(self, constants: RoomConstants):
        self.constants = constants
        self.max_stride = 5

    def perturb_state(self, state: state_def.State):
        while True:
            k = np.random.choice(
                [k for k in state.objs if room_type(k) != Semantics.Exterior]
            )
            state_ = deepcopy(state)
            rn = uniform()
            try:
                if room_type(k) == Semantics.Staircase:
                    indices = self.move_staircase(state_)
                elif rn < 0.4:
                    indices = self.extrude_room_out(state_, k)
                elif rn < 0.8:
                    indices = self.extrude_room_in(state_, k)
                else:
                    indices = self.swap_room(state_, k)
            except NotImplementedError:
                indices = set()
            if len(indices) > 0:
                break
        if any(room_type(i) != Semantics.Staircase for i in indices):
            if not self.constants.fixed_contour:
                update_contour(state_, state, indices)
            for k in indices:
                update_shared(state_, k)
                update_exterior(state_, k)
        return state_

    def extrude_room(self, state, k, out=True):
        coords = np.array(state[k].polygon.exterior.coords[:])
        n = len(coords) - 1
        lengths = np.linalg.norm(coords[:-1] - coords[1:], axis=-1)
        weights = np.sqrt(lengths)
        m = np.random.choice(n, p=weights / weights.sum())
        coords = np.concatenate([coords, coords[1:-1]])
        u, v = coords[m : m + 2]
        r = uniform()
        mid = self.constants.unit_cast(u + uniform(0.35, 0.65) * (v - u))
        if (
            max(np.linalg.norm(u - mid), np.linalg.norm(v - mid))
            < self.constants.segment_margin
        ):
            if r < 0.2:
                u = mid
            elif r < 0.4:
                v = mid
        stride = self.constants.unit * (np.random.randint(self.max_stride) + 1)
        theta = np.arctan2(v[1] - u[1], v[0] - u[0])
        theta_ = theta - np.pi / 2 if out else theta + np.pi / 2
        u_ = u[0] + stride * np.cos(theta_), u[1] + stride * np.sin(theta_)
        v_ = v[0] + stride * np.cos(theta_), v[1] + stride * np.sin(theta_)
        if out:
            c = shapely.Polygon([v, u, u_, v_])
            if self.constants.fixed_contour:
                c = self.constants.canonicalize(
                    c.intersection(
                        state[room_name(Semantics.Exterior, room_level(k))].polygon
                    )
                )
            s = self.constants.canonicalize(shapely.union(state[k].polygon, c))
        else:
            c = shapely.Polygon([u, v, v_, u_])
            s = self.constants.canonicalize(shapely.difference(state[k].polygon, c))
        return c, s, stride, theta

    def extrude_room_out(self, state, k):
        c, s, stride, theta = self.extrude_room(state, k, True)
        indices = {k}
        for r in state[k].relations:
            l = r.target_name
            p = state[l].polygon
            if shapely.distance(p, c) <= stride:
                state[l].polygon = self.constants.canonicalize(p.difference(c))
                indices.add(l)
        state[k].polygon = s
        return indices

    def extrude_room_in(self, state, k):
        c, s, stride, theta = self.extrude_room(state, k, False)
        indices = {k}
        for r in state[k].relations:
            l = r.target_name
            p = state[l].polygon
            if shapely.distance(p, c) <= stride:
                if np.abs(np.abs(theta) - np.pi / 2) < 0.1:
                    q = shapely.affinity.translate(p, -stride, 0).union(
                        shapely.affinity.translate(p, stride, 0)
                    )
                else:
                    q = shapely.affinity.translate(p, 0, -stride).union(
                        shapely.affinity.translate(p, 0, stride)
                    )
                inter = q.intersection(c)
                if inter.area > 0.1:
                    state[l].polygon = self.constants.canonicalize(p.union(inter))
                    indices.add(l)
        state[k].polygon = s
        return indices

    def swap_room(self, state, k):
        j = np.random.choice(
            [r.target_name for r in state[k].relations if r.value.length > 0]
        )
        state[k].polygon, state[j].polygon = state[j].polygon, state[k].polygon
        return {k, j}

    def move_staircase(self, state):
        p = state[room_name(Semantics.Staircase, 0)].polygon
        if uniform() < 0.2:
            p = shapely.affinity.rotate(p, 90)
            if p.coords[0][0] != self.constants.unit_cast(p.coords[0][0]):
                p = shapely.affinity.translate(
                    p, self.constants.unit / 2, self.constants.unit / 2
                )
        else:
            p = shapely.affinity.translate(
                p,
                self.constants.unit
                * np.random.randint(1 - self.max_stride, self.max_stride),
                self.constants.unit
                * np.random.randint(1 - self.max_stride, self.max_stride),
            )
        if state[room_name(Semantics.Exterior, len(state.graphs) - 1)].polygon.contains(
            p.buffer(-0.1)
        ):
            names = set()
            for i, _ in enumerate(state.graphs):
                names.add(room_name(Semantics.Staircase, i))
            for n in names:
                state[n].polygon = p
            return names
        return set()

    @staticmethod
    def plot(state):
        plt.clf()
        _, axes = plt.subplots(1, len(state.graphs))
        centroids = {}
        for k, o in valid_rooms(state):
            i = room_level(k)
            plot_geometry(axes[i], o.polygon, np.random.uniform(0, 1, 3))
            centroids[k] = o.polygon.centroid
        for i, g in enumerate(state.graphs):
            for k, ns in g.valid_neighbours.items():
                for n in ns:
                    if k < n:
                        axes[i].plot(centroids[k], centroids[n], "k-")
        plt.show()
