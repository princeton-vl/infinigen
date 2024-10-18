# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei

import numpy as np
import shapely
from numpy.random import uniform
from shapely import box

from infinigen.core.constraints.example_solver.room.base import (
    room_level,
    room_name,
    room_type,
)
from infinigen.core.constraints.example_solver.room.solidifier import max_mls
from infinigen.core.constraints.example_solver.room.utils import (
    update_exterior,
    update_shared,
)
from infinigen.core.tags import Semantics
from infinigen.core.util.math import normalize
from infinigen.core.util.random import log_uniform
from infinigen.core.util.random import random_general as rg


class ContourFactory:
    def __init__(self, consgraph):
        self.consgraph = consgraph
        self.constants = consgraph.constants
        self.n_trials = 1000
        self.long_prob = "bool", 0.5
        self.corner_prob = "weighted_choice", (1, "round"), (2, "sharp"), (7, "none")
        self.maximal_radius = 4

    def add_staircase(self, contour):
        x, y = contour.boundary.xy
        x_, x__ = np.min(x), np.max(x)
        y_, y__ = np.min(y), np.max(y)
        for _ in range(self.n_trials):
            area = self.constants.segment_margin * self.constants.wall_height * 6
            skewness = log_uniform(0.5, 0.8)
            if uniform() < 0.5:
                skewness = 1 / skewness
            width, height = (
                self.constants.unit_cast(np.sqrt(area * skewness).item()),
                self.constants.unit_cast(np.sqrt(area / skewness).item()),
            )
            x = self.constants.unit_cast(uniform(x_, x__ - width))
            y = self.constants.unit_cast(uniform(y_, y__ - height))
            b = box(x, y, x + width, y + height)
            if contour.contains(b):
                return b
        else:
            raise ValueError("Invalid staircase")

    @staticmethod
    def get_length(polygon, point):
        coords = np.array(polygon.exterior.coords)[:-1]
        indices = np.nonzero(np.abs(coords - point[np.newaxis]).sum(-1) < 0.1)[0]
        if len(indices) > 0:
            i = indices[0]
            q = np.linalg.norm(coords[i] - coords[(i + 1) % len(coords)])
            r = np.linalg.norm(coords[i] - coords[(i - 1) % len(coords)])
            return min(q, r)
        return None

    def decorate(self, state):
        if self.constants.fixed_contour:
            if rg(self.long_prob):
                slope = uniform(0.05, 0.2)
                indices = set()
                for k, obj_st in state.objs.items():
                    if room_type(k) not in [Semantics.StaircaseRoom]:
                        p = obj_st.polygon
                        x, y = np.array(p.exterior.coords).T
                        y -= np.where(np.abs(x) < 0.1, slope * x, 0)
                        q = shapely.Polygon(np.stack([x, y], -1))
                        if np.abs(p.area - q.area) > 1e-6:
                            state[k].polygon = q
                            indices.add(k)
                for k in indices:
                    update_shared(state, k)
        else:
            for i in reversed(range(len(state.graphs))):
                exterior_name = room_name(Semantics.Exterior, i)
                exterior = self.constants.canonicalize(state[exterior_name].polygon)
                c = exterior.centroid.coords[0]
                coords = np.array(exterior.exterior.coords)
                for p in coords:
                    if (
                        shapely.Point(p)
                        .buffer(0.1, cap_style="square")
                        .intersection(exterior)
                        .area
                        > 2 * 0.1**2
                    ):
                        continue
                    inside_point = (
                        p[0] + (c[0] - p[0]) * 1e-2,
                        p[1] + (c[1] - p[1]) * 1e-2,
                    )
                    if not exterior.contains(shapely.Point(inside_point)):
                        continue
                    length = self.get_length(exterior, p)
                    if length is None:
                        continue
                    k, l = None, None
                    for k, obj_st in state.objs.items():
                        if (
                            room_type(k)
                            not in [Semantics.Staircase, Semantics.Exterior]
                            and room_level(k) == i
                        ):
                            l = self.get_length(obj_st.polygon, p)
                            if l is not None:
                                break
                    if k is None or l is None:
                        continue
                    length = min(min(length, l), self.maximal_radius)
                    directions = [
                        (x_, y_)
                        for x_ in [-1, 1]
                        for y_ in [-1, 1]
                        if exterior.contains(shapely.Point(p[0] + x_, p[1] + y_))
                    ]
                    if len(directions) == 0:
                        continue
                    direction = directions[0]
                    corner_func = rg(self.corner_prob)
                    if corner_func == "none":
                        continue
                    length -= self.constants.unit
                    while length > 0:
                        q = p[0] + direction[0] * length, p[1] + direction[1] * length
                        cutter = shapely.box(
                            min(p[0], q[0]),
                            min(p[1], q[1]),
                            max(p[0], q[0]),
                            max(p[1], q[1]),
                        )
                        quad_segs = (
                            1 if corner_func == "sharp" else np.random.randint(4, 7)
                        )
                        exterior_ = self.constants.canonicalize(
                            exterior.difference(cutter)
                            .union(shapely.Point(q).buffer(length, quad_segs=quad_segs))
                            .buffer(0)
                        )
                        new = self.constants.canonicalize(
                            state[k].polygon.intersection(exterior_).buffer(0)
                        )
                        if all(
                            new.buffer(-m + 1e-2).geom_type == "Polygon"
                            for m in np.linspace(0, 0.75, 4)
                        ):
                            x, y, x_, y_ = max_mls(new.exterior)
                            if np.linalg.norm([x - x_, y - y_]) >= 1:
                                break
                        length -= self.constants.unit
                    if length <= 0:
                        continue
                    if i < len(state.graphs) - 1:
                        if not exterior_.contains(
                            state[room_name(Semantics.Exterior, i + 1)].polygon
                        ):
                            continue
                    coords = np.array(new.exterior.coords)
                    diff = normalize(coords[1:] - coords[:-1])
                    diff_ = diff[list(range(1, len(diff))) + [0]]
                    if np.any((diff * diff_).sum(-1) < -0.01):
                        continue
                    state[k].polygon = new
                    state[exterior_name].polygon = exterior_
                    update_exterior(state, k)
