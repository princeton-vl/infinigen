# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Lingjie Mei: primary author
# - Karhan Kayan: fix constants

import numpy as np
import shapely
from shapely import MultiLineString
from shapely.ops import shared_paths

from infinigen.core.constraints.example_solver.state_def import State
from infinigen.core.tags import Semantics

from .base import room_level, room_name, valid_rooms


def update_shared(state: State, i: str):
    for r in state[i].relations:
        r.value = shared(state[i].polygon, state[r.target_name].polygon)
    for k, o in valid_rooms(state):
        if room_level(k) == room_level(i) and k != i:
            r = next(r for r in o.relations if r.target_name == i)
            r.value = shared(o.polygon, state[i].polygon)


def update_contour(state_: State, state: State, indices: set[str]):
    i = next(iter(indices))
    exterior = room_name(Semantics.Exterior, room_level(i))
    minus = shapely.union_all([state[k].polygon for k in indices])
    plus = shapely.union_all([state_[k].polygon for k in indices])
    state_[exterior].polygon = state[exterior].polygon.difference(minus).union(plus)


def update_exterior(state: State, i: str):
    exterior = room_name(Semantics.Exterior, room_level(i))
    r = next(r for r in state[exterior].relations if r.target_name == i)
    v = state[i].polygon.exterior
    for q in state[i].relations:
        if q.value.length > 1e-6:
            v = v.difference(q.value)
    v = shapely.force_2d(v)
    if v.geom_type == "MultiLineString":
        r.value = v
    elif v.length > 0:
        r.value = MultiLineString([v])
    else:
        r.value = MultiLineString(v)


def mls_ccw(mls: MultiLineString, state: State, i: str):
    exterior = state[i].polygon.exterior
    coords = np.array(exterior.coords[:-1])
    mls_ = []
    for ls in mls.geoms:
        u, v = ls.coords[:2]
        x = np.argmin(np.linalg.norm(coords - np.array(u)[np.newaxis], axis=-1))
        y = np.argmin(np.linalg.norm(coords - np.array(v)[np.newaxis], axis=-1))
        mls_.append(ls if x < y else shapely.reverse(ls))
    return MultiLineString(mls_)


def update_staircase(state: State, i: str):
    pholder = room_name(Semantics.Staircase, room_level(i))
    r = next(r for r in state[pholder].relations if r.target_name == i)
    inter = state[i].polygon.intersection(state[pholder].polygon).area
    r.value = inter / state[pholder].polygon.area


def shared(s, t):
    with np.errstate(invalid="ignore"):
        forward, backward = shared_paths(s.boundary, t.boundary).geoms
    if forward.length > 0:
        return forward
    elif backward.length > 0:
        return backward
    else:
        return shapely.MultiLineString()
