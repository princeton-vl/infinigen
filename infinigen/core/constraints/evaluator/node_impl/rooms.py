# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lingjie Mei
import numpy as np

from infinigen.assets.utils.shapes import segment_filter
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints.evaluator.node_impl.impl_bindings import (
    register_node_impl,
)
from infinigen.core.constraints.example_solver import state_def
from infinigen.core.constraints.example_solver.room.base import (
    room_level,
    room_name,
    room_type,
)
from infinigen.core.tags import Semantics
from infinigen.core.util.math import normalize


def separate_floors(state):
    for i, g in enumerate(state.graphs):
        yield state_def.State({k: v for k, v in state.items() if room_level(k) == i}), g


def abs_distance(x, y):
    z = [0] * 4
    z[0 if y[0] > x[0] else 1] = np.abs(y[0] - x[0])
    z[2 if y[1] > x[1] else 3] = np.abs(y[1] - x[1])
    return np.array(z)


def reduce(x, reduce_fn):
    match reduce_fn:
        case "max":
            return np.max(x)
        case "mean":
            return np.mean(x)
        case "sum":
            return np.sum(x)
        case _:
            return np.nan


@register_node_impl(cl.access_angle)
def access_angle_impl(cons: cl.access_angle, state: state_def.State, child_vals: dict):
    x = next(iter(child_vals["objs"]))
    if len(state.graphs) == 0:
        return 0
    graph = state.graphs[room_level(x)]
    if graph.root == x:
        return 0
    root = np.array(state[graph.root].polygon.centroid.coords)[0]
    co = np.array(state[x].polygon.centroid.coords)[0]
    angles = [np.pi]
    for n in graph.valid_neighbours[x]:
        co_ = np.array(state[n].polygon.centroid.coords)[0]
        angles.append(
            np.arccos((1 - 1e-6) * normalize(co_ - co) @ normalize(root - co))
        )
    return np.min(angles)


@register_node_impl(cl.aspect_ratio)
def aspect_ratio_impl(cons: cl.aspect_ratio, state: state_def.State, child_vals: dict):
    c = next(iter(child_vals["objs"]))
    x, y, xx, yy = state[c].polygon.bounds
    a = (xx - x) / (yy - y)
    return max(a, 1 / a)


@register_node_impl(cl.convexity)
def convexity_impl(cons: cl.convexity, state: state_def, child_vals: dict):
    x = next(iter(child_vals["objs"]))
    return state[x].polygon.convex_hull.area / state[x].polygon.area


@register_node_impl(cl.area)
def area_impl(cons: cl.area, state: state_def, child_vals: dict):
    return sum(state[x].polygon.area for x in child_vals["objs"])


@register_node_impl(cl.n_verts)
def n_verts_impl(cons: cl.n_verts, state: state_def, child_vals: dict):
    return sum(len(state[x].polygon.exterior.coords) for x in child_vals["objs"])


@register_node_impl(cl.shared_length)
def shared_length_impl(cons: cl.shared_length, state: state_def, child_vals: dict):
    s = 0
    for x in child_vals["objs"]:
        for y in child_vals["objs_"]:
            if room_type(x) != Semantics.Exterior:
                x, y = y, x
            s += next(r for r in state[x].relations if r.target_name == y).value.length
    return s


@register_node_impl(cl.shared_n_verts)
def shared_n_verts_impl(cons: cl.shared_length, state: state_def, child_vals: dict):
    s = 0
    for x in child_vals["objs"]:
        for y in child_vals["objs_"]:
            if room_type(y) == Semantics.Exterior:
                x, y = y, x
            mls = next(r for r in state[x].relations if r.target_name == y).value
            for ls in mls.geoms:
                s += len(ls.coords)
    return s


@register_node_impl(cl.grid_line_count)
def grid_line_count_impl(cons: cl.grid_line_count, state: state_def, child_vals: dict):
    skeletons = set()
    for k in child_vals["objs"]:
        obj_st = state[k]
        for u, v in obj_st.polygon.exterior.coords:
            if cons.direction == "x":
                skeletons.add(int(u / cons.constants.unit))
            else:
                skeletons.add(int(v / cons.constants.unit))
    return len(skeletons)


@register_node_impl(cl.narrowness)
def narrowness_impl(cons: cl.narrowness, state: state_def, child_vals: dict):
    x = next(iter(child_vals["objs"]))
    p = state[x].polygon.buffer(0)
    count = 0
    unit = cons.constants.unit
    for i in range(int(cons.thresh / unit)):
        with np.errstate(invalid="ignore"):
            q = p.buffer(-unit / 2 * i, join_style="mitre", cap_style="flat").buffer(0)
            if not (q.is_valid and q.geom_type == "Polygon" and q.area > 0):
                count += 100
            else:
                count += (
                    p.length
                    - q.buffer(
                        unit / 2 * i, join_style="mitre", cap_style="flat"
                    ).length
                )
    return count


@register_node_impl(cl.intersection)
def intersection_impl(cons: cl.intersection, state: state_def, child_vals: dict):
    x = next(iter(child_vals["objs"]))
    return sum(
        state[x].polygon.intersection(state[y].polygon).area
        for y in child_vals["objs_"]
    )


@register_node_impl(cl.graph_coherent)
def graph_coherent_impl(cons: cl.graph_coherent, state: state_def, child_vals: dict):
    count = 0
    for i, graph in enumerate(state.graphs):
        for k, ns in graph.neighbours.items():
            for r in state[k].relations:
                if r.target_name in ns and not segment_filter(
                    r.value, cons.constants.segment_margin
                ):
                    count += 2 if room_type(k) == Semantics.Exterior else 1
        total_area = sum(
            state[k].polygon.area for k, _ in graph.valid_neighbours.items()
        )
        if (
            abs(total_area - state[room_name(Semantics.Exterior, i)].polygon.area)
            > 0.01
        ):
            count += 1
        if i > 0 and not cons.constants.fixed_contour:
            if not state[room_name(Semantics.Exterior, i - 1)].polygon.contains(
                state[room_name(Semantics.Exterior, i)].polygon
            ):
                count += 1
    return count


@register_node_impl(cl.same_level)
def same_level_impl(cons: cl.same_level, state: state_def, child_vals: dict):
    x = next(iter(child_vals["objs"]))
    return {k for k, _ in state.objs.items() if room_level(k) == room_level(x)}


@register_node_impl(cl.length)
def length_impl(cons: cl.same_level, state: state_def, child_vals: dict):
    return sum(state[k].polygon.length for k in child_vals["objs"])


@register_node_impl(cl.rand)
def rand_impl(cons: cl.rand, state: state_def, child_vals: dict):
    count = child_vals["count"]
    match cons.type:
        case "bool" | "bern":
            if count == 1:
                return -np.log(cons.args)
            elif count == 0:
                return -np.log(1 - cons.args)
        case "categorical" | "cat":
            if count < len(cons.args):
                return -np.log(cons.args[count] + 1e-20)
    return -np.log(1e-100)
