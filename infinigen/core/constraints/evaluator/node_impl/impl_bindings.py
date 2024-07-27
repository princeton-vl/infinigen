# Copyright (C) 2024, Princeton University.

# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Karhan Kayan: geometry impl bindings
# - Alexander Raistrick: impl interface, set_reasoning / operator impls
# - Lingjie Mei: bugfix

import functools
import logging
import math

import numpy as np

from infinigen.core import tags as t
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints import reasoning as r
from infinigen.core.constraints.evaluator import domain_contains
from infinigen.core.constraints.example_solver import state_def

from . import symmetry, trimesh_geometry

logger = logging.getLogger(__name__)

node_impls = {}


def statenames_to_blnames(state, names):
    return [state.objs[n].obj.name for n in names]


def register_node_impl(node_cls):
    def decorator(func):
        node_impls[node_cls] = func
        return func

    return decorator


def generic_impl_interface(cons: cl.Node, state: state_def.State, child_vals: dict):
    pass


@register_node_impl(cl.constant)
def constant_impl(cons: cl.Node, state: state_def.State, child_vals: dict):
    return cons.value


@register_node_impl(cl.ScalarOperatorExpression)
@register_node_impl(cl.BoolOperatorExpression)
def operator_impl(
    cons: cl.ScalarOperatorExpression | cl.BoolOperatorExpression,
    state: state_def.State,
    child_vals: dict,
):
    operands = [child_vals[f"operands[{i}]"] for i in range(len(cons.operands))]

    try:
        if (
            len(operands) == 1
            or "numpy" in cons.func.__class__.__name__
            or "lambda" in cons.func.__name__
        ):
            return cons.func(*operands)
        else:
            return functools.reduce(cons.func, operands)
    except ZeroDivisionError as e:
        raise ZeroDivisionError(f"{e} in {cons=}, {operands=}")


@register_node_impl(cl.center_stable_surface_dist)
def center_stable_surface_impl(
    cons: cl.center_stable_surface_dist, state: state_def.State, child_vals: dict
):
    objs = child_vals["objs"]
    return trimesh_geometry.center_stable_surface(state.trimesh_scene, objs, state)


@register_node_impl(cl.accessibility_cost)
def accessibility_impl(
    cons: cl.accessibility_cost,
    state: state_def.State,
    child_vals: dict,
    use_collision_impl: bool = True,
):
    objs = statenames_to_blnames(state, child_vals["objs"])
    others = statenames_to_blnames(state, child_vals["others"])
    if len(objs) == 0:
        return 0

    if use_collision_impl:
        logger.debug("accessibility_cost_cuboid_penetration(%s, %s)", objs, others)
        res = trimesh_geometry.accessibility_cost_cuboid_penetration(
            state.trimesh_scene,
            objs,
            others,
            cons.normal,
            cons.dist,
            bvh_cache=state.bvh_cache,
        )
    else:
        logger.debug("accessibility_cost(%s, %s)", objs, others)
        res = trimesh_geometry.accessibility_cost(
            state.trimesh_scene, objs, others, cons.normal
        )
    return res


@register_node_impl(cl.distance)
def min_distance_impl(
    cons: cl.Node, state: state_def.State, child_vals: dict, others_tags: set = None
):
    objs = statenames_to_blnames(state, child_vals["objs"])
    others = statenames_to_blnames(state, child_vals["others"])

    if len(objs) == 0 or len(others) == 0:
        logger.debug("min_distance had no targets")
        return 0

    logger.debug("min_distance_impl(%s, %s)", objs, others)

    res = trimesh_geometry.min_dist(
        state.trimesh_scene,
        a=objs,
        b=others,
        b_tags=others_tags,
        bvh_cache=state.bvh_cache,
    )

    if res.dist < 0:
        return 0

    return res.dist


@register_node_impl(cl.min_distance_internal)
def min_distance_internal_impl(
    cons: cl.min_distance_internal, state: state_def.State, child_vals: dict
):
    objs = statenames_to_blnames(state, child_vals["objs"])
    if len(objs) <= 1:
        return 0
    return trimesh_geometry.min_dist(state.trimesh_scene, a=objs).dist


@register_node_impl(cl.min_dist_2d)
def min_dist_2d_impl(cons: cl.min_dist_2d, state: state_def.State, child_vals: dict):
    a = statenames_to_blnames(state, child_vals["objs"])
    b = statenames_to_blnames(state, child_vals["others"])
    if len(a) == 0 or len(b) == 0:
        return 0
    return trimesh_geometry.min_dist_2d(state.trimesh_scene, a, b)


@register_node_impl(cl.focus_score)
def focus_score_impl(
    cons: cl.focus_score,
    state: state_def.State,
    child_vals: dict,
):
    a = statenames_to_blnames(state, child_vals["objs"])
    b = statenames_to_blnames(state, child_vals["others"])

    if len(a) == 0 or len(b) == 0:
        return 0

    return trimesh_geometry.focus_score(state, a=a, b=b)


@register_node_impl(cl.angle_alignment_cost)
def angle_alignment_impl(
    cons: cl.angle_alignment_cost,
    state: state_def.State,
    child_vals: dict,
    others_tags: set = None,
):
    a = statenames_to_blnames(state, child_vals["objs"])
    b = statenames_to_blnames(state, child_vals["others"])
    if len(a) == 0 or len(b) == 0:
        return 0
    return trimesh_geometry.angle_alignment_cost(state, a, b, others_tags)


@register_node_impl(cl.freespace_2d)
def freespace_2d_impl(cons: cl.freespace_2d, state: state_def.State, child_vals: dict):
    return trimesh_geometry.freespace_2d()


@register_node_impl(cl.rotational_asymmetry)
def rotational_asymmetry_impl(
    cons: cl.rotational_asymmetry, state: state_def.State, child_vals: dict
):
    objs = statenames_to_blnames(state, child_vals["objs"])
    if len(objs) <= 1:
        return 0
    return symmetry.compute_total_rotation_asymmetry(objs)


@register_node_impl(cl.reflectional_asymmetry)
def reflectional_asymmetry_impl(
    cons: cl.reflectional_asymmetry,
    state: state_def.State,
    child_vals: dict,
    use_long_plane: bool = True,
):
    objs = statenames_to_blnames(state, child_vals["objs"])
    others = statenames_to_blnames(state, child_vals["others"])
    if len(objs) <= 1:
        return 0
    return trimesh_geometry.reflectional_asymmetry_score(
        state.trimesh_scene, objs, others, use_long_plane
    )


@register_node_impl(cl.coplanarity_cost)
def coplanarity_cost_impl(
    cons: cl.coplanarity_cost, state: state_def.State, child_vals: dict
):
    objs = child_vals["objs"]
    if len(objs) <= 1:
        return 0
    return trimesh_geometry.coplanarity_cost(state.trimesh_scene, objs)


@register_node_impl(cl.tagged)
def tagged_impl(cons: cl.tagged, state: state_def.State, child_vals: dict):
    res = {o for o in child_vals["objs"] if t.satisfies(state.objs[o].tags, cons.tags)}

    # logger.debug('tagged(%s) produced %s from %i candidates', cons.tags, res, len(child_vals['objs']))

    return res


@register_node_impl(cl.count)
def count_impl(cons: cl.count, state: state_def.State, child_vals: dict):
    return len(child_vals["objs"])


@register_node_impl(cl.in_range)
def in_range_impl(cons: cl.in_range, state: state_def.State, child_vals: dict):
    x = child_vals["val"]
    return x <= cons.high and x >= cons.low


@register_node_impl(cl.related_to)
def related_children_impl(
    cons: cl.related_to, state: state_def.State, child_vals: dict
):
    r = cons.relation
    children: set[str] = child_vals["child"]
    parents: set[str] = child_vals["parent"]

    res = set()
    for o in children:
        if any(
            rs.relation.implies(r) and rs.target_name in parents
            for rs in state.objs[o].relations
        ):
            res.add(o)

    # logger.debug('related_to %s produced %s from %i candidates', cons.relation, res, len(children))

    return res


@register_node_impl(cl.excludes)
def excludes_impl(cons: cl.excludes, state: state_def.State, child_vals: dict):
    return {o for o in child_vals["objs"] if state.objs[o].tags.isdisjoint(cons.tags)}


@register_node_impl(cl.volume)
def volume_impl(cons: cl.volume, state: state_def.State, child_vals: dict):
    objs = child_vals["objs"]

    res = 0
    for o in objs:
        s = state.objs[o]
        dims = sorted(list(s.obj.dimensions), reverse=True)

        if isinstance(cons.dims, int):
            dims = dims[: cons.dims]
        elif isinstance(cons.dims, tuple):
            dims = np.array(dims)[np.array(cons.dims)]
        else:
            raise TypeError(f"Unexpected {type(cons.dims)=}")

        res += math.prod(dims) + 1e-10

    return res


@register_node_impl(cl.hinge)
def hinge_impl(cons: cl.hinge, state: state_def.State, child_vals: dict):
    x = child_vals["val"]

    if x < cons.low:
        return cons.low - x
    elif x > cons.high:
        return x - cons.high
    else:
        return 0


@register_node_impl(cl.union)
def union_impl(cons: cl.union, state: state_def.State, child_vals: dict):
    return {
        o for o in child_vals["objs"] if not state.objs[o].tags.isdisjoint(cons.tags)
    }


@register_node_impl(r.FilterByDomain)
def filter_by_domain_impl(
    cons: r.FilterByDomain, state: state_def.State, child_vals: dict
) -> set[str]:
    return {
        o
        for o in child_vals["objs"]
        if domain_contains.domain_contains(cons.filter, state, state.objs[o])
    }
