# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: 
# - Karhan Kayan: geometry impl bindings
# - Alexander Raistrick: impl interface, set_reasoning / operator impls
# - Lingjie Mei: bugfix

import math
import logging
import functools

import gin
import numpy as np

from infinigen.core.constraints import (
    constraint_language as cl,
    reasoning as r
)
from infinigen.core.constraints.example_solver import state_def
from infinigen.core.constraints.evaluator import domain_contains

from . import trimesh_geometry, symmetry

logger = logging.getLogger(__name__)

node_impls = {}

def statenames_to_blnames(state, names):
    return [state.objs[n].obj.name for n in names]

def register_node_impl(node_cls):
    def decorator(func):
        node_impls[node_cls] = func
        return func
    return decorator

def generic_impl_interface(
    cons: cl.Node, 
    state: state_def.State, 
    child_vals: dict
):
    pass

@register_node_impl(cl.constant)
def constant_impl(
    cons: cl.Node, 
    state: state_def.State, 
    child_vals: dict
):
    return cons.value

@register_node_impl(cl.ScalarOperatorExpression)
@register_node_impl(cl.BoolOperatorExpression)
def operator_impl(
    cons: cl.ScalarOperatorExpression | cl.BoolOperatorExpression, 
    state: state_def.State, 
    child_vals: dict
):
    
    operands = [
        child_vals[f'operands[{i}]']
        for i in range(len(cons.operands))
    ]

    try:
        if (
            isinstance(cons.func, np.ufunc)
            or len(operands) == 1
        ):
            return cons.func(*operands)
        else:
            return functools.reduce(cons.func, operands)
    except ZeroDivisionError as e:
        raise ZeroDivisionError(f'{e} in {cons=}, {operands=}')
@register_node_impl(cl.center_stable_surface_dist)
    cons: cl.center_stable_surface_dist, 
    child_vals: dict,
    use_collision_impl: bool = True
    
    
    if use_collision_impl:
        res = trimesh_geometry.accessibility_cost_cuboid_penetration(
            state.trimesh_scene, 
            objs, 
            others, 
            cons.normal, 
            cons.dist, 
            bvh_cache=state.bvh_cache
        )
    else:
        res = trimesh_geometry.accessibility_cost(
            state.trimesh_scene, objs, others, cons.normal
        )
    return res
@register_node_impl(cl.distance)
    cons: cl.Node, 
    state: state_def.State, 
    others_tags: set = None
):

    objs = statenames_to_blnames(state, child_vals['objs'])
    others = statenames_to_blnames(state, child_vals['others'])

        logger.debug('min_distance had no targets')
    
    res = trimesh_geometry.min_dist(
        state.trimesh_scene, 
        a=objs, 
        b=others, 
        b_tags=others_tags,
        bvh_cache=state.bvh_cache
    )

    if res.dist < 0:
        return 0

    return res.dist

@register_node_impl(cl.min_distance_internal)
    cons: cl.min_distance_internal, 
    state: state_def.State, 
    child_vals: dict
):
    objs = statenames_to_blnames(state, child_vals['objs'])
    if len(objs) <= 1:
    return trimesh_geometry.min_dist(
    ).dist

@register_node_impl(cl.min_dist_2d)
def min_dist_2d_impl(
    state: state_def.State, 
    child_vals: dict
):
    a = statenames_to_blnames(state, child_vals['objs'])
    b = statenames_to_blnames(state, child_vals['others'])
    return trimesh_geometry.min_dist_2d(
    )

@register_node_impl(cl.focus_score)
def focus_score_impl(
    state: state_def.State, 
    child_vals: dict,
):

    a = statenames_to_blnames(state, child_vals['objs'])
    b = statenames_to_blnames(state, child_vals['others'])
    
    if len(a) == 0 or len(b) == 0:
        return 0
    
    return trimesh_geometry.focus_score(
        a=a,
        b=b
    )

def angle_alignment_impl(
    state: state_def.State, 
):
    a = statenames_to_blnames(state, child_vals['objs'])
    b = statenames_to_blnames(state, child_vals['others'])
    if len(a) == 0 or len(b) == 0:
        return 0
    )

@register_node_impl(cl.freespace_2d)
def freespace_2d_impl(
    state: state_def.State, 
    child_vals: dict
):
    return trimesh_geometry.freespace_2d()

@register_node_impl(cl.rotational_asymmetry)
def rotational_asymmetry_impl(
    state: state_def.State, 
    child_vals: dict
):
    objs = statenames_to_blnames(state, child_vals['objs'])
    if len(objs) <= 1:
        return 0
    return symmetry.compute_total_rotation_asymmetry(objs)

    use_long_plane: bool = True,

@register_node_impl(cl.tagged)
def tagged_impl(
    state: state_def.State, 
    child_vals: dict
):
    res = {
        o for o in child_vals['objs']
        if t.satisfies(state.objs[o].tags, cons.tags)
    }

    #logger.debug('tagged(%s) produced %s from %i candidates', cons.tags, res, len(child_vals['objs']))

    return res

    state: state_def.State, 

    state: state_def.State, 
@register_node_impl(cl.related_to)
    cons: cl.related_to,
    state: state_def.State,
    child_vals: dict
):
    children: set[str] = child_vals['child']
    parents: set[str] = child_vals['parent']
    res = set()
    for o in children:
            rs.relation.implies(r) and rs.target_name in parents
        ):
            res.add(o)

    #logger.debug('related_to %s produced %s from %i candidates', cons.relation, res, len(children))

    return res
@register_node_impl(cl.excludes)
def excludes_impl(
    cons: cl.excludes,
    state: state_def.State,
    return {
        o for o in child_vals['objs']
        if state.objs[o].tags.isdisjoint(cons.tags)
    }

@register_node_impl(cl.volume)
def volume_impl(
    cons: cl.volume,
    state: state_def.State,
    child_vals: dict
):
    objs = child_vals['objs']
    
    res = 0
    for o in objs:

        s = state.objs[o]
        dims = sorted(list(s.obj.dimensions), reverse=True)

        if isinstance(cons.dims, int):
            dims = dims[:cons.dims]
        elif isinstance(cons.dims, tuple):
            dims = np.array(dims)[np.array(cons.dims)]
        else:
            raise TypeError(f'Unexpected {type(cons.dims)=}')
            
        res += math.prod(dims)

    return res

@register_node_impl(cl.hinge)
def hinge_impl(
    cons: cl.hinge,
    state: state_def.State,
    child_vals: dict
):
    x = child_vals['val']

    if x < cons.low:
        return cons.low - x
    elif x > cons.high:
        return x - cons.high
    else:
        return 0
    
@register_node_impl(r.FilterByDomain)
def filter_by_domain_impl(
    cons: r.FilterByDomain,
    state: state_def.State,
    child_vals: dict
) -> set[str]:
    
    return {
        o
        for o in child_vals['objs']
        if domain_contains.domain_contains(cons.filter, state, state.objs[o])
    }
