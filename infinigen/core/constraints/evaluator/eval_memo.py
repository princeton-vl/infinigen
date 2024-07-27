# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import logging

from infinigen.core import tags as t
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints.example_solver import moves
from infinigen.core.constraints.example_solver.state_def import ObjectState, State

logger = logging.getLogger(__name__)


def memo_key(n: cl.Node):
    match n:
        case cl.item(var):
            return var
        case cl.scene():
            return cl.scene
        case _:
            return id(n)


def evict_memo_for_obj(node: cl.Problem, memo: dict, obj: ObjectState):
    recvals = [evict_memo_for_obj(child, memo, obj) for _, child in node.children()]
    res = any(recvals)

    match node:
        case cl.tagged(_, tags):
            if not t.implies(obj.tags, tags):
                res = False
        case cl.scene():
            res = True
        case _:
            pass

    key = memo_key(node)
    if res and key in memo:
        del memo[key]

    return res


def reset_bvh_cache(state, filter_name=None):
    """
    filter_name: if specified, only get rid of things containing this
    """

    static_tags = {t.Semantics.Room, t.Semantics.Cutter}

    def keep_key(k):
        names, tags = k

        if filter_name is not None:
            obj = state.objs[filter_name].obj
            return obj.name not in names

        for n in names:
            if n not in state.objs:
                return False
            ostate = state.objs[n]
            if not ostate.tags.intersection(static_tags):
                return False

        return True

    prev_keys = list(state.bvh_cache.keys())
    for k in prev_keys:
        res = keep_key(k)
        if res:
            continue
        del state.bvh_cache[k]

    logger.debug(
        f"reset_bvh_cache evicted {len(prev_keys) - len(state.bvh_cache)} out of {len(prev_keys)} orig"
    )


def evict_memo_for_move(
    problem: cl.Problem, state: State, memo: dict, move: moves.Move
):
    match move:
        case (
            moves.TranslateMove(names)
            | moves.RotateMove(names)
            | moves.Addition(names=names)
            | moves.ReinitPoseMove(names=names)
            | moves.RelationPlaneChange(names=names)
            | moves.Resample(names=names)
        ):
            for name in names:
                assert name is not None, move
                evict_memo_for_obj(problem, memo, state.objs[name])
                reset_bvh_cache(state, filter_name=name)
        case moves.Deletion(name):
            # TODO hack - delete everything since we cant evict for specific obj after it has ben deleted
            # easily fixable with more work / refactoring
            for k in list(memo.keys()):
                del memo[k]
            reset_bvh_cache(state)
        case _:
            raise NotImplementedError(f"Unsure what to evict for {move=}")
