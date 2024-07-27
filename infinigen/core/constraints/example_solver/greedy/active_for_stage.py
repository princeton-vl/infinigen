# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import logging

from infinigen.core import tags as t
from infinigen.core.constraints import reasoning as r
from infinigen.core.constraints.evaluator import domain_contains
from infinigen.core.constraints.example_solver import state_def
from infinigen.core.util import blender as butil

logger = logging.getLogger(__name__)


def find_ancestors_of_type(
    state: state_def.State, objkey: str, filter_type: r.Domain, seen: set = None
) -> set[str]:
    """
    Find objkeys of all ancestors of `objkey` which match `filter_type`

    Object `A` is a parent of `objkey` if there exists a sequence of objects `A, B, C, ..., objkey`
    where A is in B's relations, B is in C's relations, etc, AND only `A` matches the filter

    Returns
    -------
    parents: set(str)
        objkeys of objects of the given type which are parents in the relation graph

    """

    if seen is None:
        seen = set()

    seen.add(objkey)

    obj = state.objs[objkey]

    if domain_contains.domain_contains(filter_type, state, obj):
        return {objkey}

    result = set()
    for rel in obj.relations:
        if rel.target_name in seen:
            continue

        result.update(find_ancestors_of_type(state, rel.target_name, filter_type, seen))

    return result


def _is_active_room_object(
    state: state_def.State, objkey: str, var_assignments: dict[t.Variable, str]
) -> bool:
    """
    Determine if an object should be active for the given assignment

    if there is a `room` var specified, `objkey` must be a descendent of that room
    if there is an `obj` var specified, `objkey must not be a descendent of any other obj

    """

    for var, assignment in var_assignments.items():
        if assignment is None:
            continue
        match var.name:
            case "room":
                room_ancestors = find_ancestors_of_type(
                    state, objkey, r.Domain({t.Semantics.Room})
                )
                if assignment not in room_ancestors:
                    logger.debug(
                        f"{objkey} is inactive due to room {room_ancestors=} {assignment=}"
                    )
                    return False
            case "obj":
                obj_ancestors = find_ancestors_of_type(
                    state, objkey, r.Domain({t.Semantics.Object})
                )
                if len(obj_ancestors) and objkey not in obj_ancestors:
                    logger.debug(
                        f"{objkey} is inactive due to obj {assignment=} {obj_ancestors=}"
                    )
                    return False
            case _:
                raise NotImplementedError(
                    f"{_is_active_room_object.__name__} encountered unknown variable {var}. "
                    "Greedy stages with vars besides room/obj are not yet supported"
                )

    return True


def set_active(state, objkey, active):
    state.objs[objkey].active = active
    for child in butil.iter_object_tree(state.objs[objkey].obj):
        child.hide_viewport = not active


def update_active_flags(state: state_def.State, var_assignments: dict[t.Variable, str]):
    count = 0
    for objkey, objstate in state.objs.items():
        active = _is_active_room_object(state, objkey, var_assignments)
        set_active(state, objkey, active)
        count += active
    return count
