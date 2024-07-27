# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import copy
import itertools
import logging
import typing

from infinigen.core import tags as t
from infinigen.core.constraints import reasoning as r
from infinigen.core.constraints.evaluator.domain_contains import objkeys_in_dom
from infinigen.core.constraints.example_solver import state_def

logger = logging.getLogger(__name__)


def _resolve_toplevel_var(
    dom: r.Domain,
    state: state_def.State,
    limits: dict[t.Variable, int] = None,
) -> typing.Iterator[str]:
    """
    Find and yield all valid substitutions of a toplevel VariableTag in a given dom

    ASSUMES: there is at most one variable in the domain, and it is at the top level
    """

    if limits is None:
        limits = {}

    vars = [ti for ti in dom.tags if isinstance(ti, t.Variable)]
    if len(vars) == 0:
        yield dom
        return
    elif len(vars) > 1:
        raise ValueError(f"More than one variable in domain {dom}")

    # valid assignments for the var are any objs satisfying everything else in the domain
    vartag = vars[0]
    result = copy.deepcopy(dom)
    result.tags.remove(vartag)
    objkeys = objkeys_in_dom(result, state)
    logger.debug(
        f"Found {len(objkeys)} valid assignments for {repr(vartag)} via {result} on "
    )

    # if the user says limit "room" to 3 and we are doing "room", apply the limit
    name_limit = limits.get(vartag, None)
    if name_limit is not None:
        objkeys = objkeys[:name_limit]

    for objkey in objkeys:
        logger.debug(f"Assigning {objkey} for {vartag}")
        yield result.with_tags(state.objs[objkey].tags)


def substitutions(
    dom: r.Domain,
    state: state_def.State,
    limits: dict[t.Variable, int] | None = None,
    nonempty: bool = False,
) -> typing.Iterator[r.Domain]:
    """Find all t.Variable in d's tags or relations, and return one Domain for each possible assignment

    limits cuts off enumeration of each varname with some integer count
    """

    child_assignment_prod = itertools.product(
        *(substitutions(dchild, state, limits, nonempty) for _, dchild in dom.relations)
    )

    i = None

    for i, dsubs in enumerate(child_assignment_prod):
        assert len(dsubs) == len(dom.relations)
        rels = [(rel, dsubs[j]) for j, (rel, _) in enumerate(dom.relations)]

        candidate = r.Domain(tags=dom.tags, relations=rels)

        yield from _resolve_toplevel_var(candidate, state, limits=limits)

    if i is None and nonempty:
        raise ValueError(f"Found no substitutions found for {dom=}")


def iterate_assignments(
    dom: r.Domain,
    state: state_def.State,
    vars: list[t.Variable],
    limits: dict[t.Variable, int] | None = None,
    nonempty: bool = False,
) -> typing.Iterator[dict[t.Variable, str]]:
    """Find all combinations of assignments for the listed vars.

    Variables will be considered IN ORDER, IE first variable can affect options for second variable,
    but not the other way around.

    Parameters
    ----------
    dom : r.Domain
        The domain to substitute variables in
    state : state_def.State
        The state to substitute variables in
    vars : list[str]
        The names of the variables to substitute
    limits : dict[str, int]
        Consider only the first N objects for each variable
    nonempty : bool
        Raise an error if no substitutions are found

    Returns
    -------
    typing.Iterator[dict[str, r.Domain]]
        Iterator over dicts of variable assignments to domains

    """

    if limits is None:
        limits = {}

    if len(vars) == 0:
        yield {}
        return

    assert isinstance(vars, list), vars
    var = vars[0]

    doms_for_var = [d for d in dom.traverse() if var in d.tags]
    if len(doms_for_var) == 0:
        yield {}
        return

    combined, *rest = doms_for_var
    for d in rest:
        combined = combined.intersection(d)
    combined = copy.deepcopy(
        combined
    )  # prevents modification of original domain if it had the var
    combined.tags.remove(var)
    if not combined.intersects(combined):
        raise ValueError(
            f"{iterate_assignments.__name__} with {var=} arrived at contradictory {combined=}"
        )

    candidates = sorted(objkeys_in_dom(combined, state))

    candidates = [
        c for c in candidates if t.Semantics.NoChildren not in state.objs[c].tags
    ]

    i = None
    for i, objkey in enumerate(candidates):
        limit = limits.get(var, None)
        if limit is not None and i >= limits[var]:
            break

        dom_objkey = r.domain_tag_substitute(
            copy.deepcopy(dom), var, combined.with_tags(t.SpecificObject(objkey))
        )
        rest_iter = iterate_assignments(
            dom_objkey,
            state,
            vars[1:],
            limits,
        )

        for rest_assignments in rest_iter:
            yield {var: objkey, **rest_assignments}

    if i is None and nonempty:
        raise ValueError(f"Found no assignments found for {dom=}")
