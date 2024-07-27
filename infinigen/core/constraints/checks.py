# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import copy
import itertools
import logging
import typing

from tqdm import tqdm

from infinigen.core import tags as t
from infinigen.core.constraints import (
    constraint_language as cl,
)
from infinigen.core.constraints import (
    reasoning as r,
)
from infinigen.core.constraints.example_solver import (
    propose_discrete,
    propose_relations,
)

logger = logging.getLogger(__name__)


def iter_domains(node: cl.Node) -> typing.Iterator[r.Domain]:
    match node:
        case cl.ObjectSetExpression():
            yield node, r.constraint_domain(node)
        case cl.Expression() | cl.Problem():
            for k, c in node.children():
                yield from iter_domains(c)
        case _:
            raise ValueError(f"iter_domains found unmatched {type(node)=} {node=}")


def bound_coverage(b: r.Bound, stages: dict[str, r.Domain]) -> list[str]:
    return [
        k for k, f in stages.items() if propose_discrete.active_for_stage(b.domain, f)
    ]


def check_coverage_errors(b: r.Bound, coverage: list, stages: dict[str, r.Domain]):
    if len(coverage) == 0:
        raise ValueError(
            f"Greedy stages did not cover all object classes! User specified bound {b} had {coverage=}"
        )

    if len(coverage) != 1:
        raise ValueError(
            f"Object class {b} was covered in more than one greedy stage! Got {coverage=}. Greedy stages must be non-overlapping"
        )

    if t.Semantics.Room in b.domain.tags:
        return  # rooms are handled separately by Lingjie's room solver, does not need bounds

    gen_options = propose_discrete.lookup_generator(b.domain.tags)
    if len(gen_options) < 1:
        raise ValueError(f"Object class {b=} had {gen_options=}")

    for k in coverage:
        logger.debug(f"Checking coverage {k=} {b.domain=} {stages[k]=}")

        if not b.domain.intersects(stages[k]):
            continue
        prop = b.domain.intersection(stages[k])

        if prop.is_recursive():
            raise ValueError(
                f"Found recursive prop domain {prop.tags=} {len(prop.relations)=}"
            )
        assert not prop.is_recursive(), prop.tags

        if not len(prop.relations):
            continue
        first, remaining, implied = propose_relations.minimize_redundant_relations(
            prop.relations
        )
        if implied:
            continue
        if isinstance(first[0], cl.AnyRelation):
            raise ValueError(f"{b=} in {stages[k]=} had underspecified {first=}")


def check_problem_greedy_coverage(prob: cl.Problem, stages: dict[str, r.Domain]):
    bounds = r.constraint_bounds(prob)

    for b in tqdm(bounds, desc="Checking greedy stages coverage"):
        coverage = bound_coverage(b, stages)
        check_coverage_errors(b, coverage, stages)


def check_unfinalized_constraints(prob: cl.Problem):
    # TODO
    return []


def check_contradictory_domains(prob: cl.Problem):
    for node, dom in iter_domains(prob):
        contradictory = not dom.satisfies(dom)
        if contradictory:
            raise ValueError(
                f"Constraint node had self-contradicting domain. \n{node=} \n{dom=}"
            )


def validate_stages(stages: dict[str, r.Domain]):
    for k, d in stages.items():
        if d.is_recursive():
            raise ValueError(f"{k=} had recursive domain")

    for (k1, d1), (k2, d2) in itertools.product(stages.items(), stages.items()):
        inter = d1.intersects(d2)
        if inter != (k1 == k2):
            raise ValueError(
                f"User provided greedy stages with keys {k1=} {k2=} which had non-empty intersection! "
                " please define greedy stages which are mutually exclusive."
            )


def check_all(
    prob: cl.Problem, greedy_stages: dict[str, r.Domain], all_vars: list[str]
):
    prob = copy.deepcopy(prob)

    # room constraints are handled separately and will not be tested in checks
    room_constraint_keys = ["node_gen", "node", "room"]

    for k in room_constraint_keys:
        if k in prob.constraints:
            prob.constraints.pop(k)
        if k in prob.score_terms:
            prob.score_terms.pop(k)

    for k, v in greedy_stages.items():
        if not isinstance(v, r.Domain):
            raise TypeError(f"Greedy stage {k=} had non-domain value {v=}")

        extras = v.all_vartags() - set(all_vars)
        if len(extras):
            raise ValueError(
                f"{k=} had extra vars {extras=}. Greedy domains may only contain vars from {all_vars}"
            )

    validate_stages(greedy_stages)

    check_problem_greedy_coverage(prob, greedy_stages)
    check_unfinalized_constraints(prob)
    check_contradictory_domains(prob)
