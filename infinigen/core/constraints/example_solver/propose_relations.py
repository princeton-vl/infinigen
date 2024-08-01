# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import logging
import typing
from pprint import pprint

import numpy as np

from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints import reasoning as r
from infinigen.core.constraints.evaluator.domain_contains import objkeys_in_dom

from . import state_def

logger = logging.getLogger(__name__)


def minimize_redundant_relations(relations: list[tuple[cl.Relation, r.Domain]]):
    """
    Given a list of relations that must be true, use the first as a constraint to tighten the remaining relations
    """

    assert len(relations) > 0

    # TODO Hacky: moves AnyRelations to the back so _hopefully_ they get implied before we get to them
    relations = sorted(
        relations, key=lambda r: isinstance(r[0], cl.AnyRelation), reverse=True
    )

    (rel, dom), *rest = relations

    # Force all remaining relations to be compatible with (rel, dom), thereby reducing their search space
    remaining_relations = []
    for r_later, d_later in rest:
        logger.debug(f"Inspecting {r_later=} {d_later=}")

        if d_later.intersects(dom):
            logger.debug(f"Intersecting {d_later} with {dom}")
            d_later = d_later.intersection(dom)

        if r.reldom_implies((rel, dom), (r_later, d_later)):
            # (rlater, dlater) is guaranteed true so long as we satisfied (rel, dom), we dont need to separately assign it
            logger.debug("Discarding since rlater,dlater it is implied")
            continue
        else:
            logger.debug(
                f"Keeping {r_later, d_later} since it is not implied by {rel, dom} "
            )
            remaining_relations.append((r_later, d_later))

    implied = any(
        r.reldom_implies(reldom_later, (rel, dom))
        for reldom_later in remaining_relations
    )

    return (rel, dom), remaining_relations, implied


def find_assignments(
    curr: state_def.State,
    relations: list[tuple[cl.Relation, r.Domain]],
    assignments: list[state_def.RelationState] = None,
) -> typing.Iterator[list[state_def.RelationState]]:
    """Iterate over possible assignments that satisfy the given relations. Some assignments may not be feasible geometrically -
    a naive implementation of this function would just enumerate all possible objects matching the assignments, and let the solver
    discover that many combinations are impossible. *This* implementation attemps to never generate guaranteed-invalid combinations in the first place.

    Complexity is pretty astronomical:
    - N^M where N is number of candidates per relation, and M is number of relations
    - reduced somewhat when relations intersect or imply eachother
    - luckily, M is typically 1, 2 or 3, as objects arent often related to lots of other objects

    TODO:
    - discover new relations constraints, which can arise from the particular choice of objects
    - prune early when object choice causes bounds to be violated

    This function essentially does a complex form of SAT-solving. It *really* shouldnt be written in python
    """

    if assignments is None:
        assignments = []
        # print('FIND ASSIGNMENTS TOPLEVEL')
        # pprint(relations)

    if len(relations) == 0:
        yield assignments
        return

    logger.debug(f"Attempting to assign {relations[0]}")

    (rel, dom), remaining_relations, implied = minimize_redundant_relations(relations)
    assert len(remaining_relations) < len(relations)

    if implied:
        logger.debug(f"Found remaining_relations implies {(rel, dom)=}, skipping it")
        yield from find_assignments(
            curr, relations=remaining_relations, assignments=assignments
        )
        return

    if isinstance(rel, cl.AnyRelation):
        pprint(relations)
        pprint([(rel, dom)] + remaining_relations)
        raise ValueError(
            f"Got {rel} as first relation. Invalid! Maybe the program is underspecified?"
        )

    candidates = objkeys_in_dom(dom, curr)

    for parent_candidate_name in candidates:
        logging.debug(f"{parent_candidate_name=}")

        parent_state = curr.objs[parent_candidate_name]
        n_parent_planes = len(
            curr.planes.get_tagged_planes(parent_state.obj, rel.parent_tags)
        )

        parent_order = np.arange(n_parent_planes)
        np.random.shuffle(parent_order)

        for parent_plane in parent_order:
            # logger.debug(f'Considering {parent_candidate_name=} {parent_plane=} {n_parent_planes=}')

            assignment = state_def.RelationState(
                relation=rel,
                target_name=parent_candidate_name,
                child_plane_idx=0,  # TODO fill in at apply()-time
                parent_plane_idx=parent_plane,
            )

            yield from find_assignments(
                curr,
                relations=remaining_relations,
                assignments=assignments + [assignment],
            )
