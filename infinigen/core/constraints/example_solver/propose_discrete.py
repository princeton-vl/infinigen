# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Alexander Raistrick: primary author
# - Karhan Kayan: fix bug to ensure deterministic behavior

import copy
import logging
from itertools import product

import gin
import numpy as np

from infinigen.core import tags as t
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints import reasoning as r
from infinigen.core.constraints import usage_lookup
from infinigen.core.constraints.evaluator.domain_contains import objkeys_in_dom
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil

from . import moves, propose_relations, state_def

logger = logging.getLogger(__name__)


class DummyCubeGenerator(AssetFactory):
    def __init__(self, seed):
        super().__init__(seed)

    def create_asset(self, *_, **__):
        return butil.spawn_cube()


def lookup_generator(preds: set[t.Semantics]):
    if t.contradiction(preds):
        raise ValueError(f"Got lookup_generator for unsatisfiable {preds=}")

    preds_pos, preds_neg = t.decompose_tags(preds)

    fac_class_tags = [x.generator for x in preds if isinstance(x, t.FromGenerator)]
    if len(fac_class_tags) > 1:
        raise ValueError(f"{preds=} had {len(fac_class_tags)=}, only 1 is allowed")
    elif len(fac_class_tags) == 1:
        (fac_class_tag,) = fac_class_tags
        usage = usage_lookup.usages_of_factory(fac_class_tag)
        remainder = preds_pos - usage - {fac_class_tag}
        if len(remainder):
            raise ValueError(
                f"Your constraint program requested {fac_class_tag} with {preds_pos}, "
                f"but used_as[] tags specify {usage} so predicates {remainder=} are unsatisfied. "
                f"Please add these remainder predicates to used_as for {fac_class_tag} so that it can be safely retrieved."
            )
        return [fac_class_tag]

    options = usage_lookup.all_factories()
    for pos_tag in preds_pos:
        options &= usage_lookup.factories_for_usage(pos_tag)
    for neg_tag in preds_neg:
        options -= usage_lookup.factories_for_usage(neg_tag)

    options = list(options)
    # sort options to ensure deterministic behavior
    options.sort(key=lambda x: x.__name__)
    np.random.shuffle(options)

    return options


def propose_addition_bound_gen(
    cons: cl.Node,
    curr: state_def.State,
    bounds: list[r.Bound],
    goal_bound_idx: r.Bound,
    gen_class: AssetFactory,
    filter_domain: r.Domain,
):
    """
    Try to propose any addition move involving the specified bound and generator
    """

    goal_bound = bounds[goal_bound_idx]
    logger.debug(
        f"attempt propose_addition for {gen_class.__name__} rels={len(goal_bound.domain.relations)}"
    )

    assert r.domain_finalized(goal_bound.domain), goal_bound
    if not active_for_stage(goal_bound.domain, filter_domain):
        raise ValueError(
            f"Attempted to propose {goal_bound} but it should not be active for {filter_domain=}"
        )
    if len(goal_bound.domain.relations) == 0:
        raise ValueError(
            f"Attempted to propose unconstrained {gen_class.__name__} with no relations"
        )

    found_tags = usage_lookup.usages_of_factory(gen_class)
    goal_pos, *_ = t.decompose_tags(goal_bound.domain.tags)
    if not t.implies(found_tags, goal_pos) and found_tags.issuperset(goal_pos):
        raise ValueError(f"Got {gen_class=} for {goal_pos=}, but it had {found_tags=}")

    prop_dom = goal_bound.domain.intersection(filter_domain)
    prop_dom.tags.update(found_tags)

    # logger.debug(f'GOAL {goal_bound.domain} \nFILTER {filter_domain}\nPROP {prop_dom}\n\n')
    logger.debug(
        "GOAL %s\n FILTER %s\n PROP %s\n",
        goal_bound.domain.repr(abbrv=True),
        filter_domain.repr(abbrv=True),
        prop_dom,
    )

    assert active_for_stage(prop_dom, filter_domain)

    search_rels = [
        rd for rd in prop_dom.relations if not isinstance(rd[0], cl.NegatedRelation)
    ]

    all_assignments = propose_relations.find_assignments(curr, search_rels)

    i = None
    for i, assignments in enumerate(all_assignments):
        logger.debug("Found assignments %d %s %s", i, len(assignments), assignments)

        yield moves.Addition(
            names=[
                f"{np.random.randint(1e6):04d}_{gen_class.__name__}"
            ],  # decided later
            gen_class=gen_class,
            relation_assignments=assignments,
            temp_force_tags=prop_dom.tags,
        )

    if i is None:
        # raise ValueError(f'Found no assignments for {prop_dom}')
        logger.debug(f"Found no assignments for {prop_dom.repr(abbrv=True)}")
        pass
    else:
        logger.debug(f"Exhausted all assignments for {gen_class=}")


def active_for_stage(prop_dom: r.Domain, filter_dom: r.Domain):
    return prop_dom.intersects(filter_dom, require_satisfies_right=True)


@gin.configurable
def preproc_bounds(
    bounds: list[r.Bound],
    state: state_def.State,
    filter: r.Domain,
    reverse=False,
    shuffle=True,
    print_bounds=False,
):
    if print_bounds:
        print(
            f"{preproc_bounds.__name__} for {filter.get_objs_named()} (total {len(bounds)}):"
        )
        for b in bounds:
            res = active_for_stage(b.domain, filter)
            if res:
                print(
                    "BOUND", res, b.domain.intersection(filter).repr(abbrv=True), "\n"
                )

    for b in bounds:
        if not r.domain_finalized(b.domain, check_anyrel=False, check_variable=True):
            raise ValueError(
                f"{preproc_bounds.__name__} found non-finalized {b.domain=}"
            )

    bounds = [b for b in bounds if active_for_stage(b.domain, filter)]

    if shuffle:
        np.random.shuffle(bounds)

    bound_counts = [len(objkeys_in_dom(b.domain, state)) for b in bounds]

    order = np.arange(len(bounds))

    def key(i):
        b = bounds[i]
        bc = bound_counts[i]
        if b.high is not None and b.high < bc:
            res = 1
        elif b.low is not None and b.low > bc:
            res = -1
        else:
            res = 0
        return -res if reverse else res

    order = sorted(order, key=key)

    return [bounds[i] for i in order if key(i) != 1]


def propose_addition(
    cons: cl.Node,
    curr: state_def.State,
    filter_domain: r.Domain,
    temperature: float,
):
    bounds = r.constraint_bounds(cons)
    bounds = preproc_bounds(bounds, curr, filter_domain)

    if len(bounds) == 0:
        logger.debug(f"Found no bounds for {filter_domain=}")
        return

    for i, bound in enumerate(bounds):
        if bound.low is None:
            # bounds with low=None are supposed to cap other bounds, not introduce new objects
            continue

        fac_options = lookup_generator(preds=bound.domain.tags)
        if len(fac_options) == 0:
            if bound.low is None or bound.low == 0:
                continue
            raise ValueError(f"Found no generators for {bound}")

        for gen_class in fac_options:
            yield from propose_addition_bound_gen(
                cons, curr, bounds, i, gen_class, filter_domain
            )

    logger.debug(f"propose_addition found no candidate moves for {bound}")


def propose_deletion(
    cons: cl.Node,
    curr: state_def.State,
    filter_domain: r.Domain,
    temperature: float,
):
    bounds = r.constraint_bounds(cons)
    bounds = preproc_bounds(bounds, curr, filter_domain, reverse=True, shuffle=True)

    if len(bounds) == 0:
        logger.debug(f"Found no bounds for {filter_domain=}")
        return

    np.random.shuffle(bounds)

    for i, bound in enumerate(bounds):
        candidates = objkeys_in_dom(bound.domain, curr)
        np.random.shuffle(candidates)
        for cand in candidates:
            yield moves.Deletion([cand])


def propose_relation_plane_change(
    cons: cl.Node,
    curr: state_def.State,
    filter_domain: r.Domain,
    temperature: float,
):
    cand_objs = objkeys_in_dom(filter_domain, curr)

    if len(cand_objs) == 0:
        logger.debug(f"Found no cand_objs for {filter_domain=}")
        return

    np.random.shuffle(cand_objs)
    for cand in cand_objs:
        for i, rels in enumerate(curr.objs[cand].relations):
            if not isinstance(rels.relation, cl.GeometryRelation):
                continue

            target_obj = curr.objs[rels.target_name].obj
            n_planes = len(
                curr.planes.get_tagged_planes(target_obj, rels.relation.parent_tags)
            )
            if n_planes <= 1:
                continue

            order = np.arange(n_planes)
            np.random.shuffle(order)
            for plane_idx in order:
                if plane_idx == rels.parent_plane_idx:
                    continue
                yield moves.RelationPlaneChange(
                    names=[cand], relation_idx=i, plane_idx=plane_idx
                )


def propose_resample(
    cons: cl.Node,
    curr: state_def.State,
    filter_domain: r.Domain,
    temperature: float,
):
    cand_objs = objkeys_in_dom(filter_domain, curr)

    if len(cand_objs) == 0:
        logger.debug(f"Found no cand_objs for {filter_domain=}")
        return

    np.random.shuffle(cand_objs)

    for cand in cand_objs:
        os = curr.objs[cand]
        if usage_lookup.has_usage(os.generator.__class__, t.Semantics.SingleGenerator):
            continue

        yield moves.Resample(names=[cand], align_corner=None)

        # corner_options = [None] + list(range(6))
        # np.random.shuffle(corner_options)
        # for c in corner_options:
        #    yield moves.Resample(name=cand, align_corner=c)


def is_swap_domains_unaffected(state: state_def.State, name1: str, name2: str):
    raise NotImplementedError()


def propose_swap(
    cons: cl.Node,
    curr: state_def.State,
    filter_domain: r.Domain,
    temperature: float,
):
    raise NotImplementedError()

    cand_objs = objkeys_in_dom(filter_domain, curr)

    if len(cand_objs) == 0:
        logger.debug(f"Found no cand_objs for {filter_domain=}")
        return

    a_objs = copy.copy(cand_objs)
    b_objs = copy.copy(cand_objs)
    np.random.shuffle(a_objs)
    np.random.shuffle(b_objs)

    for a, b in product(a_objs, b_objs):
        if a == b:
            continue
        if not is_swap_domains_unaffected(curr, a, b):
            continue
        yield moves.Swap(names=[a, b])
