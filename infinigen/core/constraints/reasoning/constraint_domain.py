# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial

from infinigen.core import tags as t
from infinigen.core.constraints import constraint_language as cl

from .domain import Domain

logger = logging.getLogger(__name__)


def constraint_domain(node: cl.ObjectSetExpression, finalize_variables=False) -> Domain:
    """Given an expression, find a compact representation of what types of objects it is applying to.

    User can compared the resulting Domain against their State and see what objects fit.
    """

    assert isinstance(node, cl.ObjectSetExpression), node

    recurse = partial(constraint_domain, finalize_variables=finalize_variables)

    match node:
        case cl.tagged(objs, tags):
            d = recurse(objs)
            d.tags.update(tags)
            if t.contradiction(d.tags):
                raise ValueError(
                    f"Contradictory tags {tags=} for {d=} while parsing constraint {node=}"
                )
            return d
        case cl.related_to(children, parents, relation):
            c_d = recurse(children)
            p_d = recurse(parents)
            c_d.add_relation(relation, p_d)
            return c_d
        case cl.scene():
            return Domain()
        case cl.item(x):
            if finalize_variables:
                return recurse(
                    node.member_of
                )  # TODO - worried about infinite recursion somehow
            else:
                return Domain(tags={t.Variable(x)})
        case FilterByDomain(objs, filter):
            return filter.intersection(recurse(objs))
        case _:
            raise NotImplementedError(node)


@dataclass
class FilterByDomain(cl.ObjectSetExpression):
    """Constraint node which says to return all objects matching a domain.

    Used as a compacted representation of the filtering performed many cl.tagged and cl.related_to calls.
    One r.Domain is sufficient to represent the effect of and combination of intersection-style filtering.

    Introduced (currently) only by greedy.filter_constraints, since that function needs to work
    with domains in order to narrow the scope of some constraints.
    """

    objs: cl.ObjectSetExpression
    filter: Domain
