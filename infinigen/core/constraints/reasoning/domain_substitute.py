# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

from __future__ import annotations

import copy
import logging

from infinigen.core import tags as t

from .constraint_domain import Domain

logger = logging.getLogger(__name__)


def domain_tag_substitute(
    domain: Domain, vartag: t.Variable, subst_domain: Domain, return_match=False
) -> Domain:
    """Return concrete substitution of `domain`, where `subst_domain` must be satisfied
    whenever `subst_tag` was present in the original.
    """

    assert isinstance(vartag, t.Variable), vartag
    domain = copy.deepcopy(domain)  # prevent modification of original

    o_match = vartag in domain.tags

    rd_sub, rd_matches = [], []
    for r, d in domain.relations:
        d, match = domain_tag_substitute(d, vartag, subst_domain, return_match=True)
        rd_sub.append((r, d))
        rd_matches.append(match)
    rd_match = any(rd_matches)

    if not (o_match or rd_match):
        return (domain, False) if return_match else domain

    domain.relations = []
    for r, d in rd_sub:
        domain.add_relation(r, d)

    if o_match:
        if vartag in domain.tags:
            domain.tags.remove(vartag)
        domain = domain.intersection(subst_domain)

    return (domain, True) if return_match else domain


def substitute_all(
    dom: Domain,
    assignments: dict[t.Variable, Domain],
) -> Domain:
    for var, d in assignments.items():
        dom = domain_tag_substitute(dom, var, d)
    return dom
