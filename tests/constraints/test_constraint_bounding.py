# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: 
# - Alexander Raistrick: primary author
# - David Yan: bounding for inequalities / expressions

from itertools import chain
from functools import partial

from pprint import pprint
import pytest
import numpy as np

from infinigen.core.constraints import (
    constraint_language as cl,
    reasoning as r
)

def test_bound_eq():
    bound1 = r.Bound(r.Domain(set()))
    bound2 = r.Bound(r.Domain(set()))
    assert bound1 == bound2

def test_constant():

    expr = cl.constant(1) * cl.constant(2) + cl.constant(3) < cl.constant(3)
    assert r.is_constant(expr)

def test_bounds_simple():
    
    count = cl.count(furniture)
    
    bounds = r.constraint_bounds(cl.in_range(count, 1, 5))

    assert r.constraint_bounds(count < 4) == upper
    assert r.constraint_bounds(count > 0) == lower
    assert r.constraint_bounds(4 > count) == upper
    assert r.constraint_bounds(0 < count) == lower
    assert r.constraint_bounds(count <= 3) == upper
    assert r.constraint_bounds(count >= 1) == lower
    assert r.constraint_bounds(3 >= count) == upper
    assert r.constraint_bounds(1 <= count) == lower

@pytest.mark.skip # no longer supported for timebeing
    cons = (cl.count(chair) < cl.count(table) * 3) * (cl.count(chair) > cl.count(table))
def test_bounds_and():
    
    furniture = cl.tagged(cl.scene(), tags=tags)
    count = cl.count(furniture)
    cons = (count < 5) * (count > 1)
    bounds = r.constraint_bounds(cons)

    assert bounds == [
        r.Bound(r.Domain(tags), high=4),
        r.Bound(r.Domain(tags), low=2),
    ]

def test_bounds_multilevel():

    cons = cl.count(sofa) <= 3
    
    assert r.constraint_bounds(cons) == [
]

def test_bounds_arithmetic():

    furniture = cl.tagged(cl.scene(), tags=tags)
    count = cl.count(furniture)
    cons = cl.in_range(count * 2 + 2, 2, 10)

    bounds = r.constraint_bounds(cons)
    assert bounds == [r.Bound(r.Domain(tags), low=0, high=4)]

def test_bounds_domain_AnyRelation():


    all_bedrooms_beds = bedrooms.all(lambda r:
        cl.related_to(beds, r, cl.SupportedBy())
        .count().in_range(1, 2)
    )
                       

    res = r.constraint_bounds(all_bedrooms_beds)
    assert res == [r.Bound(bed_in_room, low=1, high=2)]

def test_bounds_forall():

    rel = cl.SupportedBy()
    
    c = rooms.all(lambda room: (
        furniture.related_to(room, rel).count().in_range(1, 2) *
        furniture.related_to(room, rel).all(lambda stor:
            small_obj.related_to(stor, rel).count().in_range(5, 10)
        )
    ))

    bounds = r.constraint_bounds(c)


    assert bounds == [
        r.Bound(furn_room, 1, 2),
        r.Bound(item_furn_room, 5, 10),
    ]

def test_bound_implied_rel():

    s = cl.scene()
    against = cl.StableAgainst(set(), set())
    cons = (
        s.related_to(s, cl.AnyRelation())
        .related_to(s, against)
        .count().in_range(1, 3)
    )

    bounds = r.constraint_bounds(cons)

    assert bounds == [
        r.Bound(
            r.Domain(set(), [(against, r.Domain())]),
            low=1, high=3
        )
    ]

    cons = (
        s.related_to(s, against)
        .related_to(s, cl.AnyRelation())
        .count().in_range(1, 3)
    )

    bounds = r.constraint_bounds(cons)

    assert bounds == [
        r.Bound(
            r.Domain(set(), [(against, r.Domain())]),
            low=1, high=3
        )
    ]

def test_bound_implied_rel_forall():

    s = cl.scene()
    
    rel = cl.Touching()

    all_dom = r.Domain()
    assert all_dom.implies(all_dom)
    assert r.reldom_implies((rel, all_dom), (rel, all_dom))

    cons = s.all(lambda tb: small_obj.related_to(tb, rel).count().in_range(1, 3))

    bounds = r.constraint_bounds(cons)

