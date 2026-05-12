# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Alexander Raistrick: primary author
# - David Yan: bounding for inequalities / expressions


import pytest

from infinigen.core import tags as t
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints import reasoning as r


def test_bound_eq():
    bound1 = r.Bound(r.Domain(set()))
    bound2 = r.Bound(r.Domain(set()))
    assert bound1 == bound2


def test_constant():
    expr = cl.constant(1) * cl.constant(2) + cl.constant(3) < cl.constant(3)
    assert r.is_constant(expr)


def test_bounds_simple():
    furniture = cl.tagged(cl.scene(), tags={t.Semantics.Furniture})
    count = cl.count(furniture)

    bounds = r.constraint_bounds(cl.in_range(count, 1, 5))
    assert bounds == [r.Bound(r.Domain({t.Semantics.Furniture}), 1, 5)]

    lower = [r.Bound(r.Domain({t.Semantics.Furniture}), low=1)]
    upper = [r.Bound(r.Domain({t.Semantics.Furniture}), high=3)]
    assert r.constraint_bounds(count < 4) == upper
    assert r.constraint_bounds(count > 0) == lower
    assert r.constraint_bounds(4 > count) == upper
    assert r.constraint_bounds(0 < count) == lower
    assert r.constraint_bounds(count <= 3) == upper
    assert r.constraint_bounds(count >= 1) == lower
    assert r.constraint_bounds(3 >= count) == upper
    assert r.constraint_bounds(1 <= count) == lower


@pytest.mark.skip  # no longer supported for timebeing
def test_bounds_compound():
    chair = cl.tagged(cl.scene(), tags={t.Semantics.Chair})
    table = cl.tagged(cl.scene(), tags={t.Semantics.Table})

    scene_state = [(r.Domain({t.Semantics.Table}), 4)]
    cons = (cl.count(chair) < cl.count(table) * 3) * (cl.count(chair) > cl.count(table))

    bounds = r.constraint_bounds(cons, scene_state)

    assert bounds == [
        r.Bound(r.Domain({t.Semantics.Chair}), high=11),
        r.Bound(r.Domain({t.Semantics.Chair}), low=5),
    ]

    bounds2 = r.constraint_bounds(
        cl.in_range(cl.count(chair), cl.count(table), cl.count(table) * 3), scene_state
    )
    assert bounds2 == [r.Bound(r.Domain({t.Semantics.Chair}), 4, 12)]


def test_bounds_and():
    tags = {t.Semantics.Furniture}
    furniture = cl.tagged(cl.scene(), tags=tags)
    count = cl.count(furniture)
    cons = (count < 5) * (count > 1)
    bounds = r.constraint_bounds(cons)

    assert bounds == [
        r.Bound(r.Domain(tags), high=4),
        r.Bound(r.Domain(tags), low=2),
    ]


def test_bounds_multilevel():
    furniture = cl.tagged(cl.scene(), tags={t.Semantics.Furniture})
    sofa = cl.tagged(furniture, tags={t.Semantics.Seating})
    cons = cl.count(sofa) <= 3

    assert r.constraint_bounds(cons) == [
        r.Bound(r.Domain({t.Semantics.Furniture, t.Semantics.Seating}), high=3)
    ]


def test_bounds_arithmetic():
    tags = {t.Semantics.Furniture}
    furniture = cl.tagged(cl.scene(), tags=tags)
    count = cl.count(furniture)
    cons = cl.in_range(count * 2 + 2, 2, 10)

    bounds = r.constraint_bounds(cons)
    assert bounds == [r.Bound(r.Domain(tags), low=0, high=4)]


def test_bounds_domain_AnyRelation():
    bedrooms = cl.scene().tagged({t.Semantics.Bedroom})
    beds = cl.scene().tagged({t.Semantics.Bed})

    all_bedrooms_beds = bedrooms.all(
        lambda r: cl.related_to(beds, r, cl.SupportedBy()).count().in_range(1, 2)
    )

    bd = r.Domain({t.Semantics.Bedroom})
    bed_in_room = r.Domain({t.Semantics.Bed}, relations=[(cl.SupportedBy(), bd)])

    res = r.constraint_bounds(all_bedrooms_beds)
    assert res == [r.Bound(bed_in_room, low=1, high=2)]


def test_bounds_forall():
    rooms = cl.scene().tagged(t.Semantics.Room)
    furniture = cl.scene().tagged(t.Semantics.Furniture)
    small_obj = cl.scene().tagged(t.Semantics.OfficeShelfItem)
    rel = cl.SupportedBy()

    c = rooms.all(
        lambda room: (
            furniture.related_to(room, rel).count().in_range(1, 2)
            * furniture.related_to(room, rel).all(
                lambda stor: small_obj.related_to(stor, rel).count().in_range(5, 10)
            )
        )
    )

    bounds = r.constraint_bounds(c)

    furn_room = r.Domain(
        {t.Semantics.Furniture}, relations=[(rel, r.Domain({t.Semantics.Room}))]
    )
    item_furn_room = r.Domain(
        {t.Semantics.OfficeShelfItem}, relations=[(rel, furn_room)]
    )

    assert bounds == [
        r.Bound(furn_room, 1, 2),
        r.Bound(item_furn_room, 5, 10),
    ]


def test_bound_implied_rel():
    s = cl.scene()
    against = cl.StableAgainst(set(), set())
    cons = (
        s.related_to(s, cl.AnyRelation()).related_to(s, against).count().in_range(1, 3)
    )

    bounds = r.constraint_bounds(cons)

    assert bounds == [r.Bound(r.Domain(set(), [(against, r.Domain())]), low=1, high=3)]

    cons = (
        s.related_to(s, against).related_to(s, cl.AnyRelation()).count().in_range(1, 3)
    )

    bounds = r.constraint_bounds(cons)

    assert bounds == [r.Bound(r.Domain(set(), [(against, r.Domain())]), low=1, high=3)]


def test_bound_implied_rel_forall():
    s = cl.scene()

    rel = cl.Touching()

    all_dom = r.Domain()
    assert all_dom.implies(all_dom)
    assert r.reldom_implies((rel, all_dom), (rel, all_dom))

    small_obj = s.tagged(t.Semantics.OfficeShelfItem).related_to(s, rel)
    cons = s.all(lambda tb: small_obj.related_to(tb, rel).count().in_range(1, 3))

    bounds = r.constraint_bounds(cons)

    assert bounds[0].domain == r.Domain(
        {t.Semantics.OfficeShelfItem}, [(rel, r.Domain())]
    )
