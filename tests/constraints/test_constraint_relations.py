# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick


from infinigen.core import tags as t
from infinigen.core.constraints import constraint_language as cl


def test_relation_implies_trivial():
    assert not cl.StableAgainst(set(), set()).implies(cl.Touching())

    sf = cl.SupportedBy({t.Subpart.SupportSurface})
    sfi = cl.SupportedBy({t.Subpart.SupportSurface, t.Subpart.Visible})

    assert sf.implies(sf)
    assert sfi.implies(sf)
    assert sf.implies(cl.AnyRelation())
    assert sfi.implies(cl.AnyRelation())
    assert not cl.AnyRelation().implies(sf)
    assert not cl.AnyRelation().implies(sfi)


def require_intersects(a: cl.Relation, b: cl.Relation, truth):
    assert a.intersects(b) == truth
    assert b.intersects(a) == truth


example = cl.StableAgainst(
    {t.Subpart.Top, -t.Subpart.Bottom}, {t.Semantics.Object, -t.Subpart.Top}
)


def test_relations_intersects_unrestricted():
    unrestricted = cl.AnyRelation()
    require_intersects(example, unrestricted, True)
    require_intersects(example, -unrestricted, False)
    require_intersects(-example, unrestricted, True)


def test_relation_intersects_mismatched_type():
    mismatch_type = cl.Touching(example.child_tags, example.parent_tags)
    require_intersects(example, mismatch_type, False)
    require_intersects(example, -mismatch_type, True)
    require_intersects(-example, mismatch_type, True)


def test_relation_intersects_superset():
    superset = cl.StableAgainst({t.Subpart.Top}, {t.Semantics.Object})
    require_intersects(example, superset, True)
    require_intersects(
        example, -superset, True
    )  # Top-Bot,Obj-Top AND NOT(Top,Obj) permits Top+Bot_Obj+Top
    require_intersects(
        -example, superset, False
    )  # Top,Obj AND NOT(Top-Bot,Obj-Top) False


def test_relation_intersects_subset():
    subset = cl.StableAgainst(
        {t.Subpart.Top, -t.Subpart.Bottom, t.Subpart.SupportSurface},
        {t.Semantics.Object, -t.Subpart.Top, -t.Subpart.Side},
    )
    require_intersects(example, subset, True)
    require_intersects(
        example, -subset, False
    )  # Top-Bot,Obj-Top AND NOT Top-Bot+Sup,Obj-Top-Side
    require_intersects(
        -example, subset, True
    )  # Top-Bot+Sup,Obj-Top-Side AND NOT Top-Bot,Obj-Top


def test_relation_intersects_intersecting():
    inter = cl.StableAgainst(
        {t.Subpart.Top, t.Subpart.Visible}, {t.Semantics.Object, t.Semantics.Furniture}
    )
    require_intersects(example, inter, True)
    require_intersects(
        example, -inter, True
    )  # Top-Bot_Obj-Top AND NOT Top+Vis_Obj+Furn. Yes, Top-Bot-Vis_Obj-Top-Furn
    require_intersects(
        -example, inter, True
    )  # Top+Vis_Obj+Furn AND NOT Top-Bot_Obj-Top.


def test_relation_intersects_contradict_child():
    contradict_child = cl.StableAgainst(
        {t.Subpart.Top, t.Subpart.Bottom}, {t.Semantics.Object}
    )
    require_intersects(example, contradict_child, False)
    require_intersects(example, -contradict_child, True)
    require_intersects(
        -example, contradict_child, True
    )  # Top+Bot,Obj AND NOT Top-Bot,Obj-Top = Top+Bot,Obj?


def test_relation_intersects_contradict_parent():
    contradict_parent = cl.StableAgainst(
        {t.Subpart.Top}, {t.Semantics.Object, t.Subpart.Top}
    )
    require_intersects(example, contradict_parent, False)
    require_intersects(example, -contradict_parent, True)
    require_intersects(-example, contradict_parent, True)


def test_relation_difference():
    assert t.difference(
        {t.Semantics.Object, -t.Subpart.Top}, {t.Semantics.Object, t.Subpart.Bottom}
    ) == {
        t.Semantics.Object,
        -t.Subpart.Top,
        -t.Subpart.Bottom,
    }

    refine = cl.StableAgainst(set(), {t.Semantics.Object, t.Subpart.Bottom})
    assert example.difference(refine) == cl.StableAgainst(
        {t.Subpart.Top, -t.Subpart.Bottom},
        {t.Semantics.Object, -t.Subpart.Top, -t.Subpart.Bottom},
    )
