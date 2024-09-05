# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick


from infinigen.core import tags as t
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints import reasoning as r
from infinigen.core.tags import Semantics


def test_domain_obj():
    furniture = r.Domain({t.Semantics.Furniture})

    sofas = r.Domain({t.Semantics.Furniture, t.Semantics.Seating})
    assert sofas.implies(furniture)
    assert not furniture.implies(sofas)

    furniture_in_livingroom = r.Domain(
        {t.Semantics.Furniture},
        relations=[(cl.SupportedBy(), r.Domain({t.Semantics.LivingRoom}))],
    )
    assert not furniture.implies(furniture_in_livingroom)
    assert furniture_in_livingroom.implies(furniture)

    furniture_in_bathroom = r.Domain(
        {t.Semantics.Furniture},
        relations=[(cl.SupportedBy(), r.Domain({t.Semantics.Bathroom}))],
    )
    assert not furniture_in_livingroom.implies(furniture_in_bathroom)
    assert not furniture_in_bathroom.implies(furniture_in_livingroom)


def test_domain_implies_complex():
    against_wall = cl.StableAgainst(
        child_tags={t.Subpart.Back},
        parent_tags={t.Subpart.Wall, t.Subpart.Interior},
        margin=0,
    )

    d = r.Domain(
        tags={Semantics.Storage},
        relations=[(against_wall, r.Domain(tags={Semantics.Room}, relations=[]))],
    )

    assert d.implies(d)
    assert not r.Domain({t.Semantics.Storage}).implies(d)
    assert not r.Domain(set(), d.relations).implies(d)
    assert d.implies(r.Domain({t.Semantics.Storage}))
    assert d.implies(r.Domain(set(), d.relations))

    # "storage related any way to any thing" is less specific
    generalize_relation = r.Domain(
        {t.Semantics.Storage}, relations=[(cl.AnyRelation(), r.Domain())]
    )
    assert d.implies(generalize_relation)
    assert not generalize_relation.implies(d)

    # "storage related any way but specifically to bedroom"
    #   is both more and less specific so not a subset either way
    different_relation = r.Domain(
        {t.Semantics.Storage},
        relations=[
            (cl.AnyRelation(), r.Domain({t.Semantics.Room, t.Semantics.Bedroom}))
        ],
    )
    assert not d.implies(different_relation)
    assert not different_relation.implies(d)

    # "storage against a bedroom wall" is more specific
    against_bedroom_wall = r.Domain(
        {t.Semantics.Storage},
        relations=[(against_wall, r.Domain({t.Semantics.Room, t.Semantics.Bedroom}))],
    )
    assert against_bedroom_wall.implies(d)
    assert not d.implies(against_bedroom_wall)


def test_domain_var_substitute():
    var = t.Variable("x")
    start = r.Domain(
        {t.Subpart.Interior, var}, relations=[(cl.AnyRelation(), r.Domain())]
    )
    subfor = r.Domain(
        {t.Semantics.Room, t.Semantics.Bedroom},
        relations=[(cl.Touching(), r.Domain({t.Semantics.Furniture}))],
    )

    assert r.domain_tag_substitute(start, var, subfor) == r.Domain(
        {t.Semantics.Room, t.Semantics.Bedroom, t.Subpart.Interior},
        relations=[(cl.Touching(), r.Domain({t.Semantics.Furniture}))],
    )

    start2 = r.Domain(
        {t.Subpart.Interior, var},
        relations=[(cl.AnyRelation(), r.Domain({t.Semantics.Lighting}))],
    )
    assert r.domain_tag_substitute(start2, var, subfor) == r.Domain(
        {t.Semantics.Room, t.Semantics.Bedroom, t.Subpart.Interior},
        relations=[
            (
                cl.AnyRelation(),
                r.Domain({t.Semantics.Lighting}),
            ),  # not implied so gets kept
            (cl.Touching(), r.Domain({t.Semantics.Furniture})),
        ],
    )


def test_domain_intersect_tags():
    obj_types = {Semantics.Object, Semantics.Room, Semantics.Cutter}
    obj = cl.scene()[Semantics.Object].excludes(obj_types)
    room = cl.scene()[Semantics.Room].excludes(obj_types)

    ld = r.constraint_domain(obj)
    assert -Semantics.Room in ld.tags

    md = r.constraint_domain(room)
    assert -Semantics.Object in md.tags

    assert not ld.intersects(md)
    assert not ld.intersects(md)


def test_domain_construction_complex():
    dom = r.Domain()
    dom.add_relation(cl.AnyRelation(), r.Domain({t.Semantics.Object}, []))
    dom.add_relation(
        cl.StableAgainst(), r.Domain({t.Semantics.Object, t.Variable("room")}, [])
    )
    dom.add_relation(-cl.AnyRelation(), r.Domain({t.Semantics.Room}, []))

    assert dom.relations[0] == (
        cl.StableAgainst(),
        r.Domain({t.Semantics.Object, t.Variable("room")}, []),
    )
    assert dom.relations[1] == (-cl.AnyRelation(), r.Domain({t.Semantics.Room}, []))
    assert len(dom.relations) == 2


def test_domain_construction_complex_2():
    rd1 = (cl.AnyRelation(), r.Domain({t.Semantics.Room, t.Semantics.DiningRoom}))
    rd2 = (cl.StableAgainst(), r.Domain({t.Semantics.Room}))

    assert r.reldom_intersects(rd1, rd2)

    dom = r.Domain()
    dom.add_relation(*rd1)
    dom.add_relation(*rd2)

    print("DOM RESULT", dom)
    assert dom.relations == [
        (cl.StableAgainst(), r.Domain({t.Semantics.Room, t.Semantics.DiningRoom}))
    ]


def test_domain_satisfies():
    a = r.Domain({t.Semantics.Object, -t.Semantics.Room})
    b = r.Domain({t.Semantics.Object, t.Semantics.Room})
    assert not b.satisfies(a)

    b = r.Domain({t.Semantics.Object, -t.Semantics.Room})
    assert b.satisfies(a)

    a.add_relation(cl.StableAgainst(), r.Domain({t.Semantics.Room}))
    assert not b.satisfies(a)

    b.add_relation(
        cl.StableAgainst(), r.Domain({t.Semantics.Room, t.Semantics.DiningRoom})
    )
    assert b.satisfies(a)

    a.add_relation(-cl.AnyRelation(), r.Domain({t.Semantics.Object}))
    assert b.satisfies(a)

    b.add_relation(cl.StableAgainst(), r.Domain({t.Semantics.Object}))
    assert not b.satisfies(a)


def test_domain_satisfies_2():
    res_dom = r.Domain(
        {Semantics.Object, Semantics.Storage, -Semantics.Room},
        [
            (
                cl.StableAgainst(
                    {
                        t.Subpart.Bottom,
                        -t.Subpart.Top,
                        -t.Subpart.Back,
                        -t.Subpart.Front,
                    },
                    {
                        t.Subpart.Visible,
                        t.Subpart.SupportSurface,
                        -t.Subpart.Ceiling,
                        -t.Subpart.Wall,
                    },
                ),
                r.Domain({Semantics.DiningRoom, Semantics.Room, -Semantics.Object}, []),
            ),
            (-cl.AnyRelation(), r.Domain({Semantics.Object, -Semantics.Room}, [])),
        ],
    )

    filter_dom = r.Domain(
        {Semantics.Object, -Semantics.Room},
        [
            (
                cl.StableAgainst(
                    {},
                    {
                        t.Subpart.SupportSurface,
                        t.Subpart.Visible,
                        -t.Subpart.Ceiling,
                        -t.Subpart.Wall,
                    },
                ),
                r.Domain({Semantics.DiningRoom, Semantics.Room, -Semantics.Object}, []),
            ),
            (-cl.AnyRelation(), r.Domain({Semantics.Object, -Semantics.Room}, [])),
        ],
    )

    assert res_dom.satisfies(filter_dom)
