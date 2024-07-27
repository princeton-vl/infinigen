# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick


from infinigen.core import tags as t
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints import reasoning as r


def test_reldom_compatible_floorwall():
    room = r.Domain({t.Semantics.Room, -t.Semantics.Object}, [])

    nofloorrel = (
        -cl.StableAgainst(
            {},
            {
                t.Subpart.SupportSurface,
                t.Subpart.Visible,
                -t.Subpart.Ceiling,
                -t.Subpart.Wall,
            },
        ),
        room,
    )

    against = cl.StableAgainst(
        {t.Subpart.Back, -t.Subpart.Top, -t.Subpart.Front},
        {
            t.Subpart.Visible,
            t.Subpart.Wall,
            -t.Subpart.SupportSurface,
            -t.Subpart.Ceiling,
        },
    )
    wallrel = (against, room)

    assert r.reldom_compatible(nofloorrel, wallrel)
    assert r.reldom_compatible(wallrel, nofloorrel)


def test_reldom_compatible_negation():
    nofloorrel = (
        -cl.StableAgainst(
            {},
            {
                t.Subpart.SupportSurface,
                t.Subpart.Visible,
                -t.Subpart.Ceiling,
                -t.Subpart.Wall,
            },
        ),
        r.Domain({t.Semantics.Room, -t.Semantics.Object}, []),
    )

    on = cl.StableAgainst(
        {t.Subpart.Bottom, -t.Subpart.Front, -t.Subpart.Top, -t.Subpart.Back},
        {
            t.Subpart.SupportSurface,
            t.Subpart.Visible,
            -t.Subpart.Wall,
            -t.Subpart.Ceiling,
        },
    )
    specific_floorrel = (on, r.Domain({t.Semantics.Room, -t.Semantics.Object}, []))

    assert r.reldom_compatible(specific_floorrel, specific_floorrel)
    assert not r.reldom_compatible(nofloorrel, specific_floorrel)
    assert not r.reldom_compatible(specific_floorrel, nofloorrel)


def test_reldom_intersects():
    onroom = (
        cl.StableAgainst(
            {t.Subpart.Bottom, -t.Subpart.Front, -t.Subpart.Top, -t.Subpart.Back},
            {
                t.Subpart.SupportSurface,
                t.Subpart.Visible,
                -t.Subpart.Wall,
                -t.Subpart.Ceiling,
            },
        ),
        r.Domain({t.Semantics.Room, -t.Semantics.Object}, []),
    )

    onlivingroom = (
        cl.StableAgainst(
            {t.Subpart.Bottom, -t.Subpart.Front, -t.Subpart.Top, -t.Subpart.Back},
            {
                t.Subpart.SupportSurface,
                t.Subpart.Visible,
                -t.Subpart.Wall,
                -t.Subpart.Ceiling,
            },
        ),
        r.Domain(
            {
                t.Semantics.LivingRoom,
                t.Semantics.Room,
                -t.Semantics.Object,
                -t.Semantics.Bedroom,
                -t.Semantics.DiningRoom,
            },
            [],
        ),
    )

    assert r.reldom_intersects(onroom, onlivingroom)


def test_reldom_negative_contradict():
    a = (-cl.AnyRelation(), r.Domain({t.Semantics.Object, -t.Semantics.Room}, []))
    assert r.reldom_compatible(a, a)
