# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

from infinigen.core import tags as t
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints import reasoning as r
from infinigen.core.constraints.example_solver import greedy, state_def


def make_dummy_state(type_counts: dict[tuple[t.Tag], int]):
    objs = {}
    for tags, count in type_counts.items():
        for i in range(count):
            name = "_".join([t.value for t in tags]) + f"_{i}"
            objs[name] = state_def.ObjectState(
                obj=None,
                generator=None,
                tags=set(tags).union([t.SpecificObject(name)]),
                relations=[],
            )

    return state_def.State(objs=objs)


def test_substitutions_no_vars():
    state = make_dummy_state(
        {
            (t.Semantics.Room,): 3,
        }
    )

    var_dom = r.Domain({t.Semantics.Room}, [])

    subs = list(greedy.substitutions(var_dom, state))
    assert len(subs) == 1


def test_substitutions_simple():
    state = make_dummy_state(
        {
            (t.Semantics.Room,): 3,
        }
    )

    var_dom = r.Domain({t.Semantics.Room, t.Variable("room")}, [])

    subs = list(greedy.substitutions(var_dom, state))
    assert len(subs) == 3
    assert t.SpecificObject("room_0") in subs[0].tags
    assert t.SpecificObject("room_1") in subs[1].tags
    assert t.SpecificObject("room_2") in subs[2].tags


def test_substitutions_child():
    state = make_dummy_state(
        {
            (t.Semantics.Room,): 4,
        }
    )

    var_dom = r.Domain(
        {t.Semantics.Object},
        [(cl.AnyRelation(), r.Domain({t.Semantics.Room, t.Variable("room")}, []))],
    )

    subs = list(greedy.substitutions(var_dom, state))
    assert len(subs) == 4


def test_substitutions_child_complex():
    state = make_dummy_state(
        {
            (t.Semantics.Room,): 4,
        }
    )

    state.objs["obj_0"] = state_def.ObjectState(
        obj=None,
        generator=None,
        tags={t.Semantics.Object, t.SpecificObject("obj_0")},
        relations=[
            state_def.RelationState(relation=cl.Touching(), target_name="room_0")
        ],
    )

    state.objs["obj_1"] = state_def.ObjectState(
        obj=None,
        generator=None,
        tags={t.Semantics.Object, t.SpecificObject("obj_1")},
        relations=[
            state_def.RelationState(relation=cl.Touching(), target_name="room_1")
        ],
    )

    var_dom = r.Domain(
        {t.Semantics.Object, t.Variable("obj")},
        [
            (cl.AnyRelation(), r.Domain({t.Semantics.Room, t.Variable("room")}, [])),
        ],
    )

    subs = list(greedy.substitutions(var_dom, state))
    print(subs)
    assert len(subs) == 2
    assert len([s for s in subs if t.SpecificObject("obj_0") in s.tags]) == 1
    assert len([s for s in subs if t.SpecificObject("obj_1") in s.tags]) == 1
