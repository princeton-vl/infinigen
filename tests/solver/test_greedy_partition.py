# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import copy

import pytest
from test_greedy_substitutions import make_dummy_state

from infinigen.core import tags as t
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints import evaluator, usage_lookup
from infinigen.core.constraints import reasoning as r
from infinigen.core.constraints.example_solver import (
    greedy,
    propose_discrete,
    state_def,
)
from infinigen_examples import generate_indoors
from infinigen_examples.constraints import home as ex
from infinigen_examples.constraints import util as cu


def test_partition_basecase_irrelevant():
    cons = cl.scene()[t.Semantics.TableDisplayItem].count().in_range(0, 1)
    res, relevant = greedy.filter_constraints(cons, r.Domain({t.Semantics.Room}, []))
    assert not relevant


def test_basecase_relevant():
    cons = cl.scene()[t.Semantics.Room].count().in_range(0, 1)
    filter = r.Domain({t.Semantics.Room}, [])
    res, relevant = greedy.filter_constraints(cons, filter)
    assert relevant


def test_partition_collapse_binop():
    cons = cl.scene()[t.Semantics.Furniture].count().in_range(0, 1) * cl.scene()[
        t.Semantics.Room
    ].count().in_range(0, 1)

    assert not greedy.filter_constraints(
        cons.operands[0], r.Domain({t.Semantics.Room}, [])
    )[1]
    assert greedy.filter_constraints(
        cons.operands[1], r.Domain({t.Semantics.Room}, [])
    )[1]

    res, relevant = greedy.filter_constraints(cons, r.Domain({t.Semantics.Room}, []))
    assert relevant

    print("RES", res)

    expect = cl.scene()[t.Semantics.Room].count().in_range(0, 1)
    assert r.expr_equal(res, expect)


def test_partition_eliminate_irrelevant():
    scene = cl.scene()
    firstpart = scene[t.Semantics.Furniture].count().in_range(0, 1)
    secondpart = scene[t.Semantics.Furniture].all(
        lambda f: (
            scene[t.Semantics.Chair]
            .related_to(f, cl.AnyRelation())
            .count()
            .in_range(0, 1)
        )
    )
    cons = firstpart * secondpart

    assert not greedy.filter_constraints(
        secondpart, r.Domain({t.Semantics.Furniture}, [])
    )[1]

    res, relevant = greedy.filter_constraints(cons, r.Domain({t.Semantics.Furniture}))
    assert relevant
    assert r.expr_equal(res, firstpart)


def test_greedy_partition_bathroom():
    usage_lookup.initialize_from_dict(ex.home_asset_usage())
    prob = ex.home_furniture_constraints()
    stages = generate_indoors.default_greedy_stages()

    bath_cons = prob.constraints["bathroom"]

    on_floor = stages["on_floor_and_wall"]
    on_floor_any = r.domain_tag_substitute(on_floor, cu.variable_room, r.Domain())
    assert greedy.filter_constraints(bath_cons, on_floor_any)[1]

    on_bathroom = r.domain_tag_substitute(
        on_floor, cu.variable_room, r.Domain({t.Semantics.Bathroom})
    )
    assert greedy.filter_constraints(bath_cons, on_bathroom)[1]


def test_greedy_partition_multilevel():
    usage_lookup.initialize_from_dict(ex.home_asset_usage())
    stages = generate_indoors.default_greedy_stages()

    bathroom = cl.scene()[{t.Semantics.Room, t.Semantics.Bathroom}].excludes(
        cu.room_types
    )
    storage = cl.scene()[t.Semantics.Storage]

    bath_cons_1 = storage.related_to(bathroom, cu.on_floor).count().in_range(0, 1)

    on_hallway = r.domain_tag_substitute(
        stages["on_floor_and_wall"], cu.variable_room, r.Domain({t.Semantics.Hallway})
    )
    assert not greedy.filter_constraints(bath_cons_1, on_hallway)[1]

    bath_cons_2 = bathroom.all(
        lambda r: storage.related_to(r, cu.on_floor).count().in_range(0, 1)
    )
    assert not greedy.filter_constraints(bath_cons_2, on_hallway)[1]

    bath_cons_3 = bathroom.all(
        lambda r: (
            storage.related_to(r).all(
                lambda s: cl.scene()[t.Semantics.Object]
                .related_to(s)
                .count()
                .in_range(0, 1)
            )
        )
    )
    assert not greedy.filter_constraints(bath_cons_3, on_hallway)[1]


def test_greedy_partition_bathroom_nofalsepositive():
    usage_lookup.initialize_from_dict(ex.home_asset_usage())
    prob = ex.home_furniture_constraints()
    stages = generate_indoors.default_greedy_stages()

    bath_cons = prob.constraints["bathroom"]

    on_hallway = r.domain_tag_substitute(
        stages["on_floor_and_wall"], cu.variable_room, r.Domain({t.Semantics.Hallway})
    )
    assert not greedy.filter_constraints(bath_cons, on_hallway)[1]


def test_greedy_partition_plants():
    usage_lookup.initialize_from_dict(ex.home_asset_usage())
    prob = ex.home_furniture_constraints()
    stages = generate_indoors.default_greedy_stages()

    plant_cons = prob.constraints["plants"]

    on_floor = stages["on_floor_and_wall"]
    on_floor_any = r.domain_tag_substitute(on_floor, cu.variable_room, r.Domain())
    assert greedy.filter_constraints(plant_cons, on_floor_any)[1]

    on_bathroom = r.domain_tag_substitute(
        on_floor, cu.variable_room, r.Domain({t.Semantics.Bathroom})
    )
    assert greedy.filter_constraints(plant_cons, on_bathroom)[1]


@pytest.mark.skip  # filter_constraints development has been abandoned until a later date
def test_objects_on_generic_obj():
    usage_lookup.initialize_from_dict(ex.home_asset_usage())
    stages = generate_indoors.default_greedy_stages()

    on_obj = stages["on_obj"]
    on_obj = r.domain_tag_substitute(on_obj, cu.variable_room, r.Domain())
    on_obj = r.domain_tag_substitute(
        on_obj,
        cu.variable_obj,
        r.Domain({t.SpecificObject("thatchair"), t.Semantics.Chair}),
    )
    print("ON_OBJ_FILTER", on_obj)

    bathroom = cl.scene()[t.Semantics.Room, t.Semantics.Bathroom]
    storage = cl.scene()[t.Semantics.Object, t.Semantics.Storage]
    prob = bathroom.all(
        lambda r: storage.related_to(r).all(
            lambda s: (
                cl.scene()[t.Semantics.Object].related_to(s).count().in_range(0, 1)
            )
        )
    )

    cons, relevant = greedy.filter_constraints(prob, on_obj)
    assert not relevant


@pytest.mark.skip  # filter_constraints development has been abandoned until a later date
def test_on_obj_coverage():
    cons = cl.scene()[t.Semantics.Room].all(
        lambda r: (
            cl.scene()[t.Semantics.Storage]
            .related_to(r)
            .all(
                lambda s: (
                    cl.scene()[t.Semantics.Object].related_to(s).count().in_range(0, 1)
                )
            )
        )
    )

    obj_in_bathroom = r.domain_tag_substitute(
        generate_indoors.default_greedy_stages()["on_obj"],
        cu.variable_room,
        r.Domain({t.Semantics.Bathroom}),
    )
    obj_in_bathroom = r.domain_tag_substitute(
        obj_in_bathroom, cu.variable_obj, r.Domain({t.Semantics.Storage})
    )

    res, relevant = greedy.filter_constraints(cons, obj_in_bathroom)
    assert relevant


@pytest.mark.skip  # filter_constraints development has been abandoned until a later date
def test_only_bathcons_coverage():
    usage_lookup.initialize_from_dict(ex.home_asset_usage())
    prob = ex.home_furniture_constraints()
    stages = generate_indoors.default_greedy_stages()

    bath_cons = prob.constraints["bathroom"]

    dom = r.domain_tag_substitute(
        stages["on_floor_and_wall"], cu.variable_room, r.Domain({t.Semantics.Bathroom})
    )
    assert greedy.filter_constraints(bath_cons, dom)[1]

    dom = r.domain_tag_substitute(
        stages["on_wall"], cu.variable_room, r.Domain({t.Semantics.Bathroom})
    )
    assert greedy.filter_constraints(bath_cons, dom)[1]

    dom = r.domain_tag_substitute(
        stages["on_obj"], cu.variable_room, r.Domain({t.Semantics.Bathroom})
    )
    dom = r.domain_tag_substitute(dom, cu.variable_obj, r.Domain({t.Semantics.Storage}))
    assert greedy.filter_constraints(bath_cons, dom)[1]


@pytest.fixture
def precompute_all_coverage():
    usage_lookup.initialize_from_dict(ex.home_asset_usage())
    prob = ex.home_furniture_constraints()
    stages = generate_indoors.default_greedy_stages()

    cons_coverage = {k: set() for k in prob.constraints.keys()}
    score_coverage = {k: set() for k in prob.score_terms.keys()}

    for k, filter in stages.items():
        for roomtype in cu.room_types:
            room_filter = r.domain_tag_substitute(
                copy.deepcopy(filter), cu.variable_room, r.Domain({roomtype})
            )

            # eliminate the var, assume any object is fine, most generous possible assumption
            room_filter = r.domain_tag_substitute(
                room_filter, cu.variable_obj, r.Domain()
            )
            for name, cons in prob.constraints.items():
                if greedy.filter_constraints(cons, room_filter)[1]:
                    cons_coverage[name].add((k, roomtype))
            for name, score in prob.score_terms.items():
                if greedy.filter_constraints(score, room_filter)[1]:
                    score_coverage[name].add((k, roomtype))

    return cons_coverage, score_coverage


@pytest.mark.skip  # filter_constraints development has been abandoned until a later date
def test_specific_coverage(precompute_all_coverage):
    cons_coverage, _ = precompute_all_coverage

    assert cons_coverage["bathroom"] == {
        ("on_floor_and_wall", t.Semantics.Bathroom),
        ("on_wall", t.Semantics.Bathroom),
        ("on_obj", t.Semantics.Bathroom),
    }

    assert cons_coverage["diningroom"] == {
        ("on_floor_and_wall", t.Semantics.DiningRoom),
        ("on_wall", t.Semantics.DiningRoom),
        ("on_obj", t.Semantics.DiningRoom),
    }

    assert cons_coverage["livingroom"] == {
        ("on_floor_and_wall", t.Semantics.LivingRoom),
        ("on_wall", t.Semantics.LivingRoom),
        ("on_obj", t.Semantics.LivingRoom),
    }


@pytest.mark.skip  # filter_constraints development has been abandoned until a later date
def test_greedy_partition_coverage(precompute_all_coverage):
    cons_coverage, score_coverage = precompute_all_coverage

    for k, v in cons_coverage.items():
        if len(cons_coverage[k]) == 0:
            raise ValueError(f"Constraint {k} has no coverage")
    for k, v in score_coverage.items():
        if len(score_coverage[k]) == 0:
            raise ValueError(f"Score term {k} has no coverage")


def get_on_diningroom_stage():
    usage_lookup.initialize_from_dict(ex.home_asset_usage())
    stages = generate_indoors.default_greedy_stages()
    on_diningroom = r.domain_tag_substitute(
        stages["on_floor_and_wall"],
        cu.variable_room,
        r.Domain({t.Semantics.DiningRoom, t.Semantics.Room}),
    )
    return on_diningroom


@pytest.mark.skip  # filter_constraints development has been abandoned until a later date
def test_greedy_partition_diningroom():
    on_diningroom = get_on_diningroom_stage()
    prob = ex.home_furniture_constraints()
    diningroom = prob.constraints["diningroom"]

    for node in diningroom.traverse():
        if isinstance(node, cl.item):
            print(node)

    res, relevant = greedy.filter_constraints(diningroom, on_diningroom)
    assert relevant

    print("FILTER", on_diningroom)
    print("RES", res)

    assert isinstance(res, cl.ForAll)
    assert res.pred.__class__ is not cl.constant


@pytest.mark.skip  # filter_constraints development has been abandoned until a later date
def test_diningroom_bounds_active():
    usage_lookup.initialize_from_dict(ex.home_asset_usage())
    stages = generate_indoors.default_greedy_stages()
    on_diningroom = r.domain_tag_substitute(
        stages["on_floor_freestanding"],
        cu.variable_room,
        r.Domain({t.Semantics.DiningRoom}),
    )

    prob = ex.home_furniture_constraints()
    diningroom = prob.constraints["diningroom"]

    bounds_before_preproc = r.constraint_bounds(diningroom)
    bounds = propose_discrete.preproc_bounds(
        bounds_before_preproc, state_def.State({}), on_diningroom
    )

    assert len(bounds) > 0


@pytest.mark.skip  # filter_constraints development has been abandoned until a later date
def test_partition_keep_constants():
    cons = cl.scene()[t.Semantics.Room].count() * 2
    res, relevant = greedy.filter_constraints(cons, r.Domain({t.Semantics.Room}, []))
    assert relevant
    assert r.expr_equal(res, cons)


@pytest.mark.skip  # filter_constraints development has been abandoned until a later date
def test_multiroom_viol():
    state = make_dummy_state({(t.Semantics.Room,): 3})

    state.objs["room_0"].tags.add(t.Semantics.DiningRoom)

    cons = cl.scene()[t.Semantics.Room].all(
        lambda r: cl.scene()[{t.Semantics.Object, t.Semantics.Storage}]
        .related_to(r, cu.on_floor)
        .count()
        == 1
    )
    prob = cl.Problem({"storage": cons}, {})

    on_diningroom = get_on_diningroom_stage()
    prob, relevant = greedy.filter_constraints(prob, on_diningroom)
    assert relevant

    print("PRED", prob.constraints["storage"].pred)
    print("OBJS", prob.constraints["storage"].objs)

    result = evaluator.evaluate_problem(prob, state)
    assert (
        result.viol_count() == 1
    )  # only one room is relevant, so only one violation applies for this stage

    state.objs["stor_1"] = state_def.ObjectState(
        obj=None,
        generator=None,
        tags={t.Semantics.Storage},
        relations=[
            state_def.RelationState(relation=cl.StableAgainst(), target_name="room_0")
        ],
    )

    result = evaluator.evaluate_problem(prob, state)
    assert result.viol_count() == 0  # only one room is relevant, and it has an obj


@pytest.mark.skip
def test_forall_furnroom():
    scene = cl.scene()
    rooms = scene[t.Semantics.Room]
    furniture = scene[t.Semantics.Furniture]

    cons = rooms.all(lambda r: furniture.related_to(r).count().in_range(0, 1))

    room = r.Domain({t.Semantics.Room}, [])
    furn = r.Domain({t.Semantics.Furniture}, [])
    furn_room = furn.with_relation(cl.AnyRelation(), room)
    furn_no_room = furn.with_relation(-cl.AnyRelation(), room)

    res, rel = greedy.filter_constraints(cons, furn)
    assert rel
    assert r.expr_equal(res, cons)

    res, rel = greedy.filter_constraints(cons, furn_room)
    assert rel
    assert r.expr_equal(res, cons)

    res, rel = greedy.filter_constraints(cons, furn_no_room)
    assert not rel


@pytest.mark.skip
def test_forall_narrow_pred():
    scene = cl.scene()
    rooms = scene[t.Semantics.Room]
    furniture = scene[t.Semantics.Furniture]

    cons = rooms.all(lambda r: furniture.related_to(r).count().in_range(0, 1))

    room = r.Domain({t.Semantics.Room}, [])
    stor = r.Domain({t.Semantics.Furniture}, [])
    stor_room = stor.with_relation(cl.AnyRelation(), room)
    stor_no_room = stor.with_relation(-cl.AnyRelation(), room)

    res, rel = greedy.filter_constraints(cons, stor)
    assert rel
    assert r.expr_equal(res, cons)

    res, rel = greedy.filter_constraints(cons, stor_room)
    assert rel
    assert r.expr_equal(res, cons)

    res, rel = greedy.filter_constraints(cons, stor_no_room)
    assert not rel


@pytest.mark.skip
def test_forall_narrow_loopvar():
    scene = cl.scene()
    rooms = scene[t.Semantics.Room]
    furniture = scene[t.Semantics.Furniture]

    cons = rooms.all(lambda r: furniture.related_to(r).count().in_range(0, 1))

    droom = r.Domain({t.Semantics.Room, t.Semantics.DiningRoom}, [])
    furn = r.Domain({t.Semantics.Furniture}, [])
    furn_room = furn.with_relation(cl.AnyRelation(), droom)
    furn_no_room = furn.with_relation(-cl.AnyRelation(), droom)

    cons_narrow = r.FilterByDomain(rooms, droom).all(
        lambda r: furniture.related_to(r).count().in_range(0, 1)
    )

    res, rel = greedy.filter_constraints(cons, furn)
    print(res)
    assert rel
    assert r.expr_equal(res, cons_narrow)

    res, rel = greedy.filter_constraints(cons, furn_room)
    assert rel
    assert r.expr_equal(res, cons_narrow)

    res, rel = greedy.filter_constraints(cons, furn_no_room)
    assert not rel


@pytest.mark.skip
def test_forall_sumconst():
    scene = cl.scene()
    rooms = scene[t.Semantics.Room]
    scene[t.Semantics.Furniture]

    sumcons = rooms.sum(lambda r: cl.constant(1))
    assert greedy.filter_constraints(sumcons, r.Domain({t.Semantics.Room}))[1]
