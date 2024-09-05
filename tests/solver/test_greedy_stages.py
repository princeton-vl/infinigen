# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

from pprint import pprint

import pytest

from infinigen.assets.objects.tableware import PlantContainerFactory
from infinigen.core import tags as t
from infinigen.core.constraints import checks, evaluator, usage_lookup
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints import reasoning as r
from infinigen.core.constraints.example_solver import (
    greedy,
    propose_discrete,
    state_def,
)
from infinigen.core.constraints.example_solver.room.base import room_name
from infinigen.core.util import blender as butil
from infinigen_examples import generate_indoors
from infinigen_examples.constraints import home as ex
from infinigen_examples.constraints import util as cu


@pytest.mark.parametrize("key", generate_indoors.default_greedy_stages().keys())
def test_stages_relations(key):
    pprint(generate_indoors.default_greedy_stages())

    v = generate_indoors.default_greedy_stages()[key]

    assert not v.is_recursive()

    if len(v.relations) != 0:
        if all(isinstance(r, cl.NegatedRelation) for r, _ in v.relations):
            raise ValueError(
                f"Stage {key} has no positive relation, {[r for r, _ in v.relations]}"
            )
        # if any(isinstance(r, cl.AnyRelation) for r, _ in v.relations):
        #    raise ValueError(f"Stage {key} has an AnyRelation which is underspecified, {v}")


# @pytest.mark.parametrize('key', generate_indoors.default_greedy_stages().keys())
# @pytest.mark.parametrize('roomtype', cu.room_types)
# def test_stage_bound_roomsubs(key: str, roomtype: t.Semantics):
#
#    stages = generate_indoors.default_greedy_stages()
#    stage = stages[key]
#    stage = r.domain_tag_substitute(stage, t.Variable('room'), r.Domain({roomtype}))
#
#    bounds = r.constraint_bounds(ex.home_furniture_constraints())


def test_validate_bounds():
    bounds = r.constraint_bounds(ex.home_furniture_constraints())

    for b in bounds:
        for rel, dom in b.domain.relations:
            if not r.domain_finalized(dom, check_anyrel=False, check_variable=True):
                raise ValueError(f"Unfinalized domain {dom=}")
            if isinstance(rel, cl.GeometryRelation):
                if rel.child_tags == set():
                    raise ValueError(f"GeometryRelation with empty child_tags in {b=}")
                if rel.parent_tags == set():
                    raise ValueError(f"GeometryRelation with empty parent_tags in {b=}")


def test_validate_stages():
    stages = generate_indoors.default_greedy_stages()

    wall = stages["on_wall"]
    floor = stages["on_floor_freestanding"]
    assert not wall.intersects(floor)

    onobj = stages["obj_ontop_obj"]
    assert not onobj.intersects(floor)

    checks.validate_stages(stages)


def test_example_intersects():
    on_wall_complex = cl.StableAgainst(
        {-t.Subpart.Top, t.Subpart.Back, -t.Subpart.Front},
        {
            t.Subpart.Wall,
            t.Subpart.Visible,
            -t.Subpart.Ceiling,
            -t.Subpart.SupportSurface,
        },
    )
    on_wall_simple = cl.StableAgainst({}, {t.Subpart.Wall})
    assert on_wall_simple.intersects(on_wall_complex)

    dom = r.Domain(
        {t.Semantics.WallDecoration, t.Semantics.Object},
        relations=[(on_wall_complex, r.Domain({t.Semantics.Room}, []))],
    )

    filter = generate_indoors.default_greedy_stages()["on_wall"]
    assert propose_discrete.active_for_stage(dom, filter)


def test_contradiction_fail():
    prob = cl.Problem(
        constraints=[
            cl.scene()[{t.Semantics.Object, -t.Semantics.Object}].count().in_range(1, 3)
        ],
        score_terms=[],
    )
    with pytest.raises(ValueError):
        checks.check_contradictory_domains(prob)


def get_walldec():
    return r.Domain(
        {t.Semantics.WallDecoration, t.Semantics.Object, -t.Semantics.Room},
        [
            (
                cl.StableAgainst(
                    {-t.Subpart.Front, -t.Subpart.Top, t.Subpart.Back},
                    {
                        -t.Subpart.SupportSurface,
                        -t.Subpart.Ceiling,
                        t.Subpart.Visible,
                        t.Subpart.Wall,
                    },
                ),
                r.Domain({t.Semantics.Room, -t.Semantics.Object}, []),
            )
        ],
    )


def test_example_walldec():
    dom = get_walldec()
    stages = generate_indoors.default_greedy_stages()

    assert not propose_discrete.active_for_stage(dom, stages["on_ceiling"])
    assert not propose_discrete.active_for_stage(dom, stages["on_floor_freestanding"])

    assert t.satisfies(dom.tags, stages["on_wall"].tags)
    print("ONWALL", stages["on_wall"])

    assert propose_discrete.active_for_stage(dom, stages["on_wall"])


def test_example_floorwall():
    on = cl.StableAgainst(
        {t.Subpart.Bottom, -t.Subpart.Front, -t.Subpart.Top, -t.Subpart.Back},
        {
            t.Subpart.SupportSurface,
            t.Subpart.Visible,
            -t.Subpart.Wall,
            -t.Subpart.Ceiling,
        },
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

    dom = r.Domain(
        {
            t.Semantics.Storage,
            t.Semantics.Furniture,
            t.Semantics.Object,
            -t.Semantics.Room,
        },
        [
            (on, r.Domain({t.Semantics.Room, -t.Semantics.Object}, [])),
            (against, r.Domain({t.Semantics.Room, -t.Semantics.Object}, [])),
        ],
    )

    stages = generate_indoors.default_greedy_stages()

    assert propose_discrete.active_for_stage(dom, stages["on_floor_and_wall"])
    assert not propose_discrete.active_for_stage(dom, stages["on_wall"])


def test_example_secondary():
    floorwall_furn = r.Domain(
        {t.Semantics.Furniture, t.Semantics.Storage, t.Semantics.Object},
        [
            (cu.on_floor, r.Domain({t.Semantics.Room, -t.Semantics.Object}, [])),
            (cu.against_wall, r.Domain({t.Semantics.Room, -t.Semantics.Object}, [])),
        ],
    )
    dom = r.Domain(
        {t.FromGenerator(PlantContainerFactory), t.Semantics.Object, -t.Semantics.Room},
        [(cl.StableAgainst({t.Subpart.Bottom}, {t.Subpart.Top}), floorwall_furn)],
    )

    stages = generate_indoors.default_greedy_stages()
    on_obj = stages["obj_ontop_obj"]
    assert propose_discrete.active_for_stage(dom, on_obj)


def test_example_sideobj():
    anyroom = r.Domain({t.Semantics.Room, -t.Semantics.Object}, [])

    objonroom = r.Domain(
        {t.Semantics.Object, t.Semantics.Table, -t.Semantics.Room},
        [(cu.on_floor, anyroom)],
    )

    dom = r.Domain(
        {t.Semantics.Object, t.Semantics.Chair, -t.Semantics.Room},
        [(cu.front_against, objonroom), (cu.on_floor, anyroom)],
    )
    stages = generate_indoors.default_greedy_stages()
    assert propose_discrete.active_for_stage(dom, stages["side_obj"])
    assert not propose_discrete.active_for_stage(dom, stages["on_floor_freestanding"])


def test_example_monitor():
    desk = r.Domain(
        {t.Semantics.Object, t.Semantics.Desk, -t.Semantics.Room},
        [
            (cu.on_floor, r.Domain({t.Semantics.Room, -t.Semantics.Object}, [])),
            (cu.against_wall, r.Domain({t.Semantics.Room, -t.Semantics.Object}, [])),
        ],
    )

    monitor = r.Domain(
        {t.Semantics.Object, t.Semantics.Chair, -t.Semantics.Room},
        [  # chair vs other tags doesnt matter
            (cu.ontop, desk),
            (cu.against_wall, r.Domain({t.Semantics.Room, -t.Semantics.Object}, [])),
        ],
    )

    stages = generate_indoors.default_greedy_stages()

    assert propose_discrete.active_for_stage(monitor, stages["obj_ontop_obj"])
    assert not propose_discrete.active_for_stage(monitor, stages["on_wall"])


def test_example_on_obj():
    not_others = {
        -t.Semantics.LivingRoom,
        -t.Semantics.Hallway,
        -t.Semantics.Closet,
        -t.Semantics.Balcony,
        -t.Semantics.Staircase,
        -t.Semantics.Garage,
        -t.Semantics.DiningRoom,
        -t.Semantics.Utility,
        -t.Semantics.Bathroom,
        -t.Semantics.Kitchen,
    }
    bedroom_storage = r.Domain(
        {t.Semantics.Object, t.Semantics.Furniture, t.Semantics.Storage},
        [
            (
                cu.on_floor,
                r.Domain({t.Semantics.Bedroom, t.Semantics.Room}.union(not_others), []),
            ),
            (
                cu.against_wall,
                r.Domain({t.Semantics.Bedroom, t.Semantics.Room}.union(not_others), []),
            ),
        ],
    )

    obj = r.Domain(
        {t.Semantics.OfficeShelfItem, t.Semantics.Object}, [(cu.on, bedroom_storage)]
    )

    onfloor = generate_indoors.default_greedy_stages()["on_floor_freestanding"]
    dining = r.domain_tag_substitute(
        onfloor, cu.variable_room, r.Domain({t.Semantics.DiningRoom})
    )

    assert not propose_discrete.active_for_stage(obj, dining)


def test_active_incorrect_room():
    onfloor = generate_indoors.default_greedy_stages()["on_floor_freestanding"]
    dining = r.domain_tag_substitute(
        onfloor, cu.variable_room, r.Domain({t.Semantics.DiningRoom})
    )

    sofa = r.Domain(
        {t.Semantics.Object, t.Semantics.Seating, -t.Semantics.Room},
        [
            (
                cu.on_floor,
                r.Domain(
                    {
                        t.Semantics.LivingRoom,
                        -t.Semantics.DiningRoom,
                        -t.Semantics.Object,
                    },
                    [],
                ),
            )
        ],
    )

    assert not propose_discrete.active_for_stage(sofa, dining)


def test_obj_on_ceilinglight():
    bounds = r.constraint_bounds(ex.home_furniture_constraints())

    ceilinglight = r.Domain(
        {t.Semantics.Object, t.Semantics.Lighting, -t.Semantics.Room},
        [(cu.hanging, r.Domain({t.Semantics.Room, -t.Semantics.Object}, []))],
    )

    active_bounds = [
        b for b in bounds if propose_discrete.active_for_stage(ceilinglight, b.domain)
    ]

    assert active_bounds == []


def test_greedy_partition_home():
    usage_lookup.initialize_from_dict(ex.home_asset_usage())
    prob = ex.home_furniture_constraints()
    checks.check_problem_greedy_coverage(prob, generate_indoors.default_greedy_stages())


def test_contradiction_home():
    prob = ex.home_furniture_constraints()
    checks.check_contradictory_domains(prob)


@pytest.mark.parametrize("rtype", sorted(cu.room_types, key=lambda x: x.name))
def test_room_has_viols_at_init(rtype):
    prob = ex.home_furniture_constraints()

    ostate_name = room_name(rtype, 0)
    state = state_def.State(
        {
            ostate_name: state_def.ObjectState(
                obj=butil.spawn_cube(),
                generator=None,
                tags={rtype, t.Semantics.Room},
                relations=[],
            )
        }
    )

    active_count = greedy.update_active_flags(state, {cu.variable_room: ostate_name})
    print("active", rtype, active_count)
    assert active_count > 0

    filter = generate_indoors.default_greedy_stages()["on_floor_freestanding"]
    filter = r.domain_tag_substitute(
        filter, cu.variable_room, r.Domain({rtype, t.Semantics.Room})
    )

    result = evaluator.evaluate_problem(prob, state, filter)

    assert result.viol_count() > 0
