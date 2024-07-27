import gin
import numpy as np

from infinigen.core.constraints import (
    constraint_language as cl,
)
from infinigen.core.constraints.constraint_language.constants import RoomConstants
from infinigen.core.tags import Semantics


@gin.configurable
def home_constraints(weights=None, fast=False):
    problem = indoor_constraints()
    constraints = problem.constraints
    score_terms = problem.score_terms

    # region ROOM SCENE GRAPH CONSTRAINTS/GRAMMAR

    constants = RoomConstants(fixed_contour=False)
    rooms = cl.scene()[Semantics.RoomContour]
    rg = rooms[Semantics.GroundFloor]
    ru = rooms[-Semantics.GroundFloor]

    constraints["node_gen"] = (
        rooms[Semantics.Root].all(
            lambda r: rooms[Semantics.LivingRoom]
            .related_to(r, cl.Traverse())
            .count()
            .in_range(1, 2, mean=1.1)
        )
        * rooms[Semantics.LivingRoom].all(
            lambda r: rooms[Semantics.Hallway]
            .related_to(r, cl.Traverse())
            .count()
            .in_range(0, 2, mean=1.2)
        )
        * rooms[Semantics.LivingRoom].all(
            lambda r: rooms[Semantics.Bedroom]
            .related_to(r, cl.Traverse())
            .count()
            .in_range(0, 2, mean=1.2)
        )
        * rooms[Semantics.LivingRoom].all(
            lambda r: rooms[Semantics.Closet]
            .related_to(r, cl.Traverse())
            .count()
            .in_range(0, 1, mean=0.2)
        )
        * rooms[Semantics.LivingRoom].all(
            lambda r: rooms[Semantics.Bathroom]
            .related_to(r, cl.Traverse())
            .count()
            .in_range(0, 1, mean=0.4)
        )
        * rooms[Semantics.LivingRoom].all(
            lambda r: rooms[Semantics.Balcony]
            .related_to(r, cl.Traverse())
            .count()
            .in_range(0, 1, mean=0.5)
        )
        * rg[Semantics.LivingRoom].all(
            lambda r: rooms[Semantics.DiningRoom]
            .related_to(r, cl.Traverse())
            .count()
            .in_range(0, 1, mean=0.8)
        )
        * rg[Semantics.LivingRoom].all(
            lambda r: rooms[Semantics.Utility]
            .related_to(r, cl.Traverse())
            .count()
            .in_range(0, 1, mean=0.2)
        )
        * rooms[Semantics.LivingRoom].all(
            lambda r: rooms[Semantics.StaircaseRoom]
            .related_to(r, cl.Traverse())
            .count()
            .in_range(0, 1, mean=0.5)
        )
        * rooms[Semantics.Hallway].all(
            lambda r: rooms[Semantics.Bedroom]
            .related_to(r, cl.Traverse())
            .count()
            .in_range(1, 2, mean=1.5)
        )
        * rooms[Semantics.Hallway].all(
            lambda r: rooms[Semantics.Closet]
            .related_to(r, cl.Traverse())
            .count()
            .in_range(0, 1, mean=0.2)
        )
        * rooms[Semantics.Hallway].all(
            lambda r: rooms[Semantics.Bathroom]
            .related_to(r, cl.Traverse())
            .count()
            .in_range(0, 1, mean=0.6)
        )
        * rg[Semantics.Hallway].all(
            lambda r: rooms[Semantics.Garage]
            .related_to(r, cl.Traverse())
            .count()
            .in_range(0, 1, mean=0.5)
        )
        * rg[Semantics.Hallway].all(
            lambda r: rooms[Semantics.Utility]
            .related_to(r, cl.Traverse())
            .count()
            .in_range(0, 1, mean=0.2)
        )
        * rooms[Semantics.Hallway].all(
            lambda r: rooms[Semantics.StaircaseRoom]
            .related_to(r, cl.Traverse())
            .count()
            .in_range(0, 1, mean=0.4)
        )
        * rg[Semantics.Utility].all(
            lambda r: rooms[Semantics.Garage]
            .related_to(r, cl.Traverse())
            .count()
            .in_range(0, 1, mean=0.2)
        )
        * rooms[Semantics.Kitchen].all(
            lambda r: rooms[Semantics.Utility]
            .related_to(r, cl.Traverse())
            .count()
            .in_range(0, 1, mean=0.2)
        )
        * rooms[Semantics.Bedroom].all(
            lambda r: rooms[Semantics.Bathroom]
            .related_to(r, cl.Traverse())
            .count()
            .in_range(0, 1, mean=0.6)
        )
        * rooms[Semantics.Bedroom].all(
            lambda r: rooms[Semantics.Closet]
            .related_to(r, cl.Traverse())
            .count()
            .in_range(0, 1, mean=0.6)
        )
        * rooms[Semantics.Closet]
        .related_to(rooms[Semantics.Bedroom], cl.Traverse())
        .all(
            lambda r: rooms[Semantics.Bathroom]
            .related_to(r, cl.Traverse())
            .count()
            .in_range(0, 1, mean=0.5)
        )
        * rooms[Semantics.DiningRoom].all(
            lambda r: rooms[Semantics.Kitchen]
            .related_to(r, cl.Traverse())
            .count()
            .in_range(1, 1, mean=1.0)
        )
        * rooms[Semantics.Balcony].all(
            lambda r: rooms[Semantics.Exterior]
            .related_to(r, cl.Traverse())
            .count()
            .in_range(1, 1, mean=1)
        )
        * rooms[Semantics.Garage].all(
            lambda r: rooms[Semantics.Exterior]
            .related_to(r, cl.Traverse())
            .count()
            .in_range(1, 1, mean=1)
        )
        * rg[Semantics.Hallway].all(
            lambda r: rooms[Semantics.Entrance]
            .related_to(r, cl.Traverse())
            .count()
            .in_range(0, 1, mean=0.8)
        )
        * rg[Semantics.LivingRoom].all(
            lambda r: rooms[Semantics.Entrance]
            .related_to(r, cl.Traverse())
            .count()
            .in_range(0, 1, mean=0.8)
        )
    )

    if fast:
        constraints["node_gen"] = (
            rooms[Semantics.Root].all(
                lambda r: rooms[Semantics.LivingRoom]
                .related_to(r, cl.Traverse())
                .count()
                .in_range(1, 2, mean=1.1)
            )
            * rooms[Semantics.LivingRoom].all(
                lambda r: rooms[Semantics.Bedroom]
                .related_to(r, cl.Traverse())
                .count()
                .in_range(1, 2, mean=1.2)
            )
            * rooms[Semantics.LivingRoom].all(
                lambda r: rooms[Semantics.Entrance]
                .related_to(r, cl.Traverse())
                .count()
                .in_range(1, 1, mean=1)
            )
            * rooms[Semantics.LivingRoom].all(
                lambda r: rooms[Semantics.Bathroom]
                .related_to(r, cl.Traverse())
                .count()
                .in_range(1, 1, mean=1)
            )
            * rg[Semantics.LivingRoom].all(
                lambda r: rooms[Semantics.DiningRoom]
                .related_to(r, cl.Traverse())
                .count()
                .in_range(1, 1, mean=1.0)
            )
            * rooms[Semantics.DiningRoom].all(
                lambda r: rooms[Semantics.Kitchen]
                .related_to(r, cl.Traverse())
                .count()
                .in_range(1, 1, mean=1.0)
            )
        )

    # endregion

    # region ROOM SCENE GRAPH SCORING

    def private_bathroom(r):
        return rooms[Semantics.Bathroom].related_to(r, cl.Traverse()).count() >= 1

    def private_bathroom_via_closet(r):
        return (
            rooms[Semantics.Bathroom]
            .related_to(
                rooms[Semantics.Closet].related_to(r, cl.Traverse()), cl.Traverse()
            )
            .count()
            >= 1
        )

    def public_bathroom_via_hallway(r):
        return (
            rooms[Semantics.Bathroom]
            .related_to(
                rooms[Semantics.Hallway].related_to(r, cl.Traverse()), cl.Traverse()
            )
            .count()
            >= 1
        )

    def public_bathroom_via_living_room(r):
        return (
            rooms[Semantics.Bathroom]
            .related_to(
                rooms[Semantics.LivingRoom].related_to(r, cl.Traverse()), cl.Traverse()
            )
            .count()
            >= 1
        )

    node_constraint = (
        (rooms[-Semantics.Exterior][-Semantics.Entrance].count().in_range(4, 15))
        * ((rg[Semantics.LivingRoom].count() >= 1) + (rg.count() == 0))
        * ((rg[Semantics.Entrance].count() >= 1) + (rg.count() == 0))
        * ((ru[Semantics.Bedroom].count() >= 2) + (ru.count() == 0))
        * (rooms[Semantics.Bedroom].count() >= 1)
        * (ru[Semantics.DiningRoom].count() == 0)
        * ((rg[Semantics.DiningRoom].count() >= 1) + (rg.count() == 0))
        * (ru[Semantics.Kitchen].count() == 0)
        * ((rg[Semantics.Kitchen].count() >= 1) + (rg.count() == 0))
        * (ru[Semantics.Garage].count() == 0)
        * rooms[Semantics.Garage].count().in_range(0, 1)
        * rooms[Semantics.Hallway].count().in_range(0, 3)
        * (
            rooms[Semantics.StaircaseRoom].count()
            == (1 if constants.n_stories > 1 else 0)
        )
        * rooms[Semantics.Bedroom].all(
            lambda r: ~private_bathroom(r) + ~private_bathroom_via_closet(r)
        )
        * rooms[Semantics.Bedroom].all(
            lambda r: private_bathroom(r)
            + private_bathroom_via_closet(r)
            + public_bathroom_via_hallway(r)
            + public_bathroom_via_living_room(r)
        )
    )
    if fast:
        node_constraint = (
            (rooms[Semantics.Entrance].count() >= 1)
            * (rooms[Semantics.StaircaseRoom].count() == 0)
            * (rooms[Semantics.LivingRoom].count() >= 1)
            * (rooms[Semantics.Kitchen].count() >= 1)
            * (rooms[Semantics.Bedroom].count() >= 1)
            * (rooms[Semantics.Bathroom].count() >= 1)
        )

    constraints["node"] = node_constraint

    all_rooms = cl.scene()[Semantics.RoomContour]
    rooms = all_rooms[-Semantics.Exterior][-Semantics.Staircase]

    def exterior(r):
        return r.same_level()[Semantics.Exterior]

    def pholder(r):
        return r.same_level()[Semantics.Staircase]

    room_term = (
        rooms[-Semantics.Utility][-Semantics.Bathroom][-Semantics.Closet]
        .sum(lambda r: (r.access_angle() - np.pi / 2).clip(0))
        .minimize(weight=5.0)
        + (
            rooms[Semantics.Kitchen].sum(
                lambda r: (r.area() / 20).log().hinge(0, 0.4).pow(2)
            )
            + rooms[Semantics.Bedroom].sum(
                lambda r: (r.area() / 40).log().hinge(0, 0.4).pow(2)
            )
            + rooms[Semantics.LivingRoom].sum(
                lambda r: (r.area() / 40).log().hinge(0, 0.4).pow(2)
            )
            + rooms[Semantics.DiningRoom].sum(
                lambda r: (r.area() / 20).log().hinge(0, 0.4).pow(2)
            )
            + rooms[Semantics.Closet].sum(
                lambda r: (r.area() / 5).log().hinge(0, 0.4).pow(2)
            )
            + rooms[Semantics.Bathroom].sum(
                lambda r: (r.area() / 8).log().hinge(0, 0.4).pow(2)
            )
            + rooms[Semantics.Utility].sum(
                lambda r: (r.area() / 5).log().hinge(0, 0.4).pow(2)
            )
            + rooms[Semantics.Garage].sum(
                lambda r: (r.area() / 25).log().hinge(0, 0.4).pow(2)
            )
            + rooms[Semantics.Balcony].sum(
                lambda r: (r.area() / 8).log().hinge(0, 0.4).pow(2)
            )
            + rooms[Semantics.StaircaseRoom].sum(
                lambda r: (r.area() / 25).log().hinge(0, 0.4).pow(2)
            )
        ).minimize(weight=500.0)
        + sum(
            rooms[tag].sum(lambda r: r.aspect_ratio().log())
            for tag in [
                Semantics.Kitchen,
                Semantics.Bedroom,
                Semantics.LivingRoom,
                Semantics.DiningRoom,
            ]
        ).minimize(weight=50.0)
        + sum(
            rooms[tag].sum(lambda r: r.aspect_ratio().log())
            for tag in [Semantics.Closet, Semantics.Bathroom]
        ).minimize(weight=40.0)
        + rooms[-Semantics.Hallway]
        .sum(lambda r: r.convexity().log())
        .minimize(weight=5.0)
        + rooms[-Semantics.Hallway]
        .sum(lambda r: (r.n_verts() - 6).clip(0).pow(1.5))
        .minimize(weight=1.0)
        + sum(
            rooms[tag].sum(
                lambda r: r.shared_length(exterior(r)) / exterior(r).length()
            )
            for tag in {
                Semantics.Bedroom,
                Semantics.Balcony,
                Semantics.DiningRoom,
                Semantics.Kitchen,
                Semantics.LivingRoom,
            }
        ).maximize(weight=10.0)
        + sum(
            rooms[tag].sum(lambda r: (r.shared_n_verts(exterior(r)) - 2).clip(0))
            for tag in {
                Semantics.Bedroom,
                Semantics.Balcony,
                Semantics.DiningRoom,
                Semantics.Kitchen,
                Semantics.LivingRoom,
            }
        ).maximize(weight=1.0)
        + (
            rooms.grid_line_count(constants, "x")
            + rooms.grid_line_count(constants, "y")
        ).minimize(weight=2.0)
        + rooms.sum(
            lambda r: r.grid_line_count(constants, "x")
            + r.grid_line_count(constants, "y")
        ).minimize(weight=2.0)
        + sum(
            rooms[tag].area() for tag in {Semantics.Hallway, Semantics.StaircaseRoom}
        ).minimize(weight=20.0)
        + rooms.excludes(
            {
                Semantics.Bathroom,
                Semantics.Utility,
                Semantics.StaircaseRoom,
                Semantics.Hallway,
                Semantics.Balcony,
            }
        )
        .sum(lambda r: r.narrowness(constants, 2.5))
        .minimize(weight=2000.0)
        + sum(
            rooms[tag].sum(lambda r: r.narrowness(constants, 2))
            for tag in {
                Semantics.Bathroom,
                Semantics.Utility,
                Semantics.StaircaseRoom,
                Semantics.Hallway,
                Semantics.Balcony,
            }
        ).minimize(weight=2000.0)
        + rooms[Semantics.StaircaseRoom]
        .sum(lambda r: r.intersection(pholder(r)) / pholder(r).area())
        .maximize(weight=50.0)
        + rooms[Semantics.StaircaseRoom]
        .sum(
            lambda r: (r.intersection(pholder(r)) / pholder(r).area()).hinge(
                constants.staircase_thresh, 1
            )
        )
        .minimize(weight=1e5)
        + rooms[Semantics.StaircaseRoom]
        .sum(lambda r: r.area() / pholder(r).area() - r.intersection(pholder(r)))
        .minimize(weight=5.0)
    )

    score_terms["room"] = room_term

    # endregion

    return cl.Problem(
        constraints=constraints,
        score_terms=score_terms,
        constants=constants,
    )


all_constraint_funcs = [home_constraints]
