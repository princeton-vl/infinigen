import gin
import numpy as np

from infinigen.core.constraints import (
    constraint_language as cl,
)
from infinigen.core.constraints.constraint_language.constants import RoomConstants
from infinigen.core.tags import Semantics
from infinigen_examples.indoor_constraint_examples import indoor_constraints


@gin.configurable
def office_constraints(weights=None, fast=False):
    problem = indoor_constraints()
    constraints = problem.constraints
    score_terms = problem.score_terms

    constants = RoomConstants(
        room_type={
            Semantics.Hallway,
            Semantics.MeetingRoom,
            Semantics.Office,
            Semantics.OpenOffice,
            Semantics.BreakRoom,
            Semantics.Restroom,
            Semantics.StaircaseRoom,
            Semantics.Utility,
        },
        aspect_ratio_range=(0.2, 0.3),
        fixed_contour=True,
    )

    scene = cl.scene()
    rooms = scene[Semantics.RoomContour]
    rg = rooms[Semantics.GroundFloor]

    n_staircase = 1 if constants.n_stories > 1 else 0
    node_constraint = (
        rooms[Semantics.Root].all(
            lambda r: rooms[Semantics.Hallway]
            .related_to(r, cl.Traverse())
            .count()
            .in_range(1, 2, mean=1.4)
        )
        * rooms[Semantics.Hallway].all(
            lambda r: rooms[Semantics.Hallway]
            .related_to(r, cl.Traverse())
            .count()
            .in_range(0, 2, mean=0.7)
        )
        * rooms[Semantics.Hallway].all(
            lambda r: rooms[Semantics.MeetingRoom]
            .related_to(r, cl.Traverse())
            .count()
            .in_range(0, 1, mean=0.6)
        )
        * rooms[Semantics.Hallway].all(
            lambda r: rooms[Semantics.Office]
            .related_to(r, cl.Traverse())
            .count()
            .in_range(0, 2, mean=0.6)
        )
        * rooms[Semantics.Hallway].all(
            lambda r: rooms[Semantics.OpenOffice]
            .related_to(r, cl.Traverse())
            .count()
            .in_range(0, 1, mean=0.6)
        )
        * rooms[Semantics.StaircaseRoom]
        .related_to(rooms[Semantics.Hallway], cl.Traverse())
        .count()
        .in_range(n_staircase, n_staircase, mean=n_staircase)
        * rooms[Semantics.Hallway].all(
            lambda r: rooms[Semantics.BreakRoom]
            .related_to(r, cl.Traverse())
            .count()
            .in_range(0, 1, mean=0.5)
        )
        * rooms[Semantics.Hallway].all(
            lambda r: rooms[Semantics.Restroom]
            .related_to(r, cl.Traverse())
            .count()
            .in_range(0, 1, mean=0.3)
        )
        * rooms[Semantics.Hallway].all(
            lambda r: rooms[Semantics.Utility]
            .related_to(r, cl.Traverse())
            .count()
            .in_range(0, 1, mean=0.4)
        )
        * rg[Semantics.Hallway].all(
            lambda r: rooms[Semantics.Entrance]
            .related_to(r, cl.Traverse())
            .count()
            .in_range(1, 1, mean=1)
        )
        * rooms[Semantics.OpenOffice].all(
            lambda r: rooms[Semantics.Office]
            .related_to(r, cl.Traverse())
            .count()
            .in_range(0, 1, mean=0.4)
        )
        * rooms[Semantics.OpenOffice].all(
            lambda r: rooms[Semantics.OpenOffice]
            .related_to(r, cl.Traverse())
            .count()
            .in_range(0, 1, mean=0.4)
        )
        * rooms[Semantics.OpenOffice].all(
            lambda r: rooms[Semantics.MeetingRoom]
            .related_to(r, cl.Traverse())
            .count()
            .in_range(0, 1, mean=0.3)
        )
        * rooms[Semantics.OpenOffice].all(
            lambda r: rooms[Semantics.Utility]
            .related_to(r, cl.Traverse())
            .count()
            .in_range(0, 1, mean=0.3)
        )
    )
    constraints["node_gen"] = node_constraint
    node_constraint = (
        (rooms[-Semantics.Exterior][-Semantics.Entrance].count().in_range(8, 20))
        * rooms.all(
            lambda r: rooms[-Semantics.Exterior][-Semantics.Entrance]
            .related_to(r, cl.Traverse())
            .count()
            <= 5
        )
        * (rooms[Semantics.Restroom].count() >= 1)
        * (rooms[Semantics.BreakRoom].count() >= 1)
        * ((rg[Semantics.Entrance].count() >= 1) + (rg.count() == 0))
    )
    if fast:
        node_constraint = (
            (rooms[-Semantics.Exterior][-Semantics.Entrance].count().in_range(1, 20))
            * ((rg[Semantics.Entrance].count() >= 1) + (rg.count() == 0))
            * (rooms[Semantics.OpenOffice].count() >= 1)
        )

    constraints["node"] = node_constraint

    all_rooms = scene[Semantics.RoomContour]
    rooms = all_rooms[-Semantics.Exterior][-Semantics.Staircase]

    def exterior(r):
        return r.same_level()[Semantics.Exterior]

    def pholder(r):
        return r.same_level()[Semantics.Staircase]

    room_term = (
        rooms[-Semantics.Utility][-Semantics.Restroom]
        .sum(lambda r: (r.access_angle() - np.pi / 2).clip(0))
        .minimize(weight=5.0)
        + (
            rooms[Semantics.MeetingRoom].sum(
                lambda r: (r.area() / 25).log().hinge(0, 0.4).pow(2)
            )
            + rooms[Semantics.Office].sum(
                lambda r: (r.area() / 15).log().hinge(0, 0.4).pow(2)
            )
            + rooms[Semantics.OpenOffice].sum(
                lambda r: (r.area() / 30).log().hinge(0, 0.4).pow(2)
            )
            + rooms[Semantics.BreakRoom].sum(
                lambda r: (r.area() / 15).log().hinge(0, 0.4).pow(2)
            )
            + rooms[Semantics.Restroom].sum(
                lambda r: (r.area() / 10).log().hinge(0, 0.4).pow(2)
            )
            + rooms[Semantics.StaircaseRoom].sum(
                lambda r: (r.area() / 25).log().hinge(0, 0.4).pow(2)
            )
            + rooms[Semantics.Utility].sum(
                lambda r: (r.area() / 5).log().hinge(0, 0.4).pow(2)
            )
        ).minimize(weight=500.0)
        + rooms.union(
            {
                Semantics.MeetingRoom,
                Semantics.Office,
                Semantics.OpenOffice,
                Semantics.BreakRoom,
            }
        )
        .sum(lambda r: r.aspect_ratio().log())
        .minimize(weight=50.0)
        + rooms.union({Semantics.Restroom})
        .sum(lambda r: r.aspect_ratio().log())
        .minimize(weight=40.0)
        + rooms[-Semantics.Hallway]
        .sum(lambda r: r.convexity().log())
        .minimize(weight=5.0)
        + rooms[-Semantics.Hallway]
        .sum(lambda r: (r.n_verts() - 6).clip(0).pow(1.5))
        .minimize(weight=1.0)
        + rooms.union(
            {
                Semantics.MeetingRoom,
                Semantics.Office,
                Semantics.OpenOffice,
                Semantics.BreakRoom,
            }
        )
        .sum(lambda r: r.shared_length(exterior(r)) / exterior(r).length())
        .maximize(weight=10.0)
        + rooms.union(
            {
                Semantics.MeetingRoom,
                Semantics.Office,
                Semantics.OpenOffice,
                Semantics.BreakRoom,
            }
        )
        .sum(lambda r: (r.shared_n_verts(exterior(r)) - 2).clip(0))
        .maximize(weight=1.0)
        + (
            rooms.grid_line_count(constants, "x")
            + rooms.grid_line_count(constants, "y")
        ).minimize(weight=2.0)
        + rooms.sum(
            lambda r: r.grid_line_count(constants, "x")
            + r.grid_line_count(constants, "y")
        ).minimize(weight=2.0)
        + rooms.union({Semantics.Hallway, Semantics.StaircaseRoom})
        .area()
        .minimize(weight=20.0)
        + rooms.excludes(
            {
                Semantics.Restroom,
                Semantics.Utility,
                Semantics.StaircaseRoom,
                Semantics.Hallway,
            }
        )
        .sum(lambda r: r.narrowness(constants, 2.5))
        .minimize(weight=2000.0)
        + rooms.union(
            {
                Semantics.Restroom,
                Semantics.Utility,
                Semantics.StaircaseRoom,
                Semantics.Hallway,
            }
        )
        .sum(lambda r: r.narrowness(constants, 2))
        .minimize(weight=2000.0)
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
        .minimize(weight=20.0)
    )

    score_terms["room"] = room_term

    return cl.Problem(
        constraints=constraints, score_terms=score_terms, constants=constants
    )


all_constraint_funcs = [office_constraints]
