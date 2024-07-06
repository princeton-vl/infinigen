import gin
import numpy as np

from infinigen.core.constraints import (
    constraint_language as cl,
)
from infinigen.core.constraints.constraint_language.constants import RoomConstants
from infinigen.core.tags import Semantics
from infinigen_examples.indoor_constraint_examples import indoor_constraints


@gin.configurable
def warehouse_constraints(weights=None, fast=False):
    problem = indoor_constraints()
    constraints = problem.constraints
    score_terms = problem.score_terms

    constants = RoomConstants(
        n_stories=1, room_type={Semantics.Warehouse,
            Semantics.Hallway,
            Semantics.FactoryOffice,
            Semantics.Garage,
            Semantics.Restroom,
            Semantics.StaircaseRoom,
            Semantics.Utility
        }, fixed_contour=False
    )

    scene = cl.scene()
    rooms = scene[Semantics.RoomContour]
    rg = rooms[Semantics.GroundFloor]

    node_constraint = (
            rooms[Semantics.Root].all(
                lambda r: rooms[Semantics.Warehouse].related_to(r, cl.Traverse()).count().in_range(1, 2, mean=1.5)
            ) *
            rooms[Semantics.Warehouse].all(
                lambda r: rooms[Semantics.Hallway].related_to(r, cl.Traverse()).count().in_range(1, 2, mean=1.1)
            ) *
            rooms[Semantics.Warehouse].all(
                lambda r: rooms[Semantics.FactoryOffice].related_to(r, cl.Traverse()).count().in_range(0, 1, mean=.7)
            ) *
            rooms[Semantics.Warehouse].all(
                lambda r: rooms[Semantics.Garage].related_to(r, cl.Traverse()).count().in_range(0, 1, mean=.8)
            ) *
            rooms[Semantics.Warehouse].all(
                lambda r: rooms[Semantics.Utility].related_to(r, cl.Traverse()).count().in_range(0, 2, mean=.8)
            ) *
            rooms[Semantics.Hallway].all(
                lambda r: rooms[Semantics.FactoryOffice].related_to(r, cl.Traverse()).count().in_range(0, 1, mean=.7)
            ) *
            rooms[Semantics.Hallway].all(
                lambda r: rooms[Semantics.StaircaseRoom].related_to(r, cl.Traverse()).count().in_range(0, 1, mean=.5)
            ) *
            rooms[Semantics.Hallway].all(
                lambda r: rooms[Semantics.Restroom].related_to(r, cl.Traverse()).count().in_range(0, 1, mean=.5)
            ) *
            rooms[Semantics.Hallway].all(
                lambda r: rooms[Semantics.Utility].related_to(r, cl.Traverse()).count().in_range(0, 1, mean=.5)
            ) *
            rooms[Semantics.Hallway].all(
                lambda r: rooms[Semantics.Entrance].related_to(r, cl.Traverse()).count().in_range(1, 1, mean=1)
            ) *
            rooms[Semantics.Garage].all(
                lambda r: rooms[Semantics.Exterior].related_to(r, cl.Traverse()).count().in_range(1, 1, mean=1)
            )
    )
    constraints['node_gen'] = node_constraint

    node_constraint = (
            rooms.all(
                lambda r: rooms[-Semantics.Exterior][-Semantics.Entrance].related_to(r, cl.Traverse()).count() <= 6
            ) *
            (rooms[Semantics.Restroom].count() >= 1) *
            ((rg[Semantics.Garage].count() >= 1) + (rg.count() == 0)) *
            ((rg[Semantics.Entrance].count() >= 1) + (rg.count() == 0)) *
            (rooms[Semantics.StaircaseRoom].count() == (1 if constants.n_stories > 1 else 0)
             ))

    if fast:
        node_constraint = (
                (rooms[-Semantics.Exterior][-Semantics.Entrance].count().in_range(1, 20)) *
                ((rg[Semantics.Entrance].count() >= 1) + (rg.count() == 0)) *
                (rooms[Semantics.Warehouse].count() >= 1)
        )

    constraints['node'] = node_constraint

    all_rooms = scene[Semantics.RoomContour]
    rooms = all_rooms[-Semantics.Exterior][-Semantics.Staircase]
    exterior = lambda r: r.same_level()[Semantics.Exterior]
    pholder = lambda r: r.same_level()[Semantics.Staircase]

    room_term = (
            rooms[-Semantics.Utility][-Semantics.Restroom].sum(lambda r: r.direct_access()).minimize(weight=5.) +
            (rooms[Semantics.Warehouse].sum(lambda r: (r.area() / 300).log().hinge(0, .4).pow(2)) +
             rooms[Semantics.Garage].sum(lambda r: (r.area() / 80).log().hinge(0, .4).pow(2)) +
             rooms[Semantics.FactoryOffice].sum(lambda r: (r.area() / 20).log().hinge(0, .4).pow(2)) +
             rooms[Semantics.Restroom].sum(lambda r: (r.area() / 10).log().hinge(0, .4).pow(2)) +
             rooms[Semantics.StaircaseRoom].sum(lambda r: (r.area() / 20).log().hinge(0, .4).pow(2)) +
             rooms[Semantics.Utility].sum(lambda r: (r.area() / 10).log().hinge(0, .4).pow(2))).minimize(weight=500.) +
            rooms.union({Semantics.FactoryOffice, Semantics.Garage, Semantics.Warehouse}).sum(
                lambda r: r.aspect_ratio()
            ).minimize(weight=50.) +
            rooms.union({Semantics.Utility, Semantics.Restroom}).sum(lambda r: r.aspect_ratio()).minimize(weight=40.) +
            rooms[-Semantics.Hallway].sum(lambda r: r.convexity().log()).minimize(weight=5.) +
            rooms[-Semantics.Hallway].sum(lambda r: (r.n_verts() - 6).clip(0).pow(1.5)).minimize(weight=1.) +
            rooms.union({Semantics.FactoryOffice, Semantics.Warehouse, Semantics.Garage}).sum(
                lambda r: r.shared_length(exterior(r)) / exterior(r).length()
            ).maximize(weight=10.) +
            rooms.union({Semantics.FactoryOffice, Semantics.Warehouse, Semantics.Garage}).sum(
                lambda r: (r.shared_n_verts(exterior(r)) - 2).clip(0)
            ).maximize(weight=1.) +
            (rooms.grid_line_count(constants, 'x') + rooms.grid_line_count(constants, 'y')).minimize(weight=2) +
            rooms.sum(lambda r: r.grid_line_count(constants, 'x') + r.grid_line_count(constants, 'y')).minimize(
                weight=2.
            ) +
            rooms.union({Semantics.Hallway, Semantics.StaircaseRoom}).area().minimize(weight=20.) +
            rooms.excludes({Semantics.Restroom, Semantics.Utility, Semantics.StaircaseRoom, Semantics.Hallway}).sum(
                lambda r: r.narrowness(constants, 2.5)
            ).minimize(weight=2000.) +
            rooms.union({Semantics.Restroom, Semantics.Utility, Semantics.StaircaseRoom, Semantics.Hallway}).sum(
                lambda r: r.narrowness(constants, 2)
            ).minimize(weight=2000.) +
            rooms[Semantics.StaircaseRoom].sum(lambda r: r.intersection(pholder(r)) / pholder(r).area()).maximize(
                weight=50.
            ) +
            rooms[Semantics.StaircaseRoom].sum(
                lambda r: (r.intersection(pholder(r)) / pholder(r).area()).hinge(constants.staircase_thresh, 1)
            ).minimize(weight=1e5) +
            rooms[Semantics.StaircaseRoom].sum(
                lambda r: r.area() / pholder(r).area() - r.intersection(pholder(r))
            ).minimize(weight=5.) +

            rooms.union({Semantics.Garage, Semantics.Warehouse}).sum(
                lambda r: rooms.union({Semantics.Garage, Semantics.Warehouse}).related_to(
                    r, cl.Traverse()
                ).shared_length(r).hinge(4, np.inf)
            ).minimize(weight=1e5)
    )

    score_terms['room'] = room_term

    return cl.Problem(
        constraints=constraints,
        score_terms=score_terms,
        constants=constants
    )


all_constraint_funcs = [
    warehouse_constraints
]
