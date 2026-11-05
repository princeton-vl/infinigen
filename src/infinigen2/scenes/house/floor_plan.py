# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lingjie Mei: original Infinigen v1 room arrangement solver
# - Alexander Raistrick: functional Infinigen2 port

"""Pure house floor-plan generation over exact planar room polygons.

Ported from the Infinigen v1 room solver (``SegmentMaker``, ``FloorPlanSolver`` and
``BlueprintSolidifier`` in ``infinigen/core/constraints/example_solver/room/``).
"""

from __future__ import annotations

from collections import defaultdict
from itertools import pairwise
from math import hypot, sqrt
from typing import NamedTuple

import procfunc as pf
import shapely
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely.ops import split, unary_union

__all__ = [
    "HouseFloorProfileResult",
    "PolygonRings",
    "WallSegment",
    "house_room_polygon",
    "room_connect_rand",
    "room_solve",
    "wall_is_exterior",
    "wall_length",
]

Point2D = tuple[float, float]
PolygonRings = tuple[tuple[Point2D, ...], ...]


class WallSegment(NamedTuple):
    start: Point2D
    end: Point2D
    room_indices: tuple[int, ...]
    thickness: float
    door_position_frac: float | None = None
    is_flat: bool = True


class HouseFloorProfileResult(NamedTuple):
    room_boundaries: tuple[PolygonRings, ...]
    walls: tuple[WallSegment, ...]


def wall_is_exterior(wall: WallSegment) -> bool:
    return len(wall.room_indices) == 1


def wall_length(wall: WallSegment) -> float:
    return hypot(wall.end[0] - wall.start[0], wall.end[1] - wall.start[1])


def house_room_polygon(boundary_rings: PolygonRings) -> BaseGeometry:
    """Return the closed room polygon used to derive the house walls."""
    exterior, *interiors = boundary_rings
    return shapely.Polygon(exterior, interiors)


def _split_cutter_rand(
    rng: pf.RNG, bounds: tuple[float, float, float, float]
) -> BaseGeometry | None:
    sizes = (bounds[2] - bounds[0], bounds[3] - bounds[1])
    axis = 0 if sizes[0] >= sizes[1] else 1
    ratio = pf.random.uniform(rng, 0.25, 0.75)
    offset = round(ratio * sizes[axis] / 0.5) * 0.5
    bound = max(0.15 * sizes[1 - axis], 1.4)
    if offset < bound or sizes[axis] - offset < bound:
        return None
    start = [bounds[0] - 1.0, bounds[1] - 1.0]
    end = [bounds[2] + 1.0, bounds[3] + 1.0]
    start[axis] = end[axis] = bounds[axis] + offset
    return shapely.LineString((start, end))


def _split_segment_rand(
    rng: pf.RNG, segments: dict[int, BaseGeometry], wall_thickness: float
) -> bool:
    rng_segment, rng_cutter = rng.spawn(2)
    keys = list(segments)
    probabilities = [sqrt(segments[key].area) for key in keys]
    key = pf.control.choice(rng_segment, list(zip(keys, probabilities, strict=True)))
    cutter = _split_cutter_rand(rng_cutter, segments[key].bounds)
    if cutter is None:
        return False
    outline = segments[key].exterior
    corners = shapely.MultiPoint(outline.coords)
    crossings = shapely.get_parts(cutter.intersection(outline))
    # a cut close to an outline corner leaves a wall stub the room inset swallows
    gaps = [corners.distance(crossing) for crossing in crossings]
    if any(1e-6 < gap < 1.5 * wall_thickness for gap in gaps):
        return False
    pieces = tuple(split(segments[key], cutter).geoms)
    if len(pieces) != 2 or any(piece.geom_type != "Polygon" for piece in pieces):
        return False
    segments[key], segments[max(segments) + 1] = pieces
    return True


def _divide_polygons(
    rng: pf.RNG, contour: BaseGeometry, n_boxes: int, wall_thickness: float
) -> dict[int, BaseGeometry]:
    segments = {0: contour}
    box_rngs = rng.spawn(n_boxes)
    for box_rng in box_rngs:
        for attempt_rng in box_rng.spawn(100):
            if _split_segment_rand(attempt_rng, segments, wall_thickness):
                break
    return segments


def _polygon_neighbours(
    segments: dict[int, BaseGeometry],
) -> tuple[dict[int, dict[int, float]], dict[int, set[int]]]:
    shared_edges: dict[int, dict[int, float]] = {key: {} for key in segments}
    attached: dict[int, set[int]] = {key: set() for key in segments}
    for key, first in segments.items():
        for other_key, second in segments.items():
            if key >= other_key:
                continue
            length = _longest_straight_length(first.boundary & second.boundary)
            shared_edges[key][other_key] = shared_edges[other_key][key] = length
            if length >= 1.4:
                attached[key].add(other_key)
                attached[other_key].add(key)
    return shared_edges, attached


def _merge_polygons(
    rng: pf.RNG,
    segments: dict[int, BaseGeometry],
    room_count: int,
) -> list[BaseGeometry] | None:
    if len(segments) < room_count:
        return None
    shared_edges, attached = _polygon_neighbours(segments)
    merge_rngs = rng.spawn(len(segments) - room_count)
    for merge_rng in merge_rngs:
        rng_segment, rng_neighbour = merge_rng.spawn(2)
        keys = list(shared_edges)
        weights = [1 / (len(attached[key]) + 1) for key in keys]
        key = pf.control.choice(rng_segment, list(zip(keys, weights, strict=True)))
        candidates = [k for k, length in shared_edges[key].items() if length > 1e-6]
        if not candidates:
            return None
        weights = [len(attached[k] - attached[key]) ** 2 + 0.5 for k in candidates]
        neighbour = pf.control.choice(
            rng_neighbour, list(zip(candidates, weights, strict=True))
        )
        merged = unary_union((segments[key], segments[neighbour])).buffer(0)
        if merged.geom_type != "Polygon":
            return None
        segments = {
            k: value for k, value in segments.items() if k not in {key, neighbour}
        } | {neighbour: merged}
        shared_edges, attached = _polygon_neighbours(segments)
    return list(segments.values())


def _straight_segments(
    geometry: BaseGeometry, adjacent: tuple[int, ...]
) -> list[tuple[Point2D, Point2D, tuple[int, ...]]]:
    merged = shapely.simplify(shapely.line_merge(geometry), 1e-9)
    return [
        (start, end, adjacent)
        for line in shapely.get_parts(merged)
        for start, end in pairwise(line.coords)
    ]


def _longest_straight_length(geometry) -> float:
    return max(
        (
            hypot(end[0] - start[0], end[1] - start[1])
            for start, end, _adjacent in _straight_segments(geometry, ())
        ),
        default=0.0,
    )


def _on_fillet(
    start: Point2D, end: Point2D, fillet_segments: tuple[tuple[Point2D, Point2D], ...]
) -> bool:
    segment = shapely.LineString((start, end))
    fillets = [shapely.LineString(fillet).buffer(1e-6) for fillet in fillet_segments]
    return any(fillet.covers(segment) for fillet in fillets)


def _walls(
    room_polygons: tuple[BaseGeometry, ...],
    wall_thickness: float,
    fillet_segments: tuple[tuple[Point2D, Point2D], ...],
) -> tuple[WallSegment, ...]:
    walls = []
    shared_by_room: dict[int, list[BaseGeometry]] = defaultdict(list)
    for first in range(len(room_polygons)):
        for second in range(first + 1, len(room_polygons)):
            shared = room_polygons[first].boundary.intersection(
                room_polygons[second].boundary
            )
            walls.extend(_straight_segments(shared, (first, second)))
            if not shared.is_empty:
                shared_by_room[first].append(shared)
                shared_by_room[second].append(shared)
    for room_index, polygon in enumerate(room_polygons):
        shared = unary_union(shared_by_room[room_index])
        exterior = polygon.boundary.difference(shared)
        walls.extend(_straight_segments(exterior, (room_index,)))
    walls = sorted(
        (
            (min(start, end), max(start, end), adjacent)
            for start, end, adjacent in walls
            if hypot(end[0] - start[0], end[1] - start[1]) > 1e-9
        ),
        key=lambda wall: (wall[2], wall[0], wall[1]),
    )
    return tuple(
        WallSegment(
            start,
            end,
            adjacent,
            wall_thickness,
            is_flat=not _on_fillet(start, end, fillet_segments),
        )
        for start, end, adjacent in walls
    )


def room_connect_rand(
    rng: pf.RNG, walls: tuple[WallSegment, ...], room_count: int
) -> tuple[WallSegment, ...]:
    """Choose door walls and positions forming a connected room graph."""
    candidates = [
        index
        for index, wall in enumerate(walls)
        if not wall_is_exterior(wall) and wall_length(wall) >= 1.4
    ]
    parent = list(range(room_count))

    def find(room: int) -> int:
        while parent[room] != room:
            room = parent[room]
        return room

    priorities = pf.random.uniform(
        rng, (0.0,) * len(candidates), (1.0,) * len(candidates)
    )
    selected = []
    for _priority, index in sorted(zip(priorities, candidates, strict=True)):
        first, second = (find(room) for room in walls[index].room_indices)
        if first != second:
            parent[second] = first
            selected.append(index)
    if len(selected) != len(parent) - 1:
        raise ValueError("Room layout has no door-width connected spanning graph")

    door_fracs = pf.random.uniform(rng, (0.2,) * len(selected), (0.8,) * len(selected))
    door_positions = dict(zip(sorted(selected), door_fracs, strict=True))
    return tuple(
        WallSegment(
            start=wall.start,
            end=wall.end,
            room_indices=wall.room_indices,
            thickness=wall.thickness,
            door_position_frac=door_positions.get(index),
            is_flat=wall.is_flat,
        )
        for index, wall in enumerate(walls)
    )


def room_solve(
    rng: pf.RNG,
    boundary_rings: PolygonRings,
    room_count: int,
    *,
    wall_thickness: float | None = None,
    fillet_segments: tuple[tuple[Point2D, Point2D], ...] = (),
) -> HouseFloorProfileResult:
    """Partition an exact outline polygon into rooms and derive their walls."""
    exterior, *interiors = boundary_rings
    contour = shapely.Polygon(exterior, interiors)
    rng_split_count, rng_wall_thickness, rng_attempts = rng.spawn(3)
    split_count = int(room_count * pf.random.uniform(rng_split_count, 1.8, 2.0))
    if wall_thickness is None:
        wall_thickness = (
            0.06 + 0.24 * pf.random.uniform(rng_wall_thickness, 0.0, 1.0) ** 3
        )
    for attempt_rng in rng_attempts.spawn(100):
        rng_divide, rng_merge = attempt_rng.spawn(2)
        divided = _divide_polygons(rng_divide, contour, split_count, wall_thickness)
        rooms = _merge_polygons(rng_merge, divided, room_count)
        if rooms is not None:
            break
    else:
        raise ValueError("Could not divide the outline into rooms")
    walls = _walls(tuple(rooms), wall_thickness, fillet_segments)
    boundaries = []
    for room in rooms:
        inset = room.buffer(-wall_thickness / 2, join_style="mitre")
        if inset.geom_type != "Polygon":
            raise ValueError("Wall inset produced an invalid finished room polygon")
        finished = orient(shapely.simplify(shapely.normalize(inset), 1e-9), 1.0)
        rings = (finished.exterior, *finished.interiors)
        boundaries.append(tuple(tuple(ring.coords[:-1]) for ring in rings))
    return HouseFloorProfileResult(tuple(boundaries), walls)
