# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""ProcFunc/Blender adapters for dependency-free house floor plans."""

from itertools import pairwise
from typing import NamedTuple

import numpy as np
import procfunc as pf
import shapely
from shapely.geometry.base import BaseGeometry

from infinigen2.scenes.house.floor_plan import (
    PolygonRings,
    WallSegment,
    house_room_polygon,
    wall_is_exterior,
    wall_length,
)
from infinigen2.util import mesh as mesh_util

__all__ = [
    "HouseCeilingCutoutResult",
    "HouseFloorMeshPlanesResult",
    "WallPlane",
    "house_ceiling_cutout",
    "house_floor_profile_to_planes",
    "WALL_STRIP_OVERHANG",
    "vertical_strip",
]


class WallPlane(NamedTuple):
    """One room-facing wall quad, with the layout data its setup needs."""

    obj: pf.MeshObject
    wall_index: int
    room_index: int
    thickness: float
    length: float
    start: tuple[float, float]
    end: tuple[float, float]


class HouseFloorMeshPlanesResult(NamedTuple):
    planes_interior: list[WallPlane]
    planes_exterior: list[WallPlane]
    floors: list[pf.MeshObject]
    ceilings: list[pf.MeshObject]
    dimensions: pf.Vector


class HouseCeilingCutoutResult(NamedTuple):
    ceiling: pf.MeshObject
    sill: pf.MeshObject
    back: pf.MeshObject


def _mesh_from_corners(
    corners: np.ndarray, sizes: np.ndarray, uvs: np.ndarray, name: str
) -> pf.MeshObject:
    mesh = mesh_util.mesh_from_corners(corners, sizes, uvs)
    mesh.item().name = name
    pf.ops.attr.write_attribute(mesh, 1.0, "crease_edge", domain="EDGE")
    return mesh


def _ring_segments(rings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords, ring = shapely.get_coordinates(rings, return_index=True)
    same_ring = ring[:-1] == ring[1:]
    return coords[:-1][same_ring], coords[1:][same_ring]


def _axis_lines(polygon: BaseGeometry, axis: int) -> np.ndarray:
    """Coordinates of the polygon's edges running along `1 - axis`, so cells follow them."""
    starts, ends = _ring_segments(shapely.get_rings(polygon))
    flat = starts[np.abs(starts[:, axis] - ends[:, axis]) < 1e-9, axis]
    bounds = np.asarray(polygon.bounds)[[axis, axis + 2]]
    lines = np.unique(np.concatenate([bounds, flat]))
    keep = (np.diff(lines, prepend=-np.inf) > 0.05) & (lines[-1] - lines > 0.05)
    keep[-1] = True
    return lines[keep]


def _polygon_faces(
    polygon: BaseGeometry, z: float, ceiling: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Flat corners and face sizes of the polygon's quad lattice, clipped at the rim."""
    xs = mesh_util.split_gaps(_axis_lines(polygon, 0), 0.8)
    ys = mesh_util.split_gaps(_axis_lines(polygon, 1), 0.8)
    x0, y0 = np.meshgrid(xs[:-1], ys[:-1], indexing="ij")
    x1, y1 = np.meshgrid(xs[1:], ys[1:], indexing="ij")
    cells = shapely.box(x0.ravel(), y0.ravel(), x1.ravel(), y1.ravel())
    inside = shapely.within(cells, polygon)
    rim = shapely.intersects(cells, polygon) & ~inside

    quad_x = np.stack([x0, x1, x1, x0], axis=-1).reshape(-1, 4)[inside]
    quad_y = np.stack([y0, y0, y1, y1], axis=-1).reshape(-1, 4)[inside]
    quad_xy = np.stack([quad_x, quad_y], axis=-1).reshape(-1, 2)

    # snap to the grid mesh_from_corners merges on, so no corner repeats after merging
    clipped = shapely.intersection(cells[rim], polygon, grid_size=1e-6)
    pieces = shapely.get_parts(clipped)
    is_area = (shapely.get_type_id(pieces) == 3) & (shapely.area(pieces) > 1e-6)
    pieces = shapely.normalize(pieces[is_area])
    if np.any(shapely.get_num_interior_rings(pieces) > 0):
        raise ValueError("Floor lattice cell clipped to a piece with a hole")
    exteriors = shapely.get_exterior_ring(pieces)
    coords, ring = shapely.get_coordinates(exteriors, return_index=True)
    not_closing = np.zeros(len(ring), dtype=bool)
    not_closing[:-1] = ring[:-1] == ring[1:]
    rim_sizes = np.bincount(ring[not_closing], minlength=len(pieces))
    # normalize() winds exteriors clockwise; flip to counter-clockwise (upward normal)
    rim_xy = coords[not_closing][mesh_util.reversed_faces(rim_sizes)]

    xy = np.concatenate([quad_xy, rim_xy])
    sizes = np.concatenate([np.full(len(quad_x), 4), rim_sizes])
    if ceiling:
        xy = xy[mesh_util.reversed_faces(sizes)]
    return np.column_stack([xy, np.full(len(xy), z)]), sizes


def _horizontal_surface(
    polygon: BaseGeometry, z: float, name: str, ceiling: bool
) -> pf.MeshObject:
    corners, sizes = _polygon_faces(polygon, z, ceiling)
    return _mesh_from_corners(corners, sizes, corners[:, :2], name)


def _side_quads(
    starts: np.ndarray, ends: np.ndarray, z: float, depth: float
) -> tuple[np.ndarray, np.ndarray]:
    """(S, 4, 3) vertical quads rising `depth` from z along each segment, with metric UVs."""
    xy = np.stack([ends, starts, starts, ends], axis=1)
    heights = np.broadcast_to([z, z, z + depth, z + depth], xy.shape[:2])
    corners = np.concatenate([xy, heights[..., None]], axis=-1)
    length = np.linalg.norm(ends - starts, axis=1)
    zeros = np.zeros_like(length)
    corner_u = np.stack([zeros, length, length, zeros], axis=1)
    corner_v = np.broadcast_to([0.0, 0.0, depth, depth], corner_u.shape)
    return corners, np.stack([corner_u, corner_v], axis=-1)


def house_ceiling_cutout(
    polygon: BaseGeometry,
    centers: np.ndarray,
    footprint: tuple[float, float],
    height: float,
    depth: float,
) -> HouseCeilingCutoutResult:
    """Pierce a room ceiling polygon with footprint boxes lying fully inside it."""
    z = height + 0.005
    half = np.asarray(footprint) / 2
    unit = np.asarray(((-1, -1), (1, -1), (1, 1), (-1, 1)))
    holes = np.asarray(centers)[:, None, :] + unit * half
    pierced = shapely.Polygon(polygon.exterior, [*polygon.interiors, *holes])
    sill, sill_uvs = _side_quads(
        holes.reshape(-1, 2), np.roll(holes, -1, axis=1).reshape(-1, 2), z, depth
    )
    rim, rim_uvs = _side_quads(*_ring_segments(shapely.get_rings(polygon)), z, depth)
    top, top_sizes = _polygon_faces(pierced, z + depth, False)
    back = np.concatenate([top, rim.reshape(-1, 3)])
    back_sizes = np.concatenate([top_sizes, np.full(len(rim), 4)])
    back_uvs = np.concatenate([top[:, :2], rim_uvs.reshape(-1, 2)])
    return HouseCeilingCutoutResult(
        ceiling=_horizontal_surface(pierced, z, "house_ceiling_cut", True),
        sill=_mesh_from_corners(
            sill.reshape(-1, 3),
            np.full(len(sill), 4),
            sill_uvs.reshape(-1, 2),
            "house_ceiling_sill",
        ),
        back=_mesh_from_corners(back, back_sizes, back_uvs, "house_ceiling_back"),
    )


def _cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _edge_wall_spans(
    walls: tuple[WallSegment, ...],
    room_index: int,
    start: tuple[float, float],
    end: tuple[float, float],
) -> list[tuple[int, tuple[float, float], tuple[float, float]]]:
    edge_start = np.asarray(start, dtype=float)
    edge_end = np.asarray(end, dtype=float)
    edge_length = float(np.linalg.norm(edge_end - edge_start))
    direction = (edge_end - edge_start) / edge_length
    candidates = []
    for wall_index, wall in enumerate(walls):
        if room_index not in wall.room_indices:
            continue
        wall_start = np.asarray(wall.start, dtype=float)
        wall_end = np.asarray(wall.end, dtype=float)
        wall_direction = (wall_end - wall_start) / wall_length(wall)
        if abs(_cross_2d(direction, wall_direction)) > 1e-6:
            continue
        distance = abs(_cross_2d(wall_start - edge_start, direction))
        if abs(distance - wall.thickness / 2) > 1e-5:
            continue
        projections = (
            float(np.dot(wall_start - edge_start, direction)),
            float(np.dot(wall_end - edge_start, direction)),
        )
        low, high = sorted(projections)
        if min(high, edge_length) - max(low, 0.0) <= 1e-6:
            continue
        candidates.append((low, high, wall_index))
    candidates.sort()
    if not candidates:
        raise ValueError(f"Finished room edge {start}->{end} has no source wall")
    cuts = [0.0]
    cuts.extend(
        (first[1] + second[0]) / 2
        for first, second in zip(candidates, candidates[1:], strict=False)
    )
    cuts.append(edge_length)
    return [
        (
            candidate[2],
            tuple(edge_start + direction * low),
            tuple(edge_start + direction * high),
        )
        for candidate, low, high in zip(candidates, cuts, cuts[1:], strict=False)
    ]


def _polygon_edges(
    polygon: BaseGeometry,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    rings = [polygon.exterior, *polygon.interiors]
    return [edge for ring in rings for edge in pairwise(ring.coords)]


WALL_STRIP_OVERHANG = 0.03


def vertical_strip(
    start: tuple[float, float], end: tuple[float, float], height: float, name: str
) -> pf.MeshObject:
    """Metric-UV wall grid from `start` to `end`, facing left of that direction."""
    # Square-room convention: first edge is vertical (V/up), second is U/along-wall.
    length = float(np.linalg.norm(np.asarray(end) - np.asarray(start)))
    along = mesh_util.split_gaps(np.array([0.0, length]), 0.8)
    up = mesh_util.split_gaps(
        np.array([-WALL_STRIP_OVERHANG, height + WALL_STRIP_OVERHANG]), 0.8
    )
    direction = (np.asarray(end) - np.asarray(start)) / length
    u0, v0 = np.meshgrid(along[:-1], up[:-1], indexing="ij")
    u1, v1 = np.meshgrid(along[1:], up[1:], indexing="ij")
    corner_u = np.stack([u0, u0, u1, u1], axis=-1).reshape(-1, 4)
    corner_v = np.stack([v0, v1, v1, v0], axis=-1).reshape(-1, 4)
    quad_uv = np.stack([corner_u, corner_v], axis=-1)
    quad_xy = np.asarray(start) + corner_u[..., None] * direction
    quad_xyz = np.concatenate([quad_xy, corner_v[..., None]], axis=-1)
    mesh = mesh_util.mesh_from_quads(quad_xyz, quad_uv)
    mesh.item().name = name
    pf.ops.attr.write_attribute(mesh, 1.0, "crease_edge", domain="EDGE")
    return mesh


def _wall_plane(
    wall_index: int,
    room_index: int,
    start: tuple[float, float],
    end: tuple[float, float],
    height: float,
    thickness: float,
) -> WallPlane:
    length = float(np.linalg.norm(np.asarray(end) - np.asarray(start)))
    name = f"house_wall.{wall_index:02d}.{room_index:02d}"
    return WallPlane(
        obj=vertical_strip(start, end, height, name),
        wall_index=wall_index,
        room_index=room_index,
        thickness=thickness,
        length=length,
        start=start,
        end=end,
    )


def house_floor_profile_to_planes(
    room_boundaries: tuple[PolygonRings, ...],
    walls: tuple[WallSegment, ...],
    height: float = 2.8,
) -> HouseFloorMeshPlanesResult:
    """Build one room-facing quad per wall side, plus per-room floors and ceilings."""
    planes = []
    floors = []
    ceilings = []
    for room_index, rings in enumerate(room_boundaries):
        polygon = house_room_polygon(rings)
        for start, end in _polygon_edges(polygon):
            spans = _edge_wall_spans(walls, room_index, start, end)
            planes += [
                _wall_plane(
                    wall_index,
                    room_index,
                    low,
                    high,
                    height,
                    walls[wall_index].thickness,
                )
                for wall_index, low, high in spans
            ]
        floor_name = f"room_floor.{room_index:02d}"
        # extend to the wall centerline so rooms meet without a gap under doorways
        surface = polygon.buffer(walls[0].thickness / 2, join_style="mitre")
        floors.append(_horizontal_surface(surface, -0.005, floor_name, False))
        ceiling_name = f"room_ceiling.{room_index:02d}"
        ceilings.append(
            _horizontal_surface(surface, height + 0.005, ceiling_name, True)
        )
    corners = np.asarray([(*wall.start, *wall.end) for wall in walls])
    extent = corners.reshape(-1, 2).max(axis=0)
    exterior = [p for p in planes if wall_is_exterior(walls[p.wall_index])]
    interior = [p for p in planes if not wall_is_exterior(walls[p.wall_index])]
    return HouseFloorMeshPlanesResult(
        planes_interior=interior,
        planes_exterior=exterior,
        floors=floors,
        ceilings=ceilings,
        dimensions=pf.Vector((*extent, height)),
    )
