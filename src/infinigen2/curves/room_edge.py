# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

from typing import NamedTuple

import numpy as np
import procfunc as pf

__all__ = [
    "RoomEdgeBoxResult",
    "RoomEdgeCurveResult",
    "room_box_wall_lengths",
    "room_edge_box",
    "room_edge_curve_rand",
]

CORNER_SEGMENTS = 8


class RoomEdgeBoxResult(NamedTuple):
    outline: pf.MeshObject


class RoomEdgeCurveResult(NamedTuple):
    curve: pf.ProcNode
    is_flat_point: pf.ProcNode[bool]
    orig_point_idx: pf.ProcNode[int]
    wall_lengths: list[float]


def _box_corners(width: float, depth: float) -> np.ndarray:
    return np.array([(0.0, 0.0), (width, 0.0), (width, depth), (0.0, depth)])


def room_box_wall_lengths(
    width: float, depth: float, corner_radii: tuple[float, float, float, float]
) -> list[float]:
    sides = np.array([width, depth, width, depth])
    radii = np.asarray(corner_radii)
    return [float(x) for x in sides - radii - np.roll(radii, -1)]


def _corner_points(
    corner: np.ndarray, incoming: np.ndarray, radius: float, is_chamfer: bool
) -> np.ndarray:
    left = np.array([-incoming[1], incoming[0]])
    t = np.linspace(0.0, 1.0, CORNER_SEGMENTS + 1)[:, None]
    start = corner - incoming * radius
    end = corner + left * radius
    if is_chamfer:
        return start + (end - start) * t
    theta = t * np.pi / 2
    center = start + left * radius
    return center + radius * (incoming * np.sin(theta) - left * np.cos(theta))


@pf.tracer.generator
def room_edge_box(
    width: float,
    depth: float,
    corner_radii: tuple[float, float, float, float],
    corner_is_chamfer: tuple[bool, bool, bool, bool],
) -> RoomEdgeBoxResult:
    corners = _box_corners(width, depth)
    incoming = corners - np.roll(corners, 1, axis=0)
    incoming /= np.linalg.norm(incoming, axis=1, keepdims=True)
    arcs = [
        _corner_points(corners[k], incoming[k], corner_radii[k], corner_is_chamfer[k])
        for k in range(4)
    ]
    points = np.concatenate([arcs[0][-1:], *arcs[1:], arcs[0]])

    interior = np.zeros(CORNER_SEGMENTS + 1, dtype=bool)
    interior[1:-1] = True
    is_flat = ~np.concatenate([[False], interior, interior, interior, interior])
    arc_wall = np.full(CORNER_SEGMENTS + 1, 0, dtype=np.int32)
    arc_wall[0] = -1
    wall_idx = np.concatenate(
        [[0], arc_wall + 1, arc_wall + 2, arc_wall + 3, (arc_wall + 4) % 4]
    ).astype(np.int32)

    line = pf.ops.primitives.mesh_line(np.column_stack([points, np.zeros(len(points))]))
    pf.ops.attr.write_attribute(line, is_flat, "is_flat_point", domain="POINT")
    pf.ops.attr.write_attribute(line, wall_idx, "orig_point_idx", domain="POINT")
    return RoomEdgeBoxResult(outline=line)


def room_edge_curve_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    corner_radii: tuple[float, float, float, float] | None = None,
    corner_is_chamfer: tuple[bool, bool, bool, bool] | None = None,
) -> RoomEdgeCurveResult:
    if dimensions is None:
        dimensions = pf.Vector(pf.random.uniform(rng, (3.0, 3.0, 2.0), (5, 5, 3)))
    width = float(dimensions.x)
    depth = float(dimensions.y)

    rounded_probability = max(0.0, pf.random.uniform(rng, -0.1, 0.5))
    chamfer_probability = pf.random.uniform(rng, 0.0, 0.5) ** 2
    shared_radius = pf.random.uniform(rng, 0.15, 1.5)
    if corner_radii is None:
        max_radius = 0.3 * min(width, depth)
        rounded_draw = pf.random.uniform(rng, (0.0,) * 4, (1.0,) * 4)
        radius = shared_radius * pf.random.uniform(rng, (0.6,) * 4, (1.4,) * 4)
        radii = np.where(
            rounded_draw < rounded_probability, np.minimum(radius, max_radius), 0.0
        )
        corner_radii = tuple(radii.tolist())
    if corner_is_chamfer is None:
        chamfer_draw = pf.random.uniform(rng, (0.0,) * 4, (1.0,) * 4)
        corner_is_chamfer = tuple((chamfer_draw < chamfer_probability).tolist())

    box = room_edge_box(width, depth, corner_radii, corner_is_chamfer)
    return RoomEdgeCurveResult(
        curve=pf.nodes.geo.mesh_to_curve(
            pf.nodes.geo.object_info(box.outline).geometry
        ),
        is_flat_point=pf.nodes.geo.input_named_attribute(
            "is_flat_point", data_type=pf.NodeDataType.BOOLEAN
        ).attribute,
        orig_point_idx=pf.nodes.geo.input_named_attribute(
            "orig_point_idx", data_type=pf.NodeDataType.INT
        ).attribute,
        wall_lengths=room_box_wall_lengths(width, depth, corner_radii),
    )
