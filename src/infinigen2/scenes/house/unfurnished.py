# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

"""Multi-room house layout, surfaces, and walls."""

from __future__ import annotations

import logging
from collections import defaultdict
from itertools import combinations
from math import dist
from typing import NamedTuple, TypeVar

import numpy as np
import procfunc as pf
import shapely
from mathutils import Matrix
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient

from infinigen2.cameras import framing
from infinigen2.lighting import sky_lighting
from infinigen2.objects import window
from infinigen2.objects.door import door_composite_rand
from infinigen2.scenes.house.floor_mesh import (
    WALL_STRIP_OVERHANG,
    HouseCeilingCutoutResult,
    WallPlane,
    house_ceiling_cutout,
    house_floor_profile_to_planes,
    vertical_strip,
)
from infinigen2.scenes.house.floor_plan import (
    PolygonRings,
    WallSegment,
    house_room_polygon,
    room_connect_rand,
    room_solve,
)
from infinigen2.scenes.house.outline import house_outline_rand
from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.placement.distribute import propagate_modifiers_to_instances
from infinigen2.scenes.placement.retry import repeat_attempts
from infinigen2.scenes.room.ceiling_features import (
    CeilingFeaturesResult,
    CeilingGridResult,
    ceiling_lamp_grid_rand,
    ceiling_lamp_lights,
    ceiling_light_bar_grid_rand,
    ceiling_light_bar_lights_rand,
    ceiling_skylight_grid_rand,
)
from infinigen2.scenes.room.room import wall_arrangement_rand
from infinigen2.scenes.room.wall_base import (
    ROOM_SUBSURF_LEVELS,
    WallResult,
    WallSurface,
    extrude_for_thickness,
    name_objects,
    shift_wall_ends,
    wall_plain_rand,
)
from infinigen2.scenes.room.wall_cutouts import (
    CutoutResult,
    arrange_window_portals,
    cutout_spaced_instances,
    wall_painting_grid_rand,
    wall_windows_rand,
)
from infinigen2.scenes.room.wall_mounts import (
    wall_board_shelf_rand,
    wall_storage_flush_rand,
)
from infinigen2.shaders.functionality_lists import (
    ceiling_material_rand,
    floor_material_rand,
    wall_material_rand,
)
from infinigen2.util import mesh as mesh_util
from infinigen2.util.scene_cleanup import delete_object, delete_objects
from infinigen2.uv_surface import grid_placement

__all__ = [
    "HouseResult",
    "HouseRoomResult",
    "HouseWallResult",
    "HouseWallSideResult",
    "house_unfurnished_rand",
    "house_walls_rand",
    "interior_wall_arrangement_rand",
]

logger = logging.getLogger(__name__)

ObjectT = TypeVar("ObjectT", bound=pf.Object)


class HouseWallSideResult(NamedTuple):
    plane: WallPlane
    result: WallResult


class HouseWallResult(NamedTuple):
    sides: tuple[HouseWallSideResult, ...]
    planes_interior: list[pf.MeshObject]
    planes_exterior: list[pf.MeshObject]
    corners: list[pf.MeshObject]
    corners_by_room: dict[int, list[pf.MeshObject]]
    backs: list[pf.MeshObject]
    sills: list[pf.MeshObject]
    storage_containers: list[pf.MeshObject]
    supports: list[pf.MeshObject]
    lights: list[pf.LightObject]
    windows: list[pf.MeshObject]
    decorations: dict[str, list[pf.MeshObject]]
    doors: dict[int, pf.MeshObject]
    doorway_centers: dict[int, tuple[float, float]]
    colliders: ccol.CollisionSet


class HouseDoorWallsResult(NamedTuple):
    sides: list[HouseWallSideResult]
    doors: dict[int, pf.MeshObject]
    door_dimensions: dict[int, pf.Vector]
    doorway_centers: dict[int, tuple[float, float]]
    colliders: ccol.CollisionSet


class HouseRoomResult(NamedTuple):
    room_index: int
    boundary_rings: PolygonRings
    all_objects: list[pf.MeshObject]
    lights: list[pf.LightObject]
    floor: pf.MeshObject
    ceiling: pf.MeshObject
    walls: list[pf.MeshObject]
    flat_walls: list[pf.MeshObject]
    neighbors: dict[int, int]
    doors: dict[int, pf.MeshObject]
    doorway_centers: dict[int, tuple[float, float]]


class HouseResult(NamedTuple):
    all_objects: list[pf.MeshObject]
    small_objects: list[pf.MeshObject]
    cameras: list[pf.CameraObject]
    lights: list[pf.LightObject]
    rooms: tuple[HouseRoomResult, ...]
    dimensions: pf.Vector
    colliders: ccol.CollisionSet
    environment: pf.World | None


def _plane_direction(plane: WallPlane) -> np.ndarray:
    return (np.asarray(plane.end) - np.asarray(plane.start)) / plane.length


def _plane_inward(plane: WallPlane) -> np.ndarray:
    direction = _plane_direction(plane)
    return np.asarray((-direction[1], direction[0]))


def _plane_surface(plane: WallPlane, height: float) -> WallSurface:
    return WallSurface(plane.obj, plane.length, height + 2 * WALL_STRIP_OVERHANG)


@pf.tracer.grammar
def _exterior_wall_rand(
    rng: pf.RNG,
    plane: WallPlane,
    wall: WallSurface,
    material: pf.Material,
    shared_window: window.WindowResult,
) -> WallResult:
    """Outer walls favour windows over the mixed arrangement."""

    def windows(rng, wall, wall_material):
        return wall_windows_rand(
            rng,
            wall,
            wall_material,
            window_obj=shared_window.mesh,
            window_portal=shared_window.light,
            top_profile_height=shared_window.profile.top_profile_height,
            bottom_profile_height=shared_window.profile.bottom_profile_height,
            wall_thickness=plane.thickness,
            reveal_depth=plane.thickness,
        )

    def mixed(rng, wall, wall_material):
        return wall_arrangement_rand(
            rng,
            wall=wall,
            wall_material=wall_material,
            window_obj=shared_window.mesh,
            window_portal=shared_window.light,
            top_profile_height=shared_window.profile.top_profile_height,
            bottom_profile_height=shared_window.profile.bottom_profile_height,
            wall_thickness=plane.thickness,
            window_reveal_depth=plane.thickness,
        )

    rng_choice, rng_feature = rng.spawn(2)
    option = pf.control.choice(rng_choice, [(windows, 3.0), (mixed, 1.0)])
    return option(rng_feature, wall, material)


def _unique_objects(objects: list[ObjectT]) -> list[ObjectT]:
    return list({obj.item().as_pointer(): obj for obj in objects}.values())


@pf.tracer.grammar
def interior_wall_arrangement_rand(
    rng: pf.RNG,
    wall: WallSurface,
    wall_material: pf.Material | None = None,
    wall_thickness: float = 0.05,
) -> WallResult:
    """Equal-weight treatment for a partition side; doors are placed separately."""
    rng_choice, rng_feature = rng.spawn(2)
    option = pf.control.choice(
        rng_choice,
        [
            (wall_plain_rand, 1.0),
            (wall_painting_grid_rand, 1.0),
            (wall_board_shelf_rand, 1.0),
            (wall_storage_flush_rand, 1.0),
        ],
    )
    return option(
        rng_feature,
        wall=wall,
        wall_material=wall_material,
        wall_thickness=wall_thickness,
    )


def _door_dimensions_rand(
    rng: pf.RNG, wall_height: float, wall_width: float
) -> pf.Vector:
    width = min(pf.random.uniform(rng, 0.85, 1.2), wall_width - 0.31)
    height = min(pf.random.uniform(rng, 2.0, 2.2), wall_height * 0.9)
    thickness = pf.random.clip_gaussian(rng, 0.0318, 0.0127, 0.0254, 0.0762)
    return pf.Vector((thickness, width, height))


def _door_asset_rand(rng: pf.RNG, dimensions: pf.Vector) -> pf.MeshObject:
    obj = door_composite_rand(rng, dimensions=dimensions).mesh
    pf.ops.object.set_transform(obj, location=(0.0, -dimensions.y / 2, 0.0))
    pf.ops.mesh.transform_apply(obj)
    return obj


def _door_center(plane: WallPlane, position: np.ndarray, door_width: float) -> float:
    half = door_width / 2
    along = float(np.dot(position - np.asarray(plane.start), _plane_direction(plane)))
    return float(np.clip(along, half + 0.1, plane.length - half - 0.1))


def _door_cutout(
    plane: WallPlane,
    center: float,
    material: pf.Material,
    door: pf.MeshObject,
    door_width: float,
) -> CutoutResult:
    half = door_width / 2
    cutout = cutout_spaced_instances(
        surface=plane.obj,
        instance=door,
        surface_material=material,
        spacing=pf.Vector((0.0, 0.0, 0.0)),
        margin_low=pf.Vector((center - half, 0.01, 0.0)),
        margin_high=pf.Vector((plane.length - center - half, 0.0, 0.0)),
        x_instances_max=1,
        wall_thickness=plane.thickness / 2,
        recess_pct=1.0,
    )
    delete_objects([cutout.trim_edges.item(), cutout.lightblocker.item()])
    return cutout


def _door_wall_result(cutout: CutoutResult, doors: list[pf.MeshObject]) -> WallResult:
    return WallResult(
        all_objects=[cutout.geom, cutout.sill, *doors],
        wall_planes=[cutout.geom],
        backs=[],
        sills=[cutout.sill],
        storage_containers=[],
        supports=[],
        lights=[],
        decorations={"door": doors},
    )


def _door_affordance_collider(
    plane: WallPlane,
    center_xy: tuple[float, float],
    width: float,
    height: float,
    wall_index: int,
) -> pf.MeshObject:
    direction = _plane_direction(plane)
    angle = float(np.arctan2(direction[1], direction[0]) - np.pi / 2)
    collider = pf.ops.primitives.mesh_cube(size=1.0)
    pf.ops.mesh.transform(collider, scale=(1.6, width, height))
    pf.ops.object.set_transform(
        collider,
        location=(*center_xy, height / 2),
        rotation_euler=(0.0, 0.0, angle),
    )
    collider.item().name = f"house_door_affordance.{wall_index:02d}"
    collider.item().hide_render = True
    collider.item().display_type = "WIRE"
    return collider


def _keep_door_leaves(leaves: list[pf.MeshObject]) -> list[pf.MeshObject]:
    return leaves


def _drop_door_leaves(leaves: list[pf.MeshObject]) -> list[pf.MeshObject]:
    delete_objects([obj.item() for obj in leaves])
    return []


def _door_walls(
    rng: pf.RNG,
    walls: tuple[WallSegment, ...],
    planes_interior: list[WallPlane],
    height: float,
    materials: list[pf.Material],
) -> HouseDoorWallsResult:
    """Cut both sides of each door partition with one shared door asset."""
    door_walls = [
        i for i, wall in enumerate(walls) if wall.door_position_frac is not None
    ]
    results = []
    doors = {}
    door_dimensions = {}
    doorway_centers = {}
    door_clearances = []
    for wall_index, rng_wall in zip(
        door_walls, rng.spawn(len(door_walls)), strict=True
    ):
        wall = walls[wall_index]
        start = np.asarray(wall.start)
        position = start + (np.asarray(wall.end) - start) * wall.door_position_frac
        sides = [p for p in planes_interior if p.wall_index == wall_index]
        rng_params, rng_door, rng_present = rng_wall.spawn(3)
        dimensions = _door_dimensions_rand(
            rng_params, height, min(side.length for side in sides)
        )
        door = _door_asset_rand(rng_door, dimensions)
        door_width = dimensions.y
        centers = [_door_center(side, position, door_width) for side in sides]
        cutouts = [
            _door_cutout(side, center, materials[side.room_index], door, door_width)
            for side, center in zip(sides, centers, strict=True)
        ]
        delete_objects([obj.item() for obj in [door, *cutouts[1].aliases]])
        leaves_fn = pf.control.choice(
            rng_present, [(_drop_door_leaves, 1.0), (_keep_door_leaves, 1.0)]
        )
        leaves = leaves_fn(cutouts[0].aliases)
        results += [
            HouseWallSideResult(sides[0], _door_wall_result(cutouts[0], leaves)),
            HouseWallSideResult(sides[1], _door_wall_result(cutouts[1], [])),
        ]
        if leaves:
            doors[wall_index] = leaves[0]
            door_dimensions[wall_index] = dimensions
        doorway_center = (
            np.asarray(sides[0].start)
            + _plane_direction(sides[0]) * centers[0]
            - _plane_inward(sides[0]) * sides[0].thickness / 2
        )
        doorway_centers[wall_index] = (
            float(doorway_center[0]),
            float(doorway_center[1]),
        )
        door_clearances.append(
            _door_affordance_collider(
                sides[0], doorway_centers[wall_index], door_width, height, wall_index
            )
        )
    return HouseDoorWallsResult(
        sides=results,
        doors=doors,
        door_dimensions=door_dimensions,
        doorway_centers=doorway_centers,
        colliders=ccol.collision_set([*doors.values(), *door_clearances]),
    )


def _open_door_rand(
    rng: pf.RNG,
    door: pf.MeshObject,
    dimensions: pf.Vector,
    wall_thickness: float,
    obstacles: ccol.CollisionSet,
    target_angle_deg: float | None,
    attempts: int = 8,
) -> float | None:
    closed = door.item().matrix_world.copy()
    attempt_index = 0

    def attempt(attempt_rng: pf.RNG) -> float | None:
        nonlocal attempt_index
        if target_angle_deg is None:
            magnitude = pf.random.clip_gaussian(attempt_rng, 135.0, 40.0, 45.0, 180.0)
        else:
            magnitude = abs(target_angle_deg)
        direction = 1 if attempt_index % 2 == 0 else -1
        angle_deg = magnitude * direction
        attempt_index += 1
        # hinge on the wall face the leaf swings towards so it clears the jamb
        half_wall = wall_thickness / 2
        hinge_x = -half_wall if direction > 0 else max(half_wall, dimensions.x)
        hinge = Matrix.Translation((hinge_x, -dimensions.y / 2, 0.0))
        rotation = Matrix.Rotation(np.deg2rad(angle_deg), 4, "Z")
        door.item().matrix_world = closed @ hinge @ rotation @ hinge.inverted()
        if ccol.intersection_test(obstacles, door):
            door.item().matrix_world = closed
            return None
        return float(angle_deg)

    if target_angle_deg == 0:
        return 0.0
    angle_deg = repeat_attempts(attempt, rng, attempts)
    if angle_deg is None:
        door.item().matrix_world = closed
        return None
    return angle_deg


def _door_obstacles(result: WallResult) -> list[pf.MeshObject]:
    return [
        *result.wall_planes,
        *result.storage_containers,
        *result.supports,
        *result.decorations.get("window", []),
    ]


def _open_doors_rand(
    rng: pf.RNG,
    sides: list[HouseWallSideResult],
    doors: dict[int, pf.MeshObject],
    door_dimensions: dict[int, pf.Vector],
    corners: list[pf.MeshObject],
    target_angle_deg: float | None,
) -> None:
    wall_indices = sorted(doors)
    door_rngs = rng.spawn(len(wall_indices))
    for wall_index, door_rng in zip(wall_indices, door_rngs, strict=True):
        results = [s.result for s in sides if s.plane.wall_index != wall_index]
        objects = [obj for r in results for obj in _door_obstacles(r)]
        own_wall = [
            obj
            for s in sides
            if s.plane.wall_index == wall_index
            for obj in (*s.result.sills, *s.result.wall_planes)
        ]
        objects += own_wall
        other_doors = [doors[index] for index in doors if index != wall_index]
        obstacles = ccol.collision_set(_unique_objects(objects + corners + other_doors))
        door = doors[wall_index]
        thickness = next(
            s.plane.thickness for s in sides if s.plane.wall_index == wall_index
        )
        angle_deg = _open_door_rand(
            door_rng,
            door,
            door_dimensions[wall_index],
            thickness,
            obstacles,
            target_angle_deg,
        )
        if angle_deg is not None:
            continue
        for side in sides:
            if side.plane.wall_index != wall_index:
                continue
            side.result.all_objects[:] = [
                obj for obj in side.result.all_objects if obj != door
            ]
            for decorations in side.result.decorations.values():
                decorations[:] = [obj for obj in decorations if obj != door]
        delete_objects([door.item()])
        del doors[wall_index]
        del door_dimensions[wall_index]


def _prism(
    section: np.ndarray, z_bounds: tuple[float, float], ring_spacing: float
) -> pf.MeshObject:
    count = len(section)
    levels = mesh_util.split_gaps(np.asarray(z_bounds, dtype=float), ring_spacing)
    nxt = np.roll(section, -1, axis=0)
    side_xy = np.broadcast_to(
        np.stack([section, nxt, nxt, section], axis=1), (len(levels) - 1, count, 4, 2)
    )
    side_z = np.stack([levels[:-1], levels[:-1], levels[1:], levels[1:]], axis=1)
    side_z = np.broadcast_to(side_z[:, None, :, None], (*side_xy.shape[:3], 1))
    sides = np.concatenate([side_xy, side_z], axis=-1).reshape(-1, 3)
    bottom = np.column_stack([section[::-1], np.full(count, levels[0])])
    top = np.column_stack([section, np.full(count, levels[-1])])
    corners = np.concatenate([bottom, top, sides])
    sizes = np.concatenate([[count, count], np.full(len(sides) // 4, 4)])
    return mesh_util.mesh_from_corners(corners, sizes, np.zeros((len(corners), 2)))


def _filler_section(
    vertex: tuple[float, float], group: list[tuple[WallPlane, np.ndarray]]
) -> np.ndarray | None:
    """Slot footprint where outer walls of different rooms meet, else None.

    Bends within one room need no filler: the wall ends overlap or are chamfered.
    """
    thickness = group[0][0].thickness
    if len({plane.room_index for plane, _point in group}) == 1:
        return None
    normals = [-_plane_inward(plane) for plane, _point in group]
    points = [point for _plane, point in group]
    points += [point + n * thickness for point, n in zip(points, normals, strict=True)]
    points += [
        np.asarray(vertex) + (a + b) * thickness / 2 / (1 + a @ b)
        for a, b in combinations(normals, 2)
    ]
    hull = shapely.MultiPoint(points).convex_hull
    if hull.geom_type != "Polygon":
        return None
    return np.asarray(orient(hull, 1.0).exterior.coords[:-1])


@pf.tracer.generator
def shell_filler(
    section: tuple[tuple[float, float], ...], height: float, material: pf.Material
) -> pf.MeshObject:
    z_bounds = (-WALL_STRIP_OVERHANG, height + WALL_STRIP_OVERHANG)
    obj = _prism(np.asarray(section), z_bounds, 0.8)
    pf.ops.modifier.bevel(obj, width=0.006, segments=1)
    pf.ops.object.set_material(
        obj, surface=material.surface, displacement=material.displacement
    )
    pf.ops.uv.cube_project(obj, uv_name="UVMap")
    mesh_util.crease_all_edges(obj)
    return obj


def _shell_fillers(
    planes_exterior: list[WallPlane],
    materials: list[pf.Material],
    walls: tuple[WallSegment, ...],
    height: float,
) -> dict[int, list[pf.MeshObject]]:
    """Fill the wedge or slot each outer-wall vertex leaves between thickened sides."""
    ends = defaultdict(list)
    for plane in planes_exterior:
        wall = walls[plane.wall_index]
        for point in (plane.start, plane.end):
            near_start = dist(point, wall.start) < dist(point, wall.end)
            vertex = wall.start if near_start else wall.end
            key = round(vertex[0], 6), round(vertex[1], 6)
            ends[key].append((plane, np.asarray(point)))
    fillers: dict[int, list[pf.MeshObject]] = defaultdict(list)
    for vertex, group in ends.items():
        section = _filler_section(vertex, group)
        if section is None:
            continue
        points = tuple((float(x), float(y)) for x, y in section)
        obj = shell_filler(points, height, materials[group[0][0].room_index])
        pf.ops.modifier.subdivide_surface(
            obj, levels=ROOM_SUBSURF_LEVELS, _skip_apply=True
        )
        for room_index in sorted({plane.room_index for plane, _point in group}):
            fillers[room_index].append(obj)
    return dict(fillers)


def _point_key(point: tuple[float, float]) -> tuple[float, float]:
    return round(point[0], 5), round(point[1], 5)


def _plane_key(plane: WallPlane) -> tuple[int, tuple[float, float]]:
    return plane.room_index, _point_key(plane.start)


def _turn(incoming: WallPlane, outgoing: WallPlane) -> float:
    a = _plane_direction(incoming)
    b = _plane_direction(outgoing)
    return float(a[0] * b[1] - a[1] * b[0])


def _successors(planes: list[WallPlane]) -> list[tuple[WallPlane, WallPlane]]:
    """Consecutive (plane, next plane) pairs around each room, which runs CCW."""
    by_start = {_plane_key(plane): plane for plane in planes}
    pairs = [
        (plane, by_start.get((plane.room_index, _point_key(plane.end))))
        for plane in planes
    ]
    return [(plane, following) for plane, following in pairs if following is not None]


def _end_shift(turn: float, thickness: float, chamfer: float) -> float:
    # inside corners reach the next wall's centerline; outside corners trim for a chamfer
    if abs(turn) < 1e-3:
        return 0.0
    if turn > 0:
        return thickness / 2
    return -chamfer


def _corner_chamfer(
    incoming: WallPlane,
    outgoing: WallPlane,
    chamfer: float,
    height: float,
    material: pf.Material,
) -> pf.MeshObject:
    corner = np.asarray(incoming.end)
    start = corner - _plane_direction(incoming) * chamfer
    end = corner + _plane_direction(outgoing) * chamfer
    strip = vertical_strip(tuple(start), tuple(end), height, "house_wall_chamfer")
    pf.ops.object.set_material(
        strip, surface=material.surface, displacement=material.displacement
    )
    mesh_util.crease_all_edges(strip)
    pf.ops.modifier.subdivide_surface(
        strip, levels=ROOM_SUBSURF_LEVELS, _skip_apply=True
    )
    return strip


def _join_room_corners(
    sides: list[HouseWallSideResult],
    planes: list[WallPlane],
    materials: list[pf.Material],
    height: float,
) -> dict[int, list[pf.MeshObject]]:
    """Close every wall-to-wall corner and return the chamfer strips per room."""
    chamfer = 0.015
    pairs = _successors(planes)
    turn_in = {
        _plane_key(following): _turn(plane, following) for plane, following in pairs
    }
    turn_out = {
        _plane_key(plane): _turn(plane, following) for plane, following in pairs
    }
    for side in sides:
        key = _plane_key(side.plane)
        low = _end_shift(turn_in.get(key, 0.0), side.plane.thickness, chamfer)
        high = _end_shift(turn_out.get(key, 0.0), side.plane.thickness, chamfer)
        direction_xy = _plane_direction(side.plane)
        direction = (float(direction_xy[0]), float(direction_xy[1]), 0.0)
        for obj in _unique_objects([*side.result.wall_planes, *side.result.backs]):
            shift_wall_ends(obj, direction, low, high)
    strips: dict[int, list[pf.MeshObject]] = defaultdict(list)
    for plane, following in pairs:
        if _turn(plane, following) > -1e-3:
            continue
        material = materials[plane.room_index]
        strip = _corner_chamfer(plane, following, chamfer, height, material)
        strips[plane.room_index].append(strip)
    return dict(strips)


def _polygon_extent(polygon: BaseGeometry) -> tuple[float, float]:
    x0, y0, x1, y1 = polygon.bounds
    return x1 - x0, y1 - y0


def _grid_centers(polygon: BaseGeometry, grid: CeilingGridResult) -> np.ndarray:
    lows = polygon.bounds[:2]
    pitch = np.add(grid.footprint, grid.spacing)
    starts = np.add(lows, grid.margin_low) + np.asarray(grid.footprint) / 2
    axes = [starts[i] + np.arange(grid.counts[i]) * pitch[i] for i in range(2)]
    centers = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)
    half = np.asarray(grid.footprint) / 2 + 0.05
    boxes = shapely.box(*(centers - half).T, *(centers + half).T)
    return centers[shapely.contains(polygon, boxes)]


def _place_on_ceiling(
    ceiling: pf.MeshObject, grid: CeilingGridResult, centers: np.ndarray
) -> list[pf.MeshObject]:
    vertices = np.column_stack((centers, np.zeros(len(centers))))
    points = pf.ops.primitives.mesh_from_numpy(vertices=vertices)
    uv_meters = pf.nodes.geo.input_named_attribute(
        name="UVMap", data_type=pf.NodeDataType.FLOAT_VECTOR
    ).attribute
    instances = grid_placement.place_instances_on_uv_grid(
        surface=ceiling,
        uv_field=uv_meters,
        grid_mesh=points,
        query_uv=pf.nodes.geo.input_position(),
        instance=grid.instance,
        secondary_axis_vector=(0, 1, 0),
        rotation_offset=grid.rotation_offset,
        normal_offset=-grid.reveal_depth * grid.recess_pct,
    )
    aliases = pf.nodes.to_aliases(instances)
    propagate_modifiers_to_instances([grid.instance], aliases)
    delete_object(points.item())
    return aliases


def _finish_surface(obj: pf.MeshObject, material: pf.Material) -> None:
    pf.ops.object.set_material(
        obj, surface=material.surface, displacement=material.displacement
    )
    pf.ops.modifier.subdivide_surface(obj, levels=ROOM_SUBSURF_LEVELS, _skip_apply=True)


def _pierced_ceiling(
    polygon: BaseGeometry,
    ceiling: pf.MeshObject,
    material: pf.Material,
    grid: CeilingGridResult,
    centers: np.ndarray,
    height: float,
    thickness: float,
) -> HouseCeilingCutoutResult:
    depth = grid.reveal_depth
    surface = polygon.buffer(thickness / 2, join_style="mitre")
    cutout = house_ceiling_cutout(surface, centers, grid.footprint, height, depth)
    _finish_surface(cutout.ceiling, material)
    _finish_surface(cutout.sill, material)
    delete_object(ceiling.item())
    return cutout


def _house_ceiling_lamps_rand(
    rng: pf.RNG,
    polygon: BaseGeometry,
    floor: pf.MeshObject,
    ceiling: pf.MeshObject,
    material: pf.Material,
    height: float,
    thickness: float,
) -> CeilingFeaturesResult:
    rng_energy, rng_grid = rng.spawn(2)
    back = extrude_for_thickness(ceiling, thickness)
    energy = polygon.area * pf.random.uniform(rng_energy, 300, 700) / 177
    grid = ceiling_lamp_grid_rand(rng_grid, _polygon_extent(polygon), energy)
    lamps = _place_on_ceiling(ceiling, grid, _grid_centers(polygon, grid))
    lights = ceiling_lamp_lights(grid.light, grid.light_offset, lamps, energy)
    templates = [grid.instance, grid.light]
    delete_objects([obj.item() for obj in templates if obj is not None])
    return CeilingFeaturesResult(floor, ceiling, [back], [], lamps, lights)


def _house_skylights_rand(
    rng: pf.RNG,
    polygon: BaseGeometry,
    floor: pf.MeshObject,
    ceiling: pf.MeshObject,
    material: pf.Material,
    height: float,
    thickness: float,
) -> CeilingFeaturesResult:
    rng_grid, rng_fallback = rng.spawn(2)
    grid = ceiling_skylight_grid_rand(rng_grid, _polygon_extent(polygon))
    centers = _grid_centers(polygon, grid)
    templates = [grid.instance, grid.light]
    if len(centers) == 0:
        delete_objects([obj.item() for obj in templates if obj is not None])
        return _house_ceiling_lamps_rand(
            rng_fallback, polygon, floor, ceiling, material, height, thickness
        )
    windows = _place_on_ceiling(ceiling, grid, centers)
    cutout = _pierced_ceiling(
        polygon, ceiling, material, grid, centers, height, thickness
    )
    portals = []
    if grid.light is not None:
        portals = arrange_window_portals(windows, grid.instance, grid.light)
    delete_objects([obj.item() for obj in templates if obj is not None])
    return CeilingFeaturesResult(
        floor, cutout.ceiling, [cutout.back], [cutout.sill], windows, portals
    )


def _house_light_bars_rand(
    rng: pf.RNG,
    polygon: BaseGeometry,
    floor: pf.MeshObject,
    ceiling: pf.MeshObject,
    material: pf.Material,
    height: float,
    thickness: float,
) -> CeilingFeaturesResult:
    extent = _polygon_extent(polygon)
    rng_grid, rng_fallback, rng_lights = rng.spawn(3)
    grid = ceiling_light_bar_grid_rand(rng_grid, extent, material)
    centers = _grid_centers(polygon, grid)
    if len(centers) == 0:
        delete_object(grid.instance.item())
        return _house_ceiling_lamps_rand(
            rng_fallback, polygon, floor, ceiling, material, height, thickness
        )
    bars = _place_on_ceiling(ceiling, grid, centers)
    cutout = _pierced_ceiling(
        polygon, ceiling, material, grid, centers, height, thickness
    )
    lights = ceiling_light_bar_lights_rand(
        rng_lights, grid.footprint, bars, polygon.area
    )
    delete_object(grid.instance.item())
    return CeilingFeaturesResult(
        floor, cutout.ceiling, [cutout.back], [cutout.sill], bars, lights
    )


@pf.tracer.grammar
def _house_ceiling_rand(
    rng: pf.RNG,
    polygon: BaseGeometry,
    floor: pf.MeshObject,
    ceiling: pf.MeshObject,
    height: float,
    thickness: float,
    floor_material: pf.Material,
) -> CeilingFeaturesResult:
    """One room's floor and ceiling, with lamps, skylights or light bars inside it."""
    vec_pos = pf.nodes.shader.geometry().position
    _, rng_ceiling, rng_choice, rng_feature = rng.spawn(4)
    _finish_surface(floor, floor_material)
    material = ceiling_material_rand(rng_ceiling, vec_pos)
    _finish_surface(ceiling, material)
    option = pf.control.choice(
        rng_choice,
        [
            (_house_ceiling_lamps_rand, 4.0),
            (_house_skylights_rand, 1.0),
            (_house_light_bars_rand, 1.0),
        ],
    )
    return option(rng_feature, polygon, floor, ceiling, material, height, thickness)


@pf.tracer.grammar
def house_walls_rand(
    rng: pf.RNG,
    walls: tuple[WallSegment, ...],
    planes_interior: list[WallPlane],
    planes_exterior: list[WallPlane],
    room_count: int,
    height: float,
    feature_min_width: float = 1.4,
    door_open_angle_deg: float | None = None,
    room_wall_materials: list[pf.Material] | None = None,
) -> HouseWallResult:
    """Treat both sides of every partition and the inside of every outer wall.

    Partition sides have no backs because the two offset faces close the wall.
    """
    (
        rng_window,
        rng_materials,
        rng_doors,
        rng_interior,
        rng_exterior,
        rng_door_open,
    ) = rng.spawn(6)
    materials = room_wall_materials
    if materials is None:
        vec_wall = pf.nodes.shader.coord().uv
        room_rngs = rng_materials.spawn(room_count)
        materials = [wall_material_rand(r, vec_wall) for r in room_rngs]
    exterior_widths = [plane.length for plane in planes_exterior]
    usable_widths = [width for width in exterior_widths if width >= feature_min_width]
    window_width = max(1.0, min(2.0, 0.5 * min(usable_widths, default=1.0)))
    window_height = max(1.0, min(2.0, 0.7 * height))
    rng_window_dimensions, rng_window_asset = rng_window.spawn(2)
    window_dimensions = window.window_dimensions_rand(
        rng_window_dimensions, width=window_width, height=window_height
    )
    shared_window = window.window_composite_rand(
        rng_window_asset, dimensions=window_dimensions
    )
    window_offset = pf.Vector((0.0, -window_dimensions.y / 2, 0.0))
    pf.ops.object.set_transform(shared_window.mesh, location=window_offset)
    pf.ops.mesh.transform_apply(shared_window.mesh)
    if shared_window.light is not None:
        pf.ops.object.set_transform(
            shared_window.light,
            location=shared_window.light.item().location + window_offset,
        )

    door_walls = _door_walls(rng_doors, walls, planes_interior, height, materials)
    door_clearances = [
        obj for obj in door_walls.colliders.objs if obj not in door_walls.doors.values()
    ]
    interior = list(door_walls.sides)
    feature_planes = [
        plane
        for plane in planes_interior
        if walls[plane.wall_index].door_position_frac is None
    ]
    interior_rngs = rng_interior.spawn(len(feature_planes))
    for plane, plane_rng in zip(feature_planes, interior_rngs, strict=True):
        arrangement = interior_wall_arrangement_rand
        if plane.length < feature_min_width:
            arrangement = wall_plain_rand
        result = arrangement(
            plane_rng,
            wall=_plane_surface(plane, height),
            wall_material=materials[plane.room_index],
            wall_thickness=plane.thickness / 2,
        )
        interior.append(HouseWallSideResult(plane, result))

    exterior = []
    exterior_rngs = rng_exterior.spawn(len(planes_exterior))
    for plane, plane_rng in zip(planes_exterior, exterior_rngs, strict=True):
        material = materials[plane.room_index]
        if plane.length < feature_min_width:
            result = wall_plain_rand(
                plane_rng,
                wall=_plane_surface(plane, height),
                wall_material=material,
                wall_thickness=plane.thickness,
            )
        else:
            surface = _plane_surface(plane, height)
            result = _exterior_wall_rand(
                plane_rng, plane, surface, material, shared_window
            )
        exterior.append(HouseWallSideResult(plane, result))

    sides = interior + exterior
    corners_by_room = _shell_fillers(planes_exterior, materials, walls, height)
    chamfers = _join_room_corners(
        sides, planes_interior + planes_exterior, materials, height
    )
    for room_index, strips in chamfers.items():
        corners_by_room.setdefault(room_index, []).extend(strips)
    corners = _unique_objects(
        [obj for objects in corners_by_room.values() for obj in objects]
    )
    _open_doors_rand(
        rng_door_open,
        sides,
        door_walls.doors,
        door_walls.door_dimensions,
        corners,
        door_open_angle_deg,
    )
    results = [side.result for side in sides]
    decorations: dict[str, list[pf.MeshObject]] = defaultdict(list)
    for result in results:
        for kind, objs in result.decorations.items():
            decorations[kind].extend(objs)
    return HouseWallResult(
        sides=tuple(sides),
        planes_interior=[obj for side in interior for obj in side.result.wall_planes],
        planes_exterior=[obj for side in exterior for obj in side.result.wall_planes],
        corners=corners,
        corners_by_room=corners_by_room,
        backs=[obj for result in results for obj in result.backs],
        sills=[obj for result in results for obj in result.sills],
        storage_containers=[obj for r in results for obj in r.storage_containers],
        supports=[obj for r in results for obj in r.supports],
        lights=[light for result in results for light in result.lights],
        windows=decorations["window"],
        decorations=dict(decorations),
        doors=door_walls.doors,
        doorway_centers=door_walls.doorway_centers,
        colliders=ccol.collision_set(
            [*door_walls.doors.values(), *door_clearances],
            cache=door_walls.colliders,
        ),
    )


def _pick_room_materials(
    rng: pf.RNG, palette: list[pf.Material], room_count: int
) -> list[pf.Material]:
    rngs = rng.spawn(room_count)
    return [palette[pf.random.randint(r, 0, len(palette))] for r in rngs]


@pf.tracer.grammar
def house_unfurnished_rand(
    rng: pf.RNG,
    dimensions: tuple[float, float] | None = None,
    room_count: int | None = None,
    height: float | None = None,
    wall_thickness: float | None = None,
    door_open_angle_deg: float | None = None,
    floor_materials: list[pf.Material] | None = None,
    wall_materials: list[pf.Material] | None = None,
) -> HouseResult:
    """Dress a house shell with surface, wall, ceiling, and lighting options.

    Every room picks its floor from `floor_materials` and its walls from
    `wall_materials`, so the house shares a small palette.
    """
    (
        rng_outline,
        rng_profile,
        rng_connections,
        rng_height,
        rng_walls,
        rng_surfaces,
        rng_sky,
        rng_room_count,
        rng_floor_palette,
        rng_wall_palette,
    ) = rng.spawn(10)
    rng_floor_make, rng_floor_pick = rng_floor_palette.spawn(2)
    rng_wall_make, rng_wall_pick = rng_wall_palette.spawn(2)
    if floor_materials is None:
        vec_pos = pf.nodes.shader.geometry().position
        floor_rngs = rng_floor_make.spawn(2)
        floor_materials = [floor_material_rand(r, vec_pos) for r in floor_rngs]
    if wall_materials is None:
        vec_wall = pf.nodes.shader.coord().uv
        wall_rngs = rng_wall_make.spawn(3)
        wall_materials = [wall_material_rand(r, vec_wall) for r in wall_rngs]
    if room_count is None:
        room_count = pf.random.randint(rng_room_count, 3, 9)
    if height is None:
        height = pf.random.clip_gaussian(rng_height, 2.8, 0.25, 2.4, 3.5)
    outline = house_outline_rand(
        rng_outline, dimensions=dimensions, room_count=room_count
    )
    profile = room_solve(
        rng_profile,
        outline.boundary_rings,
        room_count,
        wall_thickness=wall_thickness,
        fillet_segments=outline.fillet_segments,
    )
    boundaries = profile.room_boundaries
    segments = room_connect_rand(rng_connections, profile.walls, len(boundaries))
    shape = house_floor_profile_to_planes(boundaries, segments, height=height)
    logger.info(f"Solved a house of {len(boundaries)} rooms")
    thickness = segments[0].thickness
    room_floor_materials = _pick_room_materials(
        rng_floor_pick, floor_materials, len(boundaries)
    )
    room_wall_materials = _pick_room_materials(
        rng_wall_pick, wall_materials, len(boundaries)
    )
    room_rngs = rng_surfaces.spawn(len(shape.ceilings))
    surfaces = [
        _house_ceiling_rand(
            room_rng,
            house_room_polygon(rings),
            floor,
            ceiling,
            height,
            thickness,
            floor_material,
        )
        for room_rng, rings, floor, ceiling, floor_material in zip(
            room_rngs,
            boundaries,
            shape.floors,
            shape.ceilings,
            room_floor_materials,
            strict=True,
        )
    ]
    floors = [surface.floor for surface in surfaces]
    ceilings = [surface.ceiling for surface in surfaces]
    fixtures = [obj for surface in surfaces for obj in surface.light_meshes]
    walls = house_walls_rand(
        rng_walls,
        segments,
        shape.planes_interior,
        shape.planes_exterior,
        len(boundaries),
        height,
        door_open_angle_deg=door_open_angle_deg,
        room_wall_materials=room_wall_materials,
    )
    sky = sky_lighting.sky_hosek_wilkie_with_sun_lamp_rand(rng_sky)
    storage_objects = _unique_objects([*walls.storage_containers, *walls.supports])

    all_objects = (
        name_objects(floors, "house_floor")
        + name_objects(ceilings, "house_ceiling")
        + name_objects([o for s in surfaces for o in s.backs], "house_ceiling_back")
        + name_objects([o for s in surfaces for o in s.sills], "house_ceiling_sill")
        + name_objects(fixtures, "house_ceiling_feature")
        + name_objects(walls.planes_interior, "house_wall_interior")
        + name_objects(walls.planes_exterior, "house_wall_exterior")
        + name_objects(walls.corners, "house_wall_corner")
        + name_objects(walls.backs, "house_wall_back")
        + name_objects(walls.sills, "house_wall_sill")
        + name_objects(storage_objects, "house_wall_storage")
    )
    for kind, objs in sorted(walls.decorations.items()):
        all_objects += name_objects(objs, f"house_wall_{kind}")
    door_rooms = {
        index: segment.room_indices
        for index, segment in enumerate(segments)
        if segment.door_position_frac is not None
    }
    rooms = []
    for room_index, surface in enumerate(surfaces):
        sides = [side for side in walls.sides if side.plane.room_index == room_index]
        flat_sides = [s for s in sides if segments[s.plane.wall_index].is_flat]
        neighbors = {
            i: second if first == room_index else first
            for i, (first, second) in door_rooms.items()
            if room_index in (first, second)
        }
        doors = {i: walls.doors[i] for i in neighbors if i in walls.doors}
        room_objects = [
            surface.floor,
            surface.ceiling,
            *surface.backs,
            *surface.sills,
            *surface.light_meshes,
            *doors.values(),
            *(obj for side in sides for obj in side.result.all_objects),
            *walls.corners_by_room.get(room_index, []),
        ]
        room_lights = [
            *surface.lights,
            *(light for side in sides for light in side.result.lights),
        ]
        rooms.append(
            HouseRoomResult(
                room_index=room_index,
                boundary_rings=boundaries[room_index],
                all_objects=_unique_objects(room_objects),
                lights=_unique_objects(room_lights),
                floor=surface.floor,
                ceiling=surface.ceiling,
                walls=[obj for side in sides for obj in side.result.wall_planes],
                flat_walls=[
                    obj for side in flat_sides for obj in side.result.wall_planes
                ],
                neighbors=neighbors,
                doors=doors,
                doorway_centers={
                    wall_index: walls.doorway_centers[wall_index]
                    for wall_index in neighbors
                },
            )
        )
    all_objects = _unique_objects(
        all_objects + [obj for room in rooms for obj in room.all_objects]
    )

    collider_objects = _unique_objects(
        walls.colliders.objs
        + walls.planes_interior
        + walls.planes_exterior
        + walls.corners
        + floors
        + ceilings
        + fixtures
        + storage_objects
        + [obj for objs in walls.decorations.values() for obj in objs]
        + list(walls.doors.values())
    )
    colliders = ccol.collision_set(collider_objects, cache=walls.colliders)
    ceiling_lights = [light for surface in surfaces for light in surface.lights]
    lights = _unique_objects([*ceiling_lights, *walls.lights, *sky.lights])
    return HouseResult(
        all_objects=all_objects,
        small_objects=[],
        cameras=[framing.camera_in_room_corner(shape.floors[0], float(height))],
        lights=lights,
        rooms=tuple(rooms),
        dimensions=shape.dimensions,
        colliders=colliders,
        environment=sky.environment,
    )
