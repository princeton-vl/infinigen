# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging
from typing import Callable, NamedTuple, Protocol, TypeVar, runtime_checkable

import numpy as np
import procfunc as pf

from infinigen2.lighting import sky_lighting
from infinigen2.objects import (
    bookcase,
    cabinet,
    lamp,
    rug,
    sofa,
    table,
    triangle_shelf,
    vase,
)
from infinigen2.scenes import collision_collection as ccol
from infinigen2.scenes.placement_utils import (
    keep_non_colliding,
    snap_to_plane,
)

__all__ = [
    "ArrangementResult",
    "MeshResult",
    "RoomFurnitureResult",
    "back_face_grounded",
    "centered_sofa_setup_rand",
    "keep_unobstructed",
    "place_dining_table",
    "random_bbox_poses_animation_rand",
    "retry_place",
    "room_furniture_rand",
    "side_table_object_rand",
    "storage_object_rand",
    "table_decoration_object_rand",
]

logger = logging.getLogger(__name__)

MR = TypeVar("MR", bound="MeshResult")


@runtime_checkable
class MeshResult(Protocol):
    @property
    def mesh(self) -> pf.MeshObject: ...


class _BareMeshResult(NamedTuple):
    mesh: pf.MeshObject


class RoomFurnitureResult(NamedTuple):
    furniture: list[pf.MeshObject]
    storage_surfaces: list[pf.MeshObject]
    lights: list[pf.LightObject]
    colliders: ccol.CollisionSet
    floor: pf.MeshObject
    storage_objects: list[pf.MeshObject]
    coffee_tables: list[pf.MeshObject]
    side_tables: list[pf.MeshObject]
    diningtable_objs: list[pf.MeshObject]
    sofas: list[pf.MeshObject]
    rugs: list[pf.MeshObject]


@pf.tracer.grammar
def random_bbox_poses_animation_rand(
    camera: pf.CameraObject,
    rng: pf.RNG,
    room_dimensions: pf.Vector,
    frame_start: int = 1,
    frame_end: int = 1,
    wall_margin: float = 0.5,
    ceil_margin: float = 0.3,
    floor_margin: float = 0.3,
) -> None:
    cam = camera.item()
    n_frames = max(frame_end - frame_start + 1, 1)
    frame_rngs = rng.spawn(n_frames)
    for i in range(n_frames):
        frame = frame_start + i
        r = frame_rngs[i]
        x = pf.random.uniform(r, wall_margin, room_dimensions.x - wall_margin)
        y = pf.random.uniform(r, wall_margin, room_dimensions.y - wall_margin)
        z = pf.random.clip_gaussian(
            r, 1.5, 0.4, floor_margin, room_dimensions.z - ceil_margin
        )
        yaw = pf.random.uniform(r, -np.pi, np.pi)
        pitch = pf.random.clip_gaussian(r, np.pi / 2, 0.3, np.pi / 4, 3 * np.pi / 4)
        roll = pf.random.clip_gaussian(r, 0.0, 0.05, -0.2, 0.2)
        cam.location = (x, y, z)
        cam.rotation_euler = (pitch, roll, yaw)
        cam.keyframe_insert("location", frame=frame)
        cam.keyframe_insert("rotation_euler", frame=frame)


@pf.tracer.grammar
def table_decoration_object_rand(
    rng: pf.RNG,
) -> MeshResult:
    func = pf.control.choice(
        rng,
        [
            (vase.vase_rand, 1.0),
        ],
    )
    result = func(rng)
    result.mesh.item().name = func.__name__
    return result


@pf.tracer.grammar
def side_table_object_rand(rng: pf.RNG) -> MeshResult:
    def triangle_shelf_sidetable_rand(
        rng: pf.RNG,
    ) -> triangle_shelf.TriangleShelfResult:
        dimensions = table.side_table_dimensions_rand(rng)
        return triangle_shelf.triangle_shelf_rand(rng, dimensions=dimensions)

    func = pf.control.choice(
        rng,
        [
            (table.side_table_rand, 1.0),
            (triangle_shelf_sidetable_rand, 100.0),
        ],
    )
    result = func(rng)
    result.mesh.item().name = func.__name__
    return result


@pf.tracer.grammar
def storage_object_rand(rng: pf.RNG) -> MeshResult:
    func = pf.control.choice(
        rng,
        [
            (bookcase.bookcase_rand, 1.0),
            (cabinet.cabinet_rand, 1.0),
        ],
    )
    result = func(rng)
    result.mesh.item().name = func.__name__
    return result


def retry_place(
    rng: pf.RNG,
    child: MR,
    colliders: ccol.CollisionSet,
    place_fn: Callable[..., None],
    attempts: int = 7,
    accept_fn: Callable[[pf.MeshObject], bool] | None = None,
    **kwargs,
) -> MR | None:
    """Re-pose `child` with `place_fn` until it clears the existing `colliders`
    (and satisfies `accept_fn`, if given), or give up and return None. Reads
    `colliders` but never extends it; keep_non_colliding owns updates and
    intra-batch collisions.
    """
    for r in rng.spawn(attempts):
        place_fn(r, child, **kwargs)
        if ccol.intersection_test(colliders, child.mesh):
            continue
        if accept_fn is not None and not accept_fn(child.mesh):
            continue
        return child
    child.mesh.item().name = child.mesh.item().name + "_FAILED_PLACEMENT"
    return None


def _snap_to_wall(
    rng: pf.RNG,
    child: MR,
    parents: list[pf.MeshObject],
    placement: float | None = None,
    margin: float | None = None,
    child_side: str | None = None,
) -> None:
    if placement is None:
        placement = pf.random.uniform(rng, 0.1, 0.9)
    if margin is None:
        margin = pf.random.clip_gaussian(rng, 0.15, 0.1, 0.1, 0.4)
    if child_side is None:
        child_side = pf.control.choice(rng, [("back", 1.0), ("left", 1.0)])
    snap_to_plane(
        child=child.mesh,
        parent=rng.choice(parents),
        placement=placement,
        margin=margin,
        child_side=child_side,
        parent_side="front",
    )


def _snap_back_front(
    rng: pf.RNG,
    child: MR,
    parents: list[pf.MeshObject],
    placement: float | None = None,
    margin: float | None = None,
) -> None:
    _snap_to_wall(
        rng, child, parents, placement=placement, margin=margin, child_side="back"
    )


def back_face_grounded(
    obj: pf.MeshObject,
    colliders: ccol.CollisionSet,
    margin: float,
    eps: float = 0.5,
) -> bool:
    """True iff every corner of `obj`'s back (-X local) face has a collider
    directly behind it within (1 + eps) * margin. Rejects placements where part
    of the back overhangs a window/door alcove (no wall behind that corner).
    """
    bmin, bmax = pf.ops.attr.bbox_min_max(obj, global_coords=False)
    corners_local = np.array(
        [
            [bmin[0], bmin[1], bmin[2]],
            [bmin[0], bmin[1], bmax[2]],
            [bmin[0], bmax[1], bmin[2]],
            [bmin[0], bmax[1], bmax[2]],
        ]
    )
    mw = np.array(obj.item().matrix_world)
    corners_world = corners_local @ mw[:3, :3].T + mw[:3, 3]
    back_normal = mw[:3, :3] @ np.array([-1.0, 0.0, 0.0])
    back_normal = back_normal / np.linalg.norm(back_normal)

    hits, ray_idx, _ = ccol.raycast(
        colliders, corners_world, np.tile(back_normal, (4, 1))
    )
    threshold = (1.0 + eps) * margin
    grounded = np.zeros(4, dtype=bool)
    for loc, ri in zip(hits, ray_idx, strict=False):
        if np.linalg.norm(loc - corners_world[ri]) <= threshold:
            grounded[ri] = True
    return bool(grounded.all())


def _snap_side_by_side(rng: pf.RNG, child: MR, parents: list[pf.MeshObject]) -> None:
    sides = pf.control.choice(rng, [(("left", "right"), 0.5), (("right", "left"), 0.5)])
    snap_to_plane(
        child.mesh,
        parent=rng.choice(parents),
        placement=rng.uniform(0.05, 0.95),
        margin=pf.random.clip_gaussian(rng, 0.07, 0.06, 0.02, 0.25),
        child_side=sides[0],
        parent_side=sides[1],
    )


def _snap_on_top(
    rng: pf.RNG,
    child: MR,
    parents: list[pf.MeshObject],
    xy_frac: tuple[float, float] = (0.5, 0.5),
) -> None:
    parent = rng.choice(list(parents))
    bbox_min, bbox_max = pf.ops.attr.bbox_min_max(parent, global_coords=True)
    child.mesh.item().location = (
        bbox_min[0] + (bbox_max[0] - bbox_min[0]) * xy_frac[0],
        bbox_min[1] + (bbox_max[1] - bbox_min[1]) * xy_frac[1],
        bbox_max[2] + 0.002,
    )


def _world_vert_bbox(obj: pf.MeshObject) -> tuple[np.ndarray, np.ndarray]:
    bmin, bmax = pf.ops.attr.bbox_min_max(obj, global_coords=True)
    return np.array(bmin), np.array(bmax)


def _collider_blocks_segment(
    a: np.ndarray, b: np.ndarray, colliders: ccol.CollisionSet
) -> bool:
    d = b - a
    dist = float(np.linalg.norm(d))
    origin = np.array([[a[0], a[1], 0.3]])
    hit, idx, _ = ccol.raycast(
        colliders, origin, np.array([[d[0] / dist, d[1] / dist, 0.0]])
    )
    return len(idx) > 0 and np.linalg.norm(hit[0] - origin[0]) < dist


def keep_unobstructed(
    results: list[MR | None],
    center: np.ndarray,
    colliders: ccol.CollisionSet,
) -> list[MR]:
    """Drop results whose center a collider separates from `center` (e.g. a wall
    between the rug and the sofa). Returns the unobstructed results; reads colliders
    but does not update them.
    """
    kept: list[MR] = []
    for r in results:
        if r is None:
            continue
        bmin, bmax = _world_vert_bbox(r.mesh)
        if _collider_blocks_segment(center[:2], (bmin[:2] + bmax[:2]) / 2, colliders):
            r.mesh.item().name = r.mesh.item().name + "_OBSTRUCTED"
            continue
        kept.append(r)
    return kept


class ArrangementResult(NamedTuple):
    sofas: list[MeshResult]
    dining_tables: list[MeshResult]
    rugs: list[pf.MeshObject]
    center_coffee_tables: list[MeshResult]


def _rug_in_front_of_storage(
    rng: pf.RNG,
    storage_objects: list[MeshResult],
) -> list[pf.MeshObject] | None:
    """Snap a rug to the front of a chosen storage unit, or None."""
    if not storage_objects:
        return None
    rug_result = rug.rug_rand(rng)
    storage = rng.choice(storage_objects)
    snap_to_plane(
        child=rug_result.mesh,
        parent=storage.mesh,
        placement=pf.random.uniform(rng, 0.35, 0.65),
        margin=pf.random.uniform(rng, 0.3, 1.0),
        child_side="back",
        parent_side="front",
        constraint_axis=pf.Vector((0, 0, 1)),
    )
    return [rug_result.mesh]


def _placed_rug(
    rng: pf.RNG,
    room_dimensions: pf.Vector,
    wall_clearance: float = 0.3,
) -> list[pf.MeshObject]:
    # size within the space left after clearance so the rug always fits
    avail_x = room_dimensions.x - 2 * wall_clearance
    avail_y = room_dimensions.y - 2 * wall_clearance
    length = pf.random.uniform(rng, min(1.0, avail_x), avail_x)
    width = pf.random.uniform(rng, min(1.0, avail_y), avail_y)
    thickness = pf.random.uniform(rng, 0.01, 0.02)
    rug_result = rug.rug_rand(rng, dimensions=pf.Vector((length, width, thickness)))
    rug_result.mesh.item().name = rug.rug_rand.__name__
    cx = pf.random.uniform(
        rng,
        wall_clearance + length / 2,
        room_dimensions.x - wall_clearance - length / 2,
    )
    cy = pf.random.uniform(
        rng, wall_clearance + width / 2, room_dimensions.y - wall_clearance - width / 2
    )
    pf.ops.object.set_transform(rug_result.mesh, location=(cx, cy, 0.001))
    return [rug_result.mesh]


def _maybe_rug(
    rng: pf.RNG,
    room_dimensions: pf.Vector,
    rug_weight: float = 1.0,
) -> list[pf.MeshObject]:
    func = pf.control.choice(
        rng,
        [
            (_placed_rug, rug_weight),
            (lambda *_, **__: [], 1.0),
        ],
    )
    return func(rng, room_dimensions=room_dimensions)


def _snap_facing_carpet(
    rng: pf.RNG,
    child: MR,
    carpet: pf.MeshObject,
    parent_side: str,
) -> None:
    snap_to_plane(
        child=child.mesh,
        parent=carpet,
        parent_side=parent_side,
        child_side="front",
        margin=pf.random.uniform(rng, -0.1, 0.5),
        placement=pf.random.uniform(rng, 0.35, 0.65),
        constraint_axis=pf.Vector((0, 0, 1)),
    )


def _place_on_floor(rng: pf.RNG, child: MR, room_dimensions: pf.Vector) -> None:
    bmin, _ = pf.ops.attr.bbox_min_max(child.mesh, global_coords=False)
    child.mesh.item().location = (
        pf.random.uniform(rng, 0.2, 0.8) * room_dimensions.x,
        pf.random.uniform(rng, 0.4, 0.6) * room_dimensions.y,
        0.001 - bmin[2],
    )


def _sofas_on_wall_setup(
    rng: pf.RNG,
    wall_planes: list[pf.MeshObject],
    floor: pf.MeshObject,
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
    storage_objects: list[MeshResult],
) -> tuple[ArrangementResult, ccol.CollisionSet]:
    del storage_objects
    n = pf.random.randint(rng, 0, 8)
    rngs = rng.spawn(n)
    sofas = [sofa.sofa_rand(rngs[i]) for i in range(n)]
    sofas = [
        retry_place(rngs[i], sofas[i], colliders, _snap_back_front, parents=wall_planes)
        for i in range(n)
    ]
    sofa_objs, colliders = keep_non_colliding(sofas, colliders)
    logger.info(f"Placed {len(sofa_objs)} wall sofas out of {n} attempts")
    rug_objs = _maybe_rug(rng, room_dimensions)
    return ArrangementResult(sofa_objs, [], rug_objs, []), colliders


def centered_sofa_setup_rand(
    rng: pf.RNG,
    wall_planes: list[pf.MeshObject],
    floor: pf.MeshObject,
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
    storage_objects: list[MeshResult],
) -> tuple[ArrangementResult, ccol.CollisionSet]:
    del wall_planes, floor
    rug_objs = _rug_in_front_of_storage(rng, storage_objects)
    if rug_objs is None:
        rug_objs = _placed_rug(rng, room_dimensions=room_dimensions, wall_clearance=1.2)
    carpet = rug_objs[0]
    cmin, cmax = _world_vert_bbox(carpet)
    center = (cmin + cmax) / 2

    # one sofa per chosen rug side, snapped to that side facing inward
    side_names = ["right", "left", "front", "back"]
    n_sides = pf.random.randint(rng, 2, 5)
    sides = [
        side_names[int(i)]
        for i in rng.choice(len(side_names), size=n_sides, replace=False)
    ]
    n = len(sides)
    rngs = rng.spawn(n)
    sofas = [sofa.sofa_rand(rngs[i]) for i in range(n)]
    sofas = [
        retry_place(
            rngs[i],
            sofas[i],
            colliders,
            _snap_facing_carpet,
            carpet=carpet,
            parent_side=sides[i],
        )
        for i in range(n)
    ]
    sofas = keep_unobstructed(sofas, center, colliders)
    sofa_objs, colliders = keep_non_colliding(sofas, colliders)
    logger.info(f"Placed {len(sofa_objs)} carpet sofas out of {n} attempts")

    def _place_coffee(rng: pf.RNG) -> list[MeshResult]:
        child = table.coffee_table_rand(rng)
        child.mesh.item().location = (
            center[0] + pf.random.uniform(rng, -0.2, 0.2),
            center[1] + pf.random.uniform(rng, -0.2, 0.2),
            0.01,
        )
        return [child]

    coffee_fn = pf.control.choice(
        rng,
        [(_place_coffee, 1.0), (lambda *_, **__: [], 1.0)],
    )
    center_coffee = coffee_fn(rng)
    center_coffee, colliders = keep_non_colliding(center_coffee, colliders)

    # sofas already ringed the carpet above; drop the rug ~1/3 of the time
    out_rugs = pf.control.choice(rng, [(rug_objs, 2.0), ([], 1.0)])

    return ArrangementResult(sofa_objs, [], out_rugs, center_coffee), colliders


def _place_in_free_floorspace(
    rng: pf.RNG,
    child: MR,
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
    clearance: float = 2.0,
    attempts: int = 7,
) -> MR | None:
    """Place `child` at a random floor location whose `clearance`x-footprint box
    (at the child's own height) clears all existing colliders. Returns the placed
    child, or None if no clear spot is found.
    """

    def footprint_clears(mesh: pf.MeshObject) -> bool:
        lo, hi = (
            np.array(b) for b in pf.ops.attr.bbox_min_max(mesh, global_coords=True)
        )
        ext = hi - lo
        transform = np.eye(4)
        transform[:3, 3] = (lo + hi) / 2
        return not ccol.box_intersection_test(
            colliders, transform, [clearance * ext[0], clearance * ext[1], ext[2]]
        )

    return retry_place(
        rng,
        child,
        colliders,
        _place_on_floor,
        attempts=attempts,
        accept_fn=footprint_clears,
        room_dimensions=room_dimensions,
    )


def place_dining_table(
    rng: pf.RNG,
    wall_planes: list[pf.MeshObject],
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
) -> tuple[list[MeshResult], ccol.CollisionSet]:
    """Place a single dining table: 2/3 in clear floor space, 1/3 snapped to a
    wall. Returns the placed result (length 0 or 1) and updated colliders.
    """
    dining_table = table.dining_table_rand(rng)

    def in_free_floorspace():
        return _place_in_free_floorspace(rng, dining_table, room_dimensions, colliders)

    def against_wall():
        return retry_place(
            rng,
            dining_table,
            colliders,
            _snap_to_wall,
            parents=wall_planes,
            margin=pf.random.uniform(rng, 0.03, 0.10),
        )

    placed = pf.control.choice(
        rng,
        [
            (in_free_floorspace, 2.0),
            (against_wall, 1.0),
        ],
    )()
    diningtable_objs, colliders = keep_non_colliding([placed], colliders)
    logger.info(f"Placed {len(diningtable_objs)} dining tables")
    return diningtable_objs, colliders


# ruff: noqa: C901
@pf.tracer.grammar
def room_furniture_rand(
    rng: pf.RNG,
    dimensions: pf.Vector,
    wall_planes: list[pf.MeshObject],
    floor: pf.MeshObject,
    frame_start: int = 1,
    frame_end: int = 1,
    extra_colliders: list[pf.MeshObject] | None = None,
    wall_storage: list[pf.MeshObject] | None = None,
) -> RoomFurnitureResult:
    (
        rng_sky,
        rng_big,
        rng_decoration_object,
        rng_table,
        rng_lamp,
        rng_table_lamp,
        rng_dining_table,
        _rng_wall_object,
        rng_plants,
        rng_rug,
        rng_dining_place,
    ) = rng.spawn(11)

    room_dimensions = dimensions

    # dedicated lanes so the sun angle/intensity are identical across sky models
    # (Nishita vs Hosek-Wilkie+lamp) for the same seed; only model-specific params differ
    rng_elev, rng_rot, rng_intensity, rng_size, rng_sky_model = rng_sky.spawn(5)
    sun_elevation_deg = pf.random.uniform(rng_elev, 5, 80)
    sun_rotation_deg = pf.random.uniform(rng_rot, 0, 360)
    sun_intensity = pf.random.uniform(rng_intensity, 0.8, 1.0)
    sun_size_deg = pf.random.clip_gaussian(rng_size, 0.5, 0.3, 0.25, 5)
    sky_env = sky_lighting.hosek_wilkie_sky_with_sun_lamp_rand(
        rng_sky_model,
        sun_elevation_deg=sun_elevation_deg,
        sun_rotation_deg=sun_rotation_deg,
        sun_intensity=sun_intensity,
        sun_size_deg=sun_size_deg,
    )
    sun_lamp = sky_env.lights[0]

    if extra_colliders is None:
        extra_colliders = []
    if wall_storage is None:
        wall_storage = []
    colliders = ccol.collision_set(wall_planes + [floor] + extra_colliders)
    logger.info(
        f"Created collision set with {len(colliders.mesh_fcl_colliders)} underlying BVHs"
    )

    # arrangement goes first so wall storage can avoid it (storage doesn't exist
    # yet, so the arrangement gets an empty storage list to anchor against)
    arrangement_func = pf.control.choice(
        rng_big,
        [
            (_sofas_on_wall_setup, 2.0),
            (centered_sofa_setup_rand, 3.0),
        ],
    )
    arrangement, colliders = arrangement_func(
        rng_big,
        wall_planes=wall_planes,
        floor=floor,
        room_dimensions=room_dimensions,
        colliders=colliders,
        storage_objects=[],
    )

    # dining table placed in half of scenes, independent of the sofa arrangement
    def with_dining_table():
        return place_dining_table(
            rng_dining_place,
            wall_planes=wall_planes,
            room_dimensions=room_dimensions,
            colliders=colliders,
        )

    dining_tables, colliders = pf.control.choice(
        rng_dining_place,
        [(with_dining_table, 1.0), (lambda: ([], colliders), 1.0)],
    )()

    n = pf.random.randint(rng_big, 4, 10)
    rngs = rng_big.spawn(n)
    storage = [storage_object_rand(rngs[i]) for i in range(n)]
    wall_margins = [pf.random.uniform(rngs[i], 0.03, 0.10) for i in range(n)]
    wall_colliders = ccol.collision_set(wall_planes)
    storage = [
        retry_place(
            rngs[i],
            storage[i],
            colliders,
            _snap_back_front,
            attempts=12,
            parents=wall_planes,
            margin=wall_margins[i],
            accept_fn=lambda m, margin=wall_margins[i]: back_face_grounded(
                m, wall_colliders, margin
            ),
        )
        for i in range(n)
    ]
    storage_objects, colliders = keep_non_colliding(storage, colliders)
    logger.info(f"Placed {len(storage_objects)} storage objects out of {n} attempts")

    sofa_meshes = [r.mesh for r in arrangement.sofas]

    # the center coffee table from the arrangement is the only coffee table now
    coffee_tables: list[MeshResult] = list(arrangement.center_coffee_tables)

    # SIDETABLES BY BIG OBJECTS
    n = min(pf.random.randint(rng_table, 0, 6), len(arrangement.sofas))
    rngs = rng_table.spawn(n)
    side_tables = [table.side_table_rand(rngs[i]) for i in range(n)]
    side_tables = [
        retry_place(
            rngs[i],
            side_tables[i],
            colliders,
            _snap_side_by_side,
            parents=sofa_meshes,
            attempts=12,
        )
        for i in range(n)
    ]
    side_tables, colliders = keep_non_colliding(side_tables, colliders)
    logger.info(f"Placed {len(side_tables)} side tables out of {n} attempts")

    # FLOOR LAMPS beside seating / floor storage (not wall-mounted storage)
    big_meshes = sofa_meshes + [r.mesh for r in storage_objects]
    n = min(pf.random.randint(rng_lamp, 0, 3), len(big_meshes))
    rngs = rng_lamp.spawn(n)
    floor_lamps = [
        lamp.lamp_rand(rngs[i], pf.random.uniform(rngs[i], 1.0, 2.0)) for i in range(n)
    ]
    floor_lamps = [
        retry_place(
            rngs[i], floor_lamps[i], colliders, _snap_side_by_side, parents=big_meshes
        )
        for i in range(n)
    ]
    floor_lamps, colliders = keep_non_colliding(floor_lamps, colliders)
    logger.info(f"Placed {len(floor_lamps)} floor lamps out of {n} attempts")
    floor_lamp_lights = [r.light for r in floor_lamps if r.light is not None]
    lights: list[pf.LightObject] = [sun_lamp]
    lights += pf.control.choice(rng_lamp, [(floor_lamp_lights, 2), ([], 1)])

    # vases/lamps sit on the tops of all tables and storage units
    surface_meshes = [
        r.mesh for r in dining_tables + coffee_tables + storage_objects + side_tables
    ]
    n = min(pf.random.randint(rng_decoration_object, 1, 4), 2 * len(surface_meshes))
    rngs = rng_decoration_object.spawn(n)
    decorations = [table_decoration_object_rand(rngs[i]) for i in range(n)]
    decorations = [
        retry_place(
            rngs[i], decorations[i], colliders, _snap_on_top, parents=surface_meshes
        )
        for i in range(n)
    ]
    decorations, colliders = keep_non_colliding(decorations, colliders)
    logger.info(f"Placed {len(decorations)} decoration objects out of {n} attempts")

    # table lamps only on side tables, never dining/coffee tables
    side_table_meshes = [r.mesh for r in side_tables + storage_objects]
    n = min(pf.random.randint(rng_table_lamp, 1, 3), len(side_table_meshes))
    rngs = rng_table_lamp.spawn(n)
    table_lamps = [lamp.desk_lamp_rand(rngs[i]) for i in range(n)]
    table_lamps = [
        retry_place(
            rngs[i], table_lamps[i], colliders, _snap_on_top, parents=side_table_meshes
        )
        for i in range(n)
    ]
    table_lamps, colliders = keep_non_colliding(table_lamps, colliders)
    logger.info(f"Placed {len(table_lamps)} table lamps out of {n} attempts")
    table_lamp_lights = [r.light for r in table_lamps if r.light is not None]
    lights += pf.control.choice(rng_table_lamp, [(table_lamp_lights, 2), ([], 1)])

    logger.info(f"Placed {len(arrangement.rugs)} rugs")

    placed_results: dict[str, list[MeshResult]] = {
        "sofa": arrangement.sofas,
        "storage": storage_objects,
        "coffee_table": coffee_tables,
        "side_table": side_tables,
        "floor_lamp": floor_lamps,
        "table_lamp": table_lamps,
        "decoration": decorations,
        "dining_table": dining_tables,
    }
    furniture_named: dict[str, list[pf.MeshObject]] = {
        "rug": arrangement.rugs,
    }
    for name, results in placed_results.items():
        furniture_named[name] = [r.mesh for r in results]

    for name, objs in furniture_named.items():
        for i, obj in enumerate(objs):
            obj.item().name = f"{name}_{i}"
            for j, slot in enumerate(obj.item().material_slots):
                if slot.material is not None:
                    slot.material.name = f"{name}_{i}_{j}"

    all_furniture = [obj for objs in furniture_named.values() for obj in objs]

    storage_surfaces = (
        [r.mesh for r in dining_tables]
        + [r.mesh for r in coffee_tables]
        + [r.mesh for r in storage_objects]
        + [r.mesh for r in side_tables]
    )

    return RoomFurnitureResult(
        furniture=all_furniture,
        storage_surfaces=storage_surfaces,
        lights=lights,
        colliders=colliders,
        floor=floor,
        storage_objects=[r.mesh for r in storage_objects],
        coffee_tables=[r.mesh for r in coffee_tables],
        side_tables=[r.mesh for r in side_tables],
        diningtable_objs=[r.mesh for r in dining_tables],
        sofas=sofa_meshes,
        rugs=list(arrangement.rugs),
    )
