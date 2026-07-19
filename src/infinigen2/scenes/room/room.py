# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import functools
import logging
from typing import NamedTuple

import procfunc as pf

from infinigen2.lighting import sky_lighting
from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.placement.culling import keep_non_colliding
from infinigen2.scenes.room.room_furniture import (
    MeshResult,
    back_face_grounded,
    dining_table_setup_rand,
    retry_place,
    snap_back_front,
    snap_on_top,
    sofa_lamps_rand,
    sofa_setup_rand,
    storage_object_rand,
    table_decoration_object_rand,
)
from infinigen2.scenes.room.room_shape import (
    room_shape_rand,
)
from infinigen2.scenes.room.room_small_objects import (
    objects_scatter_rand,
    objects_scattered_on_surface,
    small_objects_collection_rand,
)
from infinigen2.scenes.room.room_surface_features import (
    ceiling_feature_rand,
    skirting_rand,
    wall_feature_rand,
)

__all__ = [
    "LivingroomResult",
    "livingroom_rand",
    "room_rand",
]

logger = logging.getLogger(__name__)


class LivingroomResult(NamedTuple):
    all_objects: list[pf.MeshObject]
    lights: list[pf.LightObject]
    colliders: ccol.CollisionSet
    floor: pf.MeshObject
    dimensions: pf.Vector | None = None


def _name_materials(obj: pf.MeshObject, base: str) -> None:
    for j, slot in enumerate(obj.item().material_slots):
        if slot.material is not None:
            slot.material.name = f"{base}_{j}"


def _rename(objs: list[pf.MeshObject], name: str) -> list[pf.MeshObject]:
    """Name each object (and its materials) `{name}.NN` and return them as a list, so
    a scene's objects can be gathered by concatenating single-category _rename calls."""
    named = list(objs)
    for i, obj in enumerate(named):
        obj.item().name = f"{name}.{i:02d}"
        _name_materials(obj, f"{name}.{i:02d}")
    return named


def _wall_storage_objects(
    rng: pf.RNG,
    wall_planes: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
) -> tuple[list[MeshResult], ccol.CollisionSet]:
    n = pf.random.randint(rng, 4, 10)
    rngs = rng.spawn(n)
    storage = [storage_object_rand(rngs[i]) for i in range(n)]
    wall_margins = [pf.random.uniform(rngs[i], 0.03, 0.10) for i in range(n)]
    wall_colliders = ccol.collision_set(wall_planes)
    placed_storage = []
    for i in range(n):
        grounded = functools.partial(
            back_face_grounded, colliders=wall_colliders, margin=wall_margins[i]
        )
        storage_obj = retry_place(
            rngs[i],
            storage[i],
            colliders,
            snap_back_front,
            attempts=12,
            parents=wall_planes,
            margin=wall_margins[i],
            accept_fn=grounded,
        )
        placed_storage.append(storage_obj)
    storage_objects, colliders = keep_non_colliding(placed_storage, colliders)
    logger.info(f"Placed {len(storage_objects)} storage objects out of {n} attempts")
    return storage_objects, colliders


def _surface_decorations(
    rng: pf.RNG,
    surface_meshes: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
) -> tuple[list[MeshResult], ccol.CollisionSet]:
    n = min(pf.random.randint(rng, 1, 4), 2 * len(surface_meshes))
    rngs = rng.spawn(n)
    decorations = [table_decoration_object_rand(rngs[i]) for i in range(n)]
    placed_decorations = []
    for i in range(n):
        decoration = retry_place(
            rngs[i],
            decorations[i],
            colliders,
            snap_on_top,
            parents=surface_meshes,
        )
        placed_decorations.append(decoration)
    decorations, colliders = keep_non_colliding(placed_decorations, colliders)
    logger.info(f"Placed {len(decorations)} decoration objects out of {n} attempts")
    return decorations, colliders


def _furnished_room_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None,
    setups: list[tuple[object, float]],
) -> LivingroomResult:
    """Build the room shell, pick one furniture centerpiece from `setups`, add wall
    storage, lamps, surface decorations and scattered small objects. `setups` is a
    weighted list of setup entrypoints (e.g. sofa_setup_rand, dining_table_setup_rand),
    each called with (rng, wall_planes, room_dimensions, colliders)."""
    # rng lanes: 0 room shell, 1 furniture, 2 small objects
    rng_room, rng_furniture, rng_small = rng.spawn(3)

    rng_shape, rng_walls, rng_ceiling, rng_skirting = rng_room.spawn(4)
    shape = room_shape_rand(rng_shape, dimensions=dimensions)
    logger.info(f"Created room shape with {len(shape.flat_walls)} flat walls")
    wall_result = wall_feature_rand(rng_walls, shape)
    logger.info(
        f"Created wall features with {len(wall_result.wall_planes)} wall planes"
    )
    ceiling_result = ceiling_feature_rand(rng_ceiling, shape)
    logger.info("Created ceiling and floor features")
    skirt_objs = skirting_rand(
        rng_skirting, walls=wall_result.wall_planes + [shape.walls]
    )
    logger.info(f"Created {len(skirt_objs)} skirting objects")

    all_room = (
        _rename([shape.floor], "room_floor")
        + _rename([ceiling_result.ceiling], "room_ceiling")
        + _rename(wall_result.wall_planes, "room_wall")
        + _rename([shape.walls], "room_wall_corners")
        + _rename(skirt_objs, "room_skirting")
        + _rename(wall_result.backs + ceiling_result.backs, "room_wall_back")
        + _rename(wall_result.sills + ceiling_result.sills, "room_wall_sill")
        + _rename(ceiling_result.light_meshes, "ceiling_light")
    )
    for name, objs in wall_result.decorations.items():
        all_room += _rename(objs, name)

    lights = ceiling_result.lights + wall_result.lights
    windows = wall_result.decorations.get("window", [])
    storage_surfaces = wall_result.storage
    windowsills = wall_result.sills + ceiling_result.sills
    room_dimensions = shape.dimensions

    rng_sky, rng_setup, rng_lamp, rng_storage_place, rng_decor = rng_furniture.spawn(5)
    sky = sky_lighting.hosek_wilkie_sky_with_sun_lamp_rand(rng_sky)
    colliders = ccol.collision_set(
        wall_result.wall_planes
        + [shape.floor, shape.walls]
        + storage_surfaces
        + windows
    )

    setup_func = pf.control.choice(rng_setup, setups)
    setup = setup_func(
        rng_setup,
        wall_planes=wall_result.wall_planes,
        room_dimensions=room_dimensions,
        colliders=colliders,
    )
    sofas = list(getattr(setup, "sofas", []))
    coffee_tables = list(getattr(setup, "coffee_tables", []))
    side_tables = list(getattr(setup, "side_tables", []))
    dining_tables = list(getattr(setup, "dining_tables", []))
    dining_chairs = list(getattr(setup, "dining_chairs", []))
    solid = [
        r.mesh
        for r in sofas + coffee_tables + side_tables + dining_tables + dining_chairs
    ]
    colliders = ccol.collision_set(colliders.objs + solid, existing=colliders)

    floor_lamps, table_lamps, lamp_lights, colliders = sofa_lamps_rand(
        rng_lamp,
        [r.mesh for r in sofas],
        [r.mesh for r in side_tables],
        colliders,
    )

    storage_objects, colliders = _wall_storage_objects(
        rng_storage_place, wall_result.wall_planes, colliders
    )

    surface_results = dining_tables + coffee_tables + side_tables + storage_objects
    surface_meshes = [r.mesh for r in surface_results]
    decorations, colliders = _surface_decorations(rng_decor, surface_meshes, colliders)

    lights = lights + sky.lights + lamp_lights

    all_furniture = (
        _rename(list(getattr(setup, "rugs", [])), "rug")
        + _rename([r.mesh for r in sofas], "sofa")
        + _rename([r.mesh for r in storage_objects], "storage")
        + _rename([r.mesh for r in coffee_tables], "coffee_table")
        + _rename([r.mesh for r in side_tables], "side_table")
        + _rename([r.mesh for r in floor_lamps], "floor_lamp")
        + _rename([r.mesh for r in table_lamps], "table_lamp")
        + _rename([r.mesh for r in decorations], "decoration")
        + _rename([r.mesh for r in dining_tables], "dining_table")
        + _rename([r.mesh for r in dining_chairs], "dining_chair")
    )

    furnished_objects = all_room + all_furniture
    base_colliders = ccol.collision_set(furnished_objects, existing=colliders)
    wall_shelves = storage_surfaces

    rng_pool, rng_place = rng_small.spawn(2)
    pool = small_objects_collection_rand(rng_pool)
    (
        _rng_dining,
        _rng_coffee,
        rng_side,
        rng_storage,
        _rng_sofa,
        _rng_rug,
        rng_shelf,
        rng_sill,
    ) = rng_place.spawn(8)

    colliders = base_colliders
    if wall_shelves:
        colliders = ccol.collision_set(
            colliders.objs + wall_shelves, existing=colliders
        )
    small_objects: list = []

    placed, colliders = objects_scattered_on_surface(
        rng_side, [r.mesh for r in side_tables], pool, colliders
    )
    small_objects += placed
    logger.info(f"Placed {len(placed)} small objects on side tables")

    placed, colliders = objects_scattered_on_surface(
        rng_storage,
        [r.mesh for r in storage_objects],
        pool,
        colliders,
        skip_prob=1 / 3,
    )
    small_objects += placed
    logger.info(f"Placed {len(placed)} small objects on floor storage")

    placed, colliders = objects_scatter_rand(
        rng_shelf, wall_shelves, pool, colliders, skip_prob=0.0
    )
    small_objects += placed
    logger.info(f"Placed {len(placed)} small objects on wall shelves")

    placed, colliders = objects_scattered_on_surface(
        rng_sill, windowsills, pool, colliders, skip_prob=2 / 3
    )
    small_objects += placed
    logger.info(f"Placed {len(placed)} small objects on windowsills")

    all_objects = furnished_objects + small_objects

    return LivingroomResult(
        all_objects=all_objects,
        lights=lights,
        colliders=ccol.collision_set(all_objects, existing=base_colliders),
        floor=shape.floor,
        dimensions=room_dimensions,
    )


@pf.tracer.grammar
def livingroom_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    frame_start: int = 1,
    frame_end: int = 1,
) -> LivingroomResult:
    """A furnished living room whose centerpiece is a sofa grouping."""
    return _furnished_room_rand(rng, dimensions, [(sofa_setup_rand, 1.0)])


@pf.tracer.grammar
def room_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    frame_start: int = 1,
    frame_end: int = 1,
) -> LivingroomResult:
    """Everything livingroom_rand does, but the furniture centerpiece is a choice
    between a sofa grouping and a dining setup (table with chairs around it)."""
    return _furnished_room_rand(
        rng,
        dimensions,
        [(sofa_setup_rand, 1.0), (dining_table_setup_rand, 1.0)],
    )
