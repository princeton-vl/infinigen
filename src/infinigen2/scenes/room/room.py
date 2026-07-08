# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging
from typing import NamedTuple

import procfunc as pf

from infinigen2.scenes import collision_collection as ccol
from infinigen2.scenes.room.room_furniture import (
    room_furniture_rand,
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
]

logger = logging.getLogger(__name__)


class LivingroomResult(NamedTuple):
    all_objects: list
    lights: list
    colliders: ccol.CollisionSet
    floor: pf.MeshObject
    dimensions: pf.Vector | None = None


@pf.tracer.grammar
def livingroom_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    frame_start: int = 1,
    frame_end: int = 1,
) -> LivingroomResult:
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

    result_categories = {
        "room_floor": [shape.floor],
        "room_ceiling": [ceiling_result.ceiling],
        "room_wall": wall_result.wall_planes,
        "room_wall_corners": [shape.walls],
        "room_skirting": skirt_objs,
        "room_wall_back": wall_result.backs + ceiling_result.backs,
        "room_wall_sill": wall_result.sills + ceiling_result.sills,
        "ceiling_light": ceiling_result.light_meshes,
    }
    result_categories.update(wall_result.decorations)
    for name, objs in result_categories.items():
        for i, obj in enumerate(objs):
            obj.item().name = f"{name}.{i:02d}"
    all_room_objects = [obj for objs in result_categories.values() for obj in objs]
    lights = ceiling_result.lights + wall_result.lights

    windows = wall_result.decorations.get("window", [])
    storage_surfaces = wall_result.storage
    windowsills = wall_result.sills + ceiling_result.sills
    room_dimensions = shape.dimensions

    furniture = room_furniture_rand(
        rng_furniture,
        dimensions=room_dimensions,
        wall_planes=wall_result.wall_planes,
        floor=shape.floor,
        frame_start=frame_start,
        frame_end=frame_end,
        extra_colliders=storage_surfaces + windows,
        wall_storage=storage_surfaces,
    )

    furnished_objects = all_room_objects + furniture.furniture
    base_colliders = ccol.collision_set(furnished_objects, existing=furniture.colliders)
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
        rng_side, furniture.side_tables, pool, colliders
    )
    small_objects += placed
    logger.info(f"Placed {len(placed)} small objects on side tables")

    placed, colliders = objects_scattered_on_surface(
        rng_storage, furniture.storage_objects, pool, colliders, skip_prob=1 / 3
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
        lights=lights + furniture.lights,
        colliders=ccol.collision_set(all_objects, existing=base_colliders),
        floor=shape.floor,
        dimensions=room_dimensions,
    )
