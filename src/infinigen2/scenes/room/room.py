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
    "livingroom_nofurniture_rand",
    "livingroom_rand",
    "livingroom_with_smallobj_rand",
]

logger = logging.getLogger(__name__)


class LivingroomResult(NamedTuple):
    all_objects: list
    cameras: list
    lights: list
    colliders: ccol.CollisionSet
    storage_objects: list
    coffee_tables: list
    side_tables: list
    diningtable_objs: list
    floor: pf.MeshObject
    storage_surfaces: list[pf.MeshObject]
    dimensions: pf.Vector | None = None
    windows: list[pf.MeshObject] | None = None
    wall_planes: list[pf.MeshObject] | None = None
    windowsills: list[pf.MeshObject] | None = None
    wall_shelves: list[pf.MeshObject] | None = None
    sofas: list[pf.MeshObject] | None = None
    rugs: list[pf.MeshObject] | None = None


def _livingroom_nofurniture_rand_impl(
    rng_room: pf.RNG,
    dimensions: pf.Vector | None = None,
) -> LivingroomResult:
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

    return LivingroomResult(
        all_objects=all_room_objects,
        cameras=[],
        lights=lights,
        colliders=ccol.collision_set(all_room_objects),
        storage_objects=[],
        coffee_tables=[],
        side_tables=[],
        diningtable_objs=[],
        floor=shape.floor,
        storage_surfaces=wall_result.storage,
        dimensions=shape.dimensions,
        windows=wall_result.decorations.get("window", []),
        wall_planes=wall_result.wall_planes,
        windowsills=wall_result.sills + ceiling_result.sills,
    )


def _livingroom_rand_impl(
    result: LivingroomResult,
    rng_furniture: pf.RNG,
    frame_start: int,
    frame_end: int,
) -> LivingroomResult:
    furniture = room_furniture_rand(
        rng_furniture,
        dimensions=result.dimensions,
        wall_planes=result.wall_planes,
        floor=result.floor,
        frame_start=frame_start,
        frame_end=frame_end,
        extra_colliders=result.storage_surfaces + (result.windows or []),
        wall_storage=result.storage_surfaces,
    )

    all_objects = result.all_objects + furniture.furniture

    return LivingroomResult(
        all_objects=all_objects,
        cameras=[],
        lights=result.lights + furniture.lights,
        colliders=ccol.collision_set(all_objects, existing=furniture.colliders),
        storage_objects=furniture.storage_objects,
        coffee_tables=furniture.coffee_tables,
        side_tables=furniture.side_tables,
        diningtable_objs=furniture.diningtable_objs,
        floor=result.floor,
        dimensions=result.dimensions,
        storage_surfaces=furniture.storage_surfaces + result.storage_surfaces,
        wall_planes=result.wall_planes,
        windowsills=result.windowsills or [],
        wall_shelves=result.storage_surfaces,
        sofas=furniture.sofas,
        rugs=furniture.rugs,
    )


def _livingroom_with_smallobj_rand_impl(
    result: LivingroomResult,
    rng_small: pf.RNG,
) -> LivingroomResult:
    rng_pool, rng_place = rng_small.spawn(2)
    pool = small_objects_collection_rand(rng_pool)
    (
        rng_dining,
        rng_coffee,
        rng_side,
        rng_storage,
        rng_sofa,
        rng_rug,
        rng_shelf,
        rng_sill,
    ) = rng_place.spawn(8)
    colliders = result.colliders
    wall_cabinets = result.wall_shelves or []
    if wall_cabinets:
        colliders = ccol.collision_set(
            colliders.objs + wall_cabinets, existing=colliders
        )
    small_objects: list = []

    """
    placed, colliders = objects_scattered_on_surface(
        rng_coffee, result.coffee_tables, pool, colliders, skip_prob=2 / 3
    )
    small_objects += placed
    logger.info(f"Placed {len(placed)} small objects on coffee tables")
    """

    placed, colliders = objects_scattered_on_surface(
        rng_side, result.side_tables, pool, colliders
    )
    small_objects += placed
    logger.info(f"Placed {len(placed)} small objects on side tables")

    placed, colliders = objects_scattered_on_surface(
        rng_storage, result.storage_objects, pool, colliders, skip_prob=1 / 3
    )
    small_objects += placed
    logger.info(f"Placed {len(placed)} small objects on floor storage")

    """
    placed, colliders = pf.control.choice(
        rng_sofa,
        [
            (
                lambda: objects_scattered_on_surface(
                    rng_sofa, result.sofas or [], pool, colliders
                ),
                1.0,
            ),
            (lambda: ([], colliders), 4.0),
        ],
    )()
    small_objects += placed

    placed, colliders = pf.control.choice(
        rng_rug,
        [
            (
                lambda: objects_scattered_on_surface(
                    rng_rug, result.rugs or [], pool, colliders
                ),
                1.0,
            ),
            (lambda: ([], colliders), 5.0),
        ],
    )()
    small_objects += placed

    placed, colliders = objects_scattered_on_surface(
        rng_dining, result.diningtable_objs, pool, colliders, skip_prob=0.5
    )
    small_objects += placed
    """

    placed, colliders = objects_scatter_rand(
        rng_shelf,
        result.wall_shelves or [],
        pool,
        colliders,
        skip_prob=0.0,
    )
    small_objects += placed
    logger.info(f"Placed {len(placed)} small objects on wall shelves")

    placed, colliders = objects_scattered_on_surface(
        rng_sill, result.windowsills or [], pool, colliders, skip_prob=2 / 3
    )
    small_objects += placed
    logger.info(f"Placed {len(placed)} small objects on windowsills")

    all_objects = result.all_objects + small_objects
    return result._replace(
        all_objects=all_objects,
        colliders=ccol.collision_set(all_objects, existing=result.colliders),
    )


# Each variant splits rng the same way (lane 0 room, 1 furniture, 2 small objects)
# and never nests the others, so spawn(3)[0] gives every variant the same room.
@pf.tracer.grammar
def livingroom_nofurniture_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
) -> LivingroomResult:
    rng_room, _rng_furniture, _rng_small = rng.spawn(3)
    return _livingroom_nofurniture_rand_impl(rng_room, dimensions=dimensions)


@pf.tracer.grammar
def livingroom_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    frame_start: int = 1,
    frame_end: int = 1,
) -> LivingroomResult:
    rng_room, rng_furniture, _rng_small = rng.spawn(3)
    result = _livingroom_nofurniture_rand_impl(rng_room, dimensions=dimensions)
    return _livingroom_rand_impl(result, rng_furniture, frame_start, frame_end)


@pf.tracer.grammar
def livingroom_with_smallobj_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    frame_start: int = 1,
    frame_end: int = 1,
) -> LivingroomResult:
    rng_room, rng_furniture, rng_small = rng.spawn(3)
    result = _livingroom_nofurniture_rand_impl(rng_room, dimensions=dimensions)
    result = _livingroom_rand_impl(result, rng_furniture, frame_start, frame_end)
    return _livingroom_with_smallobj_rand_impl(result, rng_small)


room_with_all_objects = livingroom_with_smallobj_rand
