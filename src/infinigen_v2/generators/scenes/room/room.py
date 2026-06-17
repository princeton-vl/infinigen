import logging
from typing import NamedTuple

import procfunc as pf

from infinigen_v2.generators.scenes import collision_collection as ccol
from infinigen_v2.generators.scenes.room.room_furniture import (
    room_furniture_distribution,
)
from infinigen_v2.generators.scenes.room.room_shape import (
    room_shape_distribution,
)
from infinigen_v2.generators.scenes.room.room_small_objects import (
    objects_scatter_distribution,
    objects_scattered_on_surface,
    small_objects_collection_distribution,
)
from infinigen_v2.generators.scenes.room.room_surface_features import (
    ceiling_feature_distribution,
    skirting_distribution,
    wall_feature_distribution,
)

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


@pf.tracer.grammar
def livingroom_nofurniture_distribution(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
) -> LivingroomResult:
    rng_shape, rng_walls, rng_ceiling, rng_skirting = rng.spawn(4)

    shape = room_shape_distribution(rng_shape, dimensions=dimensions)
    logger.info(f"Created room shape with {len(shape.flat_walls)} flat walls")

    wall_result = wall_feature_distribution(rng_walls, shape)
    logger.info(
        f"Created wall features with {len(wall_result.wall_planes)} wall planes"
    )

    ceiling_result = ceiling_feature_distribution(rng_ceiling, shape)
    logger.info("Created ceiling and floor features")

    skirt_objs = skirting_distribution(
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


@pf.tracer.grammar
def livingroom_distribution(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    frame_start: int = 1,
    frame_end: int = 1,
) -> LivingroomResult:
    rng_room, rng_furniture = rng.spawn(2)

    result = livingroom_nofurniture_distribution(rng_room, dimensions=dimensions)

    furniture = room_furniture_distribution(
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


@pf.tracer.grammar
def livingroom_with_smallobj_distribution(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    frame_start: int = 1,
    frame_end: int = 1,
) -> LivingroomResult:
    rng_room, rng_small = rng.spawn(2)

    result = livingroom_distribution(
        rng_room, dimensions=dimensions, frame_start=frame_start, frame_end=frame_end
    )

    rng_pool, rng_place = rng_small.spawn(2)
    pool = small_objects_collection_distribution(rng_pool)
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

    placed, colliders = objects_scattered_on_surface(
        rng_coffee, result.coffee_tables, pool, colliders, skip_prob=2 / 3
    )
    small_objects += placed
    logger.info(f"Placed {len(placed)} small objects on coffee tables")

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

    placed, colliders = objects_scatter_distribution(
        rng_shelf,
        result.wall_shelves or [],
        pool,
        colliders,
        skip_prob=1 / 3,
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


room_with_all_objects = livingroom_with_smallobj_distribution
