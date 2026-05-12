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
    room_small_objects_distribution,
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

    skirt_objs = skirting_distribution(rng_skirting, shape)
    logger.info(f"Created {len(skirt_objs)} skirting objects")

    result_categories = {
        "room_floor": [shape.floor],
        "room_ceiling": [shape.ceiling],
        "room_wall": wall_result.wall_planes,
        "room_wall_corners": [shape.walls],
        "room_skirting": skirt_objs,
        "room_wall_back": wall_result.extras + [ceiling_result.extras[0]],
        "ceiling_light": ceiling_result.extras[1:],
    }
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
        storage_surfaces=[],
        dimensions=shape.dimensions,
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

    wall_planes = [
        o for o in result.all_objects if o.item().name.startswith("room_wall.")
    ]
    furniture = room_furniture_distribution(
        rng_furniture,
        dimensions=result.dimensions,
        wall_planes=wall_planes,
        floor=result.floor,
        frame_start=frame_start,
        frame_end=frame_end,
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
        storage_surfaces=furniture.storage_surfaces,
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
    small_objects = room_small_objects_distribution(
        rng_small,
        storage_surfaces=result.storage_surfaces,
        floor=result.floor,
        colliders=result.colliders,
    )

    all_objects = result.all_objects + small_objects
    return result._replace(
        all_objects=all_objects,
        colliders=ccol.collision_set(all_objects, existing=result.colliders),
    )


room_with_all_objects = livingroom_with_smallobj_distribution
