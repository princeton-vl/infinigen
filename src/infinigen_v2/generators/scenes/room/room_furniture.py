import logging
from typing import NamedTuple, Protocol, TypeVar, runtime_checkable

import numpy as np
import procfunc as pf

from infinigen_v2.generators.lighting import sky_lighting
from infinigen_v2.generators.objects import (
    bookcase,
    cabinet,
    desk,
    lamp,
    rug,
    sofa,
    table,
    triangle_shelf,
    vase,
    wall_art,
)
from infinigen_v2.generators.scenes import collision_collection as ccol
from infinigen_v2.generators.scenes.placement_utils import (
    repeat_attempts,
    snap_to_plane,
)
from infinigen_v2.util.instance import instanced_objects

logger = logging.getLogger(__name__)

MR = TypeVar("MR", bound="MeshResult")


@runtime_checkable
class MeshResult(Protocol):
    @property
    def mesh(self) -> pf.MeshObject: ...


class BareMeshResult(NamedTuple):
    mesh: pf.MeshObject


def _extract_lights(results: list[MeshResult]) -> list[pf.LightObject]:
    return [r.light for r in results if hasattr(r, "light") and r.light is not None]


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


@pf.tracer.grammar
def random_bbox_poses_animation_distribution(
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
def dining_table_setup_distribution(
    rng: pf.RNG,
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
) -> tuple[list[MeshResult], ccol.CollisionSet] | None:
    dims = table.table_dimensions_distribution(rng)  # TODO correlate with room dims

    dining_table = table.dining_table_distribution(rng, dimensions=dims)
    dining_table.mesh.item().name = table.dining_table_distribution.__name__

    def _place_dining_table(
        rng: pf.RNG,
        dining_table: table.TableResult,
        room_dimensions: pf.Vector,
        colliders: ccol.CollisionSet,
    ) -> tuple[list[MeshResult], ccol.CollisionSet] | None:
        position_frac = (
            pf.random.uniform(rng, 0.2, 0.8),
            pf.random.uniform(rng, 0.4, 0.6),
        )
        dining_table.mesh.item().location = (
            position_frac[0] * room_dimensions.x,
            position_frac[1] * room_dimensions.y,
            0.001,
        )
        if ccol.intersection_test(colliders, dining_table.mesh):
            # TODO - test for large distance around table, then continue onwards with chairs
            return None
        colliders = ccol.collision_set(
            colliders.objs + [dining_table.mesh], existing=colliders
        )
        return [dining_table], colliders

    return repeat_attempts(
        _place_dining_table,
        rng=rng,
        dining_table=dining_table,
        room_dimensions=room_dimensions,
        colliders=colliders,
    )


@pf.tracer.grammar
def table_decoration_object_distribution(
    rng: pf.RNG,
) -> MeshResult:
    func = pf.control.choice(
        rng,
        [
            (vase.vase_distribution, 1.0),
        ],
    )
    result = func(rng)
    result.mesh.item().name = func.__name__
    return result


@pf.tracer.grammar
def side_table_object_distribution(rng: pf.RNG) -> MeshResult:
    def triangle_shelf_sidetable_distribution(
        rng: pf.RNG,
    ) -> triangle_shelf.TriangleShelfResult:
        dimensions = table.side_table_dimensions_distribution(rng)
        return triangle_shelf.triangle_shelf_distribution(rng, dimensions=dimensions)

    func = pf.control.choice(
        rng,
        [
            (table.side_table_distribution, 1.0),
            (triangle_shelf_sidetable_distribution, 100.0),
        ],
    )
    result = func(rng)
    result.mesh.item().name = func.__name__
    return result


@pf.tracer.grammar
def storage_object_distribution(rng: pf.RNG) -> MeshResult:
    func = pf.control.choice(
        rng,
        [
            (bookcase.bookcase_distribution, 1.0),
            (cabinet.cabinet_distribution, 1.0),
        ],
    )
    result = func(rng)
    result.mesh.item().name = func.__name__
    return result


@pf.tracer.grammar
def room_art_distribution(rng: pf.RNG, room_dimensions: pf.Vector) -> MeshResult:
    height = pf.random.uniform(rng, 0.2, 0.8) * room_dimensions.z
    aspect = pf.random.uniform(rng, 0.5, 2.0)
    thickness = pf.random.uniform(rng, 0.01, 0.05)
    dimensions = pf.Vector((thickness, aspect * height, height))

    func = pf.control.choice(
        rng,
        [
            (wall_art.wall_art_distribution, 2.0),
            (wall_art.mirror_distribution, 0.0),
        ],
    )
    result = func(rng, dimensions=dimensions)
    result.mesh.item().name = func.__name__
    return result


def update_non_colliding(
    objs: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
) -> tuple[list[pf.MeshObject], ccol.CollisionSet]:
    result = []
    for i, obj in enumerate(objs):
        if ccol.intersection_test(colliders, obj):
            obj.item().name = obj.item().name + f"_{i}_COLLIDED"
            continue
        colliders = ccol.collision_set(colliders.objs + [obj], existing=colliders)
        result.append(obj)

    n_rejected = len(objs) - len(result)
    if objs:
        logger.info(
            f"update_non_colliding: {n_rejected}/{len(objs)} rejected ({100 * n_rejected // len(objs)}%)"
        )
    return result, colliders


def place_back_front(
    rng: pf.RNG,
    child: MR,
    parents: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
    placement: float | None = None,
    margin: float | None = None,
) -> tuple[MR, ccol.CollisionSet] | None:
    if placement is None:
        placement = pf.random.uniform(rng, 0.1, 0.9)
    if margin is None:
        margin = pf.random.clip_gaussian(rng, 0.15, 0.1, 0.1, 0.4)

    snap_to_plane(
        child=child.mesh,
        parent=rng.choice(parents),
        placement=placement,
        margin=margin,
        child_side="back",
        parent_side="front",
    )

    if ccol.intersection_test(colliders, child.mesh):
        return None
    colliders = ccol.collision_set(colliders.objs + [child.mesh], existing=colliders)
    return child, colliders


def place_side_by_side(
    rng,
    child: MR,
    parents: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
    sides: tuple[str, str] | None = None,
    placement: float | None = None,
    margin: float | None = None,
) -> tuple[MR, ccol.CollisionSet] | None:
    if sides is None:
        sides = pf.control.choice(
            rng, [(("left", "right"), 0.5), (("right", "left"), 0.5)]
        )
    if placement is None:
        placement = rng.uniform(0.05, 0.95)
    if margin is None:
        margin = pf.random.clip_gaussian(rng, 0.07, 0.06, 0.02, 0.25)
    snap_to_plane(
        child.mesh,
        parent=rng.choice(parents),
        placement=placement,
        margin=margin,
        child_side=sides[0],
        parent_side=sides[1],
    )
    if ccol.intersection_test(colliders, child.mesh):
        return None
    colliders = ccol.collision_set(colliders.objs + [child.mesh], existing=colliders)
    return child, colliders


def place_ontop_centered(
    rng: pf.RNG,
    child: MR,
    parents: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
    xy_frac: tuple[float, float] | None = None,
) -> tuple[MR, ccol.CollisionSet] | None:
    if xy_frac is None:
        xy_frac = (0.5, 0.5)
    parent = rng.choice(list(parents))
    bbox_min, bbox_max = pf.ops.attr.bbox_min_max(parent, global_coords=True)
    center_x = bbox_min[0] + (bbox_max[0] - bbox_min[0]) * xy_frac[0]
    center_y = bbox_min[1] + (bbox_max[1] - bbox_min[1]) * xy_frac[1]
    child.mesh.item().location = (center_x, center_y, bbox_max[2] + 0.002)
    if ccol.intersection_test(colliders, child.mesh):
        return None
    colliders = ccol.collision_set(colliders.objs + [child.mesh], existing=colliders)
    return child, colliders


def instance_and_collide(
    rng: pf.RNG,
    parent: pf.MeshObject,
    children: pf.Collection,
    colliders: ccol.CollisionSet,
    density: float | None = None,
    offset: tuple[float, float, float] = (0, 0, 0.002),
) -> tuple[list[pf.MeshObject], ccol.CollisionSet]:
    if density is None:
        density = pf.random.uniform(rng, 0.1, 0.3)
    density *= pf.random.uniform(rng, 0.1, 1.0)
    instances = instanced_objects(
        rng=rng,
        parent=parent,
        child=children,
        density=density,
        offset=offset,
    )
    instances, colliders = update_non_colliding(instances, colliders)
    return instances, colliders


def centered_from_col(
    rng: pf.RNG,
    parent: pf.MeshObject,
    children: pf.Collection,
    colliders: ccol.CollisionSet,
) -> tuple[list[pf.MeshObject], ccol.CollisionSet] | None:
    children_list = list(children)
    if not children_list:
        logger.warning("centered_from_col: empty collection, skipping")
        return None
    child = BareMeshResult(mesh=pf.ops.object.alias(rng.choice(children_list)))
    res = place_ontop_centered(
        rng=rng,
        child=child,
        parents=[parent],
        colliders=colliders,
    )
    if res is not None:
        return [res[0].mesh], res[1]
    return None


def _floor_ceil_margin_height(
    rng: pf.RNG,
    obj: pf.MeshObject,
    room_dimensions: pf.Vector,
    floor_ceil_margin: float = 0.1,
) -> float:
    bmin, bmax = pf.ops.attr.bbox_min_max(obj)
    hmin = floor_ceil_margin - bmin[2]
    hmax = room_dimensions.z - bmax[2] - floor_ceil_margin
    return pf.random.uniform(rng, min(hmin, hmax), max(hmin, hmax))


# ruff: noqa: C901
@pf.tracer.grammar
def room_furniture_distribution(
    rng: pf.RNG,
    dimensions: pf.Vector,
    wall_planes: list[pf.MeshObject],
    floor: pf.MeshObject,
    frame_start: int = 1,
    frame_end: int = 1,
) -> RoomFurnitureResult:
    (
        rng_sky,
        rng_big,
        rng_decoration_object,
        rng_table,
        rng_lamp,
        rng_table_lamp,
        rng_dining_table,
        rng_wall_object,
        rng_plants,
        rng_rug,
    ) = rng.spawn(10)

    room_dimensions = dimensions

    sun_elevation_deg = pf.random.uniform(rng_sky, 5, 80)
    _sky_shader = sky_lighting.nishita_sky_distribution(
        rng_sky, sun_elevation_deg=sun_elevation_deg
    )

    colliders = ccol.collision_set(wall_planes + [floor])
    logger.info(
        f"Created collision set with {len(colliders.mesh_fcl_colliders)} underlying BVHs"
    )

    n_sofas = pf.random.randint(rng_big, 0, 8)
    sofa_objs: list[MeshResult] = []
    for _i in range(n_sofas):
        child = sofa.sofa_distribution(rng_big)
        child.mesh.item().name = sofa.sofa_distribution.__name__
        res = repeat_attempts(
            place_back_front,
            rng=rng_big,
            child=child,
            parents=wall_planes,
            colliders=colliders,
        )
        if res is not None:
            sofa_objs.append(res[0])
            colliders = res[1]
    logger.info(f"Placed {len(sofa_objs)} sofas out of {n_sofas} attempts")

    sofa_meshes = [r.mesh for r in sofa_objs]

    n_coffee_tables = min(pf.random.randint(rng_table, 0, 2), len(sofa_objs))
    coffee_tables: list[MeshResult] = []
    for _i in range(n_coffee_tables):
        child = table.coffee_table_distribution(rng_table)
        child.mesh.item().name = table.coffee_table_distribution.__name__
        res = repeat_attempts(
            place_back_front,
            rng=rng_table,
            child=child,
            parents=sofa_meshes,
            colliders=colliders,
            margin=pf.random.clip_gaussian(rng_table, 0.25, 0.2, 0.15, 0.75),
        )
        if res is not None:
            coffee_tables.append(res[0])
            colliders = res[1]
    logger.info(
        f"Placed {len(coffee_tables)} coffee tables out of {n_coffee_tables} attempts"
    )

    # SIDETABLES BY BIG OBJECTS
    n_side_tables = min(pf.random.randint(rng_table, 0, 6), len(sofa_objs))
    side_tables: list[MeshResult] = []
    for _i in range(n_side_tables):
        child = table.side_table_distribution(rng_table)
        child.mesh.item().name = table.side_table_distribution.__name__
        res = repeat_attempts(
            place_side_by_side,
            rng=rng_table,
            child=child,
            parents=sofa_meshes,
            colliders=colliders,
        )
        if res is not None:
            side_tables.append(res[0])
            colliders = res[1]
    logger.info(
        f"Placed {len(side_tables)} side tables out of {n_side_tables} attempts"
    )

    n_storage_objects = pf.random.randint(rng_big, 4, 10)
    storage_objects: list[MeshResult] = []
    for _i in range(n_storage_objects):
        res = repeat_attempts(
            place_back_front,
            rng=rng_big,
            child=storage_object_distribution(rng_big),
            parents=wall_planes,
            colliders=colliders,
            margin=0.02,
        )
        if res is not None:
            storage_objects.append(res[0])
            colliders = res[1]
    logger.info(
        f"Placed {len(storage_objects)} storage objects out of {n_storage_objects} attempts"
    )

    # FLOOR LAMPS BY BIG OBJECTS
    big_meshes = sofa_meshes + [r.mesh for r in storage_objects]

    floor_lamps: list[MeshResult] = []
    n_floor_lamps = min(pf.random.randint(rng_lamp, 0, 2), len(big_meshes))
    for _i in range(n_floor_lamps):
        height = pf.random.uniform(rng_lamp, 1.0, 2.0)
        child = lamp.lamp_distribution(rng_lamp, height)
        child.mesh.item().name = lamp.lamp_distribution.__name__
        res = repeat_attempts(
            place_side_by_side,
            rng=rng_lamp,
            child=child,
            parents=big_meshes,
            colliders=colliders,
        )
        if res is not None:
            floor_lamps.append(res[0])
            colliders = res[1]
    logger.info(
        f"Placed {len(floor_lamps)} floor lamps out of {n_floor_lamps} attempts"
    )
    floor_lamp_lights = _extract_lights(floor_lamps)
    lights: list[pf.LightObject] = []
    lights += pf.control.choice(rng_lamp, [(floor_lamp_lights, 0.5), ([], 0.5)])

    # LAMPS AND STUFF GO ON SIDETABLES
    surface_meshes = [r.mesh for r in side_tables + storage_objects]
    n_decoration_objs = min(
        pf.random.randint(rng_decoration_object, 0, 4),
        len(surface_meshes),
    )
    decoration_objs: list[MeshResult] = []
    for _i in range(n_decoration_objs):
        child = table_decoration_object_distribution(rng_decoration_object)
        res = repeat_attempts(
            place_ontop_centered,
            rng=rng_decoration_object,
            child=child,
            parents=surface_meshes,
            colliders=colliders,
        )
        if res is not None:
            decoration_objs.append(res[0])
            colliders = res[1]
    logger.info(
        f"Placed {len(decoration_objs)} decoration objects out of {n_decoration_objs} attempts"
    )

    # TABLE LAMPS ON SIDETABLES/STORAGE
    n_table_lamps = min(pf.random.randint(rng_table_lamp, 0, 2), len(surface_meshes))
    table_lamps: list[MeshResult] = []
    for _i in range(n_table_lamps):
        child = lamp.desk_lamp_distribution(rng_table_lamp)
        child.mesh.item().name = lamp.desk_lamp_distribution.__name__
        res = repeat_attempts(
            place_ontop_centered,
            rng=rng_table_lamp,
            child=child,
            parents=surface_meshes,
            colliders=colliders,
        )
        if res is not None:
            table_lamps.append(res[0])
            colliders = res[1]
    logger.info(
        f"Placed {len(table_lamps)} table lamps out of {n_table_lamps} attempts"
    )
    table_lamp_lights = _extract_lights(table_lamps)
    lights += pf.control.choice(rng_table_lamp, [(table_lamp_lights, 0.5), ([], 0.5)])

    n_wall_objects = pf.random.randint(rng_wall_object, 0, 5)
    wall_objects: list[MeshResult] = []
    for _i in range(n_wall_objects):
        child = room_art_distribution(rng_wall_object, room_dimensions=room_dimensions)
        child.mesh.item().location.z = _floor_ceil_margin_height(
            rng_wall_object, child.mesh, room_dimensions, 0.1
        )
        res = repeat_attempts(
            place_back_front,
            rng=rng_wall_object,
            child=child,
            parents=wall_planes,
            colliders=colliders,
            margin=0.02,
        )
        if res is not None:
            wall_objects.append(res[0])
            colliders = res[1]
    logger.info(
        f"Placed {len(wall_objects)} wall objects out of {n_wall_objects} attempts"
    )

    def desk_against_wall_distribution(
        rng: pf.RNG,
        room_dimensions: pf.Vector,
        colliders: ccol.CollisionSet,
    ) -> tuple[list[MeshResult], ccol.CollisionSet] | None:
        del room_dimensions
        child = desk.desk_distribution(rng)
        child.mesh.item().name = desk.desk_distribution.__name__
        res = repeat_attempts(
            place_back_front,
            rng=rng,
            child=child,
            parents=wall_planes,
            colliders=colliders,
            margin=0.05,
        )
        if res is None:
            return None
        return [res[0]], res[1]

    diningtable_func = pf.control.choice(
        rng_dining_table,
        [
            (dining_table_setup_distribution, 2),
            (desk_against_wall_distribution, 1),
            (lambda *_, **__: None, 3),
        ],
    )
    diningtable_res = diningtable_func(
        rng_dining_table,
        room_dimensions=room_dimensions,
        colliders=colliders,
    )
    if diningtable_res is not None:
        logger.info(f"Placed {len(diningtable_res[0])} dining tables")
        diningtable_objs = diningtable_res[0]
        colliders = diningtable_res[1]
    else:
        diningtable_objs = []

    def _placed_rug(
        rng: pf.RNG,
        room_floor: pf.MeshObject,
        colliders: ccol.CollisionSet,
    ) -> list[pf.MeshObject]:
        rug_result = rug.rug_distribution(rng)
        rug_result.mesh.item().name = rug.rug_distribution.__name__
        xy_frac = (
            pf.random.uniform(rng, 1 / 3, 2 / 3),
            pf.random.uniform(rng, 1 / 3, 2 / 3),
        )
        place_ontop_centered(
            rng=rng,
            child=rug_result,
            parents=[room_floor],
            colliders=colliders,
            xy_frac=xy_frac,
        )
        return [rug_result.mesh]

    rug_func = pf.control.choice(
        rng_rug,
        [
            (_placed_rug, 1.0),
            (lambda *_, **__: [], 1.0),
        ],
    )
    rug_objs = rug_func(
        rng_rug,
        room_floor=floor,
        colliders=colliders,
    )
    logger.info(f"Placed {len(rug_objs)} rugs")

    placed_results: dict[str, list[MeshResult]] = {
        "sofa": sofa_objs,
        "storage": storage_objects,
        "coffee_table": coffee_tables,
        "side_table": side_tables,
        "floor_lamp": floor_lamps,
        "table_lamp": table_lamps,
        "decoration": decoration_objs,
        "wall_art": wall_objects,
        "dining_table": diningtable_objs,
    }
    furniture_named: dict[str, list[pf.MeshObject]] = {
        "rug": rug_objs,
    }
    for name, results in placed_results.items():
        furniture_named[name] = [r.mesh for r in results]

    for _name, objs in furniture_named.items():
        for _i, obj in enumerate(objs):
            for j, slot in enumerate(obj.item().material_slots):
                if slot.material is not None:
                    slot.material.name = f"{obj.item().name}_{j}"

    all_furniture = [obj for objs in furniture_named.values() for obj in objs]

    storage_surfaces = (
        [r.mesh for r in diningtable_objs]
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
        diningtable_objs=[r.mesh for r in diningtable_objs],
    )
