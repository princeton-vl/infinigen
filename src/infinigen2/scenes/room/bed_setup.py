# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import math
from collections.abc import Sequence
from typing import NamedTuple, cast

import procfunc as pf

from infinigen2.objects import cushion, desk, lamp, storage, table
from infinigen2.objects.bed import BedResult, bed_rand
from infinigen2.scenes.placement import collision
from infinigen2.scenes.placement.culling import keep_non_colliding
from infinigen2.scenes.placement.distribute import instances_along_line
from infinigen2.scenes.placement.snap import snap_to_plane
from infinigen2.scenes.setup_utils import (
    BareMeshResult,
    MeshResult,
    clear_of,
    retry_place,
    snap_back_front,
)
from infinigen2.util.errors import RejectedScene
from infinigen2.util.mesh import center_footprint
from infinigen2.util.scene_cleanup import delete_object

__all__ = [
    "BedSetupResult",
    "bed_dimensions_rand",
    "bed_setup_multi_rand",
    "bed_setup_rand",
    "table_bedside_composite_rand",
]


class BedSetupResult(NamedTuple):
    bed: pf.MeshObject
    mattress: pf.MeshObject | None
    bedside_tables: list[pf.MeshObject]
    bedside_lamps: list[pf.MeshObject]
    pillows: list[pf.MeshObject]
    lights: list[pf.LightObject]
    all_objects: list[pf.MeshObject]
    colliders: collision.CollisionSet
    storage_containers: list[pf.MeshObject]
    supports: list[pf.MeshObject]
    storages: list[pf.MeshObject]


def table_bedside_composite_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
) -> table.TableResult:
    """Bedside-sized storage, regular table, or desk with under-top storage."""
    rng_depth, rng_width, rng_height, rng_style, rng_body = rng.spawn(5)
    if dimensions is None:
        dimensions = pf.Vector(
            (
                pf.random.uniform(rng_depth, 0.35, 0.48),
                pf.random.uniform(rng_width, 0.40, 0.47),
                pf.random.uniform(rng_height, 0.55, 0.70),
            )
        )
    body = pf.control.choice(
        rng_style,
        [
            (storage.storage_composite_rand, 1.0),
            (table.table_dining_rand, 1.0),
            (desk.desk_with_top_storage_rand, 1.0),
        ],
    )
    result = body(rng_body, dimensions=dimensions)
    center_footprint(result.mesh)
    return table.TableResult(mesh=result.mesh)


def bed_dimensions_rand(
    rng: pf.RNG,
    bbox_min: pf.Vector | None = None,
    bbox_max: pf.Vector | None = None,
    area: float | None = None,
    aspect: float | None = None,
) -> pf.Vector:
    rng_room, rng_size, rng_thickness = rng.spawn(3)
    if bbox_min is None or bbox_max is None:
        room_size = pf.random.uniform(rng_room, 0.0, 1.0)
    else:
        room_area = (bbox_max.x - bbox_min.x) * (bbox_max.y - bbox_min.y)
        room_size = min(max((room_area - 11.0) / 11.0, 0.0), 1.0)
    size = 0.5 * room_size + pf.random.uniform(rng_size, 0.0, 0.5)
    nominal_width = 0.97 + 1.06 * size
    nominal_length = 1.91 + 0.58 * size**2
    if area is None:
        area = nominal_width * nominal_length
    if aspect is None:
        aspect = nominal_length / nominal_width
    length = math.sqrt(area * aspect)
    width = max(0.6, min(math.sqrt(area / aspect), 2.1))
    if bbox_min is not None and bbox_max is not None:
        minimum_span = min(bbox_max.x - bbox_min.x, bbox_max.y - bbox_min.y)
        length = min(length, max(1.2, minimum_span - 0.6))
        width = max(0.6, min(width, minimum_span - 0.9))
    thickness = pf.random.uniform(rng_thickness, 0.20, 0.24)
    return pf.Vector((length, width, thickness))


def _place_bed_against_wall(
    rng: pf.RNG,
    bed_result: MeshResult,
    wall_planes: list[pf.MeshObject],
    colliders: collision.CollisionSet,
    furniture: Sequence[pf.MeshObject],
) -> MeshResult | None:
    margin = pf.random.uniform(rng, 0.05, 0.2)
    clearance = pf.random.uniform(rng, 0.5, 0.9)

    def accept(mesh: pf.MeshObject) -> bool:
        return clear_of(mesh, furniture, clearance)

    return retry_place(
        rng,
        bed_result,
        colliders,
        _snap_bed_against_wall,
        attempts=64,
        accept_fn=accept,
        parents=wall_planes,
        margin=margin,
    )


def _snap_bed_against_wall(
    rng: pf.RNG,
    child: MeshResult,
    parents: list[pf.MeshObject],
    margin: float,
) -> None:
    snap_back_front(
        rng,
        child,
        parents,
        placement=pf.random.uniform(rng, 0.0, 1.0),
        margin=margin,
    )


def _delete_bed_result(result: BedResult) -> None:
    delete_object(result.mattress_child.item())
    delete_object(result.mesh.item())


def _generated_bed_rand(
    rng_bed: pf.RNG,
    rng_place: pf.RNG,
    dimensions: pf.Vector | None,
    wall_planes: list[pf.MeshObject] | None,
    colliders: collision.CollisionSet,
    furniture: Sequence[pf.MeshObject],
) -> BedResult:
    result = bed_rand(rng_bed, dimensions=dimensions)
    if not wall_planes:
        return result
    placed = _place_bed_against_wall(
        rng_place, result, wall_planes, colliders, furniture
    )
    if placed is not None:
        return result
    _delete_bed_result(result)
    raise RejectedScene("Could not place bed against a wall")


def _bed_pillows_rand(
    rng: pf.RNG, mattress: pf.MeshObject, dimensions: pf.Vector
) -> list[pf.MeshObject]:
    rng, rng_pillow = rng.spawn(2)
    max_count = max(1, int(dimensions.y / 0.6))
    count = pf.random.randint(rng, max(1, max_count - 1), max_count + 1)
    gap = pf.random.uniform(rng, 0.0, 0.04)
    long_side = min(pf.random.uniform(rng, 0.66, 0.92), dimensions.y / count - gap)
    short_side = pf.random.uniform(rng, 0.48, 0.53)
    loft = pf.random.uniform(rng, 0.15, 0.22)
    pillow = cushion.cushion_bed_pillow_rand(
        rng_pillow, size=pf.Vector((short_side, long_side, loft))
    )
    x = -dimensions.x * 0.5 + short_side * 0.5 + 0.04
    z = dimensions.z * 0.5 - loft * 0.05
    half_span = count * (long_side + gap) * 0.5
    pillows = instances_along_line(
        pillow.mesh, mattress, (x, -half_span, z), (x, half_span, z), count
    )
    return pillows


def bed_setup_rand(
    rng: pf.RNG,
    bed: pf.MeshObject | None = None,
    mattress: pf.MeshObject | None = None,
    wall_planes: list[pf.MeshObject] | None = None,
    bbox_min: pf.Vector | None = None,
    bbox_max: pf.Vector | None = None,
    colliders: collision.CollisionSet | None = None,
    bed_dimensions: pf.Vector | None = None,
    furniture: Sequence[pf.MeshObject] | None = None,
) -> BedSetupResult:
    (
        r_dimensions,
        r_bed,
        r_bed_place,
        r_table,
        r_table_placement,
        r_table_choice,
        r_lamp,
        r_lamp_choices,
        r_pillows,
    ) = rng.spawn(9)
    if furniture is None:
        furniture = []
    dimensions = bed_dimensions
    if dimensions is None:
        dimensions = bed_dimensions_rand(r_dimensions, bbox_min, bbox_max)
    generated_bed = bed is None
    if colliders is None:
        wall_objects = cast(list[pf.Object], wall_planes or [])
        colliders = collision.collision_set(wall_objects)
    if bed is None:
        bed_result = _generated_bed_rand(
            r_bed,
            r_bed_place,
            dimensions,
            wall_planes,
            colliders,
            furniture,
        )
        bed_mesh = bed_result.mesh
        mattress = bed_result.mattress_child
    else:
        bed_mesh = bed
        external = [
            obj
            for obj in colliders.objs
            if obj.item() is not bed_mesh.item()
            and (mattress is None or obj.item() is not mattress.item())
        ]
        colliders = collision.collision_set(external, cache=colliders)
        original_matrix = bed_mesh.item().matrix_world.copy()
        original_name = bed_mesh.item().name
        placement_result = BareMeshResult(bed_mesh)
        placed: MeshResult | None = placement_result
        if wall_planes:
            placed = _place_bed_against_wall(
                r_bed_place,
                placement_result,
                wall_planes,
                colliders,
                furniture,
            )
        if placed is None:
            bed_mesh.item().matrix_world = original_matrix
            bed_mesh.item().name = original_name
            raise RejectedScene("Could not place bed against a wall")
        bed_mesh = placed.mesh
    objects = []
    objects.append(bed_mesh)
    if mattress is not None:
        objects.append(mattress)
    anchor_objects = list(objects)
    if any(collision.intersection_test(colliders, obj) for obj in anchor_objects):
        raise RejectedScene("Bed collides at its selected placement")

    side_table = table_bedside_composite_rand(r_table).mesh
    table_candidates = []
    for (parent_side, child_side), r_gap in zip(
        [("left", "right"), ("right", "left")],
        r_table_placement.spawn(2),
        strict=True,
    ):
        table_candidates.append(
            snap_to_plane(
                child=pf.ops.object.alias(side_table),
                parent=bed_mesh,
                placement=0.0,
                child_side=child_side,
                parent_side=parent_side,
                margin=pf.random.uniform(r_gap, 0.05, 0.10),
            )
        )
    left, right = table_candidates
    bedside_tables = pf.control.choice(
        r_table_choice,
        [
            ([], 0.09),
            ([left], 0.21),
            ([right], 0.21),
            ([left, right], 0.49),
        ],
    )
    objects += bedside_tables

    tmin, tmax = pf.ops.attr.bbox_min_max(side_table, global_coords=False)
    table_size = pf.Vector(tmax) - pf.Vector(tmin)
    lamp_template = lamp.lamp_desk_rand(
        r_lamp,
        base_radius=0.125 * min(table_size[0], table_size[1]),
    )
    setup_colliders = collision.collision_set(
        colliders.objs + anchor_objects,
        cache=colliders,
    )
    kept_tables, setup_colliders = keep_non_colliding(
        bedside_tables,
        setup_colliders,
        key=lambda obj: obj,
    )
    kept = {table_mesh.item() for table_mesh in kept_tables}
    lamps: list[lamp.LampResult] = []
    for table_mesh, lamp_rng in zip(
        bedside_tables,
        r_lamp_choices.spawn(len(bedside_tables)),
        strict=True,
    ):
        if not pf.control.choice(lamp_rng, [(True, 0.5), (False, 0.5)]):
            continue
        if table_mesh.item() not in kept:
            continue
        lamp_mesh = pf.ops.object.alias(lamp_template.mesh)
        light_obj = pf.ops.object.duplicate(lamp_template.light, linked=True)
        assert light_obj is not None
        light = cast(pf.LightObject, light_obj)
        light.item().parent = lamp_mesh.item()
        light.item().matrix_parent_inverse.identity()
        light.item().location = lamp_template.light.item().location
        snap_to_plane(
            child=lamp_mesh,
            parent=table_mesh,
            child_side="bottom",
            parent_side="top",
            margin=0.003,
            constraint_axis=None,
        )
        light_matrix = light.item().matrix_world.copy()
        light.item().parent = None
        light.item().matrix_world = light_matrix
        lamps.append(lamp.LampResult(mesh=lamp_mesh, light=light))

    lamps, setup_colliders = keep_non_colliding(lamps, setup_colliders)
    bedside_tables = kept_tables
    bedside_lamps = [result.mesh for result in lamps]
    lights = [result.light for result in lamps]
    pillows = []
    if generated_bed and mattress is not None:
        pillows = _bed_pillows_rand(r_pillows, mattress, dimensions)
    objects = [*anchor_objects, *bedside_tables, *bedside_lamps, *pillows]
    setup_colliders = collision.collision_set(
        colliders.objs + objects,
        cache=setup_colliders,
    )
    return BedSetupResult(
        bed=bed_mesh,
        mattress=mattress,
        bedside_tables=bedside_tables,
        bedside_lamps=bedside_lamps,
        pillows=pillows,
        lights=lights,
        all_objects=objects,
        colliders=setup_colliders,
        storage_containers=[],
        supports=bedside_tables + ([mattress] if mattress is not None else []),
        storages=bedside_tables,
    )


def bed_setup_multi_rand(
    rng: pf.RNG,
    wall_planes: list[pf.MeshObject],
    bbox_min: pf.Vector,
    bbox_max: pf.Vector,
    colliders: collision.CollisionSet,
) -> list[BedSetupResult]:
    rng_count, rng_beds = rng.spawn(2)
    setups: list[BedSetupResult] = []
    for bed_rng in rng_beds.spawn(pf.random.randint(rng_count, 2, 4)):
        rng_area, rng_dimensions, rng_bed, rng_aspect = bed_rng.spawn(4)
        area = pf.random.uniform(rng_area, 1.8, 2.2)
        aspect = pf.random.uniform(rng_aspect, 1.85, 2.05)
        dimensions = bed_dimensions_rand(
            rng_dimensions, bbox_min, bbox_max, area=area, aspect=aspect
        )
        furniture = [obj for setup in setups for obj in setup.all_objects]
        try:
            setup = bed_setup_rand(
                rng_bed,
                wall_planes=wall_planes,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                colliders=colliders,
                bed_dimensions=dimensions,
                furniture=furniture,
            )
        except RejectedScene:
            if not setups:
                raise
            break
        setups.append(setup)
        colliders = setup.colliders
    return setups
