# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import math
from typing import NamedTuple, cast

import bpy
import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.objects import chair, storage
from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.placement.distribute import propagate_modifiers_to_instances
from infinigen2.scenes.placement.retry import repeat_attempts
from infinigen2.scenes.room.bathroom_setup import bathroom_sink_setup_rand
from infinigen2.scenes.room.cocktail_table_setup import table_cocktail_setup_rand
from infinigen2.scenes.room.decoration_objects import (
    decoration_collection_primitives_and_real_rand,
    scatter_small_objects_on_containers,
    scatter_small_objects_on_support_tops,
)
from infinigen2.scenes.room.desk_setup import desk_setup_rand
from infinigen2.scenes.room.dining_table_setup import table_dining_with_chairs_rand
from infinigen2.scenes.room.room import RoomResult, room_unfurnished_rand
from infinigen2.scenes.room.sofa_setup import sofa_setup_centered_rand
from infinigen2.scenes.setup_utils import (
    back_face_grounded,
    sofa_object_rand,
    storage_object_rand,
)
from infinigen2.util.scene_cleanup import delete_object

__all__ = [
    "SetupGridResult",
    "WallRowResult",
    "bookshelf_aisle_rows_rand",
    "grid_centered_sofa_rand",
    "chair_banks_rand",
    "grid_cocktail_table_rand",
    "desk_aisle_rows_rand",
    "desk_grid_rand",
    "grid_dining_table_rand",
    "indoor_space_rand",
    "seat_wall_row_rand",
    "setup_grid_in_region",
    "setup_grid_instances",
    "sink_wall_row_rand",
    "sofa_grid_rand",
    "storage_aisle_rows_rand",
    "wall_row_rand",
    "wall_rows_rand",
]


class SetupGridResult(NamedTuple):
    all_objects: list[pf.MeshObject]
    storage_containers: list[pf.MeshObject]
    supports: list[pf.MeshObject]
    storages: list[pf.MeshObject]


class WallRowResult(NamedTuple):
    grid: SetupGridResult
    yaw: float


def _empty_grid() -> SetupGridResult:
    return SetupGridResult([], [], [], [])


def _merge_grids(grids: list[SetupGridResult]) -> SetupGridResult:
    return SetupGridResult(
        [obj for grid in grids for obj in grid.all_objects],
        [obj for grid in grids for obj in grid.storage_containers],
        [obj for grid in grids for obj in grid.supports],
        [obj for grid in grids for obj in grid.storages],
    )


def _role_aliases(
    unit_objects: list[pf.MeshObject],
    role_objects: list[pf.MeshObject],
    groups: list[tuple[pf.MeshObject, ...]],
) -> list[pf.MeshObject]:
    indices = [i for i, obj in enumerate(unit_objects) if obj in role_objects]
    return [group[i] for group in groups for i in indices]


def _center_unit(objects: list[pf.MeshObject]) -> pf.Vector:
    bounds = [pf.ops.attr.bbox_min_max(obj, global_coords=True) for obj in objects]
    lo = pf.Vector([min(b[0][i] for b in bounds) for i in range(3)])
    hi = pf.Vector([max(b[1][i] for b in bounds) for i in range(3)])
    shift = pf.Vector((-lo.x, -(lo.y + hi.y) / 2.0, 0.0))
    for obj in objects:
        obj.item().location += shift
    return hi - lo


@pf.nodes.node_function
def setup_grid_instances(
    collection: t.SocketOrVal[pf.Collection],
    count: t.SocketOrVal[int],
    rows: t.SocketOrVal[float],
    columns_per_pair: t.SocketOrVal[float],
    column_pitch: t.SocketOrVal[float],
    row_pitch: t.SocketOrVal[float],
    rows_per_block: t.SocketOrVal[float],
    row_aisle: t.SocketOrVal[float],
    start: t.SocketOrVal[pf.Vector],
    center: t.SocketOrVal[pf.Vector],
    yaw: t.SocketOrVal[float],
) -> pf.ProcNode[t.Instances]:
    index = pf.nodes.geo.input_index().astype(dtype=float)
    column = pf.nodes.math.floor(index / rows)
    row = index - column * rows
    pair = pf.nodes.math.floor(column / columns_per_pair)
    turned = column - pair * columns_per_pair
    block = pf.nodes.math.floor(row / rows_per_block)
    position = pf.nodes.math.combine_xyz(
        x=pair * column_pitch, y=row * row_pitch + block * row_aisle
    )
    points = pf.nodes.geo.points(count=count, position=position + start)
    instances = pf.nodes.geo.instance_on_points(
        points=points,
        instance=pf.nodes.geo.collection_info(collection),
        rotation=pf.nodes.math.combine_xyz(z=turned * math.pi),
    )
    return pf.nodes.geo.transform(
        instances, translation=center, rotation=pf.nodes.math.combine_xyz(z=yaw)
    )


def _extent(n: int, size: float, gap: float, per_block: int, aisle: float) -> float:
    return n * size + (n - 1) * gap + ((n - 1) // per_block) * aisle


def _fit_count(
    length: float, size: float, gap: float, per_block: int, aisle: float
) -> int:
    n = max(1, int((length + gap) // (size + gap)))
    while n > 1 and _extent(n, size, gap, per_block, aisle) > length:
        n -= 1
    return n


def _group_ok(
    group: tuple[pf.MeshObject, ...],
    rear: list[int],
    colliders: ccol.CollisionSet,
    wall_colliders: ccol.CollisionSet | None,
) -> bool:
    if wall_colliders is not None and not all(
        back_face_grounded(group[i], wall_colliders, 0.02) for i in rear
    ):
        return False
    return not any(ccol.intersection_test(colliders, obj) for obj in group)


def setup_grid_in_region(
    unit: SetupGridResult,
    unit_size: pf.Vector,
    center: pf.Vector,
    depth: float,
    length: float,
    yaw: float,
    gap: pf.Vector,
    aisle: float,
    rows_per_block: int,
    back_to_back: bool,
    colliders: ccol.CollisionSet,
    wall_colliders: ccol.CollisionSet | None,
) -> SetupGridResult:
    columns_per_pair = 2 if back_to_back else 1
    pair_size = unit_size.x * columns_per_pair
    n_pairs = _fit_count(depth, pair_size, gap.x, 1, 0.0)
    n_rows = _fit_count(length, unit_size.y, gap.y, rows_per_block, aisle)
    extent_x = _extent(n_pairs, pair_size, gap.x, 1, 0.0)
    extent_y = _extent(n_rows, unit_size.y, gap.y, rows_per_block, aisle)
    start = pf.Vector(
        (
            (columns_per_pair - 1) * unit_size.x - extent_x / 2.0,
            (unit_size.y - extent_y) / 2.0,
            0.0,
        )
    )

    members = []
    for obj in unit.all_objects:
        collection = pf.Collection([obj], name="setup_grid")
        instances = setup_grid_instances(
            collection=collection,
            count=n_pairs * columns_per_pair * n_rows,
            rows=float(n_rows),
            columns_per_pair=float(columns_per_pair),
            column_pitch=pair_size + gap.x,
            row_pitch=unit_size.y + gap.y,
            rows_per_block=float(rows_per_block),
            row_aisle=aisle,
            start=start,
            center=center,
            yaw=yaw,
        )
        aliases = pf.nodes.to_aliases(instances)
        bpy.data.collections.remove(collection.item())
        propagate_modifiers_to_instances([obj], aliases)
        members.append(aliases)
    rear = [
        i
        for i, obj in enumerate(unit.all_objects)
        if pf.ops.attr.bbox_min_max(obj, global_coords=True)[0][0] <= 0.015
    ]
    kept = []
    for group in zip(*members, strict=True):
        if _group_ok(group, rear, colliders, wall_colliders):
            kept.append(group)
            continue
        for obj in group:
            delete_object(obj.item())
    for i, group in enumerate(kept):
        for obj, template in zip(group, unit.all_objects, strict=True):
            obj.item().name = f"{template.item().name}_{i:03d}"
    return SetupGridResult(
        [obj for group in kept for obj in group],
        _role_aliases(unit.all_objects, unit.storage_containers, kept),
        _role_aliases(unit.all_objects, unit.supports, kept),
        _role_aliases(unit.all_objects, unit.storages, kept),
    )


def _wall_row_center(
    yaw: float,
    room_dimensions: pf.Vector,
    depth: float,
    start: float,
    length: float,
    wall_length: float,
) -> pf.Vector:
    normal = pf.Vector((math.cos(yaw), math.sin(yaw), 0.0))
    along = pf.Vector((-math.sin(yaw), math.cos(yaw), 0.0))
    room_center = pf.Vector((room_dimensions.x / 2.0, room_dimensions.y / 2.0, 0.0))
    wall_distance = abs(normal.dot(room_center))
    return (
        room_center
        - normal * (wall_distance - 0.02 - depth / 2.0)
        + along * (start + (length - wall_length) / 2.0)
    )


def _wall_row_of_unit_rand(
    rng: pf.RNG,
    unit: SetupGridResult,
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
    wall_colliders: ccol.CollisionSet,
    yaw: float,
) -> WallRowResult:
    spacing = pf.random.uniform(rng, 0.05, 0.6)
    cell_count = pf.random.randint(rng, 1, 4)
    first_cell = pf.random.randint(rng, 0, 4 - cell_count)
    unit_size = _center_unit(unit.all_objects)
    wall_length = (
        abs(math.sin(yaw)) * room_dimensions.x + abs(math.cos(yaw)) * room_dimensions.y
    )
    row_length = max(wall_length * cell_count / 3.0, unit_size.y)
    row_start = min(wall_length * first_cell / 3.0, wall_length - row_length)
    center = _wall_row_center(
        yaw, room_dimensions, unit_size.x, row_start, row_length, wall_length
    )
    grid = setup_grid_in_region(
        unit=unit,
        unit_size=unit_size,
        center=center,
        depth=unit_size.x,
        length=row_length,
        yaw=yaw,
        gap=pf.Vector((0.0, spacing, 0.0)),
        aisle=0.0,
        rows_per_block=1,
        back_to_back=False,
        colliders=colliders,
        wall_colliders=wall_colliders,
    )
    for obj in unit.all_objects:
        delete_object(obj.item())
    return WallRowResult(grid, yaw)


@pf.tracer.grammar
def sink_wall_row_rand(
    rng: pf.RNG,
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
    wall_colliders: ccol.CollisionSet,
    yaw: float,
) -> WallRowResult:
    rng, rng_setup = rng.spawn(2)
    setup = bathroom_sink_setup_rand(
        rng_setup,
        bbox_min=pf.Vector((0.0, 0.0, 0.0)),
        bbox_max=room_dimensions,
    )
    sinks = [part.mesh for part in setup.bathroom_sinks]
    storages = [part.mesh for part in setup.storages + setup.wall_storage]
    unit = SetupGridResult(
        setup.all_objects, sinks + storages, sinks + storages, storages
    )
    return _wall_row_of_unit_rand(
        rng, unit, room_dimensions, colliders, wall_colliders, yaw
    )


@pf.tracer.grammar
def seat_wall_row_rand(
    rng: pf.RNG,
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
    wall_colliders: ccol.CollisionSet,
    yaw: float,
) -> WallRowResult:
    rng, rng_choice, rng_seat = rng.spawn(3)
    seat_func = pf.control.choice(
        rng_choice,
        [
            (chair.chair_rand, 1.0),
            (chair.chair_bench_rand, 0.33),
            (sofa_object_rand, 1.0),
        ],
    )
    seat = seat_func(rng_seat).mesh
    unit = SetupGridResult([seat], [], [], [])
    return _wall_row_of_unit_rand(
        rng, unit, room_dimensions, colliders, wall_colliders, yaw
    )


@pf.tracer.grammar
def wall_row_rand(
    rng: pf.RNG,
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
    wall_colliders: ccol.CollisionSet,
    yaw: float,
) -> WallRowResult:
    rng_choice, rng_row = rng.spawn(2)
    row_func = pf.control.choice(
        rng_choice, [(sink_wall_row_rand, 1.0), (seat_wall_row_rand, 2.33)]
    )
    return row_func(rng_row, room_dimensions, colliders, wall_colliders, yaw)


def _wall_row_attempt_rand(
    rng: pf.RNG,
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
    wall_colliders: ccol.CollisionSet,
) -> WallRowResult | None:
    rng_yaw, rng_row = rng.spawn(2)
    yaw = pf.control.choice(
        rng_yaw,
        [(0.0, 1.0), (math.pi / 2.0, 1.0), (math.pi, 1.0), (-math.pi / 2.0, 1.0)],
    )
    row = wall_row_rand(rng_row, room_dimensions, colliders, wall_colliders, yaw)
    if not row.grid.all_objects:
        return None
    return row


@pf.tracer.grammar
def wall_rows_rand(
    rng: pf.RNG,
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
    wall_colliders: ccol.CollisionSet,
) -> SetupGridResult:
    rng_first, rng_second_choice, rng_second = rng.spawn(3)
    first = repeat_attempts(
        _wall_row_attempt_rand,
        rng_first,
        4,
        room_dimensions,
        colliders,
        wall_colliders,
    )
    if first is None:
        return _empty_grid()
    colliders = ccol.collision_set(
        colliders.objs + first.grid.all_objects, cache=colliders
    )
    second_func = pf.control.choice(
        rng_second_choice,
        [(lambda *_: WallRowResult(_empty_grid(), 0.0), 2.0), (wall_row_rand, 3.0)],
    )
    second = second_func(
        rng_second, room_dimensions, colliders, wall_colliders, first.yaw + math.pi
    )
    return _merge_grids([first.grid, second.grid])


def _facing_grid_rand(
    rng: pf.RNG,
    unit: SetupGridResult,
    unit_size: pf.Vector,
    center: pf.Vector,
    size: pf.Vector,
    gap: pf.Vector,
    aisle: float,
    rows_per_block: int,
    back_to_back: bool,
    colliders: ccol.CollisionSet,
) -> SetupGridResult:
    yaw = pf.control.choice(
        rng,
        [(0.0, 1.0), (math.pi / 2.0, 1.0), (math.pi, 1.0), (-math.pi / 2.0, 1.0)],
    )
    depth = abs(math.cos(yaw)) * size.x + abs(math.sin(yaw)) * size.y
    length = abs(math.sin(yaw)) * size.x + abs(math.cos(yaw)) * size.y
    return setup_grid_in_region(
        unit=unit,
        unit_size=unit_size,
        center=center,
        depth=depth,
        length=length,
        yaw=yaw,
        gap=gap,
        aisle=aisle,
        rows_per_block=rows_per_block,
        back_to_back=back_to_back,
        colliders=colliders,
        wall_colliders=None,
    )


def _split_grid_rand(
    rng: pf.RNG,
    unit: SetupGridResult,
    unit_size: pf.Vector,
    center: pf.Vector,
    size: pf.Vector,
    gap: pf.Vector,
    aisle: float,
    rows_per_block: int,
    back_to_back: bool,
    colliders: ccol.CollisionSet,
) -> SetupGridResult:
    rng, rng_first, rng_second = rng.spawn(3)
    split = pf.random.uniform(rng, 0.35, 0.65)
    split_gap = pf.random.uniform(rng, 1.0, 1.6)
    axis = pf.Vector((1.0, 0.0, 0.0) if size.x >= size.y else (0.0, 1.0, 0.0))
    total = size.dot(axis)
    first = total * split - split_gap / 2.0
    second = total - first - split_gap
    first_grid = _facing_grid_rand(
        rng_first,
        unit=unit,
        unit_size=unit_size,
        center=center + axis * (first - total) / 2.0,
        size=size - axis * (total - first),
        gap=gap,
        aisle=aisle,
        rows_per_block=rows_per_block,
        back_to_back=back_to_back,
        colliders=colliders,
    )
    second_grid = _facing_grid_rand(
        rng_second,
        unit=unit,
        unit_size=unit_size,
        center=center + axis * (total - second) / 2.0,
        size=size - axis * (total - second),
        gap=gap,
        aisle=aisle,
        rows_per_block=rows_per_block,
        back_to_back=back_to_back,
        colliders=colliders,
    )
    return _merge_grids([first_grid, second_grid])


def _interior_grid_rand(
    rng: pf.RNG,
    unit: SetupGridResult,
    unit_size: pf.Vector,
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
    gap: pf.Vector,
    aisle: float,
    rows_per_block: int,
    back_to_back: bool,
) -> SetupGridResult:
    rng, rng_layout, rng_grid = rng.spawn(3)
    margin = pf.random.uniform(rng, 0.4, 0.8)
    center = pf.Vector((room_dimensions.x / 2.0, room_dimensions.y / 2.0, 0.0))
    size = pf.Vector(
        (room_dimensions.x - 2 * margin, room_dimensions.y - 2 * margin, 0.0)
    )
    layout_func = pf.control.choice(
        rng_layout, [(_facing_grid_rand, 1.0), (_split_grid_rand, 1.0)]
    )
    grid = layout_func(
        rng_grid,
        unit=unit,
        unit_size=unit_size,
        center=center,
        size=size,
        gap=gap,
        aisle=aisle,
        rows_per_block=rows_per_block,
        back_to_back=back_to_back,
        colliders=colliders,
    )
    for obj in unit.all_objects:
        delete_object(obj.item())
    return grid


def _aisle_rows_of_unit_rand(
    rng: pf.RNG,
    unit: SetupGridResult,
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
) -> SetupGridResult:
    rng, rng_grid = rng.spawn(2)
    unit_size = _center_unit(unit.all_objects)
    gap = pf.Vector((pf.random.uniform(rng, 0.7, 1.3), 0.0, 0.0))
    aisle = pf.random.uniform(rng, 0.9, 1.5)
    rows_per_block = pf.random.randint(rng, 3, 10)
    return _interior_grid_rand(
        rng_grid,
        unit=unit,
        unit_size=unit_size,
        room_dimensions=room_dimensions,
        colliders=colliders,
        gap=gap,
        aisle=aisle,
        rows_per_block=rows_per_block,
        back_to_back=True,
    )


@pf.tracer.grammar
def desk_aisle_rows_rand(
    rng: pf.RNG,
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
) -> SetupGridResult:
    rng, rng_setup = rng.spawn(2)
    setup = desk_setup_rand(rng_setup)
    unit = SetupGridResult(
        setup.all_objects, setup.storage_containers, setup.supports, setup.storages
    )
    return _aisle_rows_of_unit_rand(rng, unit, room_dimensions, colliders)


@pf.tracer.grammar
def storage_aisle_rows_rand(
    rng: pf.RNG,
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
) -> SetupGridResult:
    rng, rng_storage = rng.spawn(2)
    mesh = storage_object_rand(rng_storage).mesh
    unit = SetupGridResult([mesh], [mesh], [mesh], [mesh])
    return _aisle_rows_of_unit_rand(rng, unit, room_dimensions, colliders)


@pf.tracer.grammar
def bookshelf_aisle_rows_rand(
    rng: pf.RNG,
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
) -> SetupGridResult:
    rng, rng_shelf, rng_rows = rng.spawn(3)
    dimensions = pf.Vector(
        (
            pf.random.uniform(rng, 0.3, 0.45),
            pf.random.uniform(rng, 0.8, 1.4),
            pf.random.uniform(rng, 1.6, 2.2),
        )
    )
    mesh = storage.storage_cell_shelf_rand(rng_shelf, dimensions=dimensions).mesh
    unit = SetupGridResult([mesh], [mesh], [mesh], [mesh])
    return _aisle_rows_of_unit_rand(rng_rows, unit, room_dimensions, colliders)


@pf.tracer.grammar
def chair_banks_rand(
    rng: pf.RNG,
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
) -> SetupGridResult:
    rng, rng_choice, rng_seat, rng_grid = rng.spawn(4)
    seat_func = pf.control.choice(
        rng_choice, [(chair.chair_rand, 3.0), (chair.chair_bench_rand, 1.0)]
    )
    seat = seat_func(rng_seat).mesh
    unit = SetupGridResult([seat], [], [], [])
    unit_size = _center_unit(unit.all_objects)
    gap = pf.Vector((pf.random.uniform(rng, 0.3, 0.6), 0.02, 0.0))
    aisle = pf.random.uniform(rng, 0.9, 1.3)
    block_length = pf.random.uniform(rng, 2.5, 7.0)
    rows_per_block = max(1, int(block_length / (unit_size.y + gap.y)))
    return _interior_grid_rand(
        rng_grid,
        unit=unit,
        unit_size=unit_size,
        room_dimensions=room_dimensions,
        colliders=colliders,
        gap=gap,
        aisle=aisle,
        rows_per_block=rows_per_block,
        back_to_back=False,
    )


def _group_gap_rand(rng: pf.RNG) -> pf.Vector:
    return pf.Vector(
        (pf.random.uniform(rng, 1.0, 2.0), pf.random.uniform(rng, 1.0, 2.0), 0.0)
    )


def _plain_grid_rand(
    rng: pf.RNG,
    unit: SetupGridResult,
    gap: pf.Vector,
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
) -> SetupGridResult:
    unit_size = _center_unit(unit.all_objects)
    return _interior_grid_rand(
        rng,
        unit=unit,
        unit_size=unit_size,
        room_dimensions=room_dimensions,
        colliders=colliders,
        gap=gap,
        aisle=0.0,
        rows_per_block=1,
        back_to_back=False,
    )


@pf.tracer.grammar
def desk_grid_rand(
    rng: pf.RNG,
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
) -> SetupGridResult:
    rng, rng_setup, rng_grid = rng.spawn(3)
    setup = desk_setup_rand(rng_setup)
    unit = SetupGridResult(
        setup.all_objects, setup.storage_containers, setup.supports, setup.storages
    )
    gap = pf.Vector(
        (pf.random.uniform(rng, 0.6, 1.5), pf.random.uniform(rng, 0.0, 1.0) ** 2, 0.0)
    )
    return _plain_grid_rand(rng_grid, unit, gap, room_dimensions, colliders)


@pf.tracer.grammar
def sofa_grid_rand(
    rng: pf.RNG,
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
) -> SetupGridResult:
    rng, rng_sofa, rng_grid = rng.spawn(3)
    sofa = sofa_object_rand(rng_sofa).mesh
    unit = SetupGridResult([sofa], [sofa], [], [])
    gap = pf.Vector(
        (pf.random.uniform(rng, 0.8, 1.5), pf.random.uniform(rng, 0.05, 0.3), 0.0)
    )
    return _plain_grid_rand(rng_grid, unit, gap, room_dimensions, colliders)


@pf.tracer.grammar
def grid_dining_table_rand(
    rng: pf.RNG,
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
) -> SetupGridResult:
    rng, rng_setup, rng_grid = rng.spawn(3)
    setup = table_dining_with_chairs_rand(rng_setup)
    unit = SetupGridResult(setup.all_objects, [], [setup.dining_table], [])
    gap = _group_gap_rand(rng)
    return _plain_grid_rand(rng_grid, unit, gap, room_dimensions, colliders)


@pf.tracer.grammar
def grid_cocktail_table_rand(
    rng: pf.RNG,
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
) -> SetupGridResult:
    rng, rng_setup, rng_grid = rng.spawn(3)
    setup = table_cocktail_setup_rand(rng_setup)
    unit = SetupGridResult(setup.all_objects, [], [setup.dining_table], [])
    gap = _group_gap_rand(rng)
    return _plain_grid_rand(rng_grid, unit, gap, room_dimensions, colliders)


@pf.tracer.grammar
def grid_centered_sofa_rand(
    rng: pf.RNG,
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
) -> SetupGridResult:
    rng, rng_setup, rng_grid = rng.spawn(3)
    setup = sofa_setup_centered_rand(rng_setup)
    unit = SetupGridResult(
        setup.all_objects, setup.storage_containers, setup.supports, setup.storages
    )
    gap = _group_gap_rand(rng)
    return _plain_grid_rand(rng_grid, unit, gap, room_dimensions, colliders)


def _interior_attempt_rand(
    rng: pf.RNG,
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
) -> SetupGridResult | None:
    rng_choice, rng_generate = rng.spawn(2)
    grid_func = pf.control.choice(
        rng_choice,
        [
            (desk_aisle_rows_rand, 2.0),
            (storage_aisle_rows_rand, 1.0),
            (bookshelf_aisle_rows_rand, 2.0),
            (chair_banks_rand, 2.0),
            (desk_grid_rand, 1.0),
            (sofa_grid_rand, 1.0),
            (grid_dining_table_rand, 1.0),
            (grid_cocktail_table_rand, 1.0),
            (grid_centered_sofa_rand, 1.0),
        ],
    )
    grid = grid_func(rng_generate, room_dimensions, colliders)
    if not grid.all_objects:
        return None
    return grid


def _room_dimensions_rand(rng: pf.RNG) -> pf.Vector:
    short_side = pf.random.uniform(rng, 4.55, 14.0)
    long_side = short_side * pf.random.uniform(rng, 1.0, 1.7)
    base_height = pf.random.clip_gaussian(rng, 3.2, 1.53, 2.2, 6.0)
    tallness = pf.random.uniform(rng, 0.0, 1.0) ** 2
    height = base_height + max(short_side - 8.0, 0.0) * 0.6 * tallness
    return pf.Vector((short_side, long_side, height))


@pf.tracer.grammar
def indoor_space_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    frame_start: int = 1,
    frame_end: int = 1,
) -> RoomResult:
    (
        rng_dimensions,
        rng_room,
        rng_walls_choice,
        rng_walls,
        rng_interior,
        rng_small,
    ) = rng.spawn(6)
    if dimensions is None:
        dimensions = _room_dimensions_rand(rng_dimensions)

    room = room_unfurnished_rand(
        rng_room,
        dimensions,
        frame_start=frame_start,
        frame_end=frame_end,
    )
    wall_colliders = ccol.collision_set(cast("list[pf.Object]", room.wall_planes))
    walls_func = pf.control.choice(
        rng_walls_choice,
        [(lambda *_: _empty_grid(), 1.0), (wall_rows_rand, 2.0)],
    )
    walls = walls_func(rng_walls, dimensions, room.colliders, wall_colliders)
    colliders = ccol.collision_set(
        room.colliders.objs + walls.all_objects, cache=room.colliders
    )
    interior = repeat_attempts(
        _interior_attempt_rand, rng_interior, 4, dimensions, colliders
    )
    if interior is None:
        interior = _empty_grid()
    grids = _merge_grids([walls, interior])
    colliders = ccol.collision_set(
        colliders.objs + interior.all_objects, cache=colliders
    )
    rng_collection, rng_supports, rng_containers = rng_small.spawn(3)
    collection = decoration_collection_primitives_and_real_rand(rng_collection)
    on_supports, _ = scatter_small_objects_on_support_tops(
        rng_supports,
        grids.supports,
        colliders,
        collection=collection,
        inter_object_collide=False,
    )
    in_containers, _ = scatter_small_objects_on_containers(
        rng_containers,
        grids.storage_containers,
        colliders,
        collection=collection,
        inter_object_collide=False,
    )
    return RoomResult(
        all_objects=room.all_objects + grids.all_objects + on_supports + in_containers,
        cameras=room.cameras,
        lights=room.lights,
        colliders=colliders,
        floor=room.floor,
        dimensions=room.dimensions,
        storage_containers=room.storage_containers,
        supports=room.supports,
        storages=room.storages,
        wall_planes=room.wall_planes,
    )
