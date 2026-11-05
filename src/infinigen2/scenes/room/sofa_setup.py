# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import functools
import logging
import math
from typing import NamedTuple

import numpy as np
import procfunc as pf
import shapely

from infinigen2.objects import cushion, monitor, rug, storage, table
from infinigen2.objects.sofa import SofaResult
from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.placement.culling import keep_non_colliding, keep_unobstructed
from infinigen2.scenes.placement.distribute import instances_along_line
from infinigen2.scenes.placement.snap import snap_to_plane
from infinigen2.scenes.setup_utils import (
    MeshResult,
    back_face_grounded,
    center_inside,
    jitter_object_rotation_rand,
    retry_place,
    side_table_object_rand,
    snap_back_front,
    snap_side_by_side,
    sofa_object_rand,
    standalone_wall_planes,
)

__all__ = [
    "SofaSetupResult",
    "TVSetupResult",
    "side_tables_rand",
    "sofa_setup_centered_rand",
    "sofa_setup_wall_anchored_rand",
    "sofa_setup_wall_rand",
    "tv_setup_rand",
    "tv_setup_wall_rand",
]

logger = logging.getLogger(__name__)


class SofaSetupResult(NamedTuple):
    all_objects: list[pf.MeshObject]
    sofas: list[SofaResult]
    rugs: list[pf.MeshObject]
    throw_pillows: list[pf.MeshObject]
    storage_containers: list[pf.MeshObject]
    supports: list[pf.MeshObject]
    storages: list[pf.MeshObject]


class TVSetupResult(NamedTuple):
    mesh: pf.MeshObject
    tv: pf.MeshObject
    all_objects: list[pf.MeshObject]


def tv_setup_rand(rng: pf.RNG) -> TVSetupResult:
    rng_diagonal, rng_size, rng_width, rng_depth, rng_height, rng_storage, rng_tv = (
        rng.spawn(7)
    )
    diagonal = pf.random.clip_gaussian(rng_diagonal, 50.0, 20.0, 24.0, 98.0) * 0.0254
    screen_dimensions = monitor.screen_dimensions_rand(rng_size, diagonal=diagonal)
    extra_width = pf.random.uniform(rng_width, 0.15, 0.35)
    storage_width = min(2.5, max(0.9, screen_dimensions.y + extra_width))
    storage_depth = pf.random.uniform(rng_depth, 0.38, 0.48)
    storage_height = pf.random.uniform(rng_height, 0.38, 0.64)
    dimensions = pf.Vector((storage_depth, storage_width, storage_height))
    storage_result = storage.storage_rand(
        rng_storage,
        dimensions=dimensions,
        n_spaces_z=1,
    )
    tv_result = monitor.monitor_rand(rng_tv, screen_dimensions=screen_dimensions)
    snap_to_plane(
        child=tv_result.mesh,
        parent=storage_result.mesh,
        child_side="bottom",
        parent_side="top",
        margin=0.002,
        constraint_axis=None,
    )
    tv_result.mesh.item().parent = storage_result.mesh.item()
    tv_result.mesh.item().matrix_parent_inverse.identity()
    return TVSetupResult(
        mesh=storage_result.mesh,
        tv=tv_result.mesh,
        all_objects=[storage_result.mesh, tv_result.mesh],
    )


def _tv_setup_accepted(
    storage_mesh: pf.MeshObject,
    tv: pf.MeshObject,
    wall_colliders: ccol.CollisionSet,
    colliders: ccol.CollisionSet,
    wall_margin: float,
) -> bool:
    grounded = back_face_grounded(
        storage_mesh,
        colliders=wall_colliders,
        margin=wall_margin,
    )
    return grounded and not ccol.intersection_test(colliders, tv)


def tv_setup_wall_rand(
    rng: pf.RNG,
    wall_planes: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
) -> list[TVSetupResult]:
    rng_tv, rng_margin, rng_place = rng.spawn(3)
    tv_setup = tv_setup_rand(rng_tv)
    wall_margin = pf.random.uniform(rng_margin, 0.03, 0.10)
    accepted = functools.partial(
        _tv_setup_accepted,
        tv=tv_setup.tv,
        wall_colliders=ccol.collision_set(wall_planes),
        colliders=colliders,
        wall_margin=wall_margin,
    )
    placed_tv = retry_place(
        rng_place,
        tv_setup,
        colliders,
        snap_back_front,
        attempts=12,
        accept_fn=accepted,
        parents=wall_planes,
        margin=wall_margin,
    )
    return [] if placed_tv is None else [placed_tv]


def _place_rug(
    rng: pf.RNG,
    child: MeshResult,
    bbox_min: pf.Vector,
    bbox_max: pf.Vector,
) -> None:
    child.mesh.item().location = (
        pf.random.uniform(rng, bbox_min.x, bbox_max.x),
        pf.random.uniform(rng, bbox_min.y, bbox_max.y),
        bbox_min.z + 0.001,
    )


def _rug_rand(
    rng: pf.RNG,
    bbox_min: pf.Vector,
    bbox_max: pf.Vector,
    colliders: ccol.CollisionSet,
    region: shapely.Polygon | None,
) -> list[pf.MeshObject]:
    rng_dims, rng_rug, rng_place = rng.spawn(3)
    length = pf.random.uniform(rng_dims, 1.6, 3.0)
    width = pf.random.uniform(rng_dims, 1.2, 2.4)
    thickness = pf.random.uniform(rng_dims, 0.01, 0.02)
    result = rug.rug_rand(rng_rug, dimensions=pf.Vector((length, width, thickness)))
    clearance = pf.Vector((1.2, 1.2, 0.0))

    def inside(mesh: pf.MeshObject) -> bool:
        return region is None or center_inside(region, mesh)

    placed = retry_place(
        rng_place,
        result,
        colliders,
        _place_rug,
        attempts=50,
        accept_fn=inside,
        bbox_min=bbox_min + clearance,
        bbox_max=bbox_max - clearance,
    )
    kept, _ = keep_non_colliding([placed], colliders)
    return [r.mesh for r in kept]


def _snap_facing_rug(
    rng: pf.RNG,
    child: MeshResult,
    rug_obj: pf.MeshObject,
    parent_side: str,
) -> None:
    snap_to_plane(
        child=child.mesh,
        parent=rug_obj,
        parent_side=parent_side,
        child_side="front",
        margin=pf.random.uniform(rng, -0.1, 0.5),
        placement=pf.random.uniform(rng, 0.35, 0.65),
        constraint_axis=pf.Vector((0, 0, 1)),
    )


def _center_coffee_table_rand(
    rng: pf.RNG,
    center: np.ndarray,
) -> list[MeshResult]:
    child = table.table_coffee_rand(rng)
    child.mesh.item().location = (
        center[0] + pf.random.uniform(rng, -0.2, 0.2),
        center[1] + pf.random.uniform(rng, -0.2, 0.2),
        0.01,
    )
    return [child]


def _throw_pillows_rand(rng: pf.RNG, sofa: SofaResult) -> list[pf.MeshObject]:
    rng, rng_pillow = rng.spawn(2)
    start = pf.Vector(sofa.back_seat_line_start)
    end = pf.Vector(sofa.back_seat_line_end)
    side = pf.random.uniform(rng, 0.4, 0.55)
    thickness = side * pf.random.uniform(rng, 0.25, 0.33)
    sink = thickness * pf.random.uniform(rng, 0.05, 0.15)
    max_count = max(1, int((end.y - start.y) / (side + 0.05)))
    count = pf.random.randint(rng, 1, min(4, max_count) + 1)
    lean = sofa.back_tilt
    offset = pf.Vector(
        (
            thickness * 0.5 * math.cos(lean) - side * 0.5 * math.sin(lean),
            0.0,
            side * 0.5 * math.cos(lean) + thickness * 0.5 * math.sin(lean) - sink,
        )
    )
    pillow = cushion.cushion_throw_pillow_rand(
        rng_pillow, size=pf.Vector((side, side, thickness))
    )
    pillows = instances_along_line(
        pillow.mesh,
        sofa.mesh,
        start + offset,
        end + offset,
        count,
        (0.0, math.pi / 2 - lean, 0.0),
    )
    return pillows


def _no_throw_pillows(rng: pf.RNG, sofa: SofaResult) -> list[pf.MeshObject]:
    return []


def _sofas_throw_pillows_rand(
    rng: pf.RNG, sofas: list[SofaResult]
) -> list[pf.MeshObject]:
    pillows = []
    for sofa, sofa_rng in zip(sofas, rng.spawn(len(sofas)), strict=True):
        rng_choice, rng_pillows = sofa_rng.spawn(2)
        func = pf.control.choice(
            rng_choice, [(_throw_pillows_rand, 2.0), (_no_throw_pillows, 1.0)]
        )
        pillows += func(rng_pillows, sofa)
    return pillows


def sofa_setup_centered_rand(
    rng: pf.RNG,
    bbox_min: pf.Vector | None = None,
    bbox_max: pf.Vector | None = None,
    colliders: ccol.CollisionSet | None = None,
    region: shapely.Polygon | None = None,
) -> SofaSetupResult:
    if bbox_min is None:
        bbox_min = pf.Vector((0.0, 0.0, 0.0))
    if bbox_max is None:
        bbox_max = pf.Vector((5.0, 5.0, 3.0))
    if colliders is None:
        colliders = ccol.collision_set([])
    rng_rug, rng_sides, rng_around = rng.spawn(3)
    rug_objs = _rug_rand(rng_rug, bbox_min, bbox_max, colliders, region)
    if not rug_objs:
        return SofaSetupResult(
            all_objects=[],
            sofas=[],
            rugs=[],
            throw_pillows=[],
            storage_containers=[],
            supports=[],
            storages=[],
        )
    sides = _rug_sides_rand(rng_sides, ["right", "left", "front", "back"], 2, 4)
    return _sofas_around_rug_rand(rng_around, rug_objs, sides, [], colliders)


def sofa_setup_wall_anchored_rand(
    rng: pf.RNG,
    wall_planes: list[pf.MeshObject] | None = None,
    bbox_min: pf.Vector | None = None,
    colliders: ccol.CollisionSet | None = None,
) -> SofaSetupResult:
    standalone_walls: list[pf.MeshObject] = []
    if wall_planes is None:
        wall_planes = standalone_wall_planes()
        standalone_walls = wall_planes
    if bbox_min is None:
        bbox_min = pf.Vector((0.0, 0.0, 0.0))
    if colliders is None:
        colliders = ccol.collision_set(wall_planes)
    elif standalone_walls:
        colliders = ccol.collision_set(
            colliders.objs + standalone_walls,
            cache=colliders,
        )
    rng_sofa, rng_place, rng_rug, rng_sides, rng_around = rng.spawn(5)
    child = sofa_object_rand(rng_sofa)
    placed = retry_place(
        rng_place,
        child,
        colliders,
        snap_back_front,
        attempts=32,
        parents=wall_planes,
    )
    wall_sofas, colliders = keep_non_colliding([placed], colliders)
    if not wall_sofas:
        return SofaSetupResult(
            all_objects=standalone_walls,
            sofas=[],
            rugs=[],
            throw_pillows=[],
            storage_containers=[],
            supports=[],
            storages=[],
        )
    rugs = _facing_rug_rand(rng_rug, wall_sofas, bbox_min.z, colliders)
    if not rugs:
        sofa_meshes = [r.mesh for r in wall_sofas]
        return SofaSetupResult(
            all_objects=standalone_walls + sofa_meshes,
            sofas=wall_sofas,
            rugs=[],
            throw_pillows=[],
            storage_containers=sofa_meshes,
            supports=[],
            storages=[],
        )
    sides = _rug_sides_rand(rng_sides, ["right", "front", "back"], 1, 4)
    rug_objs = [r.mesh for r in rugs]
    around = _sofas_around_rug_rand(rng_around, rug_objs, sides, wall_sofas, colliders)
    return SofaSetupResult(
        all_objects=standalone_walls + around.all_objects,
        sofas=around.sofas,
        rugs=around.rugs,
        throw_pillows=around.throw_pillows,
        storage_containers=around.storage_containers,
        supports=around.supports,
        storages=around.storages,
    )


def _rug_sides_rand(rng: pf.RNG, names: list[str], low: int, high: int) -> list[str]:
    n = pf.random.randint(rng, low, high)
    return [names[int(i)] for i in rng.choice(len(names), size=n, replace=False)]


def _sofas_around_rug_rand(
    rng: pf.RNG,
    rug_objs: list[pf.MeshObject],
    sides: list[str],
    wall_sofas: list[SofaResult],
    colliders: ccol.CollisionSet,
) -> SofaSetupResult:
    (
        rng_sofas,
        rng_jitter_amount,
        rng_jitter_objects,
        rng_coffee,
        rng_output,
        rng_pillows,
    ) = rng.spawn(6)
    rug_obj = rug_objs[0]
    cmin, cmax = (
        np.array(bound)
        for bound in pf.ops.attr.bbox_min_max(rug_obj, global_coords=True)
    )
    center = (cmin + cmax) / 2

    placed_sofas = []
    for side, sofa_rng in zip(sides, rng_sofas.spawn(len(sides)), strict=True):
        rng_sofa, rng_place = sofa_rng.spawn(2)
        child = sofa_object_rand(rng_sofa)
        placed = retry_place(
            rng_place,
            child,
            colliders,
            _snap_facing_rug,
            rug_obj=rug_obj,
            parent_side=side,
        )
        placed_sofas.append(placed)
    placed_sofas = keep_unobstructed(placed_sofas, center, colliders)
    around_sofas, colliders = keep_non_colliding(placed_sofas, colliders)
    logger.info(f"Placed {len(around_sofas)} carpet sofas out of {len(sides)} attempts")

    max_angle = math.radians(5) * pf.random.uniform(rng_jitter_amount, 0.0, 1.0) ** 3
    sofa_rngs = rng_jitter_objects.spawn(len(around_sofas))
    for sofa_obj, sofa_rng in zip(around_sofas, sofa_rngs, strict=True):
        jitter_object_rotation_rand(sofa_rng, sofa_obj.mesh, max_angle, colliders)

    sofa_objs = wall_sofas + around_sofas
    center_coffee = _center_coffee_table_rand(rng_coffee, center)
    center_coffee, _ = keep_non_colliding(center_coffee, colliders)
    pillows = _sofas_throw_pillows_rand(rng_pillows, sofa_objs)
    out_rugs = pf.control.choice(rng_output, [(rug_objs, 2.0), ([], 1.0)])
    all_objects = [r.mesh for r in sofa_objs + center_coffee]
    return SofaSetupResult(
        all_objects=all_objects + pillows + out_rugs,
        sofas=sofa_objs,
        rugs=out_rugs,
        throw_pillows=pillows,
        storage_containers=[r.mesh for r in sofa_objs],
        supports=[r.mesh for r in center_coffee] + out_rugs,
        storages=[r.mesh for r in center_coffee],
    )


def side_tables_rand(
    rng: pf.RNG,
    sofas: list[SofaResult],
    colliders: ccol.CollisionSet,
    max_tables: int,
) -> list[MeshResult]:
    rng_count, rng_tables = rng.spawn(2)
    n = min(pf.random.randint(rng_count, 1, max_tables + 1), len(sofas))
    sofa_meshes = [r.mesh for r in sofas]
    placed = []
    for table_rng in rng_tables.spawn(n):
        rng_table, rng_place = table_rng.spawn(2)
        child = side_table_object_rand(rng_table)
        result = retry_place(
            rng_place, child, colliders, snap_side_by_side, 12, parents=sofa_meshes
        )
        placed.append(result)
    side_tables, _ = keep_non_colliding(placed, colliders)
    logger.info(f"Placed {len(side_tables)} side tables out of {n} attempts")
    return side_tables


def _place_in_front_of_sofa(
    rng: pf.RNG,
    child: MeshResult,
    sofas: list[pf.MeshObject],
    margin_min: float,
    margin_max: float,
) -> None:
    snap_to_plane(
        child=child.mesh,
        parent=rng.choice(sofas),
        parent_side="front",
        child_side="left",
        margin=pf.random.uniform(rng, margin_min, margin_max),
        placement=pf.random.uniform(rng, 0.4, 0.6),
        constraint_axis=pf.Vector((0, 0, 1)),
    )


def _no_items(
    rng: pf.RNG,
    sofas: list[SofaResult],
    floor_z: float,
    colliders: ccol.CollisionSet,
) -> list[MeshResult]:
    return []


def _center_side_tables_rand(
    rng: pf.RNG,
    sofas: list[SofaResult],
    floor_z: float,
    colliders: ccol.CollisionSet,
) -> list[MeshResult]:
    return side_tables_rand(rng, sofas[:1], colliders, max_tables=2)


def _facing_coffee_table_rand(
    rng: pf.RNG,
    sofas: list[SofaResult],
    floor_z: float,
    colliders: ccol.CollisionSet,
) -> list[MeshResult]:
    rng_table, rng_place = rng.spawn(2)
    child = table.table_coffee_rand(rng_table)
    child.mesh.item().location = (0.0, 0.0, floor_z + 0.01)
    placed = retry_place(
        rng_place,
        child,
        colliders,
        _place_in_front_of_sofa,
        sofas=[r.mesh for r in sofas],
        margin_min=0.35,
        margin_max=0.6,
    )
    kept, _ = keep_non_colliding([placed], colliders)
    return kept


def _facing_rug_rand(
    rng: pf.RNG,
    sofas: list[SofaResult],
    floor_z: float,
    colliders: ccol.CollisionSet,
) -> list[MeshResult]:
    rng_dims, rng_rug, rng_place = rng.spawn(3)
    length = pf.random.uniform(rng_dims, 1.6, 3.0)
    width = pf.random.uniform(rng_dims, 1.2, 2.4)
    thickness = pf.random.uniform(rng_dims, 0.01, 0.02)
    result = rug.rug_rand(rng_rug, dimensions=pf.Vector((length, width, thickness)))
    result.mesh.item().location = (0.0, 0.0, floor_z + 0.001)
    placed = retry_place(
        rng_place,
        result,
        colliders,
        _place_in_front_of_sofa,
        sofas=[r.mesh for r in sofas],
        margin_min=0.02,
        margin_max=0.3,
    )
    kept, _ = keep_non_colliding([placed], colliders)
    return kept


def sofa_setup_wall_rand(
    rng: pf.RNG,
    wall_planes: list[pf.MeshObject] | None = None,
    bbox_min: pf.Vector | None = None,
    colliders: ccol.CollisionSet | None = None,
) -> SofaSetupResult:
    standalone_walls: list[pf.MeshObject] = []
    if wall_planes is None:
        wall_planes = standalone_wall_planes()
        standalone_walls = wall_planes
    if bbox_min is None:
        bbox_min = pf.Vector((0.0, 0.0, 0.0))
    if colliders is None:
        colliders = ccol.collision_set(wall_planes)
    elif standalone_walls:
        colliders = ccol.collision_set(
            colliders.objs + standalone_walls,
            cache=colliders,
        )
    (
        rng_count,
        rng_sofas,
        rng_side_choice,
        rng_side,
        rng_coffee_choice,
        rng_coffee,
        rng_rug_choice,
        rng_rug,
        rng_pillows,
    ) = rng.spawn(9)

    n = pf.random.randint(rng_count, 1, 8)
    placed_sofas = []
    for sofa_rng in rng_sofas.spawn(n):
        rng_sofa, rng_place = sofa_rng.spawn(2)
        child = sofa_object_rand(rng_sofa)
        result = retry_place(
            rng_place, child, colliders, snap_back_front, 32, parents=wall_planes
        )
        placed_sofas.append(result)
    sofa_objs, colliders = keep_non_colliding(placed_sofas, colliders)
    logger.info(f"Placed {len(sofa_objs)} wall sofas out of {n} attempts")
    if not sofa_objs:
        return SofaSetupResult(
            all_objects=standalone_walls,
            sofas=[],
            rugs=[],
            throw_pillows=[],
            storage_containers=[],
            supports=[],
            storages=[],
        )
    rug_colliders = colliders

    side_func = pf.control.choice(
        rng_side_choice, [(_center_side_tables_rand, 2.0), (_no_items, 1.0)]
    )
    side_tables = side_func(rng_side, sofa_objs, bbox_min.z, colliders)
    colliders = ccol.collision_set(
        colliders.objs + [r.mesh for r in side_tables], cache=colliders
    )
    coffee_func = pf.control.choice(
        rng_coffee_choice, [(_facing_coffee_table_rand, 2.0), (_no_items, 1.0)]
    )
    coffee_tables = coffee_func(rng_coffee, sofa_objs, bbox_min.z, colliders)
    rug_func = pf.control.choice(
        rng_rug_choice, [(_facing_rug_rand, 1.0), (_no_items, 1.0)]
    )
    rugs = rug_func(rng_rug, sofa_objs, bbox_min.z, rug_colliders)

    tables = [r.mesh for r in side_tables + coffee_tables]
    rug_objs = [r.mesh for r in rugs]
    pillows = _sofas_throw_pillows_rand(rng_pillows, sofa_objs)
    return SofaSetupResult(
        all_objects=(
            standalone_walls + [r.mesh for r in sofa_objs] + tables + rug_objs + pillows
        ),
        sofas=sofa_objs,
        rugs=rug_objs,
        throw_pillows=pillows,
        storage_containers=[r.mesh for r in sofa_objs],
        supports=tables + rug_objs,
        storages=tables,
    )
