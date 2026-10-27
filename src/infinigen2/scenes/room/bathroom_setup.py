# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
import procfunc as pf

from infinigen2.objects import (
    bathroom_hardware,
    bathtub,
    sink,
    storage,
    tap,
    toilet,
    wall_art,
)
from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.placement.culling import keep_non_colliding
from infinigen2.scenes.placement.retry import repeat_attempts
from infinigen2.scenes.placement.snap import snap_to_plane
from infinigen2.scenes.setup_utils import (
    BareMeshResult,
    MeshResult,
    back_face_grounded,
    retry_place,
    snap_back_front,
    standalone_wall_planes,
)
from infinigen2.util.scene_cleanup import delete_object, delete_objects

__all__ = [
    "BathroomSetupResult",
    "BathroomSinkSetupResult",
    "bathroom_existing_sink_setup_rand",
    "bathroom_setup_rand",
    "bathroom_sink_setup_rand",
    "snap_wall_hardware_rand",
]

logger = logging.getLogger(__name__)

_WALL_MARGIN = 0.003
_TAP_BACK_OFFSET = 0.05
_TAP_SURFACE_INSET = 0.01


class _BathtubSetupResult(NamedTuple):
    bathtubs: list[MeshResult]
    taps: list[MeshResult]
    hardware: list[MeshResult]


class _SinkSupportResult(NamedTuple):
    supports: list[MeshResult]
    storages: list[MeshResult]


class _SinkUnit(NamedTuple):
    sink: MeshResult
    tap: MeshResult
    supports: list[MeshResult]
    storages: list[MeshResult]


class _WallFeatureResult(NamedTuple):
    mirrors: list[MeshResult]
    wall_storage: list[MeshResult]


class BathroomSinkSetupResult(NamedTuple):
    bathroom_sinks: list[MeshResult]
    sink_taps: list[MeshResult]
    sink_supports: list[MeshResult]
    mirrors: list[MeshResult]
    wall_storage: list[MeshResult]
    storages: list[MeshResult]
    all_objects: list[pf.MeshObject]


class BathroomSetupResult(NamedTuple):
    named_objects: dict[str, list[pf.MeshObject]]
    all_objects: list[pf.MeshObject]
    colliders: ccol.CollisionSet
    temporary_objects: list[pf.MeshObject]
    storage_containers: list[pf.MeshObject]
    supports: list[pf.MeshObject]
    storages: list[pf.MeshObject]


def _snap_fixture_against_wall(
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


def _place_tap_at_back(
    tap_obj: pf.MeshObject,
    fixture: pf.MeshObject,
    back_offset: float = _TAP_BACK_OFFSET,
) -> None:
    minimum, maximum = pf.ops.attr.bbox_min_max(fixture, global_coords=False)
    location = pf.Vector(
        (
            minimum[0] + back_offset,
            (minimum[1] + maximum[1]) / 2.0,
            maximum[2] - _TAP_SURFACE_INSET,
        )
    )
    tap_obj.item().matrix_world = fixture.item().matrix_world @ pf.Matrix.Translation(
        location
    )


def _mesh_objects(results: list[MeshResult]) -> list[pf.MeshObject]:
    return [result.mesh for result in results]


def _with_colliders(
    colliders: ccol.CollisionSet,
    objects: list[pf.MeshObject],
) -> ccol.CollisionSet:
    return ccol.collision_set(colliders.objs + objects, cache=colliders)


def _toilet_against_wall_rand(
    rng: pf.RNG,
    wall_planes: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
) -> list[MeshResult]:
    rng_toilet, rng_place = rng.spawn(2)
    result = toilet.toilet_rand(rng_toilet)
    minimum, _ = pf.ops.attr.bbox_min_max(result.mesh, global_coords=False)
    pf.ops.object.set_transform(result.mesh, location=(0.0, 0.0, 0.001 - minimum[2]))
    wall_colliders = ccol.collision_set(wall_planes)

    def valid(fixture: pf.MeshObject) -> bool:
        return bathroom_fixture_placement_valid(
            fixture, wall_colliders, colliders, _WALL_MARGIN
        )

    placed = retry_place(
        rng_place,
        result,
        colliders,
        _snap_fixture_against_wall,
        attempts=32,
        accept_fn=valid,
        parents=wall_planes,
        margin=_WALL_MARGIN,
    )
    if placed is None:
        delete_object(result.mesh.item())
        return []
    return [placed]


def _bathtub_wall_hardware_rand(rng: pf.RNG) -> list[MeshResult]:
    return [bathroom_hardware.hardware_bathroom_rand(rng)]


def _place_bathtub_composite(
    rng: pf.RNG,
    child: MeshResult,
    parents: list[pf.MeshObject],
    child_side: str,
    margin: float,
    child_matrix: pf.Matrix,
    tap_result: MeshResult,
    tap_matrix: pf.Matrix,
    hardware: list[MeshResult],
) -> None:
    rng_wall, rng_place, rng_offset, rng_hardware_height = rng.spawn(4)
    wall = rng_wall.choice(parents)
    placement = pf.random.uniform(rng_place, 0.0, 1.0)
    wall_offset = pf.random.uniform(rng_offset, 0.0, 0.2)
    snap_to_plane(
        child.mesh,
        wall,
        placement=placement,
        margin=margin + wall_offset,
        child_side=child_side,
        parent_side="front",
    )
    delta = child.mesh.item().matrix_world @ child_matrix.inverted()
    tap_result.mesh.item().matrix_world = delta @ tap_matrix
    bathtub_top = pf.ops.attr.bbox_min_max(child.mesh, global_coords=True)[1][2]
    for result in hardware:
        hardware_minimum, _ = pf.ops.attr.bbox_min_max(result.mesh, global_coords=False)
        location = pf.Vector(result.mesh.item().location)
        location.z = bathtub_top + pf.random.uniform(rng_hardware_height, 0.15, 0.4)
        location.z -= hardware_minimum[2]
        pf.ops.object.set_transform(result.mesh, location=location)
        snap_to_plane(
            result.mesh,
            wall,
            placement=placement,
            margin=margin,
            child_side="back",
            parent_side="front",
        )


def _bathtub_components_valid(
    bathtub_mesh: pf.MeshObject,
    tap_result: MeshResult,
    hardware: list[MeshResult],
    colliders: ccol.CollisionSet,
    wall_colliders: ccol.CollisionSet,
    margin: float,
) -> bool:
    if ccol.intersection_test(colliders, tap_result.mesh):
        return False
    component_colliders = _with_colliders(colliders, [bathtub_mesh, tap_result.mesh])
    for result in hardware:
        if not back_face_grounded(result.mesh, wall_colliders, margin):
            return False
        if ccol.intersection_test(component_colliders, result.mesh):
            return False
    return True


def _bathtub_against_wall_rand(
    rng: pf.RNG,
    wall_planes: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
) -> _BathtubSetupResult:
    (
        rng_side,
        rng_bathtub,
        rng_tap,
        rng_hardware_choice,
        rng_hardware,
        rng_place,
        rng_tap_placement,
    ) = rng.spawn(7)
    child_side = pf.control.choice(rng_side, [("left", 1.0), ("right", 1.0)])
    result = bathtub.bathtub_rand(rng_bathtub)
    result.mesh.item().rotation_mode = "XYZ"
    pf.ops.object.set_transform(result.mesh, rotation_euler=(0.0, 0.0, 0.0))
    tap_result = tap.tap_rand(rng_tap)
    tap_minimum, _ = pf.ops.attr.bbox_min_max(tap_result.mesh, global_coords=False)
    tap_back_margin = pf.random.uniform(rng_tap_placement, 0.01, 0.1)
    _place_tap_at_back(tap_result.mesh, result.mesh, tap_back_margin - tap_minimum[0])
    hardware_func = pf.control.choice(
        rng_hardware_choice,
        [(_bathtub_wall_hardware_rand, 1.0), (lambda _: [], 1.0)],
    )
    hardware = hardware_func(rng_hardware)
    result_matrix = result.mesh.item().matrix_world.copy()
    tap_matrix = tap_result.mesh.item().matrix_world.copy()
    wall_colliders = ccol.collision_set(wall_planes)

    def valid(bathtub_mesh: pf.MeshObject) -> bool:
        return _bathtub_components_valid(
            bathtub_mesh,
            tap_result,
            hardware,
            colliders,
            wall_colliders,
            _WALL_MARGIN,
        )

    placed = retry_place(
        rng_place,
        result,
        colliders,
        _place_bathtub_composite,
        attempts=32,
        accept_fn=valid,
        parents=wall_planes,
        child_side=child_side,
        margin=_WALL_MARGIN,
        child_matrix=result_matrix,
        tap_result=tap_result,
        tap_matrix=tap_matrix,
        hardware=hardware,
    )
    if placed is None:
        delete_objects(
            [obj.item() for obj in _mesh_objects([result, tap_result, *hardware])]
        )
        return _BathtubSetupResult([], [], [])
    return _BathtubSetupResult([result], [tap_result], hardware)


def _bathtub_objects(setup: _BathtubSetupResult) -> list[pf.MeshObject]:
    return _mesh_objects(setup.bathtubs + setup.taps + setup.hardware)


def _snap_hardware_adjacent_rand(
    rng: pf.RNG,
    hardware: MeshResult,
    parents: list[pf.MeshObject],
) -> None:
    rng_sides, rng_target, rng_height, rng_offset, rng_gap = rng.spawn(5)
    child_side, parent_side, constraint_local = pf.control.choice(
        rng_sides,
        [
            (("right", "left", (0.0, 0.0, 1.0)), 1.0),
            (("left", "right", (0.0, 0.0, 1.0)), 1.0),
            (("bottom", "top", (0.0, 1.0, 0.0)), 1.0),
        ],
    )
    target = rng_target.choice(parents)
    target_minimum, target_maximum = pf.ops.attr.bbox_min_max(
        target, global_coords=False
    )
    hardware_minimum, hardware_maximum = pf.ops.attr.bbox_min_max(
        hardware.mesh, global_coords=False
    )
    target_width = target_maximum[1] - target_minimum[1]
    lateral_offset = target_width * pf.random.uniform(rng_offset, -0.35, 0.35)
    target_center = (target_minimum[2] + target_maximum[2]) / 2.0
    hardware_center = (hardware_minimum[2] + hardware_maximum[2]) / 2.0
    height_offset = target_center - hardware_center
    height_offset += pf.random.uniform(rng_height, -0.2, 0.2)
    target_rotation = target.item().matrix_world.to_3x3()
    location = pf.Vector(target.item().location)
    location += target_rotation @ pf.Vector((0.0, lateral_offset, 0.0))
    location.z += height_offset
    pf.ops.object.set_transform(
        hardware.mesh,
        location=location,
        rotation_euler=target.item().rotation_euler,
    )
    snap_to_plane(
        hardware.mesh,
        target,
        placement=0.0,
        margin=pf.random.uniform(rng_gap, 0.04, 0.18),
        child_side=child_side,
        parent_side=parent_side,
        constraint_axis=target_rotation @ pf.Vector(constraint_local),
    )


def snap_wall_hardware_rand(
    rng: pf.RNG,
    hardware: MeshResult,
    parents: list[pf.MeshObject],
    wall_colliders: ccol.CollisionSet,
    colliders: ccol.CollisionSet,
) -> MeshResult | None:
    """Wall-mount `hardware` beside or above one of `parents`, retrying until its
    back sits flat on `wall_colliders` and it clears `colliders`."""

    def grounded(obj: pf.MeshObject) -> bool:
        return back_face_grounded(obj, wall_colliders, _WALL_MARGIN)

    return retry_place(
        rng,
        hardware,
        colliders,
        _snap_hardware_adjacent_rand,
        attempts=8,
        accept_fn=grounded,
        parents=parents,
    )


def _bathroom_hardware_objects_rand(
    rng: pf.RNG,
    parents: list[pf.MeshObject],
    wall_planes: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
) -> tuple[list[MeshResult], ccol.CollisionSet]:
    if not parents:
        return [], colliders
    rng_count, rng_hardware = rng.spawn(2)
    n = pf.random.randint(rng_count, 0, 4)
    placed_hardware = []
    wall_colliders = ccol.collision_set(wall_planes)
    for rng_item in rng_hardware.spawn(n):
        rng_generate, rng_place = rng_item.spawn(2)
        hardware = bathroom_hardware.hardware_bathroom_rand(rng_generate)
        placed = snap_wall_hardware_rand(
            rng_place, hardware, parents, wall_colliders, colliders
        )
        if placed is None:
            delete_object(hardware.mesh.item())
            continue
        placed_hardware.append(placed)
        colliders = _with_colliders(colliders, [placed.mesh])
    logger.info(
        "Placed %d bathroom hardware out of %d attempts", len(placed_hardware), n
    )
    return placed_hardware, colliders


def _name_materials(obj: pf.MeshObject, base: str) -> None:
    for i, slot in enumerate(obj.item().material_slots):
        if slot.material is not None:
            slot.material.name = f"{base}_{i}"


def _name_objects(
    objects: list[pf.MeshObject],
    category: str,
) -> list[pf.MeshObject]:
    for i, obj in enumerate(objects):
        name = f"{category}_{i:02d}"
        obj.item().name = name
        _name_materials(obj, name)
    return objects


def _fixture_result(
    sink_setup: BathroomSinkSetupResult,
    toilets: list[MeshResult],
    bathtub_setup: _BathtubSetupResult,
    hardware: list[MeshResult],
    colliders: ccol.CollisionSet,
    temporary_objects: list[pf.MeshObject],
) -> BathroomSetupResult:
    sinks = _mesh_objects(sink_setup.bathroom_sinks)
    bathtubs = _mesh_objects(bathtub_setup.bathtubs)
    toilet_objects = _mesh_objects(toilets)
    storages = _mesh_objects(sink_setup.storages)
    named_objects = {
        "bathroom_sink": _name_objects(sinks, "bathroom_sink"),
        "sink_tap": _name_objects(_mesh_objects(sink_setup.sink_taps), "sink_tap"),
        "sink_support": _name_objects(
            _mesh_objects(sink_setup.sink_supports), "sink_support"
        ),
        "mirror": _name_objects(_mesh_objects(sink_setup.mirrors), "mirror"),
        "bathroom_wall_storage": _name_objects(
            _mesh_objects(sink_setup.wall_storage), "bathroom_wall_storage"
        ),
        "toilet": _name_objects(toilet_objects, "toilet"),
        "bathtub": _name_objects(bathtubs, "bathtub"),
        "bathtub_tap": _name_objects(_mesh_objects(bathtub_setup.taps), "bathtub_tap"),
        "bathtub_hardware": _name_objects(
            _mesh_objects(bathtub_setup.hardware), "bathtub_hardware"
        ),
        "bathroom_hardware": _name_objects(
            _mesh_objects(hardware), "bathroom_hardware"
        ),
    }
    all_objects = [obj for objects in named_objects.values() for obj in objects]
    return BathroomSetupResult(
        named_objects=named_objects,
        all_objects=all_objects,
        colliders=colliders,
        temporary_objects=temporary_objects,
        storage_containers=sinks + bathtubs + storages,
        supports=sinks + bathtubs + toilet_objects + storages,
        storages=storages,
    )


def _bathroom_around_sink_rand(
    rng: pf.RNG,
    sink_setup: BathroomSinkSetupResult,
    wall_planes: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
) -> BathroomSetupResult:
    rng_toilet, rng_bathtub, rng_hardware = rng.spawn(3)
    sinks = _mesh_objects(sink_setup.bathroom_sinks)
    sink_clearances = [bathroom_fixture_clearance(obj) for obj in sinks]
    colliders = _with_colliders(colliders, sink_setup.all_objects + sink_clearances)
    toilets = _toilet_against_wall_rand(rng_toilet, wall_planes, colliders)
    toilet_objects = _mesh_objects(toilets)
    toilet_clearances = [bathroom_fixture_clearance(obj) for obj in toilet_objects]
    colliders = _with_colliders(colliders, toilet_objects + toilet_clearances)
    bathtub_setup = _bathtub_against_wall_rand(rng_bathtub, wall_planes, colliders)
    colliders = _with_colliders(colliders, _bathtub_objects(bathtub_setup))
    hardware_targets = (
        sinks
        + _mesh_objects(sink_setup.mirrors)
        + _mesh_objects(sink_setup.wall_storage)
        + toilet_objects
    )
    hardware, colliders = _bathroom_hardware_objects_rand(
        rng_hardware, hardware_targets, wall_planes, colliders
    )
    if not sinks or not toilets:
        logger.warning("Bathroom setup is missing a sink or toilet")
    return _fixture_result(
        sink_setup,
        toilets,
        bathtub_setup,
        hardware,
        colliders,
        sink_clearances + toilet_clearances,
    )


def _generated_sink_bathroom_rand(
    rng: pf.RNG,
    wall_planes: list[pf.MeshObject],
    room_height: float,
    colliders: ccol.CollisionSet,
) -> BathroomSetupResult:
    rng_sink, rng_rest = rng.spawn(2)
    sink_setup = _placed_sink_setup_rand(rng_sink, wall_planes, room_height, colliders)
    return _bathroom_around_sink_rand(rng_rest, sink_setup, wall_planes, colliders)


def _bathroom_setup_demo_rand(rng: pf.RNG) -> BathroomSetupResult:
    wall = standalone_wall_planes(length=5.0, height=3.0)[0]
    setup = _generated_sink_bathroom_rand(rng, [wall], 3.0, ccol.collision_set([wall]))
    delete_object(wall.item())
    ground = pf.ops.primitives.mesh_single_vertex()
    ground.item().name = "bathroom_setup_ground"
    named_objects = dict(setup.named_objects)
    named_objects["bathroom_setup_ground"] = [ground]
    all_objects = [obj for objects in named_objects.values() for obj in objects]
    return BathroomSetupResult(
        named_objects=named_objects,
        all_objects=all_objects,
        colliders=ccol.collision_set(all_objects + setup.temporary_objects),
        temporary_objects=setup.temporary_objects,
        storage_containers=setup.storage_containers,
        supports=setup.supports,
        storages=setup.storages,
    )


@pf.tracer.grammar
def bathroom_setup_rand(
    rng: pf.RNG,
    wall_planes: list[pf.MeshObject] | None = None,
    bbox_min: pf.Vector | None = None,
    bbox_max: pf.Vector | None = None,
    colliders: ccol.CollisionSet | None = None,
) -> BathroomSetupResult:
    if wall_planes is None:
        return _bathroom_setup_demo_rand(rng)
    if bbox_min is None:
        bbox_min = pf.Vector((0.0, 0.0, 0.0))
    if bbox_max is None:
        bbox_max = pf.Vector((5.0, 5.0, 3.0))
    if colliders is None:
        colliders = ccol.collision_set(wall_planes)
    room_height = bbox_max.z - bbox_min.z
    return _generated_sink_bathroom_rand(rng, wall_planes, room_height, colliders)


@pf.tracer.grammar
def bathroom_existing_sink_setup_rand(
    rng: pf.RNG,
    sink_obj: pf.MeshObject,
    wall_planes: list[pf.MeshObject],
    bbox_min: pf.Vector | None = None,
    bbox_max: pf.Vector | None = None,
    colliders: ccol.CollisionSet | None = None,
) -> BathroomSetupResult:
    if bbox_min is None:
        bbox_min = pf.Vector((0.0, 0.0, 0.0))
    if bbox_max is None:
        bbox_max = pf.Vector((5.0, 5.0, 3.0))
    if colliders is None:
        colliders = ccol.collision_set(wall_planes)
    rng_sink, rng_rest = rng.spawn(2)
    sink_setup = _existing_sink_setup_rand(
        rng_sink, sink_obj, wall_planes, bbox_max.z - bbox_min.z, colliders
    )
    return _bathroom_around_sink_rand(rng_rest, sink_setup, wall_planes, colliders)


def _translate_objects(objects: list[pf.MeshObject], translation: pf.Vector) -> None:
    for obj in objects:
        location = pf.Vector(obj.item().location) + translation
        pf.ops.object.set_transform(obj, location=location)


def _sink_bathroom_for_setup_rand(
    rng: pf.RNG,
    width: float,
    size: float,
    depth: float,
    surface_material: pf.Material | None,
    metal_material: pf.Material | None,
) -> MeshResult:
    return bathtub.sink_bathroom_rand(
        rng,
        width=width,
        size=size,
        depth=depth,
        surface_material=surface_material,
        metal_material=metal_material,
    )


def _sink_kitchen_for_setup_rand(
    rng: pf.RNG,
    width: float,
    size: float,
    depth: float,
    surface_material: pf.Material | None,
    metal_material: pf.Material | None,
) -> MeshResult:
    del metal_material
    rim_margin = 0.03
    tap_margin = 0.11
    result = sink.sink_rand(
        rng,
        material=surface_material,
        width=width - rim_margin,
        depth=size - rim_margin - tap_margin,
        upper_height=depth,
        margin=rim_margin,
        water_tap_margin=tap_margin,
    )
    delete_object(result.cutter.item())
    return BareMeshResult(result.mesh)


def _bathroom_storage_rand(
    rng: pf.RNG,
    dimensions: pf.Vector,
) -> storage.StorageResult:
    rng_choice, rng_storage = rng.spawn(2)
    storage_func = pf.control.choice(
        rng_choice,
        [
            (storage.storage_composite_rand, 1.0),
            (storage.storage_cabinet_with_door_rand, 1.0),
        ],
    )
    return storage_func(rng_storage, dimensions=dimensions)


def _pedestal_sink_support_rand(
    rng: pf.RNG,
    support_height: float,
    sink_width: float,
    sink_depth: float,
    surface_material: pf.Material | None,
) -> _SinkSupportResult:
    rng_dimensions, rng_shape = rng.spawn(2)
    obj = bathtub.sink_pedestal(
        height=support_height,
        top_radius=sink_width * pf.random.uniform(rng_dimensions, 0.075, 0.1),
        bottom_radius=sink_width * pf.random.uniform(rng_dimensions, 0.1, 0.3),
        is_circular=pf.control.choice(rng_shape, [(True, 1.0), (False, 1.0)]),
        material=surface_material,
    ).mesh
    pf.ops.modifier.subdivide_surface(obj, levels=2, _skip_apply=True)
    pf.ops.object.set_transform(obj, location=(sink_depth / 2.0, 0.0, 0.001))
    return _SinkSupportResult([bathtub.SinkPedestalResult(mesh=obj)], [])


def _cabinet_sink_support_rand(
    rng: pf.RNG,
    support_height: float,
    sink_width: float,
    sink_depth: float,
    surface_material: pf.Material | None,
) -> _SinkSupportResult:
    del surface_material
    rng_inset, rng_storage = rng.spawn(2)
    base_inset = pf.random.uniform(rng_inset, 0.0, 0.1)
    support_depth = max(sink_depth - base_inset, 0.1)
    support_width = max(sink_width - base_inset, 0.1)
    cabinet = _bathroom_storage_rand(
        rng_storage,
        pf.Vector((support_depth, support_width, support_height)),
    )
    cabinet.mesh.item().name = "sink_cabinet"
    pf.ops.object.set_transform(
        cabinet.mesh,
        location=(0.0, -support_width / 2.0, 0.001),
    )
    return _SinkSupportResult([cabinet], [cabinet])


def _sink_support_rand(
    rng: pf.RNG,
    support_height: float,
    sink_width: float,
    sink_depth: float,
    surface_material: pf.Material | None,
) -> _SinkSupportResult:
    rng_choice, rng_support = rng.spawn(2)
    support_func = pf.control.choice(
        rng_choice,
        [
            (lambda *_: _SinkSupportResult([], []), 1.0),
            (_pedestal_sink_support_rand, 1.0),
            (_cabinet_sink_support_rand, 1.0),
        ],
    )
    return support_func(
        rng_support, support_height, sink_width, sink_depth, surface_material
    )


def _sink_unit_rand(
    rng: pf.RNG,
    height: float | None = None,
    width: float | None = None,
    size: float | None = None,
    depth: float | None = None,
    surface_material: pf.Material | None = None,
    metal_material: pf.Material | None = None,
    tap_material: pf.Material | None = None,
) -> _SinkUnit:
    rng_dimensions, rng_sink_choice, rng_sink, rng_tap, rng_support = rng.spawn(5)
    if height is None:
        height = pf.random.uniform(rng_dimensions, 0.8, 0.9)
    if width is None:
        width = pf.random.uniform(rng_dimensions, 0.5, 0.75)
    if size is None:
        size = pf.random.uniform(rng_dimensions, 0.4, 0.55)
    if depth is None:
        depth = pf.random.uniform(rng_dimensions, 0.14, 0.22)
    sink_func = pf.control.choice(
        rng_sink_choice,
        [(_sink_bathroom_for_setup_rand, 3.0), (_sink_kitchen_for_setup_rand, 1.0)],
    )
    sink_result = sink_func(
        rng_sink, width, size, depth, surface_material, metal_material
    )
    tap_result = tap.tap_rand(rng_tap, material=tap_material)
    _place_tap_at_back(tap_result.mesh, sink_result.mesh)
    minimum, maximum = pf.ops.attr.bbox_min_max(sink_result.mesh, global_coords=False)
    support_height = height - (maximum[2] - minimum[2])
    support = _sink_support_rand(
        rng_support,
        support_height,
        maximum[1] - minimum[1],
        maximum[0] - minimum[0],
        surface_material,
    )
    translation = pf.Vector(
        (
            -minimum[0],
            -(minimum[1] + maximum[1]) / 2.0,
            support_height + 0.001 - minimum[2],
        )
    )
    _translate_objects([sink_result.mesh, tap_result.mesh], translation)
    return _SinkUnit(sink_result, tap_result, support.supports, support.storages)


def _existing_sink_unit_rand(
    rng: pf.RNG,
    sink_obj: pf.MeshObject,
) -> _SinkUnit:
    rng_tap, rng_support = rng.spawn(2)
    tap_result = tap.tap_rand(rng_tap)
    _place_tap_at_back(tap_result.mesh, sink_obj)
    minimum, maximum = pf.ops.attr.bbox_min_max(sink_obj, global_coords=False)
    sink_matrix = sink_obj.item().matrix_world
    origin = sink_matrix.translation
    floor_local = sink_matrix.inverted() @ pf.Vector((origin.x, origin.y, 0.001))
    support = _sink_support_rand(
        rng_support,
        max(minimum[2] - floor_local.z, 0.05),
        maximum[1] - minimum[1],
        maximum[0] - minimum[0],
        None,
    )
    unit_origin = pf.Vector(
        (minimum[0], (minimum[1] + maximum[1]) / 2.0, floor_local.z - 0.001)
    )
    unit_matrix = sink_matrix @ pf.Matrix.Translation(unit_origin)
    for result in support.supports:
        result.mesh.item().matrix_world = unit_matrix @ result.mesh.item().matrix_world
    sink_result = BareMeshResult(sink_obj)
    return _SinkUnit(sink_result, tap_result, support.supports, support.storages)


def _attach_above_sink(
    obj: pf.MeshObject,
    sink_obj: pf.MeshObject,
    gap: float,
) -> None:
    sink_minimum, sink_maximum = pf.ops.attr.bbox_min_max(sink_obj, global_coords=False)
    obj_minimum, obj_maximum = pf.ops.attr.bbox_min_max(obj, global_coords=False)
    translation = pf.Vector(
        (
            sink_minimum[0] - obj_minimum[0],
            (sink_minimum[1] + sink_maximum[1]) / 2.0
            - (obj_minimum[1] + obj_maximum[1]) / 2.0,
            sink_maximum[2] + gap - obj_minimum[2],
        )
    )
    pf.ops.object.set_transform(obj, location=translation)
    obj.item().matrix_world = sink_obj.item().matrix_world @ obj.item().matrix_world


def _available_height(
    sink_obj: pf.MeshObject,
    room_height: float,
    gap: float,
) -> float:
    _, sink_world_maximum = pf.ops.attr.bbox_min_max(sink_obj, global_coords=True)
    return room_height - sink_world_maximum[2] - gap - 0.02


def _mirror_wall_feature_rand(
    rng: pf.RNG,
    sink_result: MeshResult,
    room_height: float,
) -> _WallFeatureResult:
    rng_width, rng_height, rng_depth, rng_gap, rng_mirror = rng.spawn(5)
    sink_minimum, sink_maximum = pf.ops.attr.bbox_min_max(
        sink_result.mesh, global_coords=False
    )
    gap = pf.random.uniform(rng_gap, 0.04, 0.2)
    height = pf.random.uniform(rng_height, 0.55, 1.8)
    height = max(
        0.55, min(height, _available_height(sink_result.mesh, room_height, gap))
    )
    sink_width = sink_maximum[1] - sink_minimum[1]
    dimensions = pf.Vector(
        (
            pf.random.uniform(rng_depth, 0.02, 0.04),
            sink_width * pf.random.uniform(rng_width, 0.75, 1.5),
            height,
        )
    )
    mirror = wall_art.mirror_rand(rng_mirror, dimensions=dimensions)
    mirror.mesh.item().name = "mirror"
    _attach_above_sink(mirror.mesh, sink_result.mesh, gap)
    return _WallFeatureResult([mirror], [])


def _storage_wall_feature_rand(
    rng: pf.RNG,
    sink_result: MeshResult,
    room_height: float,
) -> _WallFeatureResult:
    rng_depth, rng_height, rng_gap, rng_storage = rng.spawn(4)
    sink_minimum, sink_maximum = pf.ops.attr.bbox_min_max(
        sink_result.mesh, global_coords=False
    )
    gap = pf.random.uniform(rng_gap, 0.4, 0.5)
    depth = pf.random.uniform(rng_depth, 0.07, 0.2)
    height = pf.random.uniform(rng_height, 0.36, 0.96)
    height = max(
        0.36, min(height, _available_height(sink_result.mesh, room_height, gap))
    )
    sink_width = sink_maximum[1] - sink_minimum[1]
    cabinet = _bathroom_storage_rand(
        rng_storage,
        dimensions=pf.Vector((depth, sink_width, height)),
    )
    cabinet.mesh.item().name = "bathroom_wall_storage"
    _attach_above_sink(cabinet.mesh, sink_result.mesh, gap)
    return _WallFeatureResult([], [cabinet])


def _wall_feature_rand(
    rng: pf.RNG,
    sink_result: MeshResult,
    room_height: float,
) -> _WallFeatureResult:
    rng_choice, rng_feature = rng.spawn(2)
    feature_func = pf.control.choice(
        rng_choice,
        [
            (lambda *_: _WallFeatureResult([], []), 0.5),
            (_mirror_wall_feature_rand, 1.0),
            (_storage_wall_feature_rand, 1.0),
        ],
    )
    return feature_func(rng_feature, sink_result, room_height)


def _sink_setup_result(
    unit: _SinkUnit,
    supports: list[MeshResult],
    storages: list[MeshResult],
    features: _WallFeatureResult,
) -> BathroomSinkSetupResult:
    all_results = (
        [unit.sink, unit.tap] + supports + features.mirrors + features.wall_storage
    )
    return BathroomSinkSetupResult(
        bathroom_sinks=[unit.sink],
        sink_taps=[unit.tap],
        sink_supports=supports,
        mirrors=features.mirrors,
        wall_storage=features.wall_storage,
        storages=storages + features.wall_storage,
        all_objects=_mesh_objects(all_results),
    )


@pf.tracer.grammar
def bathroom_sink_setup_rand(
    rng: pf.RNG,
    bbox_min: pf.Vector | None = None,
    bbox_max: pf.Vector | None = None,
    height: float | None = None,
    width: float | None = None,
    size: float | None = None,
    depth: float | None = None,
    surface_material: pf.Material | None = None,
    metal_material: pf.Material | None = None,
    tap_material: pf.Material | None = None,
) -> BathroomSinkSetupResult:
    if bbox_min is None:
        bbox_min = pf.Vector((0.0, 0.0, 0.0))
    if bbox_max is None:
        bbox_max = pf.Vector((5.0, 5.0, 3.0))
    rng_unit, rng_feature = rng.spawn(2)
    unit = _sink_unit_rand(
        rng_unit,
        height=height,
        width=width,
        size=size,
        depth=depth,
        surface_material=surface_material,
        metal_material=metal_material,
        tap_material=tap_material,
    )
    features = _wall_feature_rand(rng_feature, unit.sink, bbox_max.z - bbox_min.z)
    return _sink_setup_result(unit, unit.supports, unit.storages, features)


def _wall_feature_attempt_rand(
    rng: pf.RNG,
    sink_result: MeshResult,
    room_height: float,
    wall_colliders: ccol.CollisionSet,
    colliders: ccol.CollisionSet,
) -> _WallFeatureResult | None:
    result = _wall_feature_rand(rng, sink_result, room_height)
    features = _mesh_objects(result.mirrors + result.wall_storage)
    grounded = all(
        back_face_grounded(obj, wall_colliders, _WALL_MARGIN) for obj in features
    )
    clear = not any(ccol.intersection_test(colliders, obj) for obj in features)
    if grounded and clear:
        return result
    delete_objects([obj.item() for obj in features])
    return None


def _sink_extras_rand(
    rng: pf.RNG,
    unit: _SinkUnit,
    wall_planes: list[pf.MeshObject],
    room_height: float,
    colliders: ccol.CollisionSet,
) -> BathroomSinkSetupResult:
    supports, feature_colliders = keep_non_colliding(unit.supports, colliders)
    kept = {result.mesh.item() for result in supports}
    delete_objects([r.mesh.item() for r in unit.supports if r.mesh.item() not in kept])
    storages = [result for result in unit.storages if result.mesh.item() in kept]
    features = repeat_attempts(
        _wall_feature_attempt_rand,
        rng,
        attempts=5,
        sink_result=unit.sink,
        room_height=room_height,
        wall_colliders=ccol.collision_set(wall_planes),
        colliders=feature_colliders,
    )
    if features is None:
        features = _WallFeatureResult([], [])
    return _sink_setup_result(unit, supports, storages, features)


def _place_sink_composite(
    rng: pf.RNG,
    child: MeshResult,
    parents: list[pf.MeshObject],
    margin: float,
    child_matrix: pf.Matrix,
    components: list[MeshResult],
    component_matrices: list[pf.Matrix],
) -> None:
    _snap_fixture_against_wall(rng, child, parents, margin)
    delta = child.mesh.item().matrix_world @ child_matrix.inverted()
    for component, matrix in zip(components, component_matrices, strict=True):
        component.mesh.item().matrix_world = delta @ matrix


def _placed_sink_setup_rand(
    rng: pf.RNG,
    wall_planes: list[pf.MeshObject],
    room_height: float,
    colliders: ccol.CollisionSet,
) -> BathroomSinkSetupResult:
    rng_unit, rng_place, rng_extras = rng.spawn(3)
    unit = _sink_unit_rand(rng_unit)
    unit.sink.mesh.item().rotation_mode = "XYZ"
    components = [unit.tap] + unit.supports
    sink_matrix = unit.sink.mesh.item().matrix_world.copy()
    component_matrices = [
        result.mesh.item().matrix_world.copy() for result in components
    ]
    wall_colliders = ccol.collision_set(wall_planes)

    def valid(fixture: pf.MeshObject) -> bool:
        return bathroom_fixture_placement_valid(
            fixture, wall_colliders, colliders, _WALL_MARGIN
        )

    placed = retry_place(
        rng_place,
        unit.sink,
        colliders,
        _place_sink_composite,
        attempts=32,
        accept_fn=valid,
        parents=wall_planes,
        margin=_WALL_MARGIN,
        child_matrix=sink_matrix,
        components=components,
        component_matrices=component_matrices,
    )
    if placed is None:
        unit_objects = _mesh_objects([unit.sink] + components)
        delete_objects([obj.item() for obj in unit_objects])
        return BathroomSinkSetupResult([], [], [], [], [], [], [])
    return _sink_extras_rand(rng_extras, unit, wall_planes, room_height, colliders)


def _existing_sink_setup_rand(
    rng: pf.RNG,
    sink_obj: pf.MeshObject,
    wall_planes: list[pf.MeshObject],
    room_height: float,
    colliders: ccol.CollisionSet,
) -> BathroomSinkSetupResult:
    rng_unit, rng_extras = rng.spawn(2)
    unit = _existing_sink_unit_rand(rng_unit, sink_obj)
    return _sink_extras_rand(rng_extras, unit, wall_planes, room_height, colliders)


def _bathroom_fixture_clearance_transform(
    fixture: pf.MeshObject,
    probe_size: float = 0.5,
    probe_height: float = 1.8,
    probe_gap: float = 0.05,
    probe_floor_clearance: float = 0.01,
) -> np.ndarray:
    minimum, maximum = pf.ops.attr.bbox_min_max(fixture, global_coords=False)
    center = (minimum + maximum) / 2.0
    center[0] = maximum[0] + probe_gap + probe_size / 2.0
    transform = np.array(fixture.item().matrix_world, dtype=np.float64)
    center_world = transform @ np.append(center, 1.0)
    center_world[2] = probe_floor_clearance + probe_height / 2.0
    transform[:3, 3] = center_world[:3]
    return transform


def bathroom_fixture_clearance(
    fixture: pf.MeshObject,
    probe_size: float = 0.5,
    probe_height: float = 1.8,
    probe_gap: float = 0.05,
    probe_floor_clearance: float = 0.01,
) -> pf.MeshObject:
    transform = _bathroom_fixture_clearance_transform(
        fixture,
        probe_size,
        probe_height,
        probe_gap,
        probe_floor_clearance,
    )
    clearance = pf.ops.primitives.mesh_cube(size=1.0)
    pf.ops.mesh.transform(
        clearance,
        scale=(probe_size, probe_size, probe_height),
    )
    clearance.item().matrix_world = pf.Matrix(transform)
    clearance.item().name = "bathroom_fixture_clearance"
    clearance.item().hide_render = True
    clearance.item().display_type = "WIRE"
    return clearance


def bathroom_fixture_front_clear(
    fixture: pf.MeshObject,
    colliders: ccol.CollisionSet,
    probe_size: float = 0.5,
    probe_height: float = 1.8,
    probe_gap: float = 0.05,
    probe_floor_clearance: float = 0.01,
) -> bool:
    transform = _bathroom_fixture_clearance_transform(
        fixture,
        probe_size,
        probe_height,
        probe_gap,
        probe_floor_clearance,
    )
    return not ccol.box_intersection_test(
        colliders,
        transform,
        size=(probe_size, probe_size, probe_height),
    )


def bathroom_fixture_placement_valid(
    fixture: pf.MeshObject,
    wall_colliders: ccol.CollisionSet,
    colliders: ccol.CollisionSet,
    wall_margin: float,
) -> bool:
    if not back_face_grounded(fixture, wall_colliders, wall_margin):
        return False
    return bathroom_fixture_front_clear(fixture, colliders)
