# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import functools
from typing import NamedTuple, cast

import numpy as np
import procfunc as pf

from infinigen2.objects import chair, monitor
from infinigen2.objects import desk as desk_object
from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.placement.culling import keep_non_colliding
from infinigen2.scenes.placement.snap import snap_to_plane
from infinigen2.scenes.setup_utils import (
    back_face_grounded,
    retry_place,
    snap_back_front,
)

__all__ = [
    "DeskSetupResult",
    "desk_setup_rand",
]


class DeskSetupResult(NamedTuple):
    desk: pf.MeshObject
    chair: pf.MeshObject
    monitors: list[pf.MeshObject]
    all_objects: list[pf.MeshObject]
    storage_containers: list[pf.MeshObject]
    supports: list[pf.MeshObject]
    storages: list[pf.MeshObject]


def _chair_pose_rand(rng: pf.RNG) -> tuple[float, float, float]:
    rng_placement, rng_margin, rng_yaw_active, rng_yaw, rng_yaw_sign = rng.spawn(5)
    placement = pf.random.clip_gaussian(rng_placement, 0.5, 0.15, 0.3, 0.7)
    margin = pf.random.clip_gaussian(rng_margin, 0.12, 0.16, -0.12, 0.45)
    yaw_active = pf.control.choice(rng_yaw_active, [(False, 2.0), (True, 1.0)])
    yaw = 0.0
    if yaw_active:
        yaw_magnitude = pf.random.uniform(rng_yaw, np.deg2rad(5), np.deg2rad(22))
        yaw_sign = pf.control.choice(rng_yaw_sign, [(-1.0, 1.0), (1.0, 1.0)])
        yaw = yaw_sign * yaw_magnitude
    return placement, margin, yaw


def _place_chair_rand(
    rng: pf.RNG, chair_result: chair.ChairResult, desk: pf.MeshObject
) -> None:
    placement, margin, yaw = _chair_pose_rand(rng)
    snap_to_plane(
        child=chair_result.mesh,
        parent=desk,
        placement=placement,
        child_side="front",
        parent_side="front",
        margin=margin,
    )
    chair_result.mesh.item().rotation_euler.z += yaw


def _place_desk_against_wall_rand(
    rng: pf.RNG,
    desk_result: desk_object.DeskResult,
    wall_planes: list[pf.MeshObject],
    margin: float,
) -> None:
    snap_back_front(
        rng,
        desk_result,
        wall_planes,
        placement=pf.random.uniform(rng, 0.10, 0.90),
        margin=margin,
    )


def _desk_monitor_rand(
    rng: pf.RNG,
    desk: pf.MeshObject,
    desk_dimensions: pf.Vector,
) -> list[pf.MeshObject]:
    rng_monitor, rng_params = rng.spawn(2)
    monitor_result = monitor.monitor_rand(rng_monitor)
    pf.ops.object.set_transform(
        monitor_result.mesh,
        rotation_euler=pf.Vector((0.0, 0.0, desk.item().matrix_world.to_euler().z)),
    )
    snap_to_plane(
        child=monitor_result.mesh,
        parent=desk,
        child_side="bottom",
        parent_side="top",
        margin=0.002,
        constraint_axis=None,
    )
    rear_offset = desk_dimensions.x * -0.5 + pf.random.uniform(rng_params, 0.18, 0.26)
    world_offset = desk.item().matrix_world.to_3x3() @ pf.Vector(
        (rear_offset, 0.0, 0.0)
    )
    pf.ops.object.set_transform(
        monitor_result.mesh,
        location=pf.Vector(monitor_result.mesh.item().location) + world_offset,
    )
    return [monitor_result.mesh]


def desk_setup_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    wall_planes: list[pf.MeshObject] | None = None,
    colliders: ccol.CollisionSet | None = None,
    attempts: int = 12,
) -> DeskSetupResult | None:
    """A desk and chair, snapped to a wall when wall planes are supplied."""
    if colliders is None:
        colliders = ccol.collision_set([])

    (
        rng_desk,
        rng_chair,
        rng_desk_pose,
        rng_chair_pose,
        rng_monitor_choice,
        rng_monitor,
        rng_dimensions,
    ) = rng.spawn(7)
    if dimensions is None:
        dimensions = desk_object.desk_dimensions_rand(rng_dimensions)
    desk_result = desk_object.desk_rand(rng_desk, dimensions=dimensions)
    rng_chair_choice, rng_chair_gen = rng_chair.spawn(2)
    chair_func = pf.control.choice(
        rng_chair_choice,
        [(chair.chair_office_rand, 2.0), (chair.chair_rand, 1.0)],
    )
    chair_result = chair_func(rng_chair_gen)

    if wall_planes:
        wall_margin = pf.random.uniform(rng_desk_pose, 0.03, 0.10)
        grounded = functools.partial(
            back_face_grounded,
            colliders=ccol.collision_set(cast(list[pf.Object], wall_planes)),
            margin=wall_margin,
        )
        desk_result = retry_place(
            rng_desk_pose,
            desk_result,
            colliders,
            _place_desk_against_wall_rand,
            attempts=attempts,
            accept_fn=grounded,
            wall_planes=wall_planes,
            margin=wall_margin,
        )
        if desk_result is None:
            return None

    monitor_fn = pf.control.choice(
        rng_monitor_choice,
        [(_desk_monitor_rand, 2.0), (lambda *_: [], 1.0)],
    )
    monitors = monitor_fn(rng_monitor, desk_result.mesh, dimensions)
    monitors, _ = keep_non_colliding(monitors, colliders, key=lambda mesh: mesh)

    chair_colliders = ccol.collision_set(
        [*colliders.objs, desk_result.mesh, *monitors],
        cache=colliders,
    )
    chair_result = retry_place(
        rng_chair_pose,
        chair_result,
        chair_colliders,
        _place_chair_rand,
        attempts=attempts,
        desk=desk_result.mesh,
    )
    if chair_result is None:
        return None
    desk = desk_result.mesh
    chair_obj = chair_result.mesh
    return DeskSetupResult(
        desk=desk,
        chair=chair_obj,
        monitors=monitors,
        all_objects=[desk, chair_obj, *monitors],
        storage_containers=[],
        supports=[desk],
        storages=[desk],
    )
