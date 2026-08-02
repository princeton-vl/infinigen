# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging
from typing import Callable, TypeVar

import numpy as np
import procfunc as pf

from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.placement.retry import repeat_attempts

__all__ = [
    "keep_non_colliding",
    "keep_unobstructed",
    "place_surrounding",
]

logger = logging.getLogger(__name__)

T = TypeVar("T")


def keep_non_colliding(
    items: list[T | None],
    colliders: ccol.CollisionSet,
    key: Callable[[T], pf.Object] = lambda x: x.mesh,
) -> tuple[list[T], ccol.CollisionSet]:
    """Keep each item whose `key(item)` mesh doesn't collide with `colliders` or an
    already-kept item, folding kept meshes into the set. Skips Nones; rejected items
    are dropped (not deleted) and removed later by scene autocleanup.
    """
    kept: list[T] = []
    for item in items:
        if item is None:
            continue
        mesh = key(item)
        if ccol.intersection_test(colliders, mesh):
            mesh.item().name = mesh.item().name + "_COLLIDE"
            continue
        colliders = ccol.collision_set(colliders.objs + [mesh], cache=colliders)
        kept.append(item)
    return kept, colliders


def _collider_blocks_segment(
    a: np.ndarray, b: np.ndarray, colliders: ccol.CollisionSet
) -> bool:
    d = b - a
    dist = float(np.linalg.norm(d))
    origin = np.array([[a[0], a[1], 0.3]])
    hit, idx, _ = ccol.raycast(
        colliders, origin, np.array([[d[0] / dist, d[1] / dist, 0.0]])
    )
    return len(idx) > 0 and np.linalg.norm(hit[0] - origin[0]) < dist


def keep_unobstructed(
    items: list[T | None],
    center: np.ndarray,
    colliders: ccol.CollisionSet,
    key: Callable[[T], pf.Object] = lambda x: x.mesh,
) -> list[T]:
    """Drop each item whose `key(item)` center a collider separates from `center`
    (e.g. a wall between a rug and the sofa, or beyond a wall-snapped table). Reads
    colliders but does not update them; skips Nones."""
    kept: list[T] = []
    for item in items:
        if item is None:
            continue
        mesh = key(item)
        bmin, bmax = (
            np.array(v) for v in pf.ops.attr.bbox_min_max(mesh, global_coords=True)
        )
        if _collider_blocks_segment(center[:2], (bmin[:2] + bmax[:2]) / 2, colliders):
            mesh.item().name = mesh.item().name + "_OBSTRUCTED"
            continue
        kept.append(item)
    return kept


def place_surrounding(
    rng: pf.RNG,
    parent: pf.MeshObject,
    arrange_fn: Callable[[pf.RNG, pf.MeshObject], list[pf.MeshObject]],
    colliders: ccol.CollisionSet,
    attempts: int = 4,
    min_kept: int = 2,
) -> tuple[list[pf.MeshObject], ccol.CollisionSet]:
    """Arrange items around `parent` with `arrange_fn(rng, parent)`, drop any a
    collider separates from the parent (a wall-snapped table's wall-side chairs land
    outside the room, with the wall between them and the table), cull the rest against
    `colliders`, and re-sample the whole arrangement via repeat_attempts until at least
    `min_kept` survive. `colliders` must exclude `parent` so the obstruction ray sees
    the wall behind it, not the parent itself. Suits chairs around a dining table or
    stools around a kitchen island, where one bulky design can collide every item away."""
    pmin, pmax = (
        np.array(v) for v in pf.ops.attr.bbox_min_max(parent, global_coords=True)
    )
    center = (pmin + pmax) / 2

    def attempt(r: pf.RNG) -> tuple[list[pf.MeshObject], ccol.CollisionSet] | None:
        items = arrange_fn(r, parent)
        items = keep_unobstructed(items, center, colliders, key=lambda m: m)
        kept, kept_colliders = keep_non_colliding(items, colliders, key=lambda m: m)
        if len(kept) < min_kept:
            return None
        return kept, kept_colliders

    result = repeat_attempts(attempt, rng, attempts)
    if result is None:
        logger.warning(f"place_surrounding kept < {min_kept} items every attempt")
        return [], colliders
    return result
