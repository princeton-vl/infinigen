# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

from typing import Callable, TypeVar

import procfunc as pf

from infinigen2.scenes.placement import collision as ccol

__all__ = [
    "keep_non_colliding",
]

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
        colliders = ccol.collision_set(colliders.objs + [mesh], existing=colliders)
        kept.append(item)
    return kept, colliders
