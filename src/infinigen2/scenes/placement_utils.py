# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging
from typing import Any, Callable, TypeVar

import bpy
import numpy as np
import procfunc as pf

from infinigen2.scenes import collision_collection as ccol

__all__ = [
    "delete_object",
    "distribute_in_bbox",
    "duplicates",
    "keep_non_colliding",
    "repeat_attempts",
    "snap_to_plane",
]

T = TypeVar("T")


def repeat_attempts(
    func: Callable[..., T | None],
    rng: pf.RNG,
    attempts: int = 5,
    *args: Any,
    **kwargs: Any,
) -> T | None:
    if attempts < 1:
        raise ValueError(f"repeat_attempts needs attempts >= 1, got {attempts}")
    rngs = rng.spawn(attempts)
    for i in range(attempts):
        res = func(rngs[i], *args, **kwargs)
        if res is not None:
            return res
    if child := kwargs.get("child"):
        if isinstance(child, pf.MeshObject):
            child.item().name = child.item().name + f"_FAILED_{i}"
    return None


logger = logging.getLogger(__name__)

_SIDES = {
    "front": (0, 1),
    "back": (0, -1),
    "left": (1, -1),
    "right": (1, 1),
    "top": (2, 1),
    "bottom": (2, -1),
}


def _project(a, b):
    return (np.dot(a, b) / np.dot(b, b)) * b


def snap_to_plane(
    child: pf.Object,
    parent: pf.Object,
    placement: float = 0.5,
    child_side: str = "back",
    parent_side: str = "front",
    margin: float = 0.1,
    constraint_axis: pf.Vector = pf.Vector((0, 0, 1)),
    overhang: bool = False,
):
    """Snap child to parent using bbox sides with canonical local coordinates.

    TODO: allow constraint_axis=None and 2D offsets within bounding box free dirs

    Args:
        child: Object to position
        parent: Object to snap to
        placement: How far left or right along to place the object, relative to the parent and the constraint axis
            e.g. if constraint_axis=(0, 0, 1), placement=0.5 means centered, 0 means left edge of parent's bbox, 1 means right edge of parent's bbox.
        child_side: Which side of child touches the parent
        parent_side: Which side of parent touches the child
        margin: Gap between surfaces
        constraint_axis: Axis to constrain the placement to.
            The location wont change along this axis, and we may rotate the object around this axis to snap it.
        overhang: If true, only the center of the childs bbox must attach to the parent, rather than the entire side of the bbox touching the parent.
    """

    constraint_axis = pf.Vector(constraint_axis).normalized()

    child_axis, child_sign = _SIDES[child_side]
    parent_axis, parent_sign = _SIDES[parent_side]

    c_normal_local = pf.Vector(np.eye(3)[child_axis] * child_sign)
    p_normal_local = pf.Vector(np.eye(3)[parent_axis] * parent_sign)

    # Transform parent's normal to world space
    p_normal_world = parent.item().matrix_world.to_3x3() @ p_normal_local
    target_normal = -p_normal_world.normalized()

    c_proj = c_normal_local - c_normal_local.project(constraint_axis)
    t_proj = target_normal - target_normal.project(constraint_axis)
    angle = np.arctan2(c_proj.cross(t_proj).dot(constraint_axis), c_proj.dot(t_proj))
    rotation_euler = pf.Vector(constraint_axis) * angle
    rotation_matrix = pf.Euler(rotation_euler, "XYZ").to_matrix()

    cbb = np.stack(pf.ops.attr.bbox_min_max(child, global_coords=False), axis=0)
    c_bbside = cbb[int(child_sign > 0), child_axis]
    c_attach_base_local = _project(c_bbside, c_normal_local)

    pbb = np.stack(pf.ops.attr.bbox_min_max(parent, global_coords=False), axis=0)
    p_bbside = pbb[int(parent_sign > 0), parent_axis]
    p_attach_base_local = _project(p_bbside, p_normal_local)

    p_ca_local = parent.item().matrix_world.to_3x3().inverted() @ constraint_axis
    p_orthogonal_dir_local = pf.Vector(np.cross(p_normal_local, p_ca_local))
    p_min_point = _project(pbb[0], p_orthogonal_dir_local)
    p_max_point = _project(pbb[1], p_orthogonal_dir_local)

    c_ca_local = rotation_matrix.inverted() @ constraint_axis
    c_orthogonal_dir_local = pf.Vector(np.cross(c_normal_local, c_ca_local))
    c_min_point = _project(cbb[0], c_orthogonal_dir_local)
    c_max_point = _project(cbb[1], c_orthogonal_dir_local)

    if not overhang:
        p_min_point = p_min_point - c_min_point
        p_max_point = p_max_point - c_max_point

    p_parent_local = (
        p_attach_base_local + p_min_point + (p_max_point - p_min_point) * placement
    )
    p_parent_attach = p_parent_local + margin * p_normal_local
    p_parent_attach_global = parent.item().matrix_world @ pf.Vector(p_parent_attach)

    c_attach_base_global = pf.Vector(rotation_matrix @ pf.Vector(c_attach_base_local))

    child_offset = pf.Vector(p_parent_attach_global) - c_attach_base_global
    child_offset = child_offset - child_offset.project(constraint_axis)
    child_location = (
        pf.Vector(child.item().location).project(constraint_axis) + child_offset
    )

    pf.ops.object.set_transform(child, child_location, rotation_euler=rotation_euler)

    return child


def _compute_grid_locations(
    box_min: pf.Vector,
    box_max: pf.Vector,
    spacing: pf.Vector,
) -> np.ndarray:
    box_size = np.array(box_max) - np.array(box_min)
    counts = np.maximum(1, (box_size / np.array(spacing)).astype(int))

    fracs = [np.arange(1, c + 1) / (c + 1) for c in counts]
    grid = np.meshgrid(*fracs, indexing="ij")
    positions = np.stack([g.ravel() for g in grid], axis=-1)
    return np.array(box_min) + positions * box_size


def duplicates(
    obj: pf.Object,
    locations: np.ndarray,
    rotations: np.ndarray | None = None,
    scales: np.ndarray | None = None,
) -> list[pf.Object]:
    orig_rot = tuple(obj.item().rotation_euler)
    orig_scale = tuple(obj.item().scale)
    result = []
    for i, loc in enumerate(locations):
        dup = pf.ops.object.duplicate(obj)
        dup.item().location = tuple(loc)
        dup.item().rotation_euler = (
            tuple(rotations[i]) if rotations is not None else orig_rot
        )
        dup.item().scale = tuple(scales[i]) if scales is not None else orig_scale
        result.append(dup)
    return result


def delete_object(obj: bpy.types.Object) -> None:
    data = getattr(obj, "data", None)
    item_type = getattr(obj, "type", None)
    name = obj.name
    bpy.data.objects.remove(obj, do_unlink=True)
    if data is None or not hasattr(data, "users") or data.users != 0:
        logger.debug(f"Deleting {name} ({item_type}) but NOT deleting its data")
        return
    else:
        logger.debug(f"Deleting {name} ({item_type}) and associated data")
    if item_type == "MESH":
        bpy.data.meshes.remove(data)
    elif item_type == "LIGHT":
        bpy.data.lights.remove(data)


def _origin_relative_extents(obj: pf.Object) -> tuple[np.ndarray, np.ndarray]:
    """World-axis-aligned bbox of obj relative to its origin, given its current
    rotation and scale. Non-mesh objects (e.g. lights) are treated as points."""
    if not isinstance(obj, pf.MeshObject):
        return np.zeros(3), np.zeros(3)
    corners = np.array(obj.item().bound_box)  # (8, 3) local
    rot = np.array(pf.Euler(obj.item().rotation_euler, "XYZ").to_matrix())  # (3, 3)
    scale = np.array(obj.item().scale)
    rotated = (corners * scale) @ rot.T
    return rotated.min(axis=0), rotated.max(axis=0)


def _uniform_location(
    rng: pf.RNG, loc_min: np.ndarray, loc_max: np.ndarray
) -> pf.Vector:
    return pf.Vector(
        tuple(
            pf.random.uniform(rng, float(loc_min[k]), float(loc_max[k]))
            for k in range(3)
        )
    )


def _sample_non_colliding(
    rng: pf.RNG,
    obj: pf.MeshObject,
    loc_min: np.ndarray,
    loc_max: np.ndarray,
    colliders: ccol.CollisionSet,
    attempts: int,
) -> pf.Vector:
    loc = _uniform_location(rng, loc_min, loc_max)
    for _ in range(attempts):
        obj.item().location = tuple(loc)
        if not ccol.intersection_test(colliders, obj):
            return loc
        loc = _uniform_location(rng, loc_min, loc_max)
    return loc


def distribute_in_bbox(
    rng: pf.RNG,
    objects: list[pf.Object],
    bbox: tuple[np.ndarray, np.ndarray],
    colliders: ccol.CollisionSet | None = None,
    attempts: int = 8,
) -> list[pf.Object]:
    """Place each object at a uniformly-random location inside `bbox`, keeping its
    current rotation and scale, fitting its bounding box within the box, and (when
    `colliders` is non-empty) rejection-sampling up to `attempts` times so mesh
    placements avoid that collision set. Mutates each object's location in place.
    """
    all_min = np.asarray(bbox[0], dtype=float)
    all_max = np.asarray(bbox[1], dtype=float)
    center = (all_min + all_max) / 2

    check = colliders is not None and ccol.n_colliders(colliders) > 0
    obj_rngs = rng.spawn(len(objects))
    for i, obj in enumerate(objects):
        r = obj_rngs[i]
        ext_min, ext_max = _origin_relative_extents(obj)
        loc_min = all_min - ext_min
        loc_max = all_max - ext_max
        too_large = loc_min > loc_max
        loc_min = np.where(too_large, center, loc_min)
        loc_max = np.where(too_large, center, loc_max)

        if check and isinstance(obj, pf.MeshObject):
            loc = _sample_non_colliding(r, obj, loc_min, loc_max, colliders, attempts)
        else:
            loc = _uniform_location(r, loc_min, loc_max)
        obj.item().location = tuple(loc)

    return objects


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
