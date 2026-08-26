# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import re

import bpy
import numpy as np
import procfunc as pf

from infinigen2.scenes.placement import collision as ccol

__all__ = [
    "distribute_in_bbox",
    "duplicates",
    "propagate_modifiers_to_instances",
]

_MODIFIER_SKIP_PROPS = ("rna_type", "type", "name")


def _copy_modifier(src: bpy.types.Modifier, dst_obj: bpy.types.Object) -> None:
    new = dst_obj.modifiers.new(src.name, src.type)
    for prop in src.bl_rna.properties:
        if prop.is_readonly or prop.identifier in _MODIFIER_SKIP_PROPS:
            continue
        setattr(new, prop.identifier, getattr(src, prop.identifier))
    if src.type != "NODES":
        return
    for key in src.keys():
        new[key] = src[key]


def _datablock_stem(name: str) -> str:
    return re.sub(r"\.\d+$", "", name)


def _templates_by_stem(
    templates: list[pf.MeshObject],
) -> dict[str, pf.MeshObject]:
    by_stem = {}
    for template in templates:
        stem = _datablock_stem(template.item().data.name)
        if stem in by_stem:
            raise ValueError(f"templates share mesh datablock stem {stem!r}")
        by_stem[stem] = template
    return by_stem


def propagate_modifiers_to_instances(
    templates: list[pf.MeshObject], instances: list[pf.MeshObject]
) -> None:
    # to_aliases copies each template's evaluated mesh, so aliases keep its datablock stem
    by_stem = _templates_by_stem(templates)
    for inst in instances:
        stem = _datablock_stem(inst.item().data.name)
        template = by_stem.get(stem)
        if template is None:
            raise ValueError(f"alias mesh stem {stem!r} matches no template")
        for mod in template.item().modifiers:
            _copy_modifier(mod, inst.item())


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
