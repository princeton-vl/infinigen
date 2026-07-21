# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging
import math
from typing import NamedTuple

import mathutils
import numpy as np
import procfunc as pf

from infinigen2.lighting.point_lighting import point_lamp_colored_rand
from infinigen2.objects import (
    ceiling_light,
    chair,
    door,
    flower,
    handles,
    lamp,
    random_primitives,
    rug,
    sofa,
    storage,
    table,
    vase,
    wall_art,
    window,
)
from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.placement.distribute import distribute_in_bbox
from infinigen2.shaders.dev import bsdf_simple_rand
from infinigen2.shaders.functionality_lists import all_materials_rand

__all__ = [
    "FloatingObjectsResult",
    "ObjWithPostprocessingResult",
    "ObjectResult",
    "floating_lights_rand",
    "floating_object_asset_rand",
    "floating_objects_rand",
    "lights_with_shared_wattage_rand",
    "obj_with_postprocessing",
    "object_rand",
    "point_lamp_colored_rand",
    "rotated_bbox_extents",
]

logger = logging.getLogger(__name__)


def _override_material(rng: pf.RNG, obj: pf.MeshObject) -> pf.MeshObject:
    # override samples coord().uv, so give every object a fresh cube-projected layer
    pf.ops.uv.cube_project(obj, uv_name="UVMap")
    vec = pf.nodes.shader.coord().uv
    mat_func = pf.control.choice(
        rng,
        [
            (bsdf_simple_rand, 1.0),
            (all_materials_rand, 1.0),
        ],
    )
    mat = mat_func(rng, vec)
    pf.ops.object.set_material(obj, surface=mat.surface, displacement=mat.displacement)
    return obj


class ObjectResult(NamedTuple):
    mesh: pf.MeshObject


@pf.tracer.grammar
def object_rand(rng: pf.RNG) -> ObjectResult:
    obj_func = pf.control.choice(
        rng,
        [
            (sofa.sofa_rand, 2.0),
            (chair.chair_rand, 4.0),
            (storage.shelves_rand, 1.0),
            (storage.cabinet_with_door_rand, 1.0),
            # (drawers.drawers_rand, 1.0),
            (table.side_table_rand, 0.25),
            (table.coffee_table_rand, 0.25),
            (table.cocktail_table_rand, 0.5),
            (table.dining_table_rand, 1.0),
            (rug.rug_rand, 1.0),
            (vase.vase_rand, 2.0),
            # (plate_rack.plate_rack_rand, 1.0),
            # (plate_rack.plate_rand, 1.0),
            # (plate_rack.plate_on_rack_rand, 1.0),
            (flower.flower_rand, 2.0),
            (door.door_with_handle_rand, 1.0),
            (lamp.lamp_rand, 1.0),
            (ceiling_light.ceiling_light_rand, 0.5),
            (lambda rng: window.window_rand(rng, include_glass_pane=False), 1.0),
            (handles.handle_rand, 1.0),
            (wall_art.wall_art_rand, 1.0),
            # (wall_art.mirror_rand, 0.0),
        ],
    )

    obj = obj_func(rng).mesh
    obj.item().name = obj_func.__name__

    override_func = pf.control.choice(
        rng,
        [
            (_override_material, 2.0),
            (lambda rng, obj: obj, 3.0),
        ],
    )
    return ObjectResult(mesh=override_func(rng, obj))


class ObjWithPostprocessingResult(NamedTuple):
    mesh: pf.MeshObject


@pf.tracer.grammar
def obj_with_postprocessing(rng: pf.RNG) -> ObjWithPostprocessingResult:
    obj = object_rand(rng).mesh
    return ObjWithPostprocessingResult(mesh=_override_material(rng, obj))


@pf.tracer.grammar
def floating_object_asset_rand(rng: pf.RNG) -> pf.MeshObject:
    asset_func = pf.control.choice(
        rng,
        [
            (random_primitives.primitives_rand, 1.0),
            (object_rand, 1.5),
        ],
    )
    return asset_func(rng).mesh


def _apply_object_scale(obj: pf.MeshObject) -> None:
    """Bake the object's scale into its mesh data so colliders see a rigid transform."""
    pf.ops.mesh.transform_apply(obj, location=False, rotation=False, scale=True)


def _sample_object_transform(
    rng: pf.RNG, obj: pf.MeshObject, size_scale: float = 1.0
) -> None:
    """Sample a random rotation and size for obj, set them, and bake the scale so the
    object presents a rigid transform for collision/placement."""
    angles = [pf.random.uniform(rng, 0, 2 * math.pi) for _ in range(3)]
    axes = [pf.random.clip_gaussian(rng, 1.0, 0.1, 0.8, 1.2) for _ in range(3)]
    scale = pf.Vector(tuple(axes))

    target_size = pf.random.clip_gaussian(
        rng, 0.75 * size_scale, 0.3 * size_scale, 0.3, 1.6 * size_scale
    )
    current_max = max(obj.item().dimensions)
    if current_max > 0:
        scale *= target_size / current_max

    obj.item().rotation_euler = tuple(angles)
    obj.item().scale = tuple(scale)
    _apply_object_scale(obj)


class FloatingObjectsResult(NamedTuple):
    all_objects: list[pf.MeshObject]
    colliders: ccol.CollisionSet


def rotated_bbox_extents(
    obj: pf.MeshObject,
    rot: pf.Vector,
    scale: pf.Vector,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (min, max) of the object's local bbox corners after applying rot and scale, centered at origin."""
    corners = np.array(obj.item().bound_box) * np.array(scale)  # (8, 3)
    rot_mat = np.array(mathutils.Euler(rot, "XYZ").to_matrix())  # (3, 3)
    rotated = corners @ rot_mat.T  # (8, 3)
    return rotated.min(axis=0), rotated.max(axis=0)


def recenter_origin_to_bounds(obj: pf.MeshObject) -> None:
    """Move the object's origin to its geometry bbox center, keeping the mesh in
    world position, so rotation pivots around the center instead of an offset corner."""
    item = obj.item()
    corners = [mathutils.Vector(c) for c in item.bound_box]
    local_center = sum(corners, mathutils.Vector()) / len(corners)
    item.data.transform(mathutils.Matrix.Translation(-local_center))
    item.location += item.matrix_world.to_3x3() @ local_center


@pf.tracer.grammar
def floating_objects_rand(
    rng: pf.RNG,
    colliders: ccol.CollisionSet | None = None,
    bbox: tuple[np.ndarray, np.ndarray] | None = None,
    floating_objects: list[pf.MeshObject] | None = None,
    volume_density: float = 0.125,
    check_collisions: bool = True,
    size_scale: float = 1.0,
) -> FloatingObjectsResult:
    """
    Args:
        colliders: Existing collision set to extend. Defaults to an empty set.
        bbox: Bounding box for placement. Defaults to a [-10, 10]^3 cube.
        floating_objects: If provided, place these objects instead of sampling new ones.
        volume_density: Roughly what percent of the box's volume should we fill when sampling.
        check_collisions: If False, do not add the placed objects to the returned collision set.
        size_scale: Multiplier on the sampled object size (mean/spread/max), e.g. 2.0 for larger props.
    """

    n_existing = ccol.n_colliders(colliders) if colliders is not None else 0
    logger.info(f"{floating_objects_rand.__name__} got {n_existing} colliders")

    if bbox is None:
        bbox = (np.full(3, -10.0), np.full(3, 10.0))
    all_min, all_max = bbox

    if floating_objects is None:
        dims = all_max - all_min
        volume = np.prod(dims)
        # relies on overall scale mean being 1, and even then quite noisy
        n_objects = int(np.ceil(volume * volume_density))
        sample_rngs = rng.spawn(n_objects)
        floating_objects = [
            floating_object_asset_rand(sample_rngs[i]) for i in range(n_objects)
        ]

    obj_rngs = rng.spawn(len(floating_objects))
    for i, obj in enumerate(floating_objects):
        if isinstance(obj, pf.MeshObject):
            _sample_object_transform(obj_rngs[i], obj, size_scale=size_scale)

    place_colliders = colliders if check_collisions else None
    result = distribute_in_bbox(rng, floating_objects, bbox, place_colliders)

    if check_collisions:
        collider_candidates = [o for o in result if isinstance(o, pf.MeshObject)]
        if colliders is None:
            colliders = ccol.collision_set(collider_candidates)
        else:
            colliders = ccol.collision_set(
                collider_candidates + colliders.objs, existing=colliders
            )

    logger.info(
        f"Collision set has {ccol.n_colliders(colliders)} colliders for {len(result)} objects"
    )

    return FloatingObjectsResult(
        all_objects=result,
        colliders=colliders,
    )


@pf.tracer.grammar
def lights_with_shared_wattage_rand(
    rng: pf.RNG,
    quantity: int | None = None,
    wattage: float | None = None,
) -> list[pf.LightObject]:
    """Create ``quantity`` colored point lights whose energies sum to ``wattage``."""
    if quantity is None:
        quantity = int(rng.integers(1, 3 + 1))

    if wattage is None:
        wattage = pf.random.uniform(rng, 50.0, 800.0)

    watt_per_light = wattage * rng.dirichlet(np.ones(quantity))

    lights = []
    for i, w in enumerate(watt_per_light):
        light = point_lamp_colored_rand(rng, energy=float(w))
        light.item().name = f"floating_light.{i:02d}"
        lights.append(light)

    return lights


# TODO: give lights sphere colliders so check_collisions=False can go away.
@pf.tracer.grammar
def floating_lights_rand(
    rng: pf.RNG,
    colliders: ccol.CollisionSet | None = None,
    bbox: tuple[np.ndarray, np.ndarray] | None = None,
    lights: list[pf.LightObject] | None = None,
) -> FloatingObjectsResult:
    """Place colored point lights in ``bbox``, sampling shared-wattage lights when none given."""
    if lights is None:
        lights = lights_with_shared_wattage_rand(rng)

    return floating_objects_rand(
        rng,
        colliders=colliders,
        bbox=bbox,
        floating_objects=lights,
        check_collisions=False,
    )
