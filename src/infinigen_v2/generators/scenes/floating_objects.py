import logging
import math
from typing import NamedTuple

import mathutils
import numpy as np
import procfunc as pf

from infinigen_v2.generators.objects import (
    bookcase,
    cabinet,
    desk,
    drawers,
    flower,
    random_primitives,
    sofa,
    table,
    triangle_shelf,
    vase,
    wall_art,
)
from infinigen_v2.generators.scenes import collision_collection as ccol
from infinigen_v2.generators.util.external_assets import pregenerated_asset_distribution

logger = logging.getLogger(__name__)


small_object_distribution = pregenerated_asset_distribution(
    "outputs/pregenerated_assets/indoor_small/*/export_*/export_*.obj"
)


@pf.tracer.grammar
def floating_object_asset_distribution(rng: pf.RNG) -> pf.MeshObject:
    obj_func = pf.control.choice(
        rng,
        [
            (sofa.sofa_distribution, 1.0),
            (bookcase.bookcase_distribution, 1.0),
            (cabinet.cabinet_distribution, 1.0),
            (drawers.drawers_distribution, 1.0),
            (desk.desk_distribution, 1.0),
            (table.side_table_distribution, 1.0),
            (table.coffee_table_distribution, 1.0),
            (table.cocktail_table_distribution, 1.0),
            (table.dining_table_distribution, 1.0),
            (vase.vase_distribution, 1.0),
            # (plate_rack.plate_rack_distribution, 1.0),
            # (plate_rack.plate_distribution, 1.0),
            # (plate_rack.plate_on_rack_distribution, 1.0),
            (flower.flower_distribution, 1.0),
            (triangle_shelf.triangle_shelf_distribution, 1.0),
            (wall_art.wall_art_distribution, 1.0),
            (wall_art.mirror_distribution, 0.0),
        ],
    )

    res_func = pf.control.choice(
        rng,
        [
            (random_primitives.primitives_distribution, 1.0),
            (obj_func, 3.0),
            # (small_object_distribution, 1.5),
        ],
    )

    result = res_func(rng)
    if result is None:
        result = random_primitives.primitives_distribution(rng)
    obj = result.mesh if hasattr(result, "mesh") else result
    if res_func is obj_func:
        obj.item().name = obj_func.__name__

    vec = pf.nodes.shader.coord().uv

    p = pf.random.uniform(rng, 0.0, 1.0)
    if p < 0.4:
        # override material samples coord().uv; UV-less furniture needs a layer
        if not obj.item().data.uv_layers:
            pf.ops.uv.cube_project(obj, uv_name="UVMap")
        if p < 0.2:
            mat = random_primitives.bsdf_simple_distribution(rng, vec)
        else:
            mat = random_primitives.all_materials_distribution(rng, vec)
        pf.ops.object.set_material(
            obj, surface=mat.surface, displacement=mat.displacement
        )

    return obj


def _apply_object_scale(obj: pf.MeshObject) -> None:
    """Bake the object's scale into its mesh data so colliders see a rigid transform."""
    pf.ops.mesh.transform_apply(obj, location=False, rotation=False, scale=True)


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


@pf.tracer.grammar
def floating_objects_distribution(
    rng: pf.RNG,
    colliders: ccol.CollisionSet | None = None,
    bbox: tuple[np.ndarray, np.ndarray] | None = None,
    floating_objects: list[pf.MeshObject] | None = None,
    volume_density: float = 0.125,
    check_collisions: bool = True,
) -> FloatingObjectsResult:
    """
    Args:
        colliders: Existing collision set to extend. Defaults to an empty set.
        bbox: Bounding box for placement. Defaults to a [-10, 10]^3 cube.
        floating_objects: If provided, place these objects instead of sampling new ones.
        volume_density: Roughly what percent of the box's volume should we fill when sampling.
        check_collisions: If False, do not add the placed objects to the returned collision set.
    """

    n_existing = ccol.n_colliders(colliders) if colliders is not None else 0
    logger.info(f"{floating_objects_distribution.__name__} got {n_existing} colliders")

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
            floating_object_asset_distribution(sample_rngs[i]) for i in range(n_objects)
        ]

    obj_rngs = rng.spawn(len(floating_objects))
    result = []
    for i, obj in enumerate(floating_objects):
        r = obj_rngs[i]

        if not isinstance(obj, pf.MeshObject):
            loc = pf.Vector(
                (
                    pf.random.uniform(r, all_min[0], all_max[0]),
                    pf.random.uniform(r, all_min[1], all_max[1]),
                    pf.random.uniform(r, all_min[2], all_max[2]),
                )
            )
            pf.ops.object.set_transform(obj, location=loc)
            result.append(obj)
            continue

        rot = pf.Vector(
            (
                pf.random.uniform(r, 0, 2 * math.pi),
                pf.random.uniform(r, 0, 2 * math.pi),
                pf.random.uniform(r, 0, 2 * math.pi),
            )
        )
        scale = pf.Vector(
            (
                pf.random.clip_gaussian(r, 1.0, 0.1, 0.8, 1.2),
                pf.random.clip_gaussian(r, 1.0, 0.1, 0.8, 1.2),
                pf.random.clip_gaussian(r, 1.0, 0.1, 0.8, 1.2),
            )
        )

        target_size = pf.random.clip_gaussian(rng, 0.75, 0.3, 0.3, 1.6)
        current_max = max(obj.item().dimensions)
        if current_max > 0:
            scale *= target_size / current_max

        rot_min, rot_max = rotated_bbox_extents(obj, rot, scale)
        loc_min = all_min - rot_min
        loc_max = all_max - rot_max
        too_large = loc_min > loc_max
        center = (all_min + all_max) / 2
        loc_min = np.where(too_large, center, loc_min)
        loc_max = np.where(too_large, center, loc_max)

        loc = pf.Vector(
            (
                pf.random.uniform(r, loc_min[0], loc_max[0]),
                pf.random.uniform(r, loc_min[1], loc_max[1]),
                pf.random.uniform(r, loc_min[2], loc_max[2]),
            )
        )

        pf.ops.object.set_transform(obj, location=loc, rotation_euler=rot, scale=scale)
        _apply_object_scale(obj)

        result.append(obj)

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
def point_lamp_colored_distribution(
    rng: pf.RNG,
    energy: float,
    color: tuple[float, float, float] | None = None,
    shadow_soft_size: float | None = None,
) -> pf.LightObject:
    """A point light whose emission uses a fully-random RGB color."""
    if color is None:
        color = tuple(pf.random.uniform(rng, 0.0, 1.0) for _ in range(3))
    if shadow_soft_size is None:
        shadow_soft_size = pf.random.uniform(rng, 0.01, 0.20)

    return pf.ops.primitives.light.point_lamp(
        energy=energy,
        color=color,
        shadow_soft_size=shadow_soft_size,
    )


# TODO: give lights sphere colliders so check_collisions=False can go away.
@pf.tracer.grammar
def floating_lights_distribution(
    rng: pf.RNG,
    colliders: ccol.CollisionSet | None = None,
    bbox: tuple[np.ndarray, np.ndarray] | None = None,
    n_lights: int | None = None,
    max_lights: int = 3,
    total_wattage: float | None = None,
) -> FloatingObjectsResult:
    """Create up to ``max_lights`` colored point lights and place them in ``bbox``."""
    if n_lights is None:
        n_lights = int(rng.integers(0, max_lights + 1))

    lights: list[pf.LightObject] = []
    if n_lights > 0:
        if total_wattage is None:
            total_wattage = pf.random.uniform(rng, 125.0, 625.0)
        fractions = rng.dirichlet(np.ones(n_lights))
        for i, frac in enumerate(fractions):
            light = point_lamp_colored_distribution(
                rng, energy=float(frac * total_wattage)
            )
            light.item().name = f"floating_light.{i:02d}"
            lights.append(light)

    return floating_objects_distribution(
        rng,
        colliders=colliders,
        bbox=bbox,
        floating_objects=lights,
        check_collisions=False,
    )
