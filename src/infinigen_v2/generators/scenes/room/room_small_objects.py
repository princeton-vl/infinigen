import logging
from functools import partial

import procfunc as pf

from infinigen_v2.generators.objects import random_primitives
from infinigen_v2.generators.scenes import collision_collection as ccol
from infinigen_v2.generators.scenes.room.room_furniture import (
    centered_from_col,
    instance_and_collide,
)

logger = logging.getLogger(__name__)


_SMALL_OFFSET = (0, 0, 0.002)


def _place_small_assets(
    rng: pf.RNG,
    small_assets: pf.Collection,
    surfaces: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
) -> tuple[list[pf.MeshObject], ccol.CollisionSet]:
    small_asset_instances = []
    for parent in surfaces:
        small_assets_option = pf.control.choice(
            rng,
            [
                (
                    partial(
                        instance_and_collide,
                        density=50.0,
                        offset=_SMALL_OFFSET,
                    ),
                    1.0,
                ),
                (centered_from_col, 0.5),
                (lambda *_, **__: None, 1),
            ],
        )
        res = small_assets_option(rng, parent, small_assets, colliders)
        if res is not None:
            small_asset_instances.extend(res[0])
            colliders = res[1]

    return small_asset_instances, colliders


def _small_primitive(rng: pf.RNG) -> pf.MeshObject:
    target_size = pf.random.uniform(rng, 0.05, 0.15)
    obj = random_primitives.primitives_distribution(rng, target_size=target_size).mesh
    # shift origin to bbox bottom so the 2mm placement offset gives a clean surface gap
    bmin, _ = pf.ops.attr.bbox_min_max(obj)
    pf.ops.object.set_transform(obj, location=(0, 0, -bmin[2]))
    pf.ops.mesh.transform_apply(obj)
    return obj


@pf.tracer.grammar
def room_small_objects_distribution(
    rng: pf.RNG,
    storage_surfaces: list[pf.MeshObject],
    floor: pf.MeshObject,
    colliders: ccol.CollisionSet,
) -> list[pf.MeshObject]:
    rng_small_assets = rng.spawn(1)[0]

    # TODO: replace with pregenerated_asset_distribution once assets are available:
    # from infinigen_v2.generators.util.external_assets import pregenerated_asset_distribution
    # small_object_distribution = pregenerated_asset_distribution(
    #     "outputs/pregenerated_assets/indoor_small/*/export_*/export_*.obj"
    # )
    n_small_asset_meshes = pf.random.randint(rng_small_assets, 3, 10)
    small_assets = pf.Collection(
        [
            _small_primitive(rng_small_assets.spawn(1)[0])
            for _ in range(n_small_asset_meshes)
        ]
    )

    small_asset_instances, colliders = _place_small_assets(
        rng=rng_small_assets,
        small_assets=small_assets,
        surfaces=storage_surfaces,
        colliders=colliders,
    )
    logger.info(
        f"Placed {len(small_asset_instances)} small assets on {len(storage_surfaces)} support objects"
    )

    # TODO: replace with real plant assets once pregenerated assets are available
    small_plant_instances: list[pf.MeshObject] = []
    logger.info(f"Placed {len(small_plant_instances)} small plants")

    all_small_objects = small_asset_instances + small_plant_instances
    new_objs = {
        "small_asset": small_asset_instances,
        "small_plant": small_plant_instances,
    }
    for name, objs in new_objs.items():
        for i, obj in enumerate(objs):
            obj.item().name = f"{name}_{i}"
            for j, slot in enumerate(obj.item().material_slots):
                if slot.material is not None:
                    slot.material.name = f"{name}_{i}_{j}"

    return all_small_objects
