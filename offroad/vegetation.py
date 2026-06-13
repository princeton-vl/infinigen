"""Vegetation / detail pass for the off-road grassland scene.

Scatters Infinigen assets (grass, rocks, optional trees/bushes) onto the
road-anchored terrain mesh, keeping everything OFF the road corridor by reading
the baked `dist_to_road` POINT attribute (same mechanism Infinigen uses for its
`MaskTag`). Also places the exact xodr boulders.

Requires the infinigen env (bpy + infinigen importable) and gin already inited
by the caller (see build_scene.py).
"""

from __future__ import annotations

import logging

import bpy
import numpy as np
from numpy.random import uniform as U

from infinigen.core import surface
from infinigen.core.nodes.node_wrangler import Nodes
from infinigen.core.util import blender as butil

logger = logging.getLogger("offroad.vegetation")


# --------------------------------------------------------------------------- #
# Off-corridor selection: reads baked `dist_to_road`, 0 on road -> 1 past buffer
# --------------------------------------------------------------------------- #
def offroad_selection(buffer_m: float = 2.0, blend_m: float = 2.5, attr="dist_to_road"):
    def sel(nw):
        dist = surface.eval_argument(nw, attr)
        mr = nw.new_node(
            Nodes.MapRange,
            input_kwargs={"Value": dist, 1: buffer_m, 2: buffer_m + blend_m},
            attrs={"interpolation_type": "SMOOTHSTEP"},
        )
        return mr

    return sel


def offroad_noise_selection(buffer_m=2.0, blend_m=2.5, noise_scale=0.06, thresh=0.45):
    """Off-corridor mask AND a noise patchiness (for clumpy scatters like rocks)."""
    base = offroad_selection(buffer_m, blend_m)

    def sel(nw):
        off = base(nw)
        noise = nw.new_node(Nodes.NoiseTexture, input_kwargs={"Scale": noise_scale})
        gt = nw.new_node(
            Nodes.Math,
            input_args=[noise.outputs["Fac"], thresh],
            attrs={"operation": "GREATER_THAN"},
        )
        return nw.scalar_multiply(off, gt)

    return sel


# --------------------------------------------------------------------------- #
# Grass
# --------------------------------------------------------------------------- #
def apply_grass(terrain, buffer_m=2.2, vol_density=2.5):
    """Grass scatter with controllable density (Infinigen's grass.Grass hardcodes
    vol_density up to 5 -> ~1.6M instances, which is far too heavy for large /
    sparse biomes). We replicate it with an explicit density."""
    import numpy as np
    from numpy.random import uniform as U

    from infinigen.assets.objects.grassland.grass_tuft import GrassTuftFactory
    from infinigen.assets.scatters.utils.wind import wind
    from infinigen.core.placement.factory import make_asset_collection
    from infinigen.core.placement.instance_scatter import scatter_instances

    logger.info(f"scattering grass (off-corridor, vol_density={vol_density})...")
    facs = [GrassTuftFactory(np.random.randint(1e7))]
    col = make_asset_collection(facs, n=10)
    scatter_obj = scatter_instances(
        base_obj=terrain, collection=col, scale=U(1, 3), scale_rand=U(0.7, 1),
        scale_rand_axi=0.1, vol_density=vol_density, ground_offset=0,
        normal_fac=U(0, 0.5), rotation_offset=wind(strength=10),
        selection=offroad_selection(buffer_m, blend_m=2.0), taper_scale=True,
    )
    return scatter_obj


def apply_ground_cover(terrain, buffer_m=2.2):
    """A second, finer ground layer (small monocots / clovers) for density."""
    from infinigen.assets.scatters import flowerplant

    try:
        flowerplant.Flowerplant().apply(
            terrain, selection=offroad_noise_selection(buffer_m, thresh=0.6)
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"ground cover skipped: {e}")


# --------------------------------------------------------------------------- #
# Rocks: scattered pebbles + the exact xodr boulders
# --------------------------------------------------------------------------- #
def apply_pebbles(terrain, buffer_m=2.0):
    from infinigen.assets.scatters import pebbles

    logger.info("scattering pebbles (off-corridor, patchy)...")
    try:
        pebbles.Pebbles().apply(
            terrain, selection=offroad_noise_selection(buffer_m, noise_scale=0.12, thresh=0.55)
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"pebbles skipped: {e}")


def place_boulders(road, collection_name="offroad_boulders"):
    """Place the exact xodr <objects> boulders as real rocks at their world pos."""
    from infinigen.assets.objects import rocks

    col = butil.get_collection(collection_name)
    placed = []
    for i, o in enumerate(road.obstacles):
        try:
            fac = rocks.BoulderFactory(int(1000 + i), coarse=False)
            obj = fac.spawn_asset(i)
            obj.location = (o["x"], o["y"], o["z"] - 0.15)
            r = max(0.2, o["radius"])
            obj.scale = (r, r, r * U(0.7, 1.0))
            obj.rotation_euler = (0, 0, U(0, 6.28))
            butil.put_in_collection(obj, col)
            placed.append(obj)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"boulder {o['id']} skipped: {e}")
    logger.info(f"placed {len(placed)} xodr boulders")
    return placed


def apply_fern(terrain, buffer_m=2.5):
    from infinigen.assets.scatters import fern

    try:
        fern.Fern().apply(terrain, selection=offroad_noise_selection(buffer_m, thresh=0.55))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"fern skipped: {e}")


# --------------------------------------------------------------------------- #
# Generic placeholder->populate scatter for any AssetFactory (trees, cactus, ...)
# --------------------------------------------------------------------------- #
def _scatter_factory(terrain, factory_cls, n, label, buffer_m, altitude=-0.05, blend=3.0):
    from infinigen.core.placement import placement

    if n <= 0:
        return None
    logger.info(f"scattering {n} {label}...")
    seed = int(np.random.randint(1e6))
    fac_coarse = factory_cls(seed, coarse=True)
    col = placement.scatter_placeholders_mesh(
        terrain, fac_coarse, overall_density=1.0, num_placeholders=n,
        selection=offroad_selection(buffer_m, blend_m=blend), altitude=altitude,
    )
    fac_fine = factory_cls(seed, coarse=False)
    new = placement.populate_collection(fac_fine, col, cameras=None)
    logger.info(f"  populated {len(new[0])} {label}")
    return new


def scatter_trees(terrain, road, n_trees=18, n_bushes=40, buffer_m=3.0):
    from infinigen.assets.objects import trees

    for cls, n, lbl in [(trees.BushFactory, n_bushes, "bushes"), (trees.TreeFactory, n_trees, "trees")]:
        try:
            _scatter_factory(terrain, cls, n, lbl, buffer_m)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"{lbl} skipped: {e}")


def scatter_cactus(terrain, road, n=14, buffer_m=3.0):
    from infinigen.assets.objects.cactus import CactusFactory

    try:
        _scatter_factory(terrain, CactusFactory, n, "cactus", buffer_m)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"cactus skipped: {e}")


def scatter_extra_boulders(terrain, road, n=40, buffer_m=1.5):
    from infinigen.assets.objects import rocks

    try:
        _scatter_factory(terrain, rocks.BoulderFactory, n, "boulders", buffer_m, altitude=-0.2)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"extra boulders skipped: {e}")


# --------------------------------------------------------------------------- #
# Snow: white terrain + snow surface layer on terrain & assets
# --------------------------------------------------------------------------- #
def apply_snow(terrain):
    import bpy

    from infinigen.assets.scatters import snow_layer

    try:
        snow = snow_layer.Snowlayer()
        snow.apply(terrain)  # terrain-only (per-asset application is too slow)
        logger.info("snow layer applied to terrain")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"snow layer skipped: {e}")


# --------------------------------------------------------------------------- #
# Biome dispatcher
# --------------------------------------------------------------------------- #
def populate_biome(preset: dict, terrain, road, buffer_m):
    toks = preset.get("scatters", [])
    if "grass" in toks:
        apply_grass(terrain, buffer_m, vol_density=preset.get("grass_density", 2.5))
    if "grass_sparse" in toks:
        apply_grass(terrain, buffer_m + 2.0, vol_density=0.3)  # sparse alpine grass
    if "ground_cover" in toks:
        apply_ground_cover(terrain, buffer_m)
    if "fern" in toks:
        apply_fern(terrain, buffer_m)
    if "pebbles" in toks:
        apply_pebbles(terrain, road.width * 0.5)
    if "cactus" in toks:
        scatter_cactus(terrain, road, n=preset.get("cactus_n", 14), buffer_m=buffer_m + 1.0)
    if "boulders_extra" in toks:
        scatter_extra_boulders(terrain, road, n=preset.get("boulder_n", 40), buffer_m=road.width * 0.5 + 0.5)
    if "trees" in toks or "trees_dense" in toks:
        scatter_trees(terrain, road, n_trees=preset.get("tree_n", 12),
                      n_bushes=preset.get("bush_n", 36), buffer_m=buffer_m + 1.0)
    place_boulders(road)  # the exact xodr boulders, always
    if "snow_layer" in toks:
        apply_snow(terrain)
