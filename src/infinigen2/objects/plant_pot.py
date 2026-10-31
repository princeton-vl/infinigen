# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import functools
from typing import NamedTuple

import procfunc as pf

from infinigen2.objects import bowl, phyllo_plant, vase
from infinigen2.shaders.base_materials import (
    dirt,
    granite,
    gravel_concrete,
    mud,
    sand,
    soil,
)

__all__ = [
    "PlantPotResult",
    "plant_pot",
    "plant_pot_large_rand",
    "plant_pot_small_rand",
]


class PlantPotResult(NamedTuple):
    mesh: pf.MeshObject


def plant_pot(
    plant: pf.MeshObject,
    base_radius: float = 0.12,
    rim_ratio: float = 1.2,
    height: float = 0.1,
    profile_fullness: float = 0.6,
    profile_slope: float = 0.45,
    rib_count: int = 12,
    rib_depth: float = 0.0,
    twist: float = 0.0,
    material: pf.Material | None = None,
    soil_material: pf.Material | None = None,
) -> PlantPotResult:
    thickness = base_radius * 0.035
    fill_height = height * 0.85
    pot = bowl.pot(
        diameter=2.0 * base_radius * rim_ratio,
        height=height,
        base_scale=1.0 / rim_ratio,
        profile_fullness=profile_fullness,
        profile_slope=profile_slope,
        rib_count=rib_count,
        rib_depth=rib_depth,
        twist=twist,
        thickness=thickness,
        inside_height=fill_height,
        u_resolution=48,
        v_resolution=8,
        material=material,
    )
    if soil_material is None:
        soil_material = granite.granite_no_displacement_preset(
            pf.nodes.shader.coord().object
        )
    pf.ops.object.set_material(pot.inside_volume, soil_material)
    pf.ops.object.set_transform(plant, location=(0.0, 0.0, fill_height - thickness))
    pf.ops.object.join(plant, pot.mesh)
    pf.ops.object.join(plant, pot.inside_volume)
    pf.ops.mesh.transform_apply(plant)
    return PlantPotResult(mesh=plant)


def _rosette_plant(
    rng: pf.RNG,
    pot_diameter: float,
    min_ratio: float,
    max_ratio: float,
    height: float | None = None,
) -> pf.MeshObject:
    rng_size, rng_angle, rng_spread, rng_droop, rng_width, rng_plant = rng.spawn(6)
    if height is None:
        height = pot_diameter * pf.random.uniform(rng_size, min_ratio, max_ratio)
    max_angle = pf.random.uniform(rng_angle, 1.0, 1.35)
    return phyllo_plant.plant_phyllo_basal_rand(
        rng_plant,
        leaf_size=1.0,
        leaf_width=pf.random.uniform(rng_width, 0.7, 1.1),
        sparse=False,
        stem_height=height * 0.04,
        stem_radius=height * 0.06,
        leaf_length=height * 0.8,
        min_angle=max_angle * pf.random.uniform(rng_spread, 0.6, 0.85),
        max_angle=max_angle,
        leaf_droop=pf.random.uniform(rng_droop, 0.0, 0.25),
        stem_curve_x_degrees=0.0,
        stem_curve_y_degrees=0.0,
        stem_warble=0.0,
    ).mesh


def _sword_plant(
    rng: pf.RNG,
    pot_diameter: float,
    min_ratio: float,
    max_ratio: float,
    height: float | None = None,
) -> pf.MeshObject:
    rng_size, rng_angle, rng_spread, rng_width, rng_count, rng_plant = rng.spawn(6)
    if height is None:
        height = pot_diameter * pf.random.uniform(rng_size, min_ratio, max_ratio)
    max_angle = pf.random.uniform(rng_angle, 1.38, 1.52)
    return phyllo_plant.plant_phyllo_basal_rand(
        rng_plant,
        leaf_size=1.0,
        leaf_width=pf.random.uniform(rng_width, 0.25, 0.45),
        sparse=False,
        count=pf.random.randint(rng_count, 8, 22),
        stem_height=height * 0.03,
        stem_radius=height * 0.05,
        leaf_length=height * 0.73,
        min_angle=max_angle - pf.random.uniform(rng_spread, 0.15, 0.4),
        max_angle=max_angle,
        leaf_droop=0.0,
        leaf_bend=0.03,
        stem_curve_x_degrees=0.0,
        stem_curve_y_degrees=0.0,
        stem_warble=0.0,
    ).mesh


def _arching_plant(
    rng: pf.RNG,
    pot_diameter: float,
    min_ratio: float,
    max_ratio: float,
    height: float | None = None,
) -> pf.MeshObject:
    rng_size, rng_angle, rng_spread, rng_droop, rng_width, rng_plant = rng.spawn(6)
    if height is None:
        height = pot_diameter * pf.random.uniform(rng_size, min_ratio, max_ratio)
    max_angle = pf.random.uniform(rng_angle, 1.3, 1.5)
    return phyllo_plant.plant_phyllo_basal_rand(
        rng_plant,
        leaf_size=1.0,
        leaf_width=pf.random.uniform(rng_width, 0.18, 0.32),
        sparse=False,
        stem_height=height * 0.05,
        stem_radius=height * 0.05,
        leaf_length=height * 0.95,
        min_angle=max_angle * pf.random.uniform(rng_spread, 0.55, 0.75),
        max_angle=max_angle,
        leaf_droop=pf.random.uniform(rng_droop, 0.15, 0.3),
        stem_curve_x_degrees=0.0,
        stem_curve_y_degrees=0.0,
        stem_warble=0.0,
    ).mesh


def _bush_plant(
    rng: pf.RNG,
    pot_diameter: float,
    min_ratio: float,
    max_ratio: float,
    height: float | None = None,
) -> pf.MeshObject:
    rng_size, rng_stem, rng_leaf, rng_base, rng_curve, rng_plant = rng.spawn(6)
    if height is None:
        height = pot_diameter * pf.random.uniform(rng_size, min_ratio, max_ratio)
    stem_height = height * pf.random.uniform(rng_stem, 0.7, 0.9)
    return phyllo_plant.plant_phyllo_ascending_rand(
        rng_plant,
        leaf_size=1.0,
        sparse=False,
        stem_height=stem_height,
        stem_radius=0.006 + height * 0.008,
        leaf_length=height * pf.random.uniform(rng_leaf, 0.28, 0.4),
        base_fraction=pf.random.uniform(rng_base, 0.0, 0.1),
        stem_curve_x_degrees=pf.random.uniform(rng_curve, -10.0, 10.0),
        stem_curve_y_degrees=0.0,
        stem_warble=0.0,
    ).mesh


def _tree_plant(
    rng: pf.RNG,
    pot_diameter: float,
    min_ratio: float,
    max_ratio: float,
    height: float | None = None,
) -> pf.MeshObject:
    rng_size, rng_stem, rng_leaf, rng_base, rng_curve, rng_plant = rng.spawn(6)
    if height is None:
        height = pot_diameter * pf.random.uniform(rng_size, min_ratio, max_ratio)
    leaf_length = height * pf.random.uniform(rng_leaf, 0.12, 0.2)
    return phyllo_plant.plant_phyllo_ascending_rand(
        rng_plant,
        leaf_size=1.0,
        sparse=False,
        stem_height=height - leaf_length * pf.random.uniform(rng_stem, 0.1, 0.4),
        stem_radius=0.008 + height * 0.007,
        leaf_length=leaf_length,
        base_fraction=pf.random.uniform(rng_base, 0.3, 0.55),
        stem_curve_x_degrees=pf.random.uniform(rng_curve, -6.0, 6.0),
        stem_curve_y_degrees=0.0,
        stem_warble=0.01,
    ).mesh


def _plant_pot_rand(
    rng: pf.RNG,
    plant: pf.MeshObject,
    diameter: float,
    rim_ratio: float,
    height: float,
) -> PlantPotResult:
    (
        rng_material,
        rng_rib_choice,
        rng_rib_depth,
        rng_twist_choice,
        rng_twist_amount,
        rng_profile_fullness,
        rng_profile_slope,
        rng_rib_count,
        rng_soil_choice,
        rng_soil_material,
    ) = rng.spawn(10)
    material = vase.vase_material_rand(rng_material, pf.nodes.shader.coord().uv)
    decorated_depth = pf.random.uniform(rng_rib_depth, 0.025, 0.07)
    rib_depth = pf.control.choice(rng_rib_choice, [(0.0, 1.0), (decorated_depth, 1.0)])
    twist_amount = pf.random.uniform(rng_twist_amount, -0.6, 0.6)
    twist = pf.control.choice(rng_twist_choice, [(0.0, 2.0), (twist_amount, 1.0)])
    soil_material_fn = pf.control.choice(
        rng_soil_choice,
        [
            (dirt.dirt_rand, 1.0),
            (mud.mud_rand, 1.0),
            (soil.soil_rand, 1.0),
            (sand.sand_rand, 1.0),
            (granite.granite_rand, 0.5),
            (gravel_concrete.gravel_concrete_rand, 0.5),
        ],
    )
    soil_material = soil_material_fn(rng_soil_material, pf.nodes.shader.coord().object)
    soil_material = pf.Material(
        surface=soil_material.surface,
        volume=soil_material.volume,
    )
    return plant_pot(
        plant=plant,
        base_radius=diameter * 0.5 / rim_ratio,
        rim_ratio=rim_ratio,
        height=height,
        profile_fullness=pf.random.uniform(rng_profile_fullness, 0.4, 0.8),
        profile_slope=pf.random.uniform(rng_profile_slope, 0.3, 0.65),
        rib_count=pf.random.randint(rng_rib_count, 6, 13),
        rib_depth=rib_depth,
        twist=twist,
        material=material,
        soil_material=soil_material,
    )


def plant_pot_large_rand(
    rng: pf.RNG,
    pot_diameter: float | None = None,
    pot_height: float | None = None,
    plant_height: float | None = None,
) -> PlantPotResult:
    (
        rng_diameter,
        rng_rim,
        rng_height,
        rng_form,
        rng_plant,
        rng_build,
    ) = rng.spawn(6)
    if pot_diameter is None:
        pot_diameter = pf.random.uniform(rng_diameter, 0.25, 0.45)
    if pot_height is None:
        pot_height = pot_diameter * pf.random.uniform(rng_height, 0.75, 1.1)
    forms = [
        (functools.partial(_tree_plant, min_ratio=2.8, max_ratio=4.0), 1.5),
        (functools.partial(_bush_plant, min_ratio=2.0, max_ratio=3.2), 1.0),
        (functools.partial(_sword_plant, min_ratio=1.4, max_ratio=3.0), 1.0),
        (functools.partial(_arching_plant, min_ratio=1.6, max_ratio=2.4), 1.0),
        (functools.partial(_rosette_plant, min_ratio=1.4, max_ratio=2.2), 0.5),
    ]
    plant_fn = pf.control.choice(rng_form, forms)
    return _plant_pot_rand(
        rng_build,
        plant=plant_fn(rng_plant, pot_diameter, height=plant_height),
        diameter=pot_diameter,
        rim_ratio=pf.random.uniform(rng_rim, 1.05, 1.25),
        height=pot_height,
    )


def plant_pot_small_rand(
    rng: pf.RNG,
    pot_diameter: float | None = None,
    pot_height: float | None = None,
    plant_height: float | None = None,
) -> PlantPotResult:
    (
        rng_diameter,
        rng_rim,
        rng_height,
        rng_form,
        rng_plant,
        rng_build,
    ) = rng.spawn(6)
    if pot_diameter is None:
        pot_diameter = pf.random.log_uniform(rng_diameter, 0.07, 0.18)
    if pot_height is None:
        pot_height = pot_diameter * pf.random.uniform(rng_height, 0.7, 1.05)
    forms = [
        (functools.partial(_rosette_plant, min_ratio=1.0, max_ratio=2.6), 1.5),
        (functools.partial(_bush_plant, min_ratio=1.2, max_ratio=2.4), 1.0),
        (functools.partial(_sword_plant, min_ratio=1.3, max_ratio=2.6), 1.0),
        (functools.partial(_arching_plant, min_ratio=0.9, max_ratio=1.8), 0.5),
    ]
    plant_fn = pf.control.choice(rng_form, forms)
    result = _plant_pot_rand(
        rng_build,
        plant=plant_fn(rng_plant, pot_diameter, height=plant_height),
        diameter=pot_diameter,
        rim_ratio=pf.random.uniform(rng_rim, 1.05, 1.3),
        height=pot_height,
    )
    result.mesh.item().name = plant_pot_small_rand.__name__
    return result
