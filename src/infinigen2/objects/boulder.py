# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Alexander Raistrick: add python bindings / pf.random statements of ANT Landscape presets

import math
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import procfunc as pf
from mathutils import Matrix

from infinigen2.objects import ant_landscape
from infinigen2.shaders.base_materials import granite, stone_smooth
from infinigen2.shaders.functionality_lists import (
    boulder_material_rand,
    decorative_material_rand,
)

__all__ = [
    "BoulderResult",
    "boulder",
    "boulder_abstract_rand",
    "boulder_cauliflower_rand",
    "boulder_crystalline_rand",
    "boulder_flatstones_rand",
    "boulder_rand",
    "boulder_ridged_rand",
    "boulder_rock_rand",
    "rock_material_rand",
    "rock_rand",
]


class BoulderResult(NamedTuple):
    mesh: pf.MeshObject


@pf.tracer.primitive(mutates=["mutates_obj"])
def _fit_and_ground(mutates_obj: pf.MeshObject, dimensions: pf.Vector) -> None:
    obj = mutates_obj.item()
    current = np.asarray(obj.dimensions, dtype=float)
    target = np.asarray(dimensions, dtype=float)
    scale = target / current
    obj.data.transform(Matrix.Diagonal((*scale, 1.0)))

    coordinates = np.array([vertex.co[:] for vertex in obj.data.vertices])
    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)
    center = (minimum + maximum) / 2
    translation = (-center[0], -center[1], -minimum[2])
    obj.data.transform(Matrix.Translation(translation))
    obj.data.update()


@pf.tracer.generator
def boulder(
    dimensions: pf.Vector = pf.Vector((1.0, 0.85, 0.65)),
    mesh_resolution: float = 0.02,
    random_seed: int = 0,
    material: pf.Material | None = None,
    tri_face: bool = False,
    noise_size: float = 1.0,
    noise_type: str = "hetero_terrain",
    basis_type: str = "PERLIN_ORIGINAL",
    vl_basis_type: str = "PERLIN_ORIGINAL",
    distortion: float = 1.0,
    noise_depth: int = 8,
    amplitude: float = 0.5,
    frequency: float = 2.0,
    dimension: float = 1.0,
    lacunarity: float = 2.0,
    offset: float = 1.0,
    gain: float = 1.0,
    height: float = 0.5,
    height_invert: bool = False,
    height_offset: float = 0.0,
    maximum: float = 1.0,
    minimum: float = -1.0,
    strata: float = 5.0,
    strata_type: str = "0",
) -> BoulderResult:
    """Create a ground-aligned ANT boulder with exact dimensions and edge length in meters."""
    horizontal = max(dimensions.x, dimensions.y)
    longitude_resolution = max(8, math.ceil(math.pi * horizontal / mesh_resolution))
    meridian = math.pi * (horizontal + dimensions.z) / 4
    latitude_resolution = max(4, math.ceil(meridian / mesh_resolution))
    mesh = ant_landscape.landscape(
        ant_terrain_name=boulder.__name__,
        at_cursor=False,
        smooth_mesh=False,
        tri_face=tri_face,
        sphere_mesh=True,
        subdivision_x=longitude_resolution,
        subdivision_y=latitude_resolution,
        mesh_size=2.0,
        random_seed=random_seed,
        water_plane=False,
        remove_double=True,
        refresh=True,
        noise_size=noise_size,
        noise_type=noise_type,
        basis_type=basis_type,
        vl_basis_type=vl_basis_type,
        distortion=distortion,
        noise_depth=noise_depth,
        amplitude=amplitude,
        frequency=frequency,
        dimension=dimension,
        lacunarity=lacunarity,
        offset=offset,
        gain=gain,
        height=height,
        height_invert=height_invert,
        height_offset=height_offset,
        maximum=maximum,
        minimum=minimum,
        strata=strata,
        strata_type=strata_type,
    )
    _fit_and_ground(mesh, dimensions)
    if material is None:
        material = pf.Material(surface=pf.nodes.shader.principled_bsdf())
    pf.ops.object.set_material(mesh, material=material)
    return BoulderResult(mesh=mesh)


def _boulder_rand_from_params(
    rng: pf.RNG,
    dimensions: pf.Vector | None,
    material: pf.Material | None,
    mesh_resolution: float,
    dimension_profile: pf.Vector = pf.Vector((1.0, 1.0, 0.7)),
    tri_face: bool = False,
    noise_size: float = 1.0,
    noise_type: str = "hetero_terrain",
    basis_type: str = "PERLIN_ORIGINAL",
    vl_basis_type: str = "PERLIN_ORIGINAL",
    distortion: float = 1.0,
    noise_depth: int = 8,
    amplitude: float = 0.5,
    frequency: float = 2.0,
    dimension: float = 1.0,
    lacunarity: float = 2.0,
    offset: float = 1.0,
    gain: float = 1.0,
    height: float = 0.5,
    height_invert: bool = False,
    height_offset: float = 0.0,
    maximum: float = 1.0,
    minimum: float = -1.0,
    strata: float = 5.0,
    strata_type: str = "0",
) -> BoulderResult:
    rd0, rd1, rd2, rd3, rng_material, rng_seed = rng.spawn(6)
    if dimensions is None:
        diameter = pf.random.uniform(rd0, 0.3, 2.0)
        x = diameter * dimension_profile.x * pf.random.uniform(rd1, 0.85, 1.15)
        y = diameter * dimension_profile.y * pf.random.uniform(rd2, 0.85, 1.15)
        z = diameter * dimension_profile.z * pf.random.uniform(rd3, 0.85, 1.15)
        dimensions = pf.Vector((x, y, z))
    if material is None:
        material = boulder_material_rand(rng_material, pf.nodes.shader.coord().object)
    return boulder(
        dimensions=dimensions,
        mesh_resolution=mesh_resolution,
        random_seed=pf.random.randint(rng_seed, 1, 1_000_000),
        material=material,
        tri_face=tri_face,
        noise_size=noise_size,
        noise_type=noise_type,
        basis_type=basis_type,
        vl_basis_type=vl_basis_type,
        distortion=distortion,
        noise_depth=noise_depth,
        amplitude=amplitude,
        frequency=frequency,
        dimension=dimension,
        lacunarity=lacunarity,
        offset=offset,
        gain=gain,
        height=height,
        height_invert=height_invert,
        height_offset=height_offset,
        maximum=maximum,
        minimum=minimum,
        strata=strata,
        strata_type=strata_type,
    )


def boulder_rock_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    material: pf.Material | None = None,
    mesh_resolution: float = 0.02,
) -> BoulderResult:
    rng_boulder, rng_params = rng.spawn(2)
    return _boulder_rand_from_params(
        rng_boulder,
        dimensions,
        material,
        mesh_resolution,
        dimension_profile=pf.Vector((1.2, 0.9, 0.65)),
        noise_size=pf.random.uniform(rng_params, 1.6, 2.4),
        noise_type="slick_rock",
        basis_type="VORONOI_F1",
        vl_basis_type="BLENDER",
        noise_depth=pf.random.randint(rng_params, 5, 8),
        gain=pf.random.uniform(rng_params, 2.4, 3.6),
        height=pf.random.uniform(rng_params, 2.0, 3.0),
        maximum=3.0,
        minimum=-1.0,
        strata=pf.random.uniform(rng_params, 11.0, 19.0),
    )


def boulder_crystalline_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    material: pf.Material | None = None,
    mesh_resolution: float = 0.02,
) -> BoulderResult:
    rng_boulder, rng_params = rng.spawn(2)
    return _boulder_rand_from_params(
        rng_boulder,
        dimensions,
        material,
        mesh_resolution,
        dimension_profile=pf.Vector((0.8, 0.75, 1.2)),
        noise_size=pf.random.uniform(rng_params, 0.8, 1.2),
        noise_type="turbulence_vector",
        basis_type="VORONOI_F4",
        vl_basis_type="BLENDER",
        noise_depth=1,
        distortion=pf.random.uniform(rng_params, 0.8, 1.2),
        height=pf.random.uniform(rng_params, 0.8, 1.2),
        maximum=2.0,
        minimum=-1.0,
    )


def boulder_abstract_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    material: pf.Material | None = None,
    mesh_resolution: float = 0.02,
) -> BoulderResult:
    rng_boulder, rng_params = rng.spawn(2)
    return _boulder_rand_from_params(
        rng_boulder,
        dimensions,
        material,
        mesh_resolution,
        dimension_profile=pf.Vector((1.35, 0.75, 0.9)),
        noise_type="planet_noise",
        basis_type="VORONOI_F1",
        vl_basis_type="VORONOI_F4",
        distortion=pf.random.uniform(rng_params, 0.85, 1.2),
        noise_depth=1,
        amplitude=pf.random.uniform(rng_params, 0.35, 0.47),
        frequency=pf.random.uniform(rng_params, 1.6, 2.05),
        height=pf.random.uniform(rng_params, 0.8, 1.2),
        height_invert=True,
        maximum=1.0,
        minimum=-1.0,
        strata=5.0,
        strata_type="3",
    )


def boulder_flatstones_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    material: pf.Material | None = None,
    mesh_resolution: float = 0.02,
) -> BoulderResult:
    rng_boulder, rng_params = rng.spawn(2)
    return _boulder_rand_from_params(
        rng_boulder,
        dimensions,
        material,
        mesh_resolution,
        dimension_profile=pf.Vector((1.4, 1.0, 0.3)),
        noise_size=pf.random.uniform(rng_params, 0.4, 0.6),
        noise_type="slick_rock",
        basis_type="VORONOI_CRACKLE",
        vl_basis_type="BLENDER",
        distortion=pf.random.uniform(rng_params, 1.0, 1.4),
        noise_depth=1,
        frequency=pf.random.uniform(rng_params, 1.3, 1.7),
        offset=pf.random.uniform(rng_params, 1.0, 1.25),
        gain=pf.random.uniform(rng_params, 1.6, 2.4),
        height=pf.random.uniform(rng_params, 0.03, 0.08),
        maximum=1.0,
        minimum=0.0,
    )


def boulder_ridged_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    material: pf.Material | None = None,
    mesh_resolution: float = 0.02,
) -> BoulderResult:
    rng_boulder, rng_params = rng.spawn(2)
    return _boulder_rand_from_params(
        rng_boulder,
        dimensions,
        material,
        mesh_resolution,
        dimension_profile=pf.Vector((1.2, 0.85, 0.65)),
        noise_type="ridged_multi_fractal",
        basis_type="BLENDER",
        vl_basis_type="VORONOI_F1",
        noise_depth=pf.random.randint(rng_params, 6, 10),
        frequency=pf.random.uniform(rng_params, 1.5, 2.0),
        dimension=pf.random.uniform(rng_params, 0.85, 1.0),
        lacunarity=pf.random.uniform(rng_params, 2.0, 2.6),
        offset=pf.random.uniform(rng_params, 0.8, 1.0),
        gain=pf.random.uniform(rng_params, 1.8, 2.4),
        height=pf.random.uniform(rng_params, 0.22, 0.38),
        maximum=0.5,
        minimum=-1.0,
        strata=pf.random.uniform(rng_params, 8.0, 14.0),
    )


def boulder_cauliflower_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    material: pf.Material | None = None,
    mesh_resolution: float = 0.02,
) -> BoulderResult:
    rng_boulder, rng_params = rng.spawn(2)
    return _boulder_rand_from_params(
        rng_boulder,
        dimensions,
        material,
        mesh_resolution,
        dimension_profile=pf.Vector((1.0, 1.0, 0.75)),
        noise_size=pf.random.uniform(rng_params, 0.4, 0.6),
        noise_type="hybrid_multi_fractal",
        basis_type="VORONOI_F1",
        vl_basis_type="BLENDER",
        noise_depth=pf.random.randint(rng_params, 5, 8),
        lacunarity=pf.random.uniform(rng_params, 4.0, 6.0),
        gain=pf.random.uniform(rng_params, 1.6, 2.4),
        height=pf.random.uniform(rng_params, 0.18, 0.32),
        height_invert=True,
        maximum=1.0,
        minimum=-1.0,
    )


def boulder_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    material: pf.Material | None = None,
    mesh_resolution: float = 0.02,
) -> BoulderResult:
    rng_choice, rng_boulder = rng.spawn(2)
    options: list[tuple[Callable[..., BoulderResult], float]] = [
        (boulder_rock_rand, 1.0),
        (boulder_crystalline_rand, 2.0),
        (boulder_abstract_rand, 2.0),
        (boulder_flatstones_rand, 2.0),
        (boulder_ridged_rand, 0.5),
        (boulder_cauliflower_rand, 0.5),
    ]
    function = pf.control.choice(rng_choice, options)
    return function(
        rng_boulder,
        dimensions=dimensions,
        material=material,
        mesh_resolution=mesh_resolution,
    )


def rock_material_rand(rng: pf.RNG) -> pf.Material:
    rng_choice, rng_material = rng.spawn(2)
    material_options = [
        (granite.granite_smooth_rand, 1.0),
        (stone_smooth.stone_smooth_rand, 1.0),
        (decorative_material_rand, 1.0),
    ]
    material_func = pf.control.choice(rng_choice, material_options)
    return material_func(rng_material, pf.nodes.shader.coord().object)


def rock_rand(rng: pf.RNG, material: pf.Material | None = None) -> BoulderResult:
    rng_scale, rng_material, rng_boulder = rng.spawn(3)
    scale = pf.random.uniform(rng_scale, 0.08, 0.14)
    if material is None:
        material = rock_material_rand(rng_material)
    result = boulder_rand(rng_boulder, material=material, mesh_resolution=0.04)
    # no-scaling exception: boulder presets' sizes are too hard to recalibrate small
    pf.ops.object.set_transform(result.mesh, scale=pf.Vector((scale, scale, scale)))
    pf.ops.mesh.transform_apply(result.mesh)
    return result
