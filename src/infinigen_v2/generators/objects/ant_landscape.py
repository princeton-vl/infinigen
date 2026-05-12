# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the root directory of this source tree.

# Authors: Mingzhe Wang
# Acknowledgement: This file draws inspiration from the original ant_landscape shader in infinigen

from typing import NamedTuple, Unpack

import numpy as np
import procfunc as pf


class LandscapeResult(NamedTuple):
    mesh: pf.MeshObject


def generalized_landscape_distribution(
    rng: np.random.Generator,
    dimensions: pf.Vector = pf.Vector((50, 50, 10)),
    mesh_resolution: float = 0.5,
    **overrides: Unpack[pf.ops.primitives.LandscapeParameters],
) -> LandscapeResult:
    subrng, paramsrng = rng.spawn(2)

    params = {
        "noise_size": rng.uniform(0.5, 1.5),
        "distortion": rng.uniform(0.5, 2.0),
        "noise_depth": rng.integers(7, 13),
        "amplitude": rng.uniform(0.4, 0.5),
        "frequency": rng.uniform(1.7, 2.0),
        "gain": rng.integers(1, 5),
        "height": rng.uniform(0.2, 1.8),
        "height_offset": rng.uniform(-0.15, 0.2),
        "maximum": rng.uniform(0.25, 1.25),
        "minimum": rng.uniform(-1.0, -0.2),
        "fx_frequency": rng.uniform(1.4, 1.9),
        "fx_amplitude": rng.uniform(0.38, 0.5),
        "fx_depth": rng.integers(0, 4),
        "fx_height": rng.uniform(0.25, 1.0),
        "fx_size": rng.uniform(1.0, 1.5),
        "fx_loc_x": rng.uniform(-1.0, 3.0),
        "fx_loc_y": rng.uniform(0.0, 2.0),
        "fx_offset": rng.uniform(0.0, 0.06),
        "fx_turb": rng.uniform(0.0, 0.5),
        "falloff_x": rng.uniform(2.0, 40.0),
        "falloff_y": rng.uniform(2.0, 40.0),
        "edge_level": rng.uniform(0.0, 0.15),
        "strata": rng.uniform(1.0, 11.0),
    }
    params.update(overrides)
    return LandscapeResult(
        mesh=pf.ops.primitives.landscape(
            rng=rng,
            dimensions=dimensions,
            mesh_resolution=mesh_resolution,
            **params,
        )
    )


def canyons_landcape_distribution(
    rng: np.random.Generator,
    dimensions: pf.Vector = pf.Vector((50, 50, 10)),
    mesh_resolution: float = 0.5,
):
    subrng, paramsrng = rng.spawn(2)

    return pf.ops.primitives.landscape(
        rng=subrng,
        dimensions=dimensions,
        mesh_resolution=mesh_resolution,
        noise_offset_y=-0.25,
        noise_size_y=1.25,
        noise_size=1.5,
        noise_type="marble_noise",
        distortion=paramsrng.normal(2, 0.05),
        hard_noise="1",
        noise_depth=12,
        marble_shape="4",
        height=0.6,
        fx_mix_mode="8",
        fx_type="20",
        fx_depth=3,
        fx_frequency=1.65,
        fx_size=1.5,
        fx_loc_x=3,
        fx_loc_y=2,
        fx_height=0.25,
        fx_offset=0.05,
        edge_falloff="2",
        edge_level=0.15,
        minimum=-0.2,
        strata_type="2",
        strata=paramsrng.integers(6, 12),
    )


def cliff_landscape_distribution(
    rng: np.random.Generator,
    dimensions: pf.Vector = pf.Vector((50, 50, 10)),
    mesh_resolution: float = 0.5,
):
    subrng, paramsrng = rng.spawn(2)

    return pf.ops.primitives.landscape(
        rng=subrng,
        dimensions=dimensions,
        mesh_resolution=mesh_resolution,
        noise_offset_y=-0.88,
        noise_offset_z=3.72529e-09,
        noise_size_x=2,
        noise_size_y=2,
        noise_type="marble_noise",
        basis_type="VORONOI_F2F1",
        distortion=paramsrng.normal(0.5, 0.01),
        noise_depth=7,
        marble_shape="6",
        height=1.8,
        height_offset=-0.15,
        falloff_x=25,
        falloff_y=25,
        maximum=1.25,
        strata=11,
    )


def mesa_landscape_distribution(
    rng: np.random.Generator,
    dimensions: pf.Vector = pf.Vector((50, 50, 10)),
    mesh_resolution: float = 0.5,
):
    subrng, paramsrng = rng.spawn(2)

    return pf.ops.primitives.landscape(
        rng=subrng,
        dimensions=dimensions,
        mesh_resolution=mesh_resolution,
        noise_size=paramsrng.uniform(0.5, 1.0),
        noise_type="shattered_hterrain",
        basis_type="VORONOI_F1",
        vl_basis_type="VORONOI_F2F1",
        distortion=paramsrng.normal(1.15, 0.01),
        hard_noise="1",
        amplitude=0.4,
        gain=4,
        height_offset=0.2,
        fx_frequency=paramsrng.uniform(1.4, 1.6),
        edge_falloff="3",
        falloff_x=3,
        falloff_y=3,
        maximum=0.25,
        strata=2.25,
        strata_type="2",
    )


def river_landscape_distribution(
    rng: np.random.Generator,
    dimensions: pf.Vector = pf.Vector((50, 50, 10)),
    mesh_resolution: float = 0.5,
):
    subrng, paramsrng = rng.spawn(2)

    return pf.ops.primitives.landscape(
        rng=subrng,
        dimensions=dimensions,
        mesh_resolution=mesh_resolution,
        noise_type="marble_noise",
        marble_bias="2",
        marble_shape="7",
        height=0.2,
        fx_frequency=paramsrng.uniform(1.4, 1.6),
        falloff_x=40,
        falloff_y=40,
        strata=paramsrng.uniform(1, 1.5),
        strata_type="1",
    )


def volcano_landscape_distribution(
    rng: np.random.Generator,
    dimensions: pf.Vector = pf.Vector((50, 50, 10)),
    mesh_resolution: float = 0.5,
):
    subrng, paramsrng = rng.spawn(2)

    return pf.ops.primitives.landscape(
        rng=subrng,
        dimensions=dimensions,
        mesh_resolution=mesh_resolution,
        noise_type="marble_noise",
        vl_basis_type="PERLIN_ORIGINAL",
        distortion=paramsrng.normal(1.5, 0.01),
        frequency=paramsrng.uniform(1.7, 1.9),
        gain=2,
        marble_bias="2",
        marble_sharp="3",
        marble_shape="1",
        height=0.6,
        fx_mix_mode="1",
        fx_type="14",
        fx_turb=0.5,
        fx_depth=2,
        fx_amplitude=0.38,
        fx_frequency=paramsrng.uniform(1.4, 1.6),
        fx_size=1.15,
        fx_loc_x=-1,
        fx_loc_y=1,
        fx_offset=0.06,
        edge_falloff="3",
        falloff_x=2,
        falloff_y=2,
        maximum=1,
        minimum=-1,
        strata=paramsrng.integers(4, 6),
    )


def mountain_landscape_distribution(
    rng: np.random.Generator,
    dimensions: pf.Vector = pf.Vector((50, 50, 10)),
    mesh_resolution: float = 0.5,
):
    subrng, paramsrng = rng.spawn(2)

    return pf.ops.primitives.landscape(
        rng=subrng,
        dimensions=dimensions,
        mesh_resolution=mesh_resolution,
        fx_height=1,
        edge_falloff="3",
        maximum=1,
        minimum=-1,
        strata=paramsrng.integers(5, 10),
    )


def landscape_category_distribution(
    rng: np.random.Generator,
    dimensions: pf.Vector = pf.Vector((50, 50, 10)),
    mesh_resolution: float = 0.5,
) -> LandscapeResult:
    choicerng, subrng = rng.spawn(2)

    options = [
        (canyons_landcape_distribution, 1),
        (cliff_landscape_distribution, 1),
        (mesa_landscape_distribution, 1),
        (river_landscape_distribution, 1),
        (volcano_landscape_distribution, 1),
        (mountain_landscape_distribution, 1),
    ]
    func = pf.control.choice(choicerng, options)
    return LandscapeResult(
        mesh=func(subrng, dimensions=dimensions, mesh_resolution=mesh_resolution)
    )
