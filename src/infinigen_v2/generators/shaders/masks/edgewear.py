from typing import NamedTuple

import procfunc as pf
from procfunc.nodes import types as t


class EdgewearMaskResult(NamedTuple):
    mask: pf.ProcNode[float]


def edgewear_distribution(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
    displacement_a: t.SocketOrVal[pf.Vector] = (0.0, 0.0, 0.0),
    displacement_b: t.SocketOrVal[pf.Vector] = (0.0, 0.0, 0.0),
    height_threshold: t.SocketOrVal[float] = -1.0,
) -> EdgewearMaskResult:
    scratch_radius = pf.random.uniform(rng, 0.01, 0.03)
    scratch_mask_randomness = pf.random.uniform(rng, 10.0, 20.0)
    scratch_density = pf.random.uniform(rng, 5.0, 10.0)
    scratch_opacity = pf.random.uniform(rng, 0.5, 1.0)

    edge_wear_mask_result = edgewear_mask(
        vector=vector,
        displacement_a=displacement_a,
        displacement_b=displacement_b,
        height_threshold=height_threshold,
        scratch_radius=scratch_radius,
        scratch_mask_randomness=scratch_mask_randomness,
        scratch_density=scratch_density,
        scratch_opacity=scratch_opacity,
    )
    return EdgewearMaskResult(mask=edge_wear_mask_result)


@pf.nodes.node_function
def edgewear_mask(
    vector: t.SocketOrVal[pf.Vector] = (0.0, 0.0, 0.0),
    displacement_a: t.SocketOrVal[pf.Vector] = (0.0, 0.0, 0.0),
    displacement_b: t.SocketOrVal[pf.Vector] = (0.0, 0.0, 0.0),
    height_threshold: t.SocketOrVal[float] = 0.0,
    scratch_radius: t.SocketOrVal[float] = 0.01,
    scratch_mask_randomness: t.SocketOrVal[float] = 0.5,
    scratch_density: t.SocketOrVal[float] = 5.0,
    scratch_opacity: t.SocketOrVal[float] = 0.2,
):
    surface_factor_2 = scratch_opacity
    noise_scale = scratch_mask_randomness
    bevel_radius = scratch_radius
    color_0_scale_1 = scratch_density
    base_displacement = displacement_a

    mapping = pf.nodes.shader.mapping(vector)

    noise = pf.nodes.texture.noise(vector=mapping, scale=noise_scale, detail=1.0)

    color_ramp = pf.nodes.color.color_ramp(
        fac=noise.fac,
        interpolation="LINEAR",
        points=[
            (0.444, (0.0, 0.0, 0.0, 1.0)),
            (0.535, (1.0, 1.0, 1.0, 1.0)),
        ],
    )

    bevel = pf.nodes.shader.bevel(
        samples=20, radius=bevel_radius, normal=(0.0, 0.0, 0.0)
    )

    surface_0_b = pf.nodes.shader.geometry()
    surface = bevel - surface_0_b.normal
    surface_b_fac = pf.nodes.math.absolute(surface.astype(dtype=float))

    color_ramp_1 = pf.nodes.color.color_ramp(
        fac=surface_b_fac,
        interpolation="LINEAR",
        points=[
            (0.069, (0.0, 0.0, 0.0, 1.0)),
            (0.156, (1.0, 1.0, 1.0, 1.0)),
        ],
    )

    surface_factor_1 = pf.nodes.math.clamp(
        color_ramp.color.astype(dtype=float) * color_ramp_1.color.astype(dtype=float)
    )
    surface_factor = pf.nodes.math.clamp(surface_factor_2 * surface_factor_1)

    mapping_1 = pf.nodes.shader.mapping(vector=vector, scale=(10.0, 1.0, 1.0))

    voronoi_distance = pf.nodes.texture.voronoi_distance(
        vector=mapping_1, scale=color_0_scale_1
    )

    mapping_2 = pf.nodes.shader.mapping(vector=vector, scale=(1.0, 10.0, 1.0))

    color_0_scale_0 = pf.nodes.math.vector_scale(
        vector=color_0_scale_1.astype(dtype=pf.Vector),
        scale=2.0,
    )

    voronoi_distance_1 = pf.nodes.texture.voronoi_distance(
        vector=mapping_2,
        scale=color_0_scale_0.astype(dtype=float),
    )

    color_ramp_4_fac = pf.nodes.math.clamp(voronoi_distance * voronoi_distance_1)
    color_ramp_4 = pf.nodes.color.color_ramp(
        fac=color_ramp_4_fac,
        interpolation="LINEAR",
        points=[
            (0.0, (1.0, 1.0, 1.0, 1.0)),
            (0.007, (0.0, 0.0, 0.0, 1.0)),
        ],
    )

    displacement_0_1 = pf.nodes.math.clamp(
        color_ramp_1.color.astype(dtype=float) * color_ramp_4.color.astype(dtype=float)
    )
    displacement_2 = pf.nodes.math.clamp(surface_factor_1 * displacement_0_1)
    displacement_0_0 = pf.nodes.math.vector_scale(
        vector=displacement_2.astype(dtype=pf.Vector), scale=2.0
    )
    displacement_1 = pf.nodes.math.clamp(
        base_displacement.astype(dtype=float) + displacement_0_0.astype(dtype=float)
    )

    wear_tear_mask = pf.nodes.math.greater_than(
        surface_factor.astype(dtype=float), 0.01
    )

    combined_mask = pf.nodes.math.multiply(
        wear_tear_mask.astype(dtype=float), displacement_1.astype(dtype=float)
    )

    return combined_mask
