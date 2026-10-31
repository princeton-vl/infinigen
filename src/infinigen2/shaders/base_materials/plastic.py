# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Alexander Raistrick, Lingjie Mei: refactor for Infinigen2

import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.shaders.util import coord

__all__ = [
    "plastic",
    "plastic_black_rubberized_preset",
    "plastic_black_translucent_preset",
    "plastic_grayscale_color_rand",
    "plastic_grayscale_rand",
    "plastic_opaque_rand",
    "plastic_rand",
    "plastic_tough_packaging_preset",
    "plastic_translucent_bumps_preset",
    "plastic_translucent_rand",
    "plastic_white_textured_preset",
]

_PLASTIC_RELIEF_RATIO = 2.0
_PLASTIC_RELIEF_TAPER_SIZE = 0.001


def plastic_rand(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
    translucence: t.SocketOrVal[float] | None = None,
    base_color: t.SocketOrVal[pf.Color] | None = None,
) -> pf.Material:
    if translucence is None:
        translucence = 0.0

    roughness_max = pf.random.clip_gaussian(rng, 0.25, 0.25, 0.01, 0.75)
    m_size = pf.random.uniform(rng, 0.0, 1.0)
    m_colorvar = pf.random.uniform(rng, 0.0, 1.0)
    m_value = pf.random.uniform(rng, 0.0, 1.0)
    m_sat = pf.random.uniform(rng, 0.0, 1.0) ** 1.4
    hue = pf.random.uniform(rng, 0.0, 1.0)

    if base_color is None:
        value = 0.02 + 0.95 * m_value**1.2
        saturation = m_sat * 0.9
        base_color = pf.color.hsv_color(hue=hue, saturation=saturation, value=value)

    c2_scale = 1.0 + (pf.random.uniform(rng, 0.5, 1.5) - 1.0) * m_colorvar

    roughness_variation = pf.random.uniform(rng, 0.0, 1.0)
    roughness_min = roughness_max * (0.5 + 0.5 * roughness_variation)

    specular = pf.random.uniform(rng, 0.2, 1.0)
    specular_min = specular * pf.random.uniform(rng, 0.35, 1.0)

    ior_opaque = pf.random.uniform(rng, 1.33, 1.55)
    ior_translucent = pf.random.uniform(rng, 1.2, 1.35)
    ior = ior_opaque + (ior_translucent - ior_opaque) * translucence

    noise_size = 0.0002 * 5000.0 ** (m_size**2)
    bumpiness = pf.random.clip_gaussian(rng, 0.48, 0.5, 0.0, 1.5)
    relief_taper = noise_size / _PLASTIC_RELIEF_TAPER_SIZE
    noise_height = bumpiness * _PLASTIC_RELIEF_RATIO / (1.0 + relief_taper**2)
    noise_detail = pf.random.uniform(rng, 0.0, 5.0)
    noise_distortion_strength = pf.random.uniform(rng, 0.4, 1.0)
    noise_distortion_size = pf.random.uniform(rng, 0.0, 1.0)
    noise_seed = pf.random.uniform(rng, -1000.0, 1000.0)

    return plastic(
        vector=vector,
        base_color=base_color,
        color_2_value=c2_scale,
        roughness_min=roughness_min,
        roughness_max=roughness_max,
        specular_min=specular_min,
        specular_max=specular,
        ior=ior,
        translucence=translucence,
        noise_size=noise_size,
        noise_detail=noise_detail,
        noise_distortion_strength=noise_distortion_strength,
        noise_distortion_size=noise_distortion_size,
        noise_height=noise_height,
        noise_seed=noise_seed,
    )


@pf.tracer.generator
def plastic(
    vector: pf.ProcNode[pf.Vector],
    base_color: t.SocketOrVal[pf.Color],
    color_2_value: float,
    roughness_min: float,
    roughness_max: float,
    specular_min: float,
    specular_max: float,
    ior: float,
    translucence: t.SocketOrVal[float],
    noise_size: float,
    noise_detail: float,
    noise_distortion_strength: float,
    noise_distortion_size: float,
    noise_height: float,
    noise_seed: float,
) -> pf.Material:
    color_2 = pf.nodes.color.hue_saturation(
        fac=1.0, color=base_color, value=color_2_value
    )
    return _plastic(
        vector=vector,
        surface_color_1=base_color,
        surface_color_2=color_2,
        surface_min_roughness=roughness_min,
        surface_max_roughness=roughness_max,
        surface_min_specular=specular_min,
        surface_max_specular=specular_max,
        surface_ior=ior,
        surface_transmission=translucence,
        subsurface_weight=0.0,
        subsurface_radius=(1.0, 0.2, 0.1),
        subsurface_scale=0.05,
        subsurface_anisotropy=0.0,
        noise_size=noise_size,
        noise_detail=noise_detail,
        noise_distortion_strength=noise_distortion_strength,
        noise_distortion_size=noise_distortion_size,
        noise_height=noise_height,
        noise_seed=noise_seed,
    )


def plastic_translucent_rand(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
) -> pf.Material:
    translucence = pf.random.uniform(rng, 0.4, 1.0)
    return plastic_rand(rng, vector, translucence=translucence)


@pf.nodes.node_function
def plastic_opaque(
    vector: pf.ProcNode[pf.Vector],
    color_1: t.SocketOrVal[pf.Color],
    color_2: t.SocketOrVal[pf.Color],
    roughness_min: t.SocketOrVal[float],
    roughness_max: t.SocketOrVal[float],
    specular_min: t.SocketOrVal[float],
    specular_max: t.SocketOrVal[float],
    ior: t.SocketOrVal[float],
    scale: t.SocketOrVal[float],
    seed: t.SocketOrVal[float],
    noise_size: t.SocketOrVal[float],
    noise_detail: t.SocketOrVal[float],
    noise_distortion_strength: t.SocketOrVal[float],
    displacement_strength: t.SocketOrVal[float],
) -> pf.Material:
    scaled_vector = pf.nodes.math.vector_scale(vector=vector, scale=scale)

    space_warp_result = coord.space_warp(
        vector=vector,
        strength=noise_distortion_strength,
        w=seed,
        size=noise_size,
        detail=noise_detail,
        roughness=0.5,
        lacunarity=2.0,
        distortion=0.0,
    )

    noise = pf.nodes.texture.noise(
        vector=space_warp_result.vector,
        scale=1.0 / noise_size,
        detail=noise_detail,
        noise_dimensions="4D",
        w=seed + 50.0,
    )

    fac = pf.nodes.math.map_range(value=noise.fac, from_max=0.8, from_min=0.2)

    surface_base_color = pf.nodes.color.mix_rgb(factor=fac, a=color_1, b=color_2)
    surface_roughness = pf.nodes.math.mix(a=roughness_min, b=roughness_max, factor=fac)
    surface_specular = pf.nodes.math.mix(a=specular_max, b=specular_min, factor=fac)

    surface = pf.nodes.shader.principled_bsdf(
        base_color=surface_base_color,
        roughness=surface_roughness,
        ior=ior,
        specular_ior_level=surface_specular,
    )

    voronoi_1 = pf.nodes.texture.voronoi_n_spheres_distance(
        vector=scaled_vector,
        scale=2.0,
        randomness=0.0,
    )
    edge_mask = pf.nodes.math.map_range(
        value=voronoi_1,
        from_min=0.0,
        from_max=0.03,
        to_min=1.0,
        to_max=0.0,
    )

    noise_2 = pf.nodes.texture.noise(
        vector=scaled_vector,
        w=seed,
        scale=2.5,
        detail=6.0,
        noise_dimensions="4D",
    )
    edge_modulation = pf.nodes.math.map_range(
        value=noise_2.fac,
        from_min=0.55,
        from_max=0.57,
    )
    displacement_1 = edge_mask * edge_modulation * -0.5

    noise_3 = pf.nodes.texture.noise(
        vector=scaled_vector,
        w=seed,
        scale=10.0,
        detail=15.0,
        distortion=0.1,
        noise_dimensions="4D",
    )
    disp_2 = pf.nodes.math.map_range(
        value=noise_3.fac,
        from_min=0.63,
        from_max=0.68,
    )
    displacement_2 = disp_2 * -1.0

    noise_4 = pf.nodes.texture.noise(
        vector=scaled_vector,
        w=seed,
        scale=200.0,
        noise_dimensions="4D",
    )
    voronoi_2 = pf.nodes.texture.voronoi(
        vector=scaled_vector,
        scale=200.0,
    )
    fine_detail = pf.nodes.math.mix(
        factor=0.4,
        a=noise_4.fac,
        b=voronoi_2.distance,
    )
    displacement_3 = fine_detail * 0.1

    noise_5 = pf.nodes.texture.noise(
        vector=scaled_vector,
        w=seed,
        scale=4.0,
        detail=1.0,
        roughness=0.45,
        noise_dimensions="4D",
    )
    displacement_4 = (noise_5.fac - 0.5) * 3.0

    noise_6 = pf.nodes.texture.noise(
        vector=scaled_vector,
        w=seed,
        scale=40.0,
        detail=15.0,
        distortion=0.1,
        noise_dimensions="4D",
    )
    disp_5_mask = pf.nodes.math.map_range(
        value=noise_6.fac,
        from_min=0.65,
        from_max=0.64,
        to_min=1.0,
        to_max=0.0,
    )
    noise_7 = pf.nodes.texture.noise(
        vector=scaled_vector,
        w=seed,
        scale=12.0,
        detail=6.0,
        noise_dimensions="4D",
    )
    disp_5_mod = pf.nodes.math.map_range(
        value=noise_7.fac,
        from_min=0.55,
        from_max=0.57,
    )
    displacement_5 = (disp_5_mask * disp_5_mod - 0.5) * -0.5

    noise_8 = pf.nodes.texture.noise(
        vector=scaled_vector,
        w=seed,
        scale=30.0,
        detail=3.0,
        roughness=0.45,
        noise_dimensions="4D",
    )
    displacement_6 = (noise_8.fac - 0.5) * 1.0

    noise_9 = pf.nodes.texture.noise(
        vector=scaled_vector,
        w=seed,
        scale=20.0,
        detail=3.0,
        roughness=0.45,
        noise_dimensions="4D",
    )
    disp_7 = pf.nodes.math.map_range(
        value=noise_9.fac,
        from_min=0.55,
        from_max=0.51,
        to_min=-0.5,
        to_max=0.5,
    )
    displacement_7 = disp_7 * 0.05

    total_displacement = (
        displacement_1
        + displacement_2
        + displacement_3
        + displacement_4
        + displacement_5
        + displacement_6
        + displacement_7
    )

    displacement = pf.nodes.shader.displacement(
        height=total_displacement * displacement_strength * 0.001,
        midlevel=0.0,
    )

    return pf.Material(
        surface=surface,
        displacement=displacement,
    )


def plastic_grayscale_color_rand(rng: pf.RNG) -> pf.Color:
    rng_band, rng_value = rng.spawn(2)
    value_range = pf.control.choice(
        rng_band,
        [
            ((0.02, 0.15), 0.6),
            ((0.35, 0.90), 0.4),
        ],
    )
    perceptual_value = pf.random.uniform(rng_value, value_range[0], value_range[1])
    return pf.color.hsv_color(
        hue=0.0,
        saturation=0.0,
        value=perceptual_value**2.2,
    )


def plastic_grayscale_rand(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
) -> pf.Material:
    base_color = plastic_grayscale_color_rand(rng)
    return plastic_rand(rng, vector, base_color=base_color)


def plastic_opaque_rand(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
    base_color: t.SocketOrVal[pf.Color | None] = None,
) -> pf.Material:
    if base_color is None:
        h = pf.random.uniform(rng, 0.0, 1.0)
        s = pf.random.uniform(rng, 0.2, 0.8)
        v = pf.random.uniform(rng, 0.01, 0.8)
        base_color = pf.color.hsv_to_rgba((h, s, v))

    roughness = pf.random.uniform(rng, 0.3, 1.0)
    specular = pf.random.uniform(rng, 0.1, 1.0)
    ior = pf.random.uniform(rng, 1.2, 1.55)

    scale = pf.random.uniform(rng, 2.0, 7.0)
    noise_size = pf.random.uniform(rng, 0.002, 0.02)
    noise_detail = pf.random.uniform(rng, 1.0, 6.0)
    noise_seed = pf.random.uniform(rng, -1000.0, 1000.0)

    roughness_variation = pf.random.uniform(rng, 0.0, 1.0)
    roughness_min = roughness * (roughness_variation * 0.3 + 0.7)

    specular_variation = pf.random.uniform(rng, 0.0, 1.0)
    specular_min = specular * (specular_variation * 0.25 + 0.75)

    color_1_value = pf.random.uniform(rng, 0.5, 1.0)
    color_1 = pf.nodes.color.hue_saturation(
        fac=1.0,
        color=base_color,
        value=color_1_value,
    )
    color_2_value = pf.random.uniform(rng, 0.9, 1.3)
    color_2 = pf.nodes.color.hue_saturation(
        fac=1.0,
        color=base_color,
        value=color_2_value,
    )

    noise_distortion_strength = pf.random.uniform(rng, 0.4, 1.0)
    grain_depth = pf.random.uniform(rng, 0.0, 1.5e-4)
    displacement_strength = grain_depth / 3.0e-3

    return plastic_opaque(
        vector=vector,
        color_1=color_1,
        color_2=color_2,
        roughness_min=roughness_min,
        roughness_max=roughness,
        specular_min=specular_min,
        specular_max=specular,
        ior=ior,
        scale=scale,
        seed=noise_seed,
        noise_size=noise_size,
        noise_detail=noise_detail,
        noise_distortion_strength=noise_distortion_strength,
        displacement_strength=displacement_strength,
    )


@pf.nodes.node_function
def _plastic(
    vector: t.SocketOrVal[pf.Vector],
    surface_color_1: t.SocketOrVal[pf.Color],
    surface_color_2: t.SocketOrVal[pf.Color],
    surface_min_roughness: t.SocketOrVal[float],
    surface_max_roughness: t.SocketOrVal[float],
    surface_min_specular: t.SocketOrVal[float],
    surface_max_specular: t.SocketOrVal[float],
    surface_ior: t.SocketOrVal[float],
    surface_transmission: t.SocketOrVal[float],
    subsurface_weight: t.SocketOrVal[float],
    subsurface_radius: t.SocketOrVal[pf.Vector],
    subsurface_scale: t.SocketOrVal[float],
    subsurface_anisotropy: t.SocketOrVal[float],
    noise_size: t.SocketOrVal[float],
    noise_detail: t.SocketOrVal[float],
    noise_distortion_strength: t.SocketOrVal[float],
    noise_distortion_size: t.SocketOrVal[float],
    noise_height: t.SocketOrVal[float],
    noise_seed: t.SocketOrVal[float],
) -> pf.Material:
    space_warp_result = coord.space_warp(
        vector=vector,
        strength=noise_distortion_strength,
        w=noise_seed,
        size=noise_size * noise_distortion_size,
        detail=noise_detail,
        roughness=0.5,
        lacunarity=2.0,
        distortion=0.0,
    )

    noise = pf.nodes.texture.noise(
        vector=space_warp_result.vector,
        scale=1.0 / noise_size,
        detail=noise_detail,
        noise_dimensions="4D",
        w=noise_seed + 50.0,
    )

    fac = pf.nodes.math.map_range(value=noise.fac, from_max=0.8, from_min=0.2)

    surface_base_color = pf.nodes.color.mix_rgb(
        factor=fac, a=surface_color_1, b=surface_color_2
    )
    surface_roughness = pf.nodes.math.mix(
        a=surface_min_roughness, b=surface_max_roughness, factor=fac
    )
    surface_specular_ior_level = pf.nodes.math.mix(
        a=surface_max_specular, b=surface_min_specular, factor=fac
    )

    principled = pf.nodes.shader.principled_bsdf(
        base_color=surface_base_color,
        roughness=surface_roughness,
        ior=surface_ior,
        subsurface_weight=subsurface_weight,
        subsurface_radius=subsurface_radius,
        subsurface_scale=subsurface_scale,
        subsurface_anisotropy=subsurface_anisotropy,
        specular_ior_level=surface_specular_ior_level,
        transmission_weight=surface_transmission,
    )

    displacement_scale = noise_size * noise_height
    displacement = pf.nodes.shader.displacement(
        height=fac,
        scale=displacement_scale * 0.5,
    )
    return pf.Material(
        surface=principled,
        displacement=displacement,
    )


@pf.nodes.node_function
def plastic_black_rubberized(
    vector: pf.ProcNode[pf.Vector],
    color_1: t.SocketOrVal[pf.Color] = pf.Color((0.139, 0.139, 0.139)),
    color_2: t.SocketOrVal[pf.Color] = pf.Color((0.205, 0.205, 0.205)),
    roughness_min: t.SocketOrVal[float] = 0.4,
    roughness_max: t.SocketOrVal[float] = 0.616667,
    specular_min: t.SocketOrVal[float] = 0.358333,
    specular_max: t.SocketOrVal[float] = 0.941667,
    ior: t.SocketOrVal[float] = 1.5,
    noise_size: t.SocketOrVal[float] = 0.0005,
    noise_detail: t.SocketOrVal[float] = 2.0,
    noise_height: t.SocketOrVal[float] = 0.0,
    noise_seed: t.SocketOrVal[float] = 0.0,
) -> pf.Material:
    return _plastic(
        vector=vector,
        surface_color_1=color_1,
        surface_color_2=color_2,
        surface_min_roughness=roughness_min,
        surface_max_roughness=roughness_max,
        surface_min_specular=specular_min,
        surface_max_specular=specular_max,
        surface_ior=ior,
        surface_transmission=0.0,
        subsurface_weight=0.0,
        subsurface_radius=(1.0, 0.2, 0.1),
        subsurface_scale=0.05,
        subsurface_anisotropy=0.0,
        noise_size=noise_size,
        noise_detail=noise_detail,
        noise_distortion_strength=1.0,
        noise_distortion_size=1.0,
        noise_height=noise_height,
        noise_seed=noise_seed,
    )


@pf.nodes.node_function
def plastic_black_translucent(
    vector: pf.ProcNode[pf.Vector],
    base_color: t.SocketOrVal[pf.Color] = pf.Color((0.113, 0.113, 0.113)),
    roughness: t.SocketOrVal[float] = 0.01,
    ior: t.SocketOrVal[float] = 1.2,
    transmission: t.SocketOrVal[float] = 0.941667,
    noise_size: t.SocketOrVal[float] = 0.0002,
    noise_height: t.SocketOrVal[float] = 0.2,
) -> pf.Material:
    return _plastic(
        vector=vector,
        surface_color_1=base_color,
        surface_color_2=base_color,
        surface_min_roughness=roughness,
        surface_max_roughness=roughness,
        surface_min_specular=1.0,
        surface_max_specular=1.0,
        surface_ior=ior,
        surface_transmission=transmission,
        subsurface_weight=0.0,
        subsurface_radius=(1.0, 0.2, 0.1),
        subsurface_scale=0.05,
        subsurface_anisotropy=0.0,
        noise_size=noise_size,
        noise_detail=0.0,
        noise_distortion_strength=1.0,
        noise_distortion_size=1.0,
        noise_height=noise_height,
        noise_seed=0.0,
    )


@pf.nodes.node_function
def plastic_translucent_bumps(
    vector: pf.ProcNode[pf.Vector],
    base_color: t.SocketOrVal[pf.Color] = pf.Color((0.621, 0.069, 0.069)),
    roughness: t.SocketOrVal[float] = 0.01,
    ior: t.SocketOrVal[float] = 1.3,
    transmission: t.SocketOrVal[float] = 1.0,
    noise_size: t.SocketOrVal[float] = 0.001,
    noise_height: t.SocketOrVal[float] = 0.12,
) -> pf.Material:
    return _plastic(
        vector=vector,
        surface_color_1=base_color,
        surface_color_2=base_color,
        surface_min_roughness=roughness,
        surface_max_roughness=roughness * 2.0,
        surface_min_specular=0.5,
        surface_max_specular=0.5,
        surface_ior=ior,
        surface_transmission=transmission,
        subsurface_weight=0.0,
        subsurface_radius=(1.0, 0.2, 0.1),
        subsurface_scale=0.05,
        subsurface_anisotropy=0.0,
        noise_size=noise_size,
        noise_detail=0.0,
        noise_distortion_strength=1.0,
        noise_distortion_size=0.0,
        noise_height=noise_height,
        noise_seed=0.0,
    )


@pf.nodes.node_function
def plastic_white_textured(
    vector: pf.ProcNode[pf.Vector],
    base_color: t.SocketOrVal[pf.Color] = pf.Color((0.319, 0.319, 0.319)),
    roughness_min: t.SocketOrVal[float] = 0.4,
    roughness_max: t.SocketOrVal[float] = 0.5,
    specular_min: t.SocketOrVal[float] = 0.642132,
    specular_max: t.SocketOrVal[float] = 0.666751,
    ior: t.SocketOrVal[float] = 1.5,
    noise_size: t.SocketOrVal[float] = 0.001,
    noise_detail: t.SocketOrVal[float] = 2.0,
    noise_height: t.SocketOrVal[float] = 1.0,
    noise_seed: t.SocketOrVal[float] = 3.3,
) -> pf.Material:
    return _plastic(
        vector=vector,
        surface_color_1=base_color,
        surface_color_2=base_color,
        surface_min_roughness=roughness_min,
        surface_max_roughness=roughness_max,
        surface_min_specular=specular_min,
        surface_max_specular=specular_max,
        surface_ior=ior,
        surface_transmission=0.0,
        subsurface_weight=0.0,
        subsurface_radius=(1.0, 0.2, 0.1),
        subsurface_scale=0.05,
        subsurface_anisotropy=0.0,
        noise_size=noise_size,
        noise_detail=noise_detail,
        noise_distortion_strength=0.5,
        noise_distortion_size=1.0,
        noise_height=noise_height,
        noise_seed=noise_seed,
    )


@pf.nodes.node_function
def plastic_tough_packaging(
    vector: pf.ProcNode[pf.Vector],
    base_color: t.SocketOrVal[pf.Color] = pf.Color((0.760727, 0.787818, 0.391346)),
    roughness: t.SocketOrVal[float] = 0.25,
    specular: t.SocketOrVal[float] = 0.4,
    ior: t.SocketOrVal[float] = 1.33,
    noise_size: t.SocketOrVal[float] = 1.0,
    noise_detail: t.SocketOrVal[float] = 3.0,
    noise_seed: t.SocketOrVal[float] = 0.0,
) -> pf.Material:
    material = _plastic(
        vector=vector,
        surface_color_1=base_color,
        surface_color_2=base_color,
        surface_min_roughness=roughness,
        surface_max_roughness=roughness,
        surface_min_specular=specular,
        surface_max_specular=specular,
        surface_ior=ior,
        surface_transmission=0.0,
        subsurface_weight=0.0,
        subsurface_radius=(1.0, 0.2, 0.1),
        subsurface_scale=0.05,
        subsurface_anisotropy=0.0,
        noise_size=noise_size,
        noise_detail=noise_detail,
        noise_distortion_strength=1.0,
        noise_distortion_size=1.0,
        noise_height=1.0,
        noise_seed=noise_seed,
    )
    return pf.Material(surface=material.surface)


def plastic_black_rubberized_preset(vector: pf.ProcNode[pf.Vector]) -> pf.Material:
    return plastic_black_rubberized(vector=vector)


def plastic_black_translucent_preset(vector: pf.ProcNode[pf.Vector]) -> pf.Material:
    return plastic_black_translucent(vector=vector)


def plastic_tough_packaging_preset(vector: pf.ProcNode[pf.Vector]) -> pf.Material:
    return plastic_tough_packaging(vector=vector)


def plastic_translucent_bumps_preset(vector: pf.ProcNode[pf.Vector]) -> pf.Material:
    return plastic_translucent_bumps(vector=vector)


def plastic_white_textured_preset(vector: pf.ProcNode[pf.Vector]) -> pf.Material:
    return plastic_white_textured(vector=vector)
