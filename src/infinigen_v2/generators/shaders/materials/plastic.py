import procfunc as pf
from procfunc.nodes import types as t

from infinigen_v2.generators.shaders.util import coord


def plastic_translucent_distribution(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
) -> pf.Material:
    h = pf.random.uniform(rng, 0.0, 1.0)
    s = pf.random.uniform(rng, 0.1, 1.0)
    v = pf.random.uniform(rng, 0.03, 0.95)
    color = pf.color.hsv_to_rgba((h, s, v))

    roughness = pf.random.uniform(rng, 0.15, 0.59)
    ior = pf.random.uniform(rng, 1.2, 1.35)
    transmission = pf.random.uniform(rng, 0.4, 1.0)

    noise_size = pf.random.log_uniform(rng, 0.0001, 2.0)
    noise_height = pf.random.uniform(rng, 0.05, 0.55)
    specular = pf.random.uniform(rng, 0.3, 1.0)
    noise_detail = pf.random.uniform(rng, 0.0, 4.0)

    noise_seed = pf.random.uniform(rng, -1000.0, 1000.0)

    roughness_variation = pf.random.uniform(rng, 1.0, 2.0)
    roughness_max = roughness * roughness_variation

    return _plastic(
        vector=vector,
        surface_color_1=color,
        surface_color_2=color,
        surface_min_roughness=roughness,
        surface_max_roughness=roughness_max,
        surface_min_specular=specular,
        surface_max_specular=specular,
        surface_ior=ior,
        surface_transmission=transmission,
        subsurface_weight=0.0,
        subsurface_radius=(1.0, 0.2, 0.1),
        subsurface_scale=0.05,
        subsurface_anisotropy=0.0,
        noise_size=noise_size,
        noise_detail=noise_detail,
        noise_distortion_strength=1.0,
        noise_distortion_size=pf.random.uniform(rng, 0.0, 1.0),
        noise_height=noise_height,
        noise_seed=noise_seed,
    )


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


def plastic_grayscale_distribution(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
) -> pf.Material:
    h = pf.random.uniform(rng, 0.0, 1.0)
    s = 0.0
    v = pf.random.log_uniform(rng, 0.02, 0.9)
    color = pf.color.hsv_to_rgba((h, s, v))
    return plastic_opaque_distribution(rng, vector, color=color)


def plastic_opaque_distribution(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
    color: t.SocketOrVal[pf.Color | None] = None,
) -> pf.Material:
    if color is None:
        h = pf.random.uniform(rng, 0.0, 1.0)
        s = pf.random.uniform(rng, 0.2, 0.8)
        v = pf.random.log_uniform(rng, 0.01, 0.8)
        color = pf.color.hsv_to_rgba((h, s, v))

    roughness = pf.random.uniform(rng, 0.3, 1.0)
    specular = pf.random.uniform(rng, 0.1, 1.0)
    ior = pf.random.uniform(rng, 1.2, 1.55)

    scale = pf.random.uniform(rng, 2.0, 7.0)
    noise_size = pf.random.log_uniform(rng, 0.002, 0.02)
    noise_detail = pf.random.uniform(rng, 1.0, 6.0)
    noise_seed = pf.random.uniform(rng, -1000.0, 1000.0)

    roughness_variation = pf.random.uniform(rng, 0.0, 1.0)
    roughness_min = roughness * (roughness_variation * 0.3 + 0.7)

    specular_variation = pf.random.uniform(rng, 0.0, 1.0)
    specular_min = specular * (specular_variation * 0.25 + 0.75)

    color_1_value = pf.random.uniform(rng, 0.5, 1.0)
    color_1 = pf.nodes.color.hue_saturation(
        fac=1.0,
        color=color,
        value=color_1_value,
    )
    color_2_value = pf.random.uniform(rng, 0.9, 1.3)
    color_2 = pf.nodes.color.hue_saturation(
        fac=1.0,
        color=color,
        value=color_2_value,
    )

    noise_distortion_strength = pf.random.uniform(rng, 0.4, 1.0)
    displacement_strength = pf.random.uniform(rng, 0.0, 4.0)

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
def bumpy_rubber(
    vector: pf.ProcNode[pf.Vector],
    base_color: t.SocketOrVal[pf.Color] = pf.Color((0.8, 0.8, 0.8)),
    scale: t.SocketOrVal[float] = 2.0,
    seed: t.SocketOrVal[float] = 0.0,
    roughness: t.SocketOrVal[float] = 0.4,
) -> pf.Material:
    scaled_vector = pf.nodes.math.vector_scale(vector=vector, scale=scale)

    noise_color = pf.nodes.texture.noise(
        vector=scaled_vector,
        w=seed,
        scale=18.0,
        detail=3.0,
        roughness=0.45,
        noise_dimensions="4D",
    )
    color_variation = pf.nodes.math.map_range(
        value=noise_color.fac,
        from_min=0.0,
        from_max=1.0,
        to_min=0.6,
        to_max=1.4,
    )
    varied_color = pf.nodes.color.hue_saturation(
        fac=1.0,
        color=base_color,
        value=color_variation,
    )

    surface = pf.nodes.shader.principled_bsdf(
        base_color=varied_color,
        specular_ior_level=0.9,
        roughness=roughness,
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
        height=total_displacement * 0.001,
        midlevel=0.0,
    )

    return pf.Material(
        surface=surface,
        displacement=displacement,
    )


def _bumpy_rubber_distribution(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
    base_color: t.SocketOrVal[pf.Color | None] = None,
    scale: t.SocketOrVal[float | None] = None,
    seed: t.SocketOrVal[float | None] = None,
    roughness: t.SocketOrVal[float | None] = None,
) -> pf.Material:
    if base_color is None:
        h = pf.random.uniform(rng, 0.0, 1.0)
        s = pf.random.uniform(rng, 0.4, 0.8)
        v = pf.random.uniform(rng, 0.15, 0.45)
        base_color = pf.color.hsv_to_rgba((h, s, v))
    else:
        h_offset = pf.random.uniform(rng, -0.05, 0.05)
        s_offset = pf.random.uniform(rng, -0.1, 0.1)
        v_offset = pf.random.uniform(rng, -0.15, 0.15)
        base_color = pf.nodes.color.hue_saturation(
            fac=1.0,
            color=base_color,
            hue=h_offset + 0.5,
            saturation=s_offset + 1.0,
            value=v_offset + 1.0,
        )
    if seed is None:
        seed = pf.random.uniform(rng, -1000.0, 1000.0)
    if roughness is None:
        roughness = pf.random.uniform(rng, 0.3, 0.55)
    if scale is None:
        scale = pf.random.uniform(rng, 2.0, 5.0)

    return bumpy_rubber(
        vector=vector,
        base_color=base_color,
        scale=scale,
        seed=seed,
        roughness=roughness,
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
        normal=(0.0, 0.0, 0.0),
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


def _plastic_black_rubberized_distribution(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
    color_1: t.SocketOrVal[pf.Color | None] = None,
    color_2: t.SocketOrVal[pf.Color | None] = None,
    roughness_min: t.SocketOrVal[float | None] = None,
    roughness_max: t.SocketOrVal[float | None] = None,
    specular_min: t.SocketOrVal[float | None] = None,
    specular_max: t.SocketOrVal[float | None] = None,
    ior: t.SocketOrVal[float | None] = None,
    noise_size: t.SocketOrVal[float | None] = None,
    noise_detail: t.SocketOrVal[float | None] = None,
    noise_height: t.SocketOrVal[float | None] = None,
    noise_seed: t.SocketOrVal[float | None] = None,
) -> pf.Material:
    if color_1 is None or color_2 is None:
        h = pf.random.uniform(rng, 0.0, 1.0)
        s = pf.random.uniform(rng, 0.0, 0.15)
        if color_1 is None:
            v1 = pf.random.uniform(rng, 0.02, 0.06)
            color_1 = pf.color.hsv_to_rgba((h, s, v1))
        else:
            h_offset = pf.random.uniform(rng, -0.02, 0.02)
            s_offset = pf.random.uniform(rng, -0.03, 0.03)
            v_offset = pf.random.uniform(rng, -0.05, 0.05)
            color_1 = pf.nodes.color.hue_saturation(
                fac=1.0,
                color=color_1,
                hue=h_offset + 0.5,
                saturation=s_offset + 1.0,
                value=v_offset + 1.0,
            )
        if color_2 is None:
            v2 = pf.random.uniform(rng, 0.05, 0.12)
            color_2 = pf.color.hsv_to_rgba((h, s, v2))
        else:
            h_offset = pf.random.uniform(rng, -0.02, 0.02)
            s_offset = pf.random.uniform(rng, -0.03, 0.03)
            v_offset = pf.random.uniform(rng, -0.05, 0.05)
            color_2 = pf.nodes.color.hue_saturation(
                fac=1.0,
                color=color_2,
                hue=h_offset + 0.5,
                saturation=s_offset + 1.0,
                value=v_offset + 1.0,
            )
    if roughness_min is None:
        roughness_min = pf.random.uniform(rng, 0.35, 0.5)
    if roughness_max is None:
        roughness_max = pf.random.uniform(rng, 0.5, 0.7)
    if specular_min is None:
        specular_min = pf.random.uniform(rng, 0.3, 0.5)
    if specular_max is None:
        specular_max = pf.random.uniform(rng, 0.6, 1.0)
    if ior is None:
        ior = pf.random.uniform(rng, 1.4, 1.55)
    if noise_size is None:
        noise_size = pf.random.uniform(rng, 0.0002, 0.002)
    if noise_detail is None:
        noise_detail = pf.random.uniform(rng, 1.0, 3.0)
    if noise_height is None:
        noise_height = pf.random.uniform(rng, 0.0, 0.3)
    if noise_seed is None:
        noise_seed = pf.random.uniform(rng, -100.0, 100.0)

    return plastic_black_rubberized(
        vector=vector,
        color_1=color_1,
        color_2=color_2,
        roughness_min=roughness_min,
        roughness_max=roughness_max,
        specular_min=specular_min,
        specular_max=specular_max,
        ior=ior,
        noise_size=noise_size,
        noise_detail=noise_detail,
        noise_height=noise_height,
        noise_seed=noise_seed,
    )


@pf.nodes.node_function
def plastic_black_translucent(
    vector: pf.ProcNode[pf.Vector],
    color: t.SocketOrVal[pf.Color] = pf.Color((0.113, 0.113, 0.113)),
    roughness: t.SocketOrVal[float] = 0.01,
    ior: t.SocketOrVal[float] = 1.2,
    transmission: t.SocketOrVal[float] = 0.941667,
    noise_size: t.SocketOrVal[float] = 0.0002,
    noise_height: t.SocketOrVal[float] = 0.2,
) -> pf.Material:
    return _plastic(
        vector=vector,
        surface_color_1=color,
        surface_color_2=color,
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


def _plastic_black_translucent_distribution(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
    color: t.SocketOrVal[pf.Color | None] = None,
    roughness: t.SocketOrVal[float | None] = None,
    ior: t.SocketOrVal[float | None] = None,
    transmission: t.SocketOrVal[float | None] = None,
    noise_size: t.SocketOrVal[float | None] = None,
    noise_height: t.SocketOrVal[float | None] = None,
) -> pf.Material:
    if color is None:
        h = pf.random.uniform(rng, 0.0, 1.0)
        s = pf.random.uniform(rng, 0.1, 0.4)
        v = pf.random.uniform(rng, 0.03, 0.12)
        color = pf.color.hsv_to_rgba((h, s, v))
    else:
        h_offset = pf.random.uniform(rng, -0.03, 0.03)
        s_offset = pf.random.uniform(rng, -0.05, 0.05)
        v_offset = pf.random.uniform(rng, -0.05, 0.05)
        color = pf.nodes.color.hue_saturation(
            fac=1.0,
            color=color,
            hue=h_offset + 0.5,
            saturation=s_offset + 1.0,
            value=v_offset + 1.0,
        )
    if roughness is None:
        roughness = pf.random.uniform(rng, 0.01, 0.02)
    if ior is None:
        ior = pf.random.uniform(rng, 1.2, 1.35)
    if transmission is None:
        transmission = pf.random.uniform(rng, 0.7, 1.0)
    if noise_size is None:
        noise_size = pf.random.uniform(rng, 0.0001, 0.0005)
    if noise_height is None:
        noise_height = pf.random.uniform(rng, 0.1, 0.3)

    return plastic_black_translucent(
        vector=vector,
        color=color,
        roughness=roughness,
        ior=ior,
        transmission=transmission,
        noise_size=noise_size,
        noise_height=noise_height,
    )


@pf.nodes.node_function
def plastic_soft_touch(
    vector: pf.ProcNode[pf.Vector],
    color: t.SocketOrVal[pf.Color] = pf.Color((0.761, 0.788, 0.391)),
    roughness: t.SocketOrVal[float] = 0.25,
    specular: t.SocketOrVal[float] = 0.4,
    ior: t.SocketOrVal[float] = 1.33,
    noise_size: t.SocketOrVal[float] = 1.0,
    noise_detail: t.SocketOrVal[float] = 3.0,
    noise_seed: t.SocketOrVal[float] = 0.0,
) -> pf.Material:
    return _plastic(
        vector=vector,
        surface_color_1=color,
        surface_color_2=color,
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


def _plastic_soft_touch_distribution(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
    color: t.SocketOrVal[pf.Color | None] = None,
    roughness: t.SocketOrVal[float | None] = None,
    specular: t.SocketOrVal[float | None] = None,
    ior: t.SocketOrVal[float | None] = None,
    noise_size: t.SocketOrVal[float | None] = None,
    noise_detail: t.SocketOrVal[float | None] = None,
    noise_seed: t.SocketOrVal[float | None] = None,
) -> pf.Material:
    if color is None:
        h = pf.random.uniform(rng, 0.0, 1.0)
        s = pf.random.uniform(rng, 0.2, 0.6)
        v = pf.random.uniform(rng, 0.25, 0.55)
        color = pf.color.hsv_to_rgba((h, s, v))
    else:
        h_offset = pf.random.uniform(rng, -0.05, 0.05)
        s_offset = pf.random.uniform(rng, -0.1, 0.1)
        v_offset = pf.random.uniform(rng, -0.15, 0.15)
        color = pf.nodes.color.hue_saturation(
            fac=1.0,
            color=color,
            hue=h_offset + 0.5,
            saturation=s_offset + 1.0,
            value=v_offset + 1.0,
        )
    if roughness is None:
        roughness = pf.random.uniform(rng, 0.5, 0.75)
    if specular is None:
        specular = pf.random.uniform(rng, 0.15, 0.3)
    if noise_size is None:
        noise_size = pf.random.uniform(rng, 0.001, 0.01)
    if noise_detail is None:
        noise_detail = pf.random.uniform(rng, 2.0, 6.0)
    if noise_seed is None:
        noise_seed = pf.random.uniform(rng, -100.0, 100.0)
    if ior is None:
        ior = 1.46

    return plastic_soft_touch(
        vector=vector,
        color=color,
        roughness=roughness,
        specular=specular,
        ior=ior,
        noise_size=noise_size,
        noise_detail=noise_detail,
        noise_seed=noise_seed,
    )


@pf.nodes.node_function
def plastic_sandblasted(
    vector: pf.ProcNode[pf.Vector],
    color: t.SocketOrVal[pf.Color] = pf.Color((0.064, 0.055, 0.012)),
    roughness: t.SocketOrVal[float] = 0.4,
    specular: t.SocketOrVal[float] = 0.2,
    ior: t.SocketOrVal[float] = 1.33,
    noise_size: t.SocketOrVal[float] = 1.0,
    noise_detail: t.SocketOrVal[float] = 3.0,
    noise_height: t.SocketOrVal[float] = 1.0,
) -> pf.Material:
    return _plastic(
        vector=vector,
        surface_color_1=color,
        surface_color_2=color,
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
        noise_height=noise_height,
        noise_seed=0.0,
    )


def _plastic_sandblasted_distribution(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
    color: t.SocketOrVal[pf.Color | None] = None,
    roughness: t.SocketOrVal[float | None] = None,
    specular: t.SocketOrVal[float | None] = None,
    ior: t.SocketOrVal[float | None] = None,
    noise_size: t.SocketOrVal[float | None] = None,
    noise_detail: t.SocketOrVal[float | None] = None,
    noise_height: t.SocketOrVal[float | None] = None,
) -> pf.Material:
    if color is None:
        h = pf.random.uniform(rng, 0.0, 1.0)
        s = pf.random.uniform(rng, 0.1, 0.5)
        v = pf.random.uniform(rng, 0.15, 0.5)
        color = pf.color.hsv_to_rgba((h, s, v))
    else:
        h_offset = pf.random.uniform(rng, -0.05, 0.05)
        s_offset = pf.random.uniform(rng, -0.1, 0.1)
        v_offset = pf.random.uniform(rng, -0.15, 0.15)
        color = pf.nodes.color.hue_saturation(
            fac=1.0,
            color=color,
            hue=h_offset + 0.5,
            saturation=s_offset + 1.0,
            value=v_offset + 1.0,
        )
    if roughness is None:
        roughness = pf.random.uniform(rng, 0.6, 0.9)
    if specular is None:
        specular = pf.random.uniform(rng, 0.1, 0.3)
    if noise_detail is None:
        noise_detail = pf.random.uniform(rng, 3.0, 6.0)
    if noise_height is None:
        noise_height = pf.random.uniform(rng, 0.5, 1.5)
    if ior is None:
        ior = 1.46
    if noise_size is None:
        noise_size = pf.random.uniform(rng, 0.005, 0.02)

    return plastic_sandblasted(
        vector=vector,
        color=color,
        roughness=roughness,
        specular=specular,
        ior=ior,
        noise_size=noise_size,
        noise_detail=noise_detail,
        noise_height=noise_height,
    )


@pf.nodes.node_function
def plastic_high_gloss(
    vector: pf.ProcNode[pf.Vector],
    color: t.SocketOrVal[pf.Color] = pf.Color((0.66, 0.788, 0.393)),
    roughness: t.SocketOrVal[float] = 0.25,
    specular: t.SocketOrVal[float] = 0.4,
    ior: t.SocketOrVal[float] = 1.33,
    transmission: t.SocketOrVal[float] = 1.0,
    noise_size: t.SocketOrVal[float] = 1.0,
    noise_detail: t.SocketOrVal[float] = 3.0,
    noise_seed: t.SocketOrVal[float] = 0.0,
) -> pf.Material:
    return _plastic(
        vector=vector,
        surface_color_1=color,
        surface_color_2=color,
        surface_min_roughness=roughness,
        surface_max_roughness=roughness,
        surface_min_specular=specular,
        surface_max_specular=specular,
        surface_ior=ior,
        surface_transmission=transmission,
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


def _plastic_high_gloss_distribution(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
    color: t.SocketOrVal[pf.Color | None] = None,
    roughness: t.SocketOrVal[float | None] = None,
    specular: t.SocketOrVal[float | None] = None,
    ior: t.SocketOrVal[float | None] = None,
    transmission: t.SocketOrVal[float | None] = None,
    noise_size: t.SocketOrVal[float | None] = None,
    noise_detail: t.SocketOrVal[float | None] = None,
    noise_seed: t.SocketOrVal[float | None] = None,
) -> pf.Material:
    if color is None:
        h = pf.random.uniform(rng, 0.0, 1.0)
        s = pf.random.uniform(rng, 0.3, 0.7)
        v = pf.random.uniform(rng, 0.5, 0.85)
        color = pf.color.hsv_to_rgba((h, s, v))
    else:
        h_offset = pf.random.uniform(rng, -0.05, 0.05)
        s_offset = pf.random.uniform(rng, -0.1, 0.1)
        v_offset = pf.random.uniform(rng, -0.15, 0.15)
        color = pf.nodes.color.hue_saturation(
            fac=1.0,
            color=color,
            hue=h_offset + 0.5,
            saturation=s_offset + 1.0,
            value=v_offset + 1.0,
        )
    if roughness is None:
        roughness = pf.random.uniform(rng, 0.25, 0.45)
    if specular is None:
        specular = pf.random.uniform(rng, 0.3, 0.5)
    if transmission is None:
        transmission = pf.random.uniform(rng, 0.4, 0.9)
    if noise_size is None:
        noise_size = pf.random.uniform(rng, 0.3, 2.0)
    if noise_detail is None:
        noise_detail = pf.random.uniform(rng, 2.0, 4.0)
    if noise_seed is None:
        noise_seed = pf.random.uniform(rng, -1000.0, 1000.0)
    if ior is None:
        ior = 1.33

    return plastic_high_gloss(
        vector=vector,
        color=color,
        roughness=roughness,
        specular=specular,
        ior=ior,
        transmission=transmission,
        noise_size=noise_size,
        noise_detail=noise_detail,
        noise_seed=noise_seed,
    )


@pf.nodes.node_function
def plastic_translucent_bumps(
    vector: pf.ProcNode[pf.Vector],
    color: t.SocketOrVal[pf.Color] = pf.Color((0.621, 0.069, 0.069)),
    roughness: t.SocketOrVal[float] = 0.01,
    ior: t.SocketOrVal[float] = 1.3,
    transmission: t.SocketOrVal[float] = 1.0,
    noise_size: t.SocketOrVal[float] = 0.001,
    noise_height: t.SocketOrVal[float] = 0.12,
) -> pf.Material:
    return _plastic(
        vector=vector,
        surface_color_1=color,
        surface_color_2=color,
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


def _plastic_translucent_bumps_distribution(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
    color: t.SocketOrVal[pf.Color | None] = None,
    roughness: t.SocketOrVal[float | None] = None,
    ior: t.SocketOrVal[float | None] = None,
    transmission: t.SocketOrVal[float | None] = None,
    noise_size: t.SocketOrVal[float | None] = None,
    noise_height: t.SocketOrVal[float | None] = None,
) -> pf.Material:
    if color is None:
        h = pf.random.uniform(rng, 0.0, 1.0)
        s = pf.random.uniform(rng, 0.3, 1.0)
        v = pf.random.uniform(rng, 0.5, 0.95)
        color = pf.color.hsv_to_rgba((h, s, v))
    else:
        h_offset = pf.random.uniform(rng, -0.05, 0.05)
        s_offset = pf.random.uniform(rng, -0.1, 0.1)
        v_offset = pf.random.uniform(rng, -0.15, 0.15)
        color = pf.nodes.color.hue_saturation(
            fac=1.0,
            color=color,
            hue=h_offset + 0.5,
            saturation=s_offset + 1.0,
            value=v_offset + 1.0,
        )
    if roughness is None:
        roughness = pf.random.uniform(rng, 0.01, 0.02)
    if ior is None:
        ior = pf.random.uniform(rng, 1.25, 1.35)
    if transmission is None:
        transmission = pf.random.uniform(rng, 0.95, 1.0)
    if noise_size is None:
        noise_size = pf.random.uniform(rng, 0.0005, 0.003)
    if noise_height is None:
        noise_height = pf.random.uniform(rng, 0.08, 0.2)

    return plastic_translucent_bumps(
        vector=vector,
        color=color,
        roughness=roughness,
        ior=ior,
        transmission=transmission,
        noise_size=noise_size,
        noise_height=noise_height,
    )


@pf.nodes.node_function
def plastic_white_textured(
    vector: pf.ProcNode[pf.Vector],
    color: t.SocketOrVal[pf.Color] = pf.Color((0.319, 0.319, 0.319)),
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
        surface_color_1=color,
        surface_color_2=color,
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


def _plastic_white_textured_distribution(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
    color: t.SocketOrVal[pf.Color | None] = None,
    roughness_min: t.SocketOrVal[float | None] = None,
    roughness_max: t.SocketOrVal[float | None] = None,
    specular_min: t.SocketOrVal[float | None] = None,
    specular_max: t.SocketOrVal[float | None] = None,
    ior: t.SocketOrVal[float | None] = None,
    noise_size: t.SocketOrVal[float | None] = None,
    noise_detail: t.SocketOrVal[float | None] = None,
    noise_height: t.SocketOrVal[float | None] = None,
    noise_seed: t.SocketOrVal[float | None] = None,
) -> pf.Material:
    if color is None:
        h = pf.random.uniform(rng, 0.0, 1.0)
        s = pf.random.uniform(rng, 0.0, 0.2)
        v = pf.random.uniform(rng, 0.85, 0.98)
        color = pf.color.hsv_to_rgba((h, s, v))
    else:
        h_offset = pf.random.uniform(rng, -0.05, 0.05)
        s_offset = pf.random.uniform(rng, -0.1, 0.1)
        v_offset = pf.random.uniform(rng, -0.1, 0.1)
        color = pf.nodes.color.hue_saturation(
            fac=1.0,
            color=color,
            hue=h_offset + 0.5,
            saturation=s_offset + 1.0,
            value=v_offset + 1.0,
        )
    if roughness_min is None:
        roughness_min = pf.random.uniform(rng, 0.3, 0.45)
    if roughness_max is None:
        roughness_max = pf.random.uniform(rng, 0.4, 0.55)
    if specular_min is None:
        specular_min = pf.random.uniform(rng, 0.5, 0.7)
    if specular_max is None:
        specular_max = pf.random.uniform(rng, 0.6, 0.75)
    if noise_size is None:
        noise_size = pf.random.uniform(rng, 0.0005, 0.003)
    if noise_detail is None:
        noise_detail = pf.random.uniform(rng, 1.5, 2.5)
    if noise_height is None:
        noise_height = pf.random.uniform(rng, 0.5, 1.5)
    if noise_seed is None:
        noise_seed = pf.random.uniform(rng, 0.0, 10.0)
    if ior is None:
        ior = pf.random.uniform(rng, 1.45, 1.55)

    return plastic_white_textured(
        vector=vector,
        color=color,
        roughness_min=roughness_min,
        roughness_max=roughness_max,
        specular_min=specular_min,
        specular_max=specular_max,
        ior=ior,
        noise_size=noise_size,
        noise_detail=noise_detail,
        noise_height=noise_height,
        noise_seed=noise_seed,
    )
