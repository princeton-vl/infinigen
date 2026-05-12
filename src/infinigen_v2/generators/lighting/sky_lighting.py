import numpy as np
import procfunc as pf


@pf.tracer.primitive
def nishita_sky(
    sun_size_deg: float = 0.5,
    sun_intensity: float = 0.6,
    sun_elevation_deg: float = 10.0,
    sun_rotation_deg: float = 0.0,
    altitude: float = 550.0,
    air_density: float = 1.0,
    dust_density: float = 1.0,
    ozone_density: float = 1.0,
    strength: float = 0.4,
    sun_disc: bool = True,
) -> pf.World:
    sky_texture = pf.nodes.shader.sky(
        sky_type="NISHITA",
        sun_size=np.deg2rad(sun_size_deg),
        sun_intensity=sun_intensity,
        sun_elevation=np.deg2rad(sun_elevation_deg),
        sun_rotation=np.deg2rad(sun_rotation_deg),
        altitude=altitude,
        air_density=air_density,
        dust_density=dust_density,
        ozone_density=ozone_density,
        sun_disc=sun_disc,
    )

    shader = pf.nodes.shader.background(sky_texture, strength=strength)
    return pf.nodes.to_environment(surface=shader)


@pf.tracer.grammar
def nishita_sky_distribution(
    rng: np.random.Generator,
    sun_elevation_deg: float | None = None,
    sun_rotation_deg: float | None = None,
    sun_intensity: float | None = None,
    sun_disc: bool = True,
) -> pf.World:
    if sun_elevation_deg is None:
        sun_elevation_deg = pf.random.uniform(rng, 5.0, 85.0)
    if sun_rotation_deg is None:
        sun_rotation_deg = pf.random.uniform(rng, 0.0, 360.0)
    if sun_intensity is None:
        sun_intensity = pf.random.uniform(rng, 0.8, 1.0)
    return nishita_sky(
        sun_size_deg=pf.random.clip_gaussian(rng, 0.5, 0.3, 0.25, 5),
        sun_intensity=sun_intensity,
        sun_elevation_deg=sun_elevation_deg,
        sun_rotation_deg=sun_rotation_deg,
        altitude=pf.random.clip_gaussian(rng, 100.0, 400.0, 0.0, 2000.0),
        air_density=pf.random.clip_gaussian(rng, 1.0, 0.2, 0.7, 1.3),
        dust_density=pf.random.clip_gaussian(rng, 1.0, 1.0, 0.1, 2.0),
        ozone_density=pf.random.clip_gaussian(rng, 1.0, 1.0, 0.1, 10.0),
        strength=pf.random.uniform(rng, 0.18, 0.22),
        sun_disc=sun_disc,
    )


@pf.tracer.primitive
def hosek_wilkie_sky(
    sun_elevation_deg: float = 45.0,
    sun_rotation_deg: float = 0.0,
    turbidity: float = 2.0,
    ground_albedo: float = 0.3,
    strength: float = 7.5,
) -> pf.World:
    euler = pf.Euler(
        (np.deg2rad(90 - sun_elevation_deg), 0, np.deg2rad(sun_rotation_deg))
    )
    sun_direction = euler.to_matrix() @ pf.Vector((0, 0, 1))

    sky_texture = pf.nodes.shader.sky(
        sky_type="HOSEK_WILKIE",
        sun_direction=sun_direction,
        turbidity=turbidity,
        ground_albedo=ground_albedo,
    )

    shader = pf.nodes.shader.background(sky_texture.color, strength=strength)
    return pf.nodes.to_environment(surface=shader)


@pf.tracer.grammar
def hosek_wilkie_sky_distribution(
    rng: np.random.Generator,
    sun_elevation_deg: float | None = None,
    sun_rotation_deg: float | None = None,
) -> pf.World:
    if sun_elevation_deg is None:
        sun_elevation_deg = pf.random.uniform(rng, 10.0, 80.0)
    if sun_rotation_deg is None:
        sun_rotation_deg = pf.random.uniform(rng, 0.0, 360.0)
    return hosek_wilkie_sky(
        sun_elevation_deg=sun_elevation_deg,
        sun_rotation_deg=sun_rotation_deg,
        turbidity=pf.random.uniform(rng, 0.0, 1.0),
        ground_albedo=pf.random.uniform(rng, 0.0, 1.0),
        strength=pf.random.uniform(rng, 5.0, 10.0),
    )


@pf.tracer.grammar
def sky_with_sun_lamp_distribution(
    rng: np.random.Generator,
    sun_elevation_deg: float | None = None,
    sun_rotation_deg: float | None = None,
    sun_intensity: float | None = None,
) -> tuple[pf.World, pf.Object]:
    if sun_elevation_deg is None:
        sun_elevation_deg = pf.random.clip_gaussian(rng, 30, 20, 10, 90)
    if sun_rotation_deg is None:
        sun_rotation_deg = pf.random.uniform(rng, 0, 360)
    if sun_intensity is None:
        sun_intensity = pf.random.uniform(rng, 0.8, 1.0)

    sky_shader = nishita_sky_distribution(
        rng,
        sun_elevation_deg=sun_elevation_deg,
        sun_rotation_deg=sun_rotation_deg,
        sun_intensity=sun_intensity,
        sun_disc=False,
    )

    sun_lamp = pf.ops.primitives.sun_lamp(
        intensity=sun_intensity * 10.0,
    )
    sun_lamp.item().rotation_euler = np.deg2rad(
        np.array([90 - sun_elevation_deg, 0, 180 - sun_rotation_deg])
    )
    sun_lamp.item().location.z = 25  # easier to see in blender UI

    return sky_shader, sun_lamp
