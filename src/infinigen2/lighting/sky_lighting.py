# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Alexander Raistrick, Zeyu Ma, Kaiyu Yang, Lingjie Mei: original Infinigen v1 sky lighting (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/lighting/sky_lighting.py)
# - Alexander Raistrick: port to v2

from typing import NamedTuple

import numpy as np
import procfunc as pf

__all__ = [
    "EnvironmentResult",
    "hosek_wilkie_sky",
    "hosek_wilkie_sky_rand",
    "hosek_wilkie_sky_with_sun_lamp_rand",
    "nishita_sky",
    "nishita_sky_rand",
    "sky_with_sun_lamp_rand",
]


class EnvironmentResult(NamedTuple):
    environment: pf.World
    lights: list


# Direct-sun irradiance relative to ambient sky strength; ~5:1 reads sunny,
# ~1:1 reads flat/overcast and the sun looks dim.
SUN_SKY_RATIO = 6.0


def _sun_warmth_color(
    rng: np.random.Generator, sun_elevation_deg: float
) -> tuple[float, float, float]:
    """Warm near the horizon, neutral overhead, with slight jitter."""
    t = float(np.clip(sun_elevation_deg / 60.0, 0.0, 1.0))
    warm = np.array([1.0, 0.60, 0.30])
    neutral = np.array([1.0, 0.95, 0.88])
    base = warm + (neutral - warm) * t
    color = np.clip(base + pf.random.uniform(rng, -0.03, 0.03), 0.0, 1.0)
    return tuple(float(c) for c in color)


@pf.tracer.primitive
def _nishita_sky(
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
    sky_texture = pf.nodes.texture.sky(
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
) -> EnvironmentResult:
    world = _nishita_sky(
        sun_size_deg=sun_size_deg,
        sun_intensity=sun_intensity,
        sun_elevation_deg=sun_elevation_deg,
        sun_rotation_deg=sun_rotation_deg,
        altitude=altitude,
        air_density=air_density,
        dust_density=dust_density,
        ozone_density=ozone_density,
        strength=strength,
        sun_disc=sun_disc,
    )
    return EnvironmentResult(environment=world, lights=[])


@pf.tracer.grammar
def nishita_sky_rand(
    rng: np.random.Generator,
    sun_elevation_deg: float | None = None,
    sun_rotation_deg: float | None = None,
    sun_intensity: float | None = None,
    sun_size_deg: float | None = None,
    sun_disc: bool = True,
) -> EnvironmentResult:
    if sun_elevation_deg is None:
        sun_elevation_deg = pf.random.uniform(rng, 5.0, 85.0)
    if sun_rotation_deg is None:
        sun_rotation_deg = pf.random.uniform(rng, 0.0, 360.0)
    if sun_intensity is None:
        sun_intensity = pf.random.uniform(rng, 0.8, 1.0)
    if sun_size_deg is None:
        sun_size_deg = pf.random.clip_gaussian(rng, 0.5, 0.3, 0.25, 5)
    world = _nishita_sky(
        sun_size_deg=sun_size_deg,
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
    return EnvironmentResult(environment=world, lights=[])


@pf.tracer.primitive
def _hosek_wilkie_sky(
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

    sky_texture = pf.nodes.texture.sky(
        sky_type="HOSEK_WILKIE",
        sun_direction=sun_direction,
        turbidity=turbidity,
        ground_albedo=ground_albedo,
    )

    shader = pf.nodes.shader.background(sky_texture, strength=strength)
    return pf.nodes.to_environment(surface=shader)


def hosek_wilkie_sky(
    sun_elevation_deg: float = 45.0,
    sun_rotation_deg: float = 0.0,
    turbidity: float = 2.0,
    ground_albedo: float = 0.3,
    strength: float = 7.5,
) -> EnvironmentResult:
    world = _hosek_wilkie_sky(
        sun_elevation_deg=sun_elevation_deg,
        sun_rotation_deg=sun_rotation_deg,
        turbidity=turbidity,
        ground_albedo=ground_albedo,
        strength=strength,
    )
    return EnvironmentResult(environment=world, lights=[])


@pf.tracer.grammar
def hosek_wilkie_sky_rand(
    rng: np.random.Generator,
    sun_elevation_deg: float | None = None,
    sun_rotation_deg: float | None = None,
) -> EnvironmentResult:
    if sun_elevation_deg is None:
        sun_elevation_deg = pf.random.uniform(rng, 10.0, 80.0)
    if sun_rotation_deg is None:
        sun_rotation_deg = pf.random.uniform(rng, 0.0, 360.0)
    world = _hosek_wilkie_sky(
        sun_elevation_deg=sun_elevation_deg,
        sun_rotation_deg=sun_rotation_deg,
        turbidity=pf.random.uniform(rng, 0.0, 1.0),
        ground_albedo=pf.random.uniform(rng, 0.0, 1.0),
        strength=pf.random.uniform(rng, 5.0, 10.0),
    )
    return EnvironmentResult(environment=world, lights=[])


@pf.tracer.grammar
def hosek_wilkie_sky_with_sun_lamp_rand(
    rng: np.random.Generator,
    sun_elevation_deg: float | None = None,
    sun_rotation_deg: float | None = None,
    sun_intensity: float | None = None,
    sun_size_deg: float | None = None,
) -> EnvironmentResult:
    r_default, r_sky, r_warmth = rng.spawn(3)
    if sun_elevation_deg is None:
        sun_elevation_deg = pf.random.uniform(r_default, 10.0, 80.0)
    if sun_rotation_deg is None:
        sun_rotation_deg = pf.random.uniform(r_default, 0.0, 360.0)
    if sun_intensity is None:
        sun_intensity = pf.random.uniform(r_default, 0.8, 1.0)
    if sun_size_deg is None:
        sun_size_deg = pf.random.clip_gaussian(r_default, 0.5, 0.3, 0.25, 5)

    sky_strength = pf.random.uniform(r_sky, 5.0, 7.0)
    world = _hosek_wilkie_sky(
        sun_elevation_deg=sun_elevation_deg,
        sun_rotation_deg=sun_rotation_deg,
        turbidity=pf.random.uniform(r_sky, 2.0, 4.0),
        ground_albedo=pf.random.uniform(r_sky, 0.2, 0.5),
        strength=sky_strength,
    )

    euler = pf.Euler(
        (np.deg2rad(90 - sun_elevation_deg), 0, np.deg2rad(sun_rotation_deg))
    )
    sun_direction = euler.to_matrix() @ pf.Vector((0, 0, 1))

    sun_lamp = pf.ops.primitives.sun_lamp(
        intensity=SUN_SKY_RATIO * sky_strength * sun_intensity
    )
    sun_lamp.item().data.angle = np.deg2rad(sun_size_deg)
    sun_lamp.item().data.color = _sun_warmth_color(r_warmth, sun_elevation_deg)
    sun_lamp.item().rotation_euler = sun_direction.to_track_quat("Z", "Y").to_euler()
    sun_lamp.item().location.z = 25

    return EnvironmentResult(environment=world, lights=[sun_lamp])


@pf.tracer.grammar
def sky_with_sun_lamp_rand(
    rng: np.random.Generator,
    sun_elevation_deg: float | None = None,
    sun_rotation_deg: float | None = None,
    sun_intensity: float | None = None,
) -> EnvironmentResult:
    if sun_elevation_deg is None:
        sun_elevation_deg = pf.random.clip_gaussian(rng, 30, 20, 10, 90)
    if sun_rotation_deg is None:
        sun_rotation_deg = pf.random.uniform(rng, 0, 360)
    if sun_intensity is None:
        sun_intensity = pf.random.uniform(rng, 0.8, 1.0)

    sky = nishita_sky_rand(
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

    return EnvironmentResult(environment=sky.environment, lights=[sun_lamp])
