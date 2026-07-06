# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

from typing import NamedTuple

import numpy as np
import procfunc as pf

from infinigen2.scenes import placement_utils

__all__ = [
    "ColoredLightsResult",
    "floating_colored_lights_rand",
    "point_lamp_colored_rand",
]


class ColoredLightsResult(NamedTuple):
    environment: pf.World
    lights: list


@pf.tracer.grammar
def point_lamp_colored_rand(
    rng: pf.RNG,
    energy: float,
    color: tuple[float, float, float] | None = None,
    shadow_soft_size: float | None = None,
) -> pf.LightObject:
    """A point light whose emission uses a fully-random RGB color."""
    if color is None:
        color = tuple(pf.random.uniform(rng, 0.0, 1.0) for _ in range(3))
    if shadow_soft_size is None:
        shadow_soft_size = pf.random.uniform(rng, 0.01, 0.20)

    return pf.ops.primitives.light.point_lamp(
        energy=energy,
        color=color,
        shadow_soft_size=shadow_soft_size,
    )


def _dark_environment() -> pf.World:
    shader = pf.nodes.shader.background(color=(0.0, 0.0, 0.0, 1.0), strength=0.0)
    return pf.nodes.to_environment(surface=shader)


@pf.tracer.grammar
def floating_colored_lights_rand(
    rng: pf.RNG,
    bbox: tuple[np.ndarray, np.ndarray] | None = None,
    n_lights: int | None = None,
    max_lights: int = 4,
    total_wattage: float | None = None,
) -> ColoredLightsResult:
    """Random-colored point lamps scattered in ``bbox`` against a black world, so a
    demo object reads as lit purely by the colored lights."""
    if bbox is None:
        bbox = (np.array([-1.5, -1.5, 0.3]), np.array([1.5, 1.5, 2.5]))

    if n_lights is None:
        n_lights = int(rng.integers(2, max_lights + 1))
    if total_wattage is None:
        total_wattage = pf.random.uniform(rng, 60.0, 200.0)

    light_rngs = rng.spawn(n_lights)
    fractions = rng.dirichlet(np.ones(n_lights))

    lights = []
    for i in range(n_lights):
        light = point_lamp_colored_rand(
            light_rngs[i], energy=float(fractions[i] * total_wattage)
        )
        light.item().name = f"colored_light.{i:02d}"
        lights.append(light)

    placement_utils.distribute_in_bbox(rng, lights, bbox)

    return ColoredLightsResult(environment=_dark_environment(), lights=lights)
