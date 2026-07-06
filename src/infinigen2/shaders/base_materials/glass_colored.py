# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Transpiled into procfunc/v2 format by Alexander Raistrick

import procfunc as pf
from procfunc.nodes import types as t

__all__ = [
    "glass_colored",
    "glass_colored_color_rand",
    "glass_colored_rand",
]


@pf.nodes.node_function
def glass_colored(
    base_color: t.SocketOrVal[pf.Color],
    roughness: t.SocketOrVal[float] = 0.0,
) -> pf.Material:
    surface = pf.nodes.shader.principled_bsdf(
        base_color=base_color,
        roughness=roughness,
        ior=1.5,
        transmission_weight=1.0,
    )
    return pf.Material(
        surface=surface,
        displacement=pf.nodes.math.constant((0.0, 0.0, 0.0)),
    )


def glass_colored_color_rand(rng: pf.RNG) -> pf.Color:
    hue = pf.random.uniform(rng, 0.0, 1.0)
    sat = pf.random.uniform(rng, 0.5, 0.9)
    val = pf.random.uniform(rng, 0.6, 0.9)
    return pf.color.hsv_color(hue=hue, saturation=sat, value=val)


def glass_colored_rand(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector] | None = None,
    base_color: t.SocketOrVal[pf.Color] | None = None,
    roughness: t.SocketOrVal[float] | None = None,
) -> pf.Material:
    del vector
    rng_choice, rng_color = rng.spawn(2)
    if base_color is None:
        base_color = pf.control.choice(
            rng_choice,
            [
                (glass_colored_color_rand(rng_color), 0.7),
                (pf.color.hsv_color(hue=0.0, saturation=0.0, value=1.0), 0.3),
            ],
        )
    if roughness is None:
        roughness = pf.random.clip_gaussian(rng, 0.0, 0.02, 0.0, 0.05)

    return glass_colored(base_color=base_color, roughness=roughness)
