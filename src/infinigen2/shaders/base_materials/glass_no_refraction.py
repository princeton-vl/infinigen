# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import procfunc as pf
from procfunc.nodes import types as t

__all__ = [
    "glass_no_refraction_rand",
]


def glass_no_refraction_rand(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
    roughness: t.SocketOrVal[float] | None = None,
) -> pf.Material:
    assert vector is not None
    if roughness is None:
        roughness = pf.random.clip_gaussian(rng, 0.05, 0.03, 0.02, 0.15)
    alpha = pf.random.clip_gaussian(rng, 0.05, 0.05, 0.0, 0.2)
    surface = pf.nodes.shader.principled_bsdf(
        roughness=roughness,
        alpha=alpha,
        metallic=0.0,
    )
    return pf.Material(
        surface=surface, displacement=pf.nodes.math.constant((0.0, 0.0, 0.0))
    )
