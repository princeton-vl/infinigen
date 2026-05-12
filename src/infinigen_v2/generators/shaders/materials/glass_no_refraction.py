# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import procfunc as pf
from procfunc.nodes import types as t


def glass_no_refraction_distribution(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
    roughness: t.SocketOrVal[float] | None = None,
) -> pf.Material:
    assert vector is not None
    if roughness is None:
        roughness = pf.random.clip_gaussian(rng, 0.0, 0.015, 0.0, 0.03)
    alpha = pf.random.clip_gaussian(rng, 0.05, 0.05, 0.0, 0.2)
    surface = pf.nodes.shader.principled_bsdf(
        roughness=roughness,
        alpha=alpha,
        metallic=1.0,
    )
    return pf.Material(surface=surface)
