# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Yiming Zuo: original Infinigen v1 nodegroup (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/materials/metal/hammered_metal.py)
# - Alexander Raistrick: transpile to procfunc/v2
# Acknowledgement: This file draws inspiration https://www.youtube.com/watch?v=82smQvoh0GE by Mix CG Arts

import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.shaders.materials.metal_brushed import (
    metal_any_color_rand,
)

__all__ = [
    "metal_hammered",
    "metal_hammered_rand",
]


@pf.nodes.node_function
def metal_hammered(
    vector: pf.ProcNode[pf.Vector],
    base_color: t.SocketOrVal[pf.Color] = pf.Color((0.17, 0.153, 0.28)),
    scale: t.SocketOrVal[float] = 0.13,
    seed: t.SocketOrVal[float] = -47.4818,
    roughness: t.SocketOrVal[float] = 0.2,
    displacement_strength: t.SocketOrVal[float] = 0.05,
    displacement_power: t.SocketOrVal[float] = 2.3,
) -> pf.Material:
    surface = pf.nodes.shader.principled_bsdf(
        base_color=base_color,
        metallic=1.0,
        roughness=roughness,
        subsurface_scale=1.0,
        subsurface_anisotropy=0.0,
        specular_ior_level=0.0,
    )

    noise = pf.nodes.texture.noise(
        vector=vector,
        w=seed,
        scale=scale * 20.0,
        detail=15.0,
        roughness=0.4,
        distortion=0.2,
        noise_dimensions="4D",
    )
    perturbed_vector = pf.nodes.color.mix_rgb(
        factor=0.01,
        a=vector.astype(dtype=pf.Color),
        b=noise.color,
        clamp_factor=False,
    )
    voronoi = pf.nodes.texture.voronoi_smooth_f1(
        vector=perturbed_vector.astype(dtype=pf.Vector),
        w=seed,
        scale=scale * 300.0,
        smoothness=0.2,
        voronoi_dimensions="4D",
    )
    height = ((voronoi.distance * scale) ** displacement_power) * displacement_strength

    displacement = pf.nodes.shader.displacement(
        height=height,
        midlevel=0.0,
        normal=(0.0, 0.0, 0.0),
    )

    return pf.Material(
        surface=surface,
        displacement=displacement,
        volume=None,
    )


def metal_hammered_rand(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
) -> pf.Material:
    r_color, r_params = rng.spawn(2)

    return metal_hammered(
        vector=vector,
        base_color=metal_any_color_rand(r_color),
        scale=pf.random.log_uniform(r_params, 0.1, 0.18),
        seed=pf.random.uniform(r_params, -1000.0, 1000.0),
        roughness=pf.random.uniform(r_params, 0.15, 0.35),
        displacement_strength=pf.random.log_uniform(r_params, 0.04, 0.09),
        displacement_power=pf.random.uniform(r_params, 2.0, 2.6),
    )
