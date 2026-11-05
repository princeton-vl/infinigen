# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Alexander Raistrick: refactor for Infinigen2

import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.shaders.displacements.chip_noise import (
    chip_displacement,
    noise_displacement,
    voronoi_displacement,
)
from infinigen2.shaders.util.coord import space_warp

__all__ = [
    "masonry_displacement_chip",
    "masonry_displacement_noise",
    "masonry_displacement_rand",
    "noise_displacement_layer",
    "voronoi_displacement_layer",
]


def _shape_layer(
    height_value,
    previous_layer,
    mask,
    height,
    clamp_min,
    clamp_max,
) -> pf.ProcNode[float]:
    height_clamped = pf.nodes.math.clamp(
        value=height_value * (height * 0.2), min=clamp_min, max=clamp_max
    )
    return (height_clamped * mask) + previous_layer


@pf.nodes.node_function
def noise_displacement_layer(
    vector: t.SocketOrVal[pf.Vector],
    previous_layer: t.SocketOrVal[float] = 0.0,
    mask: t.SocketOrVal[float] = 1.0,
    w: t.SocketOrVal[float] = 0.0,
    size: t.SocketOrVal[float] = 1.0,
    detail: t.SocketOrVal[float] = 4.0,
    warp_strength: t.SocketOrVal[float] = 1.0,
    warp_size: t.SocketOrVal[float] = 1.0,
    warp_detail: t.SocketOrVal[float] = 2.0,
    height: t.SocketOrVal[float] = 1.0,
    clamp_min: t.SocketOrVal[float] = 0.0,
    clamp_max: t.SocketOrVal[float] = 999.0,
) -> pf.ProcNode[float]:
    warp = space_warp(
        vector=vector,
        strength=warp_strength,
        w=w,
        size=size * warp_size,
        detail=warp_detail,
    )
    height_value = noise_displacement(
        vector=warp.vector, w=w, size=size, detail=detail, lacunarity=2.0
    )
    return _shape_layer(
        height_value=height_value,
        previous_layer=previous_layer,
        mask=mask,
        height=height,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
    )


@pf.nodes.node_function
def _chip_displacement_layer(
    vector: t.SocketOrVal[pf.Vector],
    previous_layer: t.SocketOrVal[float] = 0.0,
    mask: t.SocketOrVal[float] = 1.0,
    w: t.SocketOrVal[float] = 0.0,
    size: t.SocketOrVal[float] = 1.0,
    detail: t.SocketOrVal[float] = 4.0,
    warp_strength: t.SocketOrVal[float] = 1.0,
    warp_size: t.SocketOrVal[float] = 1.0,
    warp_detail: t.SocketOrVal[float] = 2.0,
    height: t.SocketOrVal[float] = 1.0,
    clamp_min: t.SocketOrVal[float] = 0.0,
    clamp_max: t.SocketOrVal[float] = 999.0,
) -> pf.ProcNode[float]:
    warp = space_warp(
        vector=vector,
        strength=warp_strength,
        w=w,
        size=size * warp_size,
        detail=warp_detail,
    )
    height_value = chip_displacement(
        vector=warp.vector, w=w, size=size, detail=detail, lacunarity=2.0
    )
    return _shape_layer(
        height_value=height_value,
        previous_layer=previous_layer,
        mask=mask,
        height=height,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
    )


@pf.nodes.node_function
def voronoi_displacement_layer(
    vector: t.SocketOrVal[pf.Vector],
    previous_layer: t.SocketOrVal[float] = 0.0,
    mask: t.SocketOrVal[float] = 1.0,
    w: t.SocketOrVal[float] = 0.0,
    size: t.SocketOrVal[float] = 1.0,
    detail: t.SocketOrVal[float] = 4.0,
    warp_strength: t.SocketOrVal[float] = 1.0,
    warp_size: t.SocketOrVal[float] = 1.0,
    warp_detail: t.SocketOrVal[float] = 2.0,
    height: t.SocketOrVal[float] = 1.0,
    clamp_min: t.SocketOrVal[float] = 0.0,
    clamp_max: t.SocketOrVal[float] = 999.0,
) -> pf.ProcNode[float]:
    warp = space_warp(
        vector=vector,
        strength=warp_strength,
        w=w,
        size=size * warp_size,
        detail=warp_detail,
    )
    height_value = voronoi_displacement(
        vector=warp.vector, w=w, size=size, detail=detail, lacunarity=2.0
    )
    return _shape_layer(
        height_value=height_value,
        previous_layer=previous_layer,
        mask=mask,
        height=height,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
    )


def masonry_displacement_rand(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
    layer_1_height: t.SocketOrVal[float] = 0.04,
    layer_1_clamp_max: t.SocketOrVal[float] | None = 0.015,
    layer_2_mask: t.SocketOrVal[float] = 1.0,
    layer_2_height: t.SocketOrVal[float] = 0.01,
) -> pf.ProcNode[pf.Vector]:
    rng, rng_layer_1 = rng.spawn(2)
    if layer_1_clamp_max is None:
        layer_1_clamp_max = pf.random.uniform(rng, 0.015, 9.715)

    masonry_func = pf.control.choice(
        rng_layer_1,
        [(masonry_displacement_noise, 1.0), (masonry_displacement_chip, 1.0)],
    )
    layer_1_size = pf.random.uniform(rng, 0.09, 0.12)
    layer_1_detail = pf.random.uniform(rng, 0, 6)
    layer_1_warp_strength = pf.random.uniform(rng, 1, 2)
    layer_1_warp_size = pf.random.uniform(rng, 0.25, 1.0)
    layer_1_warp_detail = pf.random.uniform(rng, 1, 4)
    layer_2_size = pf.random.uniform(rng, 0.1, 0.2)
    layer_2_detail = pf.random.uniform(rng, 4, 8)
    layer_2_warp_size = pf.random.uniform(rng, 0.2, 1.0)
    layer_2_warp_detail = pf.random.uniform(rng, 1, 4)

    return masonry_func(
        vector=vector,
        layer_1_size=layer_1_size,
        layer_1_detail=layer_1_detail,
        layer_1_warp_strength=layer_1_warp_strength,
        layer_1_warp_size=layer_1_warp_size,
        layer_1_warp_detail=layer_1_warp_detail,
        layer_1_height=layer_1_height,
        layer_1_clamp_max=layer_1_clamp_max,
        layer_2_mask=layer_2_mask,
        layer_2_size=layer_2_size,
        layer_2_detail=layer_2_detail,
        layer_2_warp_size=layer_2_warp_size,
        layer_2_warp_detail=layer_2_warp_detail,
        layer_2_height=layer_2_height,
    )


def _masonry_layer_2(
    vector: t.SocketOrVal[pf.Vector],
    layer_1: pf.ProcNode[float],
    layer_2_mask: t.SocketOrVal[float],
    layer_2_size: float,
    layer_2_detail: float,
    layer_2_warp_size: float,
    layer_2_warp_detail: float,
    layer_2_height: t.SocketOrVal[float],
) -> pf.ProcNode[pf.Vector]:
    layer_2 = noise_displacement_layer(
        vector=vector,
        previous_layer=layer_1,
        mask=layer_2_mask,
        w=0.0,
        size=layer_2_size,
        detail=layer_2_detail,
        warp_strength=1.0,
        warp_size=layer_2_warp_size,
        warp_detail=layer_2_warp_detail,
        height=layer_2_height,
        clamp_min=-12.0,
        clamp_max=999.0,
    )
    return pf.nodes.shader.displacement(height=layer_2, midlevel=0.0)


@pf.tracer.generator
def masonry_displacement_noise(
    vector: t.SocketOrVal[pf.Vector],
    layer_1_size: float,
    layer_1_detail: float,
    layer_1_warp_strength: float,
    layer_1_warp_size: float,
    layer_1_warp_detail: float,
    layer_1_height: t.SocketOrVal[float],
    layer_1_clamp_max: t.SocketOrVal[float],
    layer_2_mask: t.SocketOrVal[float],
    layer_2_size: float,
    layer_2_detail: float,
    layer_2_warp_size: float,
    layer_2_warp_detail: float,
    layer_2_height: t.SocketOrVal[float],
) -> pf.ProcNode[pf.Vector]:
    layer_1 = noise_displacement_layer(
        vector=vector,
        mask=1.0,
        w=0.0,
        size=layer_1_size,
        detail=layer_1_detail,
        warp_strength=layer_1_warp_strength,
        warp_size=layer_1_warp_size,
        warp_detail=layer_1_warp_detail,
        height=layer_1_height,
        clamp_min=0.0,
        clamp_max=layer_1_clamp_max,
    )
    return _masonry_layer_2(
        vector=vector,
        layer_1=layer_1,
        layer_2_mask=layer_2_mask,
        layer_2_size=layer_2_size,
        layer_2_detail=layer_2_detail,
        layer_2_warp_size=layer_2_warp_size,
        layer_2_warp_detail=layer_2_warp_detail,
        layer_2_height=layer_2_height,
    )


@pf.tracer.generator
def masonry_displacement_chip(
    vector: t.SocketOrVal[pf.Vector],
    layer_1_size: float,
    layer_1_detail: float,
    layer_1_warp_strength: float,
    layer_1_warp_size: float,
    layer_1_warp_detail: float,
    layer_1_height: t.SocketOrVal[float],
    layer_1_clamp_max: t.SocketOrVal[float],
    layer_2_mask: t.SocketOrVal[float],
    layer_2_size: float,
    layer_2_detail: float,
    layer_2_warp_size: float,
    layer_2_warp_detail: float,
    layer_2_height: t.SocketOrVal[float],
) -> pf.ProcNode[pf.Vector]:
    layer_1 = _chip_displacement_layer(
        vector=vector,
        mask=1.0,
        w=0.0,
        size=layer_1_size,
        detail=layer_1_detail,
        warp_strength=layer_1_warp_strength,
        warp_size=layer_1_warp_size,
        warp_detail=layer_1_warp_detail,
        height=layer_1_height,
        clamp_min=0.0,
        clamp_max=layer_1_clamp_max,
    )
    return _masonry_layer_2(
        vector=vector,
        layer_1=layer_1,
        layer_2_mask=layer_2_mask,
        layer_2_size=layer_2_size,
        layer_2_detail=layer_2_detail,
        layer_2_warp_size=layer_2_warp_size,
        layer_2_warp_detail=layer_2_warp_detail,
        layer_2_height=layer_2_height,
    )
