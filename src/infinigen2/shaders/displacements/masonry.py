# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.shaders.displacements.chip_noise import (
    DisplacementLayerResult,
    chip_displacement,
    noise_displacement,
    voronoi_displacement,
)
from infinigen2.shaders.util.coord import space_warp

__all__ = [
    "masonry_displacement_rand",
    "noise_displacement_layer",
    "voronoi_displacement_layer",
]


def _shape_layer(
    height_value,
    vector,
    previous_layer,
    mask,
    height,
    clamp_min,
    clamp_max,
) -> DisplacementLayerResult:
    height_clamped = pf.nodes.math.clamp(
        value=height_value * (height * 0.2), min=clamp_min, max=clamp_max
    )
    return DisplacementLayerResult(
        height=(height_clamped * mask) + previous_layer,
        displacement=vector,
    )


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
) -> DisplacementLayerResult:
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
        vector=vector,
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
) -> DisplacementLayerResult:
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
        vector=vector,
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
) -> DisplacementLayerResult:
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
        vector=vector,
        previous_layer=previous_layer,
        mask=mask,
        height=height,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
    )


def masonry_displacement_rand(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
    layer_1_height: t.SocketOrVal[float],
    layer_1_clamp_max: t.SocketOrVal[float],
    layer_2_mask: t.SocketOrVal[float],
    layer_2_height: t.SocketOrVal[float],
):
    if layer_1_clamp_max is None:
        layer_1_clamp_max = pf.random.uniform(rng, 0.015, 9.715)

    layer_1_func = pf.control.choice(
        rng,
        [
            (noise_displacement_layer, 1.0),
            (_chip_displacement_layer, 1.0),
        ],
    )

    layer_1_result = layer_1_func(
        vector=vector,
        mask=1.0,
        w=0.0,
        size=pf.random.uniform(rng, 0.09, 0.12),
        detail=pf.random.uniform(rng, 0, 6),
        warp_strength=pf.random.uniform(rng, 1, 2),
        warp_size=pf.random.uniform(rng, 0.25, 1.0),
        warp_detail=pf.random.uniform(rng, 1, 4),
        height=layer_1_height,
        clamp_min=0.0,
        clamp_max=layer_1_clamp_max,
    )

    layer_2_result = noise_displacement_layer(
        vector=layer_1_result.displacement,
        previous_layer=layer_1_result.height,
        mask=layer_2_mask,
        w=0.0,
        size=pf.random.uniform(rng, 0.1, 0.2),
        detail=pf.random.uniform(rng, 4, 8),
        warp_strength=1.0,
        warp_size=pf.random.uniform(rng, 0.2, 1.0),
        warp_detail=pf.random.uniform(rng, 1, 4),
        height=layer_2_height,
        clamp_min=-12.0,
        clamp_max=999.0,
    )

    displacement = pf.nodes.shader.displacement(
        height=layer_2_result.height,
        midlevel=0.0,
        normal=(0.0, 0.0, 0.0),
    )
    return DisplacementLayerResult(
        displacement=displacement,
        height=layer_2_result.height,
    )
