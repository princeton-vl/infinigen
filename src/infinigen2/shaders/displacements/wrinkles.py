# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

from typing import NamedTuple

import numpy as np
import procfunc as pf
from procfunc.nodes import types as t
from procfunc.nodes.util.bpy_node_info import NodeDataType

__all__ = [
    "wrinkles_carpet",
    "wrinkles_carpet_preset",
    "wrinkles_crease",
    "wrinkles_crease_dense",
    "wrinkles_crease_dense_preset",
    "wrinkles_crease_preset",
    "wrinkles_crumpled",
    "wrinkles_crumpled_preset",
    "wrinkles_fabric_rand",
    "wrinkles_paper_crumpled",
    "wrinkles_paper_crumpled_preset",
    "wrinkles_paper_rand",
    "wrinkles_paper_soft",
    "wrinkles_paper_soft_preset",
    "wrinkles_rug",
    "wrinkles_rug_preset",
    "wrinkles_rug_rand",
]


class _WrinkleResult(NamedTuple):
    height: pf.ProcNode[float]
    factor: pf.ProcNode[float]


@pf.nodes.node_function
def _coord_warp(
    vector: t.SocketOrVal[pf.Vector],
    strength: t.SocketOrVal[float],
    w: t.SocketOrVal[float],
    size: t.SocketOrVal[float],
    detail: t.SocketOrVal[float],
    roughness: t.SocketOrVal[float],
    lacunarity: t.SocketOrVal[float] = 2.0,
) -> pf.ProcNode[pf.Vector]:
    roughness = pf.nodes.math.clamp(roughness)
    noise = pf.nodes.texture.noise(
        vector=vector,
        scale=0.5 / size,
        detail=detail,
        roughness=roughness,
        lacunarity=lacunarity,
        noise_dimensions="4D",
        w=w,
    )
    offset = noise.color.astype(dtype=pf.Vector) - (0.5, 0.5, 0.5)
    return vector + offset * (strength * size)


@pf.nodes.node_function
def _s_curve(
    value: t.SocketOrVal[float],
    curve_type: t.SocketOrVal[float] = 1.0,
) -> pf.ProcNode[float]:
    circular_low = 0.5 - pf.nodes.math.sqrt(0.25 - value**2.0)
    circular_high = pf.nodes.math.sqrt(0.25 - (1.0 - value) ** 2.0) + 0.5
    circular = pf.nodes.math.mix(
        a=circular_low,
        b=circular_high,
        factor=value > 0.5,
        data_type=NodeDataType.FLOAT,
    )
    smooth = value**2.0 * 3.0 - value**3.0 * 2.0
    return pf.nodes.math.mix(
        a=circular,
        b=smooth,
        factor=curve_type - 1.0,
        data_type=NodeDataType.FLOAT,
    )


@pf.nodes.node_function
def _height_blend(
    height_a: t.SocketOrVal[float],
    height_b: t.SocketOrVal[float],
    mix_mode: t.SocketOrVal[int] = 1,
    blend_distance: t.SocketOrVal[float] = 0.0,
    mask: t.SocketOrVal[float] = 1.0,
) -> pf.ProcNode[float]:
    maximum = pf.nodes.math.maximum(a=height_a, b=height_b)
    half_blend = blend_distance * 0.5
    blend_value = pf.nodes.math.map_range(
        value=height_b - height_a,
        from_min=-half_blend,
        from_max=half_blend,
        data_type=NodeDataType.FLOAT,
    )
    smooth_max = pf.nodes.math.mix(
        a=height_a,
        b=height_b,
        factor=_s_curve(blend_value),
        data_type=NodeDataType.FLOAT,
    )
    maximum = pf.nodes.math.mix(
        a=maximum,
        b=smooth_max,
        factor=blend_distance > 0.0,
        data_type=NodeDataType.FLOAT,
    )
    combined = pf.nodes.math.mix(
        a=height_a + height_b,
        b=maximum,
        factor=mix_mode.astype(dtype=float) - 1.0,
        data_type=NodeDataType.FLOAT,
    )
    return pf.nodes.math.mix(
        a=height_a,
        b=combined,
        factor=mask,
        data_type=NodeDataType.FLOAT,
    )


@pf.nodes.node_function
def _wrinkles_soft(
    vector: t.SocketOrVal[pf.Vector],
    base_height: t.SocketOrVal[float] = 0.0,
    mix_mode: t.SocketOrVal[int] = 1,
    blend_distance: t.SocketOrVal[float] = 0.0,
    w: t.SocketOrVal[float] = 0.0,
    size: t.SocketOrVal[float] = 0.03,
    roughness: t.SocketOrVal[float] = 0.8,
    spread: t.SocketOrVal[float] = 1.0,
    stretch: t.SocketOrVal[float] = 3.0,
    rotation: t.SocketOrVal[pf.Vector] = (0.0, 0.0, 0.0),
    height: t.SocketOrVal[float] = 0.0,
    steepness: t.SocketOrVal[float] = 1.0,
    distortion_size: t.SocketOrVal[float] = 1.0,
    distortion_strength: t.SocketOrVal[float] = 1.0,
    distortion_detail: t.SocketOrVal[float] = 0.0,
    distortion_roughness: t.SocketOrVal[float] = 0.65,
    mask: t.SocketOrVal[float] = 1.0,
) -> _WrinkleResult:
    warped = _coord_warp(
        vector=vector,
        strength=distortion_strength,
        w=w,
        size=size * distortion_size * 0.5,
        detail=distortion_detail,
        roughness=distortion_roughness,
        lacunarity=0.1,
    )
    rotated = pf.nodes.math.vector_rotate_euler(vector=warped, rotation=rotation)
    stretched = rotated / pf.nodes.math.combine_xyz(x=1.0, y=stretch, z=1.0)
    noise = pf.nodes.texture.noise(
        vector=pf.nodes.math.combine_xyz(x=w, y=w, z=w) + stretched,
        scale=1.0 / size,
        detail=0.0,
        roughness=roughness,
        offset=0.0,
        gain=0.0,
        noise_type="HYBRID_MULTIFRACTAL",
    )
    value = pf.nodes.math.map_range(
        value=pf.nodes.math.absolute(noise.fac),
        from_max=0.75,
        data_type=NodeDataType.FLOAT,
    )
    value = pf.nodes.math.map_range(
        value=value,
        from_min=1.0 - spread,
        data_type=NodeDataType.FLOAT,
    )
    factor = _s_curve(value, curve_type=2.0) ** steepness
    result = _height_blend(
        height_a=base_height,
        height_b=height * factor,
        mix_mode=mix_mode,
        blend_distance=blend_distance,
        mask=mask,
    )
    return _WrinkleResult(result, factor)


@pf.nodes.node_function
def _wrinkles_sharp(
    vector: t.SocketOrVal[pf.Vector],
    base_height: t.SocketOrVal[float] = 0.0,
    mix_mode: t.SocketOrVal[int] = 1,
    w: t.SocketOrVal[float] = 0.0,
    size: t.SocketOrVal[float] = 0.03,
    detail: t.SocketOrVal[float] = 0.0,
    roughness: t.SocketOrVal[float] = 0.6,
    spread: t.SocketOrVal[float] = 0.9,
    stretch: t.SocketOrVal[float] = 1.0,
    rotation: t.SocketOrVal[pf.Vector] = (0.0, 0.0, 0.0),
    height: t.SocketOrVal[float] = 0.0,
    steepness: t.SocketOrVal[float] = 1.0,
    distortion_size: t.SocketOrVal[float] = 1.0,
    distortion_strength: t.SocketOrVal[float] = 1.0,
    distortion_detail: t.SocketOrVal[float] = 0.0,
    distortion_roughness: t.SocketOrVal[float] = 0.35,
    mask: t.SocketOrVal[float] = 1.0,
) -> _WrinkleResult:
    warped = _coord_warp(
        vector=vector,
        strength=distortion_strength,
        w=w,
        size=size * distortion_size * 0.5,
        detail=distortion_detail,
        roughness=distortion_roughness,
    )
    rotated = pf.nodes.math.vector_rotate_euler(vector=warped, rotation=rotation)
    stretched = rotated / pf.nodes.math.combine_xyz(x=1.0, y=stretch, z=1.0)
    voronoi = pf.nodes.texture.voronoi(
        vector=pf.nodes.math.combine_xyz(z=w) + stretched,
        scale=1.0 / size,
        detail=detail,
        roughness=roughness,
        feature="F2",
        normalize=True,
    )
    factor = pf.nodes.math.map_range(
        value=voronoi.distance,
        from_min=0.1,
        from_max=0.5,
        data_type=NodeDataType.FLOAT,
    )
    factor = pf.nodes.math.map_range(
        value=factor,
        from_min=1.0 - spread,
        data_type=NodeDataType.FLOAT,
    )
    factor = factor**steepness
    result = _height_blend(
        height_a=base_height,
        height_b=factor * height,
        mix_mode=mix_mode,
        mask=mask,
    )
    return _WrinkleResult(result, factor)


@pf.nodes.node_function
def _wrinkles_lines(
    vector: t.SocketOrVal[pf.Vector],
    base_height: t.SocketOrVal[float] = 0.0,
    w: t.SocketOrVal[float] = 0.0,
    length: t.SocketOrVal[float] = 0.03,
    thickness: t.SocketOrVal[float] = 0.003,
    rounded: t.SocketOrVal[float] = 0.2,
    random_rotation: t.SocketOrVal[float] = 0.0,
    stretch: t.SocketOrVal[float] = 1.0,
    rotation: t.SocketOrVal[pf.Vector] = (0.0, 0.0, 0.0),
    height: t.SocketOrVal[float] = 0.0,
    steepness: t.SocketOrVal[float] = 1.0,
    distortion_size: t.SocketOrVal[float] = 1.0,
    distortion_strength: t.SocketOrVal[float] = 1.0,
    distortion_detail: t.SocketOrVal[float] = 0.0,
    distortion_roughness: t.SocketOrVal[float] = 0.65,
) -> _WrinkleResult:
    warped = _coord_warp(
        vector=vector,
        strength=distortion_strength,
        w=w,
        size=length * distortion_size * 0.5,
        detail=distortion_detail,
        roughness=distortion_roughness,
    )
    rotated = pf.nodes.math.vector_rotate_euler(vector=warped, rotation=rotation)
    stretched = rotated / pf.nodes.math.combine_xyz(x=1.0, y=stretch, z=1.0)
    cell_vector = stretched + pf.nodes.math.combine_xyz(z=w)
    scale = 1.0 / length
    cell = pf.nodes.texture.voronoi(vector=cell_vector, scale=scale, normalize=True)
    cell_random = pf.nodes.texture.white_noise(cell.color.astype(dtype=pf.Vector))
    random_euler = pf.nodes.math.map_range(
        value=cell_random.color.astype(dtype=pf.Vector),
        from_min=(0.0, 0.0, 0.0),
        from_max=(1.0, 1.0, 1.0),
        to_min=(-np.pi, -np.pi, -np.pi),
        to_max=(np.pi, np.pi, np.pi),
        data_type=NodeDataType.FLOAT_VECTOR,
    )
    local = pf.nodes.math.vector_rotate_euler(
        vector=cell_vector - cell.position,
        rotation=random_euler * random_rotation,
    )
    cross = pf.nodes.math.map_range(
        value=pf.nodes.math.absolute(local.y),
        from_max=thickness,
        to_min=1.0,
        to_max=0.0,
        data_type=NodeDataType.FLOAT,
    )
    round_start = 1.0 - rounded
    curved = pf.nodes.math.float_curve(
        factor=1.0,
        value=pf.nodes.math.map_range(
            value=cross,
            from_min=round_start,
            data_type=NodeDataType.FLOAT,
        ),
        curve=np.array(
            [[0.0, 0.0], [0.2545, 0.2625], [0.7136, 0.6937], [1.0, 0.7687]],
            dtype=np.float64,
        ),
    )
    curved = pf.nodes.math.map_range(
        value=curved,
        to_min=round_start,
        data_type=NodeDataType.FLOAT,
    )
    cross = pf.nodes.math.mix(
        a=cross,
        b=curved,
        factor=cross > round_start,
        data_type=NodeDataType.FLOAT,
    )
    cell_distance = pf.nodes.texture.voronoi_distance(
        vector=cell_vector,
        scale=scale,
        normalize=True,
    )
    fade = pf.nodes.math.float_curve(
        factor=1.0,
        value=pf.nodes.math.map_range(
            value=cell_distance,
            from_max=0.25,
            data_type=NodeDataType.FLOAT,
        ),
        curve=np.array(
            [[0.0, 0.0], [0.0818, 0.0563], [0.8409, 0.9313], [1.0, 1.0]],
            dtype=np.float64,
        ),
    )
    factor = cross**steepness * fade
    result = _height_blend(height_a=base_height, height_b=factor * height)
    return _WrinkleResult(result, factor)


def _displacement(height: t.SocketOrVal[float]) -> pf.ProcNode[pf.Vector]:
    return pf.nodes.shader.displacement(height=height, midlevel=0.0)


@pf.tracer.generator
def wrinkles_carpet(
    vector: t.SocketOrVal[pf.Vector],
    size_scale: t.SocketOrVal[float],
    height_scale: t.SocketOrVal[float],
    phase: t.SocketOrVal[float],
) -> pf.ProcNode[pf.Vector]:
    mask_noise = pf.nodes.texture.noise(
        vector=vector,
        scale=2.0 / size_scale,
        detail=0.0,
        noise_dimensions="4D",
        w=60.6 + phase,
    )
    mask = pf.nodes.math.map_range(
        value=mask_noise.fac, from_min=0.25, from_max=0.75, data_type=NodeDataType.FLOAT
    )
    first = _wrinkles_soft(
        vector=vector,
        w=phase,
        size=0.1 * size_scale,
        roughness=0.8,
        spread=0.89032257,
        stretch=5.0,
        height=0.04 * height_scale,
        distortion_size=2.0,
        distortion_detail=2.0,
        distortion_roughness=0.3,
        mask=mask,
    )
    second = _wrinkles_soft(
        vector=vector,
        base_height=first.height,
        mix_mode=2,
        blend_distance=0.2 * height_scale,
        w=76.8 + phase,
        size=0.1 * size_scale,
        spread=0.8372093,
        stretch=5.0,
        rotation=(0.0, 0.0, np.pi / 2),
        height=0.05 * height_scale,
        distortion_size=2.0,
        distortion_detail=1.0,
        distortion_roughness=0.3,
    )
    return _displacement(second.height)


@pf.tracer.generator
def wrinkles_rug(
    vector: t.SocketOrVal[pf.Vector],
    size_scale: t.SocketOrVal[float],
    height_scale: t.SocketOrVal[float],
    phase: t.SocketOrVal[float],
) -> pf.ProcNode[pf.Vector]:
    mask_noise = pf.nodes.texture.noise(
        vector=vector,
        scale=2.0 / size_scale,
        detail=0.0,
        noise_dimensions="4D",
        w=60.6 + phase,
    )
    mask = pf.nodes.math.map_range(
        value=mask_noise.fac, from_min=0.25, from_max=0.75, data_type=NodeDataType.FLOAT
    )
    result = _wrinkles_soft(
        vector=vector,
        w=phase,
        size=0.1 * size_scale,
        spread=0.7612903,
        stretch=4.0,
        height=0.02 * height_scale,
        distortion_size=2.0,
        distortion_detail=4.0,
        distortion_roughness=0.3,
        mask=mask,
    )
    return _displacement(result.height)


@pf.tracer.generator
def wrinkles_paper_soft(
    vector: t.SocketOrVal[pf.Vector],
    size_scale: t.SocketOrVal[float],
    height_scale: t.SocketOrVal[float],
    phase: t.SocketOrVal[float],
) -> pf.ProcNode[pf.Vector]:
    first = _wrinkles_sharp(
        vector=vector,
        w=32.1 + phase,
        size=0.08 * size_scale,
        detail=1.0,
        roughness=0.6084906,
        stretch=1.5,
        height=-0.008 * height_scale,
        distortion_size=0.5,
        distortion_detail=4.0,
        distortion_roughness=0.5,
    )
    second = _wrinkles_sharp(
        vector=vector,
        base_height=first.height,
        w=phase,
        size=0.05 * size_scale,
        detail=1.0,
        roughness=0.6084906,
        stretch=1.5,
        height=0.003 * height_scale,
        distortion_size=0.5,
        distortion_detail=4.0,
        distortion_roughness=0.5,
    )
    return _displacement(second.height)


@pf.tracer.generator
def wrinkles_paper_crumpled(
    vector: t.SocketOrVal[pf.Vector],
    size_scale: t.SocketOrVal[float],
    height_scale: t.SocketOrVal[float],
    phase: t.SocketOrVal[float],
) -> pf.ProcNode[pf.Vector]:
    first = _wrinkles_sharp(
        vector=vector,
        w=32.1 + phase,
        size=0.08 * size_scale,
        detail=1.0,
        roughness=0.6084906,
        spread=1.0,
        stretch=1.2,
        height=-0.02 * height_scale,
        distortion_strength=1.5,
        distortion_detail=4.0,
        distortion_roughness=0.5,
    )
    mask_noise = pf.nodes.texture.noise(
        vector=vector, noise_dimensions="4D", w=25.7 + phase
    )
    mask = pf.nodes.math.map_range(
        value=mask_noise.fac, from_min=0.3, from_max=0.7, data_type=NodeDataType.FLOAT
    )
    second = _wrinkles_sharp(
        vector=vector,
        base_height=first.height,
        w=phase,
        size=0.05 * size_scale,
        detail=2.0,
        roughness=0.5,
        spread=1.0,
        stretch=1.2,
        height=0.04 * height_scale,
        distortion_size=0.5,
        distortion_detail=6.0,
        distortion_roughness=0.5,
        mask=mask,
    )
    return _displacement(second.height)


@pf.tracer.generator
def wrinkles_crumpled(
    vector: t.SocketOrVal[pf.Vector],
    size_scale: t.SocketOrVal[float],
    height_scale: t.SocketOrVal[float],
    phase: t.SocketOrVal[float],
) -> pf.ProcNode[pf.Vector]:
    first = _wrinkles_sharp(
        vector=vector,
        mix_mode=0,
        w=phase,
        size=0.08 * size_scale,
        roughness=0.5,
        height=0.02 * height_scale,
        distortion_size=0.5,
        distortion_strength=2.0,
        distortion_detail=2.0,
        distortion_roughness=0.65,
    )
    second = _wrinkles_sharp(
        vector=vector,
        base_height=first.height,
        mix_mode=0,
        w=180.5 + phase,
        size=0.06 * size_scale,
        roughness=0.5,
        height=0.008 * height_scale,
        distortion_size=0.5,
        distortion_strength=2.0,
        distortion_detail=1.0,
        distortion_roughness=0.65,
    )
    third = _wrinkles_lines(
        vector=vector,
        base_height=second.height,
        w=phase,
        length=0.32 * size_scale,
        thickness=0.04 * size_scale,
        rounded=0.039408844,
        random_rotation=1.0,
        height=-0.008 * height_scale,
        steepness=2.0,
        distortion_size=0.5,
    )
    fourth = _wrinkles_lines(
        vector=vector,
        base_height=third.height,
        w=36.7 + phase,
        length=0.32 * size_scale,
        thickness=0.04 * size_scale,
        random_rotation=1.0,
        height=-0.008 * height_scale,
        steepness=2.0,
        distortion_size=0.5,
    )
    fifth = _wrinkles_soft(
        vector=vector,
        base_height=fourth.height,
        blend_distance=0.15 * height_scale,
        w=0.34 + phase,
        size=0.16 * size_scale,
        roughness=0.5,
        spread=0.67980295,
        stretch=5.0,
        height=0.02 * height_scale,
        distortion_size=2.0,
    )
    return _displacement(fifth.height)


@pf.tracer.generator
def wrinkles_crease(
    vector: t.SocketOrVal[pf.Vector],
    size_scale: t.SocketOrVal[float],
    height_scale: t.SocketOrVal[float],
    phase: t.SocketOrVal[float],
) -> pf.ProcNode[pf.Vector]:
    sharp = _wrinkles_sharp(
        vector=vector,
        mix_mode=0,
        w=phase,
        size=0.1 * size_scale,
        detail=2.0,
        roughness=0.5,
        stretch=1.5,
        height=0.015 * height_scale,
        distortion_detail=4.0,
        distortion_roughness=0.5169951,
    )
    line_1 = _wrinkles_lines(
        vector=vector,
        base_height=sharp.height,
        w=-19.1 + phase,
        length=0.1 * size_scale,
        thickness=0.028 * size_scale,
        rounded=0.0817734,
        random_rotation=0.773399,
        stretch=3.0,
        height=-0.004 * height_scale,
        steepness=2.0,
        distortion_size=2.0,
        distortion_strength=0.5,
        distortion_detail=4.0,
        distortion_roughness=0.3,
    )
    line_2 = _wrinkles_lines(
        vector=vector,
        base_height=line_1.height,
        w=109.9 + phase,
        length=0.105 * size_scale,
        thickness=0.012 * size_scale,
        rounded=0.0817734,
        random_rotation=0.773399,
        stretch=6.0,
        height=-0.0015 * height_scale,
        steepness=2.0,
        distortion_size=2.0,
        distortion_strength=0.3,
        distortion_detail=4.0,
        distortion_roughness=0.3,
    )
    line_3 = _wrinkles_lines(
        vector=vector,
        base_height=line_2.height,
        w=379.9 + phase,
        length=0.09 * size_scale,
        thickness=0.012 * size_scale,
        rounded=0.0817734,
        random_rotation=0.773399,
        stretch=6.0,
        height=0.001 * height_scale,
        steepness=2.0,
        distortion_size=2.0,
        distortion_strength=0.3,
        distortion_detail=4.0,
        distortion_roughness=0.3,
    )
    line_4 = _wrinkles_lines(
        vector=vector,
        base_height=line_3.height,
        w=109.9 + phase,
        length=0.04 * size_scale,
        thickness=0.012 * size_scale,
        rounded=0.0817734,
        random_rotation=0.4827586,
        stretch=6.0,
        rotation=(0.0, 0.0, np.pi / 6),
        height=0.0008 * height_scale,
        steepness=2.0,
        distortion_size=3.0,
        distortion_detail=2.0,
        distortion_roughness=0.3,
    )
    soft = _wrinkles_soft(
        vector=vector,
        base_height=line_4.height,
        w=0.34 + phase,
        size=0.12 * size_scale,
        spread=0.9162562,
        stretch=5.0,
        height=0.008 * height_scale,
        distortion_size=2.0,
    )
    return _displacement(soft.height)


@pf.tracer.generator
def wrinkles_crease_dense(
    vector: t.SocketOrVal[pf.Vector],
    size_scale: t.SocketOrVal[float],
    height_scale: t.SocketOrVal[float],
    phase: t.SocketOrVal[float],
) -> pf.ProcNode[pf.Vector]:
    sharp_1 = _wrinkles_sharp(
        vector=vector,
        mix_mode=0,
        w=phase,
        size=0.05 * size_scale,
        detail=1.0,
        roughness=0.5,
        spread=1.0,
        stretch=2.0,
        height=0.015 * height_scale,
        distortion_detail=4.0,
        distortion_roughness=0.5169951,
    )
    sharp_2 = _wrinkles_sharp(
        vector=vector,
        base_height=sharp_1.height,
        mix_mode=0,
        w=84.0 + phase,
        size=0.05 * size_scale,
        detail=1.0,
        roughness=0.5,
        spread=1.0,
        stretch=2.0,
        rotation=(0.0, 0.0, 0.19896752),
        height=-0.01 * height_scale,
        distortion_detail=4.0,
        distortion_roughness=0.5169951,
    )
    line_1 = _wrinkles_lines(
        vector=vector,
        base_height=sharp_2.height,
        w=-19.1 + phase,
        length=0.05 * size_scale,
        thickness=0.01 * size_scale,
        rounded=0.0817734,
        random_rotation=0.38423643,
        stretch=6.0,
        height=-0.002 * height_scale,
        steepness=2.0,
        distortion_size=4.0,
        distortion_detail=4.0,
        distortion_roughness=0.3,
    )
    line_2 = _wrinkles_lines(
        vector=vector,
        base_height=line_1.height,
        w=109.9 + phase,
        length=0.105 * size_scale,
        thickness=0.012 * size_scale,
        rounded=0.0817734,
        random_rotation=0.5862069,
        stretch=5.0,
        height=0.0015 * height_scale,
        steepness=2.0,
        distortion_size=4.0,
        distortion_strength=0.3,
        distortion_detail=4.0,
        distortion_roughness=0.3,
    )
    line_3 = _wrinkles_lines(
        vector=vector,
        base_height=line_2.height,
        w=200.0 + phase,
        length=0.05 * size_scale,
        thickness=0.012 * size_scale,
        rounded=0.0817734,
        random_rotation=0.5418719,
        stretch=10.0,
        rotation=(0.0, 0.0, 1.3089969),
        height=0.0015 * height_scale,
        steepness=2.0,
        distortion_size=4.0,
        distortion_strength=0.5,
        distortion_detail=4.0,
        distortion_roughness=0.3,
    )
    line_4 = _wrinkles_lines(
        vector=vector,
        base_height=line_3.height,
        w=300.0 + phase,
        length=0.05 * size_scale,
        thickness=0.012 * size_scale,
        rounded=0.0817734,
        random_rotation=0.9064039,
        stretch=10.0,
        rotation=(0.0, 0.0, 0.9424778),
        height=0.0015 * height_scale,
        steepness=2.0,
        distortion_size=6.0,
        distortion_strength=0.8,
        distortion_detail=4.0,
        distortion_roughness=0.3,
    )
    return _displacement(line_4.height)


def wrinkles_carpet_preset(
    vector: t.SocketOrVal[pf.Vector],
) -> pf.ProcNode[pf.Vector]:
    return wrinkles_carpet(vector, size_scale=1.0, height_scale=1.0, phase=0.0)


def wrinkles_crease_preset(
    vector: t.SocketOrVal[pf.Vector],
) -> pf.ProcNode[pf.Vector]:
    return wrinkles_crease(vector, size_scale=1.0, height_scale=1.0, phase=0.0)


def wrinkles_crease_dense_preset(
    vector: t.SocketOrVal[pf.Vector],
) -> pf.ProcNode[pf.Vector]:
    return wrinkles_crease_dense(vector, size_scale=1.0, height_scale=1.0, phase=0.0)


def wrinkles_crumpled_preset(
    vector: t.SocketOrVal[pf.Vector],
) -> pf.ProcNode[pf.Vector]:
    return wrinkles_crumpled(vector, size_scale=1.0, height_scale=1.0, phase=0.0)


def wrinkles_paper_crumpled_preset(
    vector: t.SocketOrVal[pf.Vector],
) -> pf.ProcNode[pf.Vector]:
    return wrinkles_paper_crumpled(vector, size_scale=1.0, height_scale=1.0, phase=0.0)


def wrinkles_paper_soft_preset(
    vector: t.SocketOrVal[pf.Vector],
) -> pf.ProcNode[pf.Vector]:
    return wrinkles_paper_soft(vector, size_scale=1.0, height_scale=1.0, phase=0.0)


def wrinkles_rug_preset(
    vector: t.SocketOrVal[pf.Vector],
) -> pf.ProcNode[pf.Vector]:
    return wrinkles_rug(vector, size_scale=1.0, height_scale=1.0, phase=0.0)


def wrinkles_fabric_rand(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
) -> pf.ProcNode[pf.Vector]:
    rng_func, rng_size, rng_height, rng_phase = rng.spawn(4)
    size_scale = pf.random.uniform(rng_size, 0.85, 1.15)
    height_scale = pf.random.uniform(rng_height, 0.8, 1.2)
    phase = pf.random.uniform(rng_phase, -100.0, 100.0)
    wrinkles_func = pf.control.choice(
        rng_func,
        [
            (wrinkles_crease, 1.0),
            (wrinkles_crease_dense, 1.0),
            (wrinkles_crumpled, 1.0),
        ],
    )
    return wrinkles_func(
        vector,
        size_scale=size_scale,
        height_scale=height_scale,
        phase=phase,
    )


def wrinkles_rug_rand(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
) -> pf.ProcNode[pf.Vector]:
    rng_func, rng_size, rng_height, rng_phase = rng.spawn(4)
    size_scale = pf.random.uniform(rng_size, 0.85, 1.15)
    height_scale = pf.random.uniform(rng_height, 0.75, 1.25)
    phase = pf.random.uniform(rng_phase, -100.0, 100.0)
    wrinkles_func = pf.control.choice(
        rng_func,
        [
            (wrinkles_carpet, 1.0),
            (wrinkles_rug, 1.0),
        ],
    )
    return wrinkles_func(
        vector,
        size_scale=size_scale,
        height_scale=height_scale,
        phase=phase,
    )


def wrinkles_paper_rand(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
) -> pf.ProcNode[pf.Vector]:
    rng_func, rng_size, rng_height, rng_phase = rng.spawn(4)
    size_scale = pf.random.uniform(rng_size, 0.85, 1.15)
    height_scale = pf.random.uniform(rng_height, 0.8, 1.2)
    phase = pf.random.uniform(rng_phase, -100.0, 100.0)
    wrinkles_func = pf.control.choice(
        rng_func,
        [
            (wrinkles_paper_crumpled, 1.0),
            (wrinkles_paper_soft, 1.0),
        ],
    )
    return wrinkles_func(
        vector,
        size_scale=size_scale,
        height_scale=height_scale,
        phase=phase,
    )
