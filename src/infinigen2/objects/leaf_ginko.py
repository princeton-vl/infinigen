# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Yiming Zuo: original Infinigen leaf ginko (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/objects/leaves/leaf_ginko.py)
# - Alexander Raistrick: refactor for Infinigen2

import math
from typing import NamedTuple

import numpy as np
import procfunc as pf
from procfunc.nodes import types as t
from procfunc.nodes.util.bpy_node_info import NodeDataType

from infinigen2.objects.leaf_coord import leaf_coord, normalize_leaf_coord
from infinigen2.shaders.base_materials.leaf import leaf_rand as leaf_material_rand

__all__ = [
    "LeafGinkoResult",
    "leaf_ginko",
    "leaf_ginkgo_rand",
]

# Coarse boundary-solved topology. The old pipeline subdivided a dense plane and
# threshold-deleted cells outside the silhouette, welding poly count to grid resolution.
# ginko_shape is affine in radius: shape(r, theta) = r * g(theta) - h(theta), so the boundary
# is the closed form r_b(theta) = h/g, recovered in-graph from two shape evals along each
# vertex direction. We warp a coarse polar fan so its rim lands exactly on r_b, keeping
# topology coarse and constant instead of carving a dense grid.
_ANGULAR_VERTS = 99
_RING_VERTS = 12
_TWO_PI = 2.0 * math.pi


@pf.nodes.node_function
def ginko_shape(
    coordinate: t.SocketOrVal[pf.Vector] = (0.0, 0.0, 0.0),
    multiplier: t.SocketOrVal[float] = 1.98,
    scale_margin: t.SocketOrVal[float] = 6.6,
) -> pf.ProcNode[float]:
    result_0_a = pf.nodes.math.vector_length(coordinate * (0.9, 1.0, 0.0))

    result_0_value = pf.nodes.texture.gradient(
        vector=coordinate, gradient_type="RADIAL"
    )
    result_0_b_value_a = pf.nodes.math.pingpong(value=result_0_value.fac, scale=0.5)
    result_0_b_value = result_0_b_value_a * multiplier
    result_0_x_a = pf.nodes.texture.noise(
        vector=None, noise_dimensions="1D", w=result_0_value.fac
    )
    result_0_x = result_0_x_a.fac * 0.3
    result_0_b_a_vector = pf.nodes.math.combine_xyz(result_0_b_value + result_0_x)
    result_0_b_a = pf.nodes.texture.wave(
        vector=result_0_b_a_vector,
        scale=scale_margin,
        distortion=5.82,
        detail=1.52,
        detail_roughness=1.0,
    )
    result_0_b_1 = result_0_b_a.fac * 0.02
    result_0_b_0 = pf.nodes.math.float_curve(
        factor=1.0,
        value=result_0_b_value,
        curve=np.array(
            [
                [0.0, 0.0],
                [0.523, 0.1156],
                [0.5805, 0.7469],
                [0.7742, 0.7719],
                [0.9461, 0.7531],
                [1.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )
    subtract = result_0_a - (result_0_b_1 + result_0_b_0)

    return subtract


@pf.nodes.node_function
def ginko_silhouette(
    geometry: pf.ProcNode,
    angle: t.SocketOrVal[float] = -1.22,
    multiplier: t.SocketOrVal[float] = 1.9,
    scale_margin: t.SocketOrVal[float] = 6.5,
) -> pf.ProcNode:
    pos = pf.nodes.geo.input_position()
    u = (pos.x + 0.5) * _TWO_PI
    v = pos.y + 0.5

    dir_u = pf.nodes.math.combine_xyz(pf.nodes.math.cos(u), pf.nodes.math.sin(u), 0.0)
    dir_r = pf.nodes.math.vector_rotate_euler(
        vector=dir_u, rotation=pf.nodes.math.combine_xyz(0.0, 0.0, angle)
    )
    s1 = ginko_shape(coordinate=dir_r, multiplier=multiplier, scale_margin=scale_margin)
    s05 = ginko_shape(
        coordinate=dir_r * 0.5, multiplier=multiplier, scale_margin=scale_margin
    )
    g = (s1 - s05) * 2.0
    r_b = pf.nodes.math.maximum((g - s1) / g, 0.0)

    return pf.nodes.geo.set_position(geometry=geometry, position=dir_u * (v * r_b))


@pf.nodes.node_function
def valid_area(
    value: t.SocketOrVal[float] = 0.5,
) -> pf.ProcNode[float]:
    result_0_value = pf.nodes.math.sign(value)

    map_range = pf.nodes.math.map_range(
        value=result_0_value,
        from_min=-1.0,
        to_max=0.0,
        to_min=1.0,
        data_type=NodeDataType.FLOAT,
    )

    return map_range


class GinkoVeinResult(NamedTuple):
    vein: pf.ProcNode[float]
    wave: pf.ProcNode[float]


@pf.nodes.node_function
def ginko_vein(
    vector: t.SocketOrVal[pf.Vector] = (0.0, 0.0, 0.0),
    scale_vein: t.SocketOrVal[float] = 80.0,
    scale_wave: t.SocketOrVal[float] = 5.0,
) -> GinkoVeinResult:
    wave_b_vector = vector - (-0.18, 0.0, 0.0)

    wave_a_1 = pf.nodes.texture.noise(wave_b_vector)
    wave_1 = pf.nodes.texture.gradient(vector=wave_b_vector, gradient_type="RADIAL")
    wave_addend_a = pf.nodes.math.pingpong(value=wave_1.fac, scale=0.5)
    wave_b = pf.nodes.math.vector_length(wave_b_vector)
    wave_addend_b = (wave_addend_a - 0.5) * -0.44
    wave_addend = wave_addend_a + (wave_b * wave_addend_b)
    vein_value_vector_x = pf.nodes.math.multiply_add(
        a=wave_a_1.fac, b=0.005, addend=wave_addend
    )

    vein_value_vector = pf.nodes.math.combine_xyz(vein_value_vector_x)
    vein_value = pf.nodes.texture.wave(
        vector=vein_value_vector,
        scale=scale_vein,
        distortion=0.6,
        detail=3.0,
        detail_scale=5.0,
        detail_roughness=1.0,
        phase_offset=-4.62,
    )
    vein = pf.nodes.math.map_range(
        value=vein_value.color.astype(dtype=float) * wave_b,
        from_max=-0.32,
        from_min=0.15,
        to_max=-0.02,
        data_type=NodeDataType.FLOAT,
    )
    wave_a_vector_x = pf.nodes.math.multiply_add(
        a=wave_a_1.fac, b=0.03, addend=wave_addend
    )

    wave_a_vector = pf.nodes.math.combine_xyz(wave_a_vector_x)
    wave_a_0 = pf.nodes.texture.wave(
        vector=wave_a_vector,
        scale=scale_wave,
        distortion=-0.42,
        detail=10.0,
        detail_roughness=1.0,
        phase_offset=-4.62,
    )
    wave = wave_a_0.fac * wave_b
    return GinkoVeinResult(vein, wave)


class GinkoStemResult(NamedTuple):
    stem: pf.ProcNode[float]
    stem_raw: pf.ProcNode[float]


@pf.nodes.node_function
def ginko_stem(
    coordinate: t.SocketOrVal[pf.Vector] = (0.0, 0.0, 0.0),
    length: t.SocketOrVal[float] = 0.64,
    value: t.SocketOrVal[float] = 0.005,
) -> GinkoStemResult:
    stem_raw_value_1 = coordinate + (0.0, 0.03, 0.0)

    stem_raw_value_value_value = pf.nodes.math.map_range(
        value=stem_raw_value_1.y,
        from_max=0.0,
        from_min=-1.0,
        data_type=NodeDataType.FLOAT,
    )
    stem_raw_value_value = pf.nodes.math.float_curve(
        factor=1.0,
        value=stem_raw_value_value_value,
        curve=np.array(
            [[0.0, 0.5], [0.2719, 0.4827], [0.767, 0.5271], [1.0, 0.5]],
            dtype=np.float64,
        ),
    )
    stem_raw_value_0 = pf.nodes.math.map_range(
        value=stem_raw_value_value,
        to_min=-1.0,
        data_type=NodeDataType.FLOAT,
    )
    stem_raw = pf.nodes.math.absolute(stem_raw_value_0 + stem_raw_value_1.x)
    stem_a_a = pf.nodes.math.map_range(
        value=stem_raw_value_1.y,
        from_max=-0.35,
        from_min=-1.72,
        to_max=0.008,
        to_min=0.03,
        interpolation_type="SMOOTHSTEP",
        data_type=NodeDataType.FLOAT,
    )
    stem_a_b_0 = pf.nodes.math.absolute(stem_raw_value_1.y + length)
    stem_a = pf.nodes.math.smooth_maximum(
        a=stem_raw - stem_a_a,
        b=stem_a_b_0 - length,
        distance=0.02,
    )
    stem = stem_a - value
    return GinkoStemResult(stem, stem_raw)


class GinkoResult(NamedTuple):
    geometry: pf.ProcNode[bool]
    vein: pf.ProcNode[float]
    shape: pf.ProcNode[float]
    wave: pf.ProcNode[float]


@pf.nodes.node_function
def ginko(
    mesh: pf.ProcNode,
    vein_length: t.SocketOrVal[float] = 0.64,
    vein_width: t.SocketOrVal[float] = 0.005,
    angle: t.SocketOrVal[float] = -1.7617,
    displacenment: t.SocketOrVal[float] = 0.5,
    multiplier: t.SocketOrVal[float] = 1.98,
    scale_vein: t.SocketOrVal[float] = 80.0,
    scale_wave: t.SocketOrVal[float] = 5.0,
    scale_margin: t.SocketOrVal[float] = 6.6,
) -> GinkoResult:
    input_position = pf.nodes.geo.input_position()

    ginko_shape_coordinate_rotation = pf.nodes.math.combine_xyz(0, 0, angle)

    ginko_shape_coordinate = pf.nodes.math.vector_rotate_euler(
        vector=input_position,
        rotation=ginko_shape_coordinate_rotation,
    )
    ginko_shape_result = ginko_shape(
        coordinate=ginko_shape_coordinate,
        multiplier=multiplier,
        scale_margin=scale_margin,
    )
    valid_area_result = valid_area(value=ginko_shape_result)

    ginko_vein_result = ginko_vein(
        vector=ginko_shape_coordinate,
        scale_vein=scale_vein,
        scale_wave=scale_wave,
    )

    capture_1 = valid_area_result * ginko_vein_result.vein

    capture_0 = pf.nodes.math.map_range(
        value=ginko_shape_result,
        from_max=0.0,
        from_min=-1.0,
        to_max=0.0,
        to_min=-5.0,
        clamp=False,
        data_type=NodeDataType.FLOAT,
    )
    capture_attribute_attribute_value = pf.nodes.math.clamp(capture_1 * capture_0)
    capture_attribute_attribute = pf.nodes.math.clamp(
        value=capture_attribute_attribute_value, max=0.01
    )
    capture_attribute = pf.nodes.geo.capture_attribute(
        geometry=mesh,
        attribute=capture_attribute_attribute,
    )
    capture_attribute_1 = pf.nodes.geo.capture_attribute(
        geometry=capture_attribute.geometry,
        attribute=ginko_shape_result,
    )
    ginko_stem_result = ginko_stem(
        coordinate=input_position, length=vein_length, value=vein_width
    )

    valid_area_result_1 = valid_area(value=ginko_stem_result.stem)

    geometry_offset_z_0 = (
        valid_area_result_1 * ginko_stem_result.stem_raw
    ) + capture_attribute_attribute

    geometry_offset = pf.nodes.math.combine_xyz(z=geometry_offset_z_0 * displacenment)
    set_position = pf.nodes.geo.set_position(
        geometry=capture_attribute_1.geometry, offset=geometry_offset
    )

    valid_area_result_2 = valid_area(value=ginko_shape_result)

    wave = valid_area_result_2 * ginko_vein_result.wave

    return GinkoResult(
        set_position, capture_attribute.attribute, capture_attribute_1.attribute, wave
    )


@pf.nodes.node_function
def apply_wave(
    geometry: pf.ProcNode,
    wave_scale_y: t.SocketOrVal[float] = 1.0,
    wave_scale_x: t.SocketOrVal[float] = 1.0,
    x_modulated: t.SocketOrVal[float] = 0.0,
) -> pf.ProcNode[pf.Vector]:
    input_position = pf.nodes.geo.input_position()

    input_position_1 = pf.nodes.geo.input_position()
    attribute_statistic = pf.nodes.geo.attribute_statistic(
        geometry=geometry,
        attribute=input_position_1.y,
        data_type=NodeDataType.FLOAT,
    )

    set_z_value_value = pf.nodes.math.map_range(
        value=input_position.y,
        from_max=attribute_statistic.max,
        from_min=attribute_statistic.min,
        data_type=NodeDataType.FLOAT,
    )

    set_z_value = pf.nodes.math.float_curve(
        factor=1.0,
        value=set_z_value_value,
        curve=np.array([[0.0, 0.5], [0.7452, 0.5772], [1.0, 0.5]], dtype=np.float64),
    )
    set_z_0 = pf.nodes.math.map_range(
        value=set_z_value, to_min=-1.0, data_type=NodeDataType.FLOAT
    )
    set_position_1_offset = pf.nodes.math.combine_xyz(z=set_z_0 * wave_scale_y)
    set_position_1 = pf.nodes.geo.set_position(
        geometry=geometry, offset=set_position_1_offset
    )
    attribute_statistic_1 = pf.nodes.geo.attribute_statistic(
        geometry=geometry,
        attribute=x_modulated,
        data_type=NodeDataType.FLOAT,
    )

    result_0_value = pf.nodes.math.map_range(
        value=x_modulated,
        from_max=attribute_statistic_1.max,
        from_min=attribute_statistic_1.min,
        data_type=NodeDataType.FLOAT,
    )

    result_0_offset_z_value = pf.nodes.math.float_curve(
        factor=1.0,
        value=result_0_value,
        curve=np.array(
            [[0.0, 0.5], [0.4, 0.512], [0.5, 0.5], [0.6, 0.512], [1.0, 0.5]],
            dtype=np.float64,
        ),
    )
    result_0_offset_z_0 = pf.nodes.math.map_range(
        value=result_0_offset_z_value,
        to_min=-1.0,
        data_type=NodeDataType.FLOAT,
    )
    result_0_offset = pf.nodes.math.combine_xyz(z=result_0_offset_z_0 * wave_scale_x)
    set_position = pf.nodes.geo.set_position(
        geometry=set_position_1, offset=result_0_offset
    )

    return set_position


@pf.nodes.node_function
def move_to_origin(
    geometry: pf.ProcNode,
) -> pf.ProcNode[pf.Vector]:
    input_position = pf.nodes.geo.input_position()

    attribute_statistic = pf.nodes.geo.attribute_statistic(
        geometry=geometry,
        attribute=input_position.y,
        data_type=NodeDataType.FLOAT,
    )

    result_0_offset = pf.nodes.math.combine_xyz(y=0.0 - attribute_statistic.min)

    set_position = pf.nodes.geo.set_position(geometry=geometry, offset=result_0_offset)

    return set_position


class GeoLeafGinkoResult(NamedTuple):
    geometry: pf.ProcNode
    coordinate: pf.ProcNode[pf.Vector]


@pf.nodes.node_function
def geo_leaf_ginko(
    vein_length: t.SocketOrVal[float] = 0.4906043,
    angle: t.SocketOrVal[float] = -1.2235969,
    multiplier: t.SocketOrVal[float] = 1.9063802,
    scale_vein: t.SocketOrVal[float] = 73.207085,
    scale_wave: t.SocketOrVal[float] = 4.8405147,
    scale_margin: t.SocketOrVal[float] = 6.528117,
) -> GeoLeafGinkoResult:
    grid = pf.nodes.geo.mesh_grid(
        size_x=1.0, size_y=1.0, vertices_x=_ANGULAR_VERTS, vertices_y=_RING_VERTS
    )
    silhouette = ginko_silhouette(
        geometry=grid.mesh,
        angle=angle,
        multiplier=multiplier,
        scale_margin=scale_margin,
    )
    welded = pf.nodes.geo.merge_by_distance(geometry=silhouette, distance=1e-4)

    input_position = pf.nodes.geo.input_position()

    capture_coordinate = pf.nodes.geo.capture_attribute(
        geometry=welded, attribute=input_position
    )

    ginko_result = ginko(
        mesh=capture_coordinate.geometry,
        vein_length=vein_length,
        angle=angle,
        multiplier=multiplier,
        scale_vein=scale_vein,
        scale_wave=scale_wave,
        scale_margin=scale_margin,
    )

    set_position_offset_z = pf.nodes.math.map_range(
        value=ginko_result.wave,
        to_max=0.04,
        data_type=NodeDataType.FLOAT,
    )

    set_position_offset = pf.nodes.math.combine_xyz(z=set_position_offset_z)
    set_position = pf.nodes.geo.set_position(
        geometry=ginko_result.geometry,
        offset=set_position_offset,
    )
    input_position_1 = pf.nodes.geo.input_position()

    apply_wave_result = apply_wave(
        geometry=set_position, wave_scale_x=0.0, x_modulated=input_position_1.x
    )

    move_to_origin_result = move_to_origin(geometry=apply_wave_result)

    return GeoLeafGinkoResult(move_to_origin_result, capture_coordinate.attribute)


class LeafGinkoResult(NamedTuple):
    mesh: pf.MeshObject


@pf.tracer.generator
def leaf_ginko(
    size: float = 0.15,
    vein_length: float = 0.45,
    angle_deg: float = -90.0,
    multiplier: float = 1.94,
    scale_vein: float = 80.0,
    scale_wave: float = 5.0,
    scale_margin: float = 6.5,
    material: pf.Material | None = None,
) -> LeafGinkoResult:
    return _leaf_ginko_mesh(
        size=size,
        vein_length=vein_length,
        angle=math.radians(angle_deg),
        multiplier=multiplier,
        scale_vein=scale_vein,
        scale_wave=scale_wave,
        scale_margin=scale_margin,
        material=material,
    )


@pf.tracer.generator
def _leaf_ginko_mesh(
    size: float,
    vein_length: float,
    angle: float,
    multiplier: float,
    scale_vein: float,
    scale_wave: float,
    scale_margin: float,
    material: pf.Material | None,
) -> LeafGinkoResult:
    if material is None:
        material = pf.Material(surface=pf.nodes.shader.principled_bsdf())

    geo = geo_leaf_ginko(
        vein_length=vein_length,
        angle=angle,
        multiplier=multiplier,
        scale_vein=scale_vein,
        scale_wave=scale_wave,
        scale_margin=scale_margin,
    )
    obj = pf.nodes.to_mesh_object_with_attributes(
        geo.geometry, attributes={"coordinate": geo.coordinate}
    )[0]
    normalize_leaf_coord(obj)

    pf.ops.object.set_material(obj, material=material)
    pf.ops.object.set_transform(obj, scale=(size, size, size))
    pf.ops.mesh.transform_apply(obj)
    return LeafGinkoResult(mesh=obj)


def leaf_ginkgo_rand(
    rng: pf.RNG,
    material: pf.Material | None = None,
) -> LeafGinkoResult:
    rngs = rng.spawn(2)

    if material is None:
        material = leaf_material_rand(rngs[0], vector=leaf_coord())

    rng_geo = rngs[1]
    obj = _leaf_ginko_mesh(
        size=pf.random.uniform(rng_geo, 0.08, 0.2),
        vein_length=pf.random.uniform(rng_geo, 0.4, 0.5),
        angle=math.radians(pf.random.uniform(rng_geo, -110.0, -70.0)),
        multiplier=pf.random.uniform(rng_geo, 1.9, 1.98),
        scale_vein=pf.random.uniform(rng_geo, 70.0, 90.0),
        scale_wave=pf.random.uniform(rng_geo, 4.0, 6.0),
        scale_margin=pf.random.uniform(rng_geo, 5.5, 7.5),
        material=material,
    ).mesh
    pf.ops.modifier.subdivide_surface(obj, levels=2, _skip_apply=True)
    return LeafGinkoResult(mesh=obj)
