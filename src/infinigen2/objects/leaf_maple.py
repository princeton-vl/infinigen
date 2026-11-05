# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Yiming Zuo: original Infinigen leaf maple (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/objects/leaves/leaf_maple.py)
# - Alexander Raistrick: refactor for Infinigen2
# Acknowledgement: This file draws inspiration https://www.youtube.com/watch?v=X9YmJ0zGWHw by Creative Shrimp

import math
from typing import NamedTuple

import numpy as np
import procfunc as pf
from procfunc.nodes import types as t
from procfunc.nodes.util.bpy_node_info import NodeDataType

from infinigen2.objects.leaf_coord import leaf_coord, normalize_leaf_coord
from infinigen2.shaders.base_materials.leaf import leaf_rand as leaf_material_rand

__all__ = [
    "LeafMapleResult",
    "leaf_maple",
    "leaf_maple_mesh",
    "leaf_maple_rand",
]

# Coarse boundary-solved topology. The old pipeline subdivided a dense plane and
# threshold-deleted cells outside the silhouette, welding poly count to grid resolution.
# maple_shape is affine in radius: shape(r, theta) = r * g(theta) - h(theta), so the lobed
# boundary is the closed form r_b(theta) = h/g, recovered in-graph from two shape evals along
# each vertex direction. We warp a coarse polar fan so its rim lands exactly on r_b; the
# pointed lobes need finer angular resolution than a rounded leaf, hence more angular verts.
_ANGULAR_VERTS = 97
_RING_VERTS = 7
_TWO_PI = 2.0 * math.pi
_HALF_PI = 0.5 * math.pi


class MapleStemResult(NamedTuple):
    stem: pf.ProcNode[float]
    stem_raw: pf.ProcNode[float]


@pf.nodes.node_function
def maple_stem(
    coordinate: t.SocketOrVal[pf.Vector] = (0.0, 0.0, 0.0),
    length: t.SocketOrVal[float] = 0.64,
    value: t.SocketOrVal[float] = 0.005,
) -> MapleStemResult:
    stem_raw_value_1 = coordinate + (0.0, 0.08, 0.0)

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
            [[0.0, 0.5], [0.2232, 0.5233], [0.7287, 0.4702], [1.0, 0.5]],
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
    return MapleStemResult(stem, stem_raw)


class MapleShapeResult(NamedTuple):
    shape: pf.ProcNode[float]
    displacement: pf.ProcNode[float]


@pf.nodes.node_function
def maple_shape(
    coordinate: t.SocketOrVal[pf.Vector] = (0.0, 0.0, 0.0),
    multiplier: t.SocketOrVal[float] = 1.96,
    noise_level: t.SocketOrVal[float] = 0.02,
) -> MapleShapeResult:
    shape_a_1 = pf.nodes.math.vector_length(coordinate * (0.9, 1.0, 0.0))

    shape_value = pf.nodes.texture.gradient(vector=coordinate, gradient_type="RADIAL")
    shape_a_value_a = pf.nodes.math.pingpong(value=shape_value.fac, scale=0.5)
    shape_a_value = shape_a_value_a * multiplier
    shape_a_0 = pf.nodes.math.float_curve(
        factor=1.0,
        value=shape_a_value,
        curve=np.array(
            [
                [0.0, 0.0],
                [0.1156, 0.075],
                [0.2109, 0.2719],
                [0.2602, 0.2344],
                [0.3633, 0.2625],
                [0.4171, 0.5545],
                [0.4336, 0.5344],
                [0.4568, 0.7094],
                [0.4749, 0.6012],
                [0.4882, 0.6636],
                [0.5352, 0.4594],
                [0.5484, 0.4375],
                [0.5648, 0.4469],
                [0.6366, 0.7331],
                [0.6719, 0.6562],
                [0.7149, 0.8225],
                [0.768, 0.6344],
                [0.7928, 0.6853],
                [0.8156, 0.5125],
                [0.8297, 0.4906],
                [0.85, 0.5125],
                [0.8988, 0.747],
                [0.9297, 0.6937],
                [0.9648, 0.8937],
                [0.9797, 0.8656],
                [0.9883, 0.8938],
                [1.0, 1.0],
            ],
            dtype=np.float64,
        ),
    )
    shape = (shape_a_1 - shape_a_0) - 0.06
    displacement_a = pf.nodes.math.float_curve(
        factor=1.0,
        value=shape_a_value,
        curve=np.array(
            [
                [0.0, 0.0],
                [0.1156, 0.075],
                [0.2109, 0.2719],
                [0.2602, 0.2344],
                [0.3633, 0.2625],
                [0.4336, 0.5344],
                [0.4568, 0.7094],
                [0.4749, 0.6012],
                [0.5352, 0.4594],
                [0.5484, 0.4375],
                [0.5648, 0.4469],
                [0.6719, 0.6562],
                [0.7149, 0.8225],
                [0.768, 0.6344],
                [0.8156, 0.5125],
                [0.8297, 0.4906],
                [0.85, 0.5125],
                [0.9297, 0.6937],
                [0.9883, 0.8938],
                [1.0, 1.0],
            ],
            dtype=np.float64,
        ),
    )

    displacement = (shape_a_1 - displacement_a) - 0.06
    return MapleShapeResult(shape, displacement)


@pf.nodes.node_function
def maple_silhouette(
    geometry: pf.ProcNode,
    angle: t.SocketOrVal[float] = -0.09,
    multiplier: t.SocketOrVal[float] = 1.92,
) -> pf.ProcNode:
    pos = pf.nodes.geo.input_position()
    u = (pos.x + 0.5) * _TWO_PI
    v = pos.y + 0.5

    dir_u = pf.nodes.math.combine_xyz(pf.nodes.math.cos(u), pf.nodes.math.sin(u), 0.0)
    dir_shape = pf.nodes.math.vector_rotate_euler(
        vector=dir_u, rotation=pf.nodes.math.combine_xyz(0.0, 0.0, angle - _HALF_PI)
    )
    s1 = maple_shape(coordinate=dir_shape, multiplier=multiplier).shape
    s05 = maple_shape(coordinate=dir_shape * 0.5, multiplier=multiplier).shape
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


@pf.nodes.node_function
def vein(
    vector: t.SocketOrVal[pf.Vector] = (0.0, 0.0, 0.0),
    angle: t.SocketOrVal[float] = 0.0,
    length: t.SocketOrVal[float] = 0.0,
    start: t.SocketOrVal[float] = 0.0,
    x_modulated: t.SocketOrVal[float] = 0.0,
    anneal: t.SocketOrVal[float] = 0.4,
    phase_offset: t.SocketOrVal[float] = 0.0,
) -> pf.ProcNode[float]:
    result_0_0_vector_x = pf.nodes.math.absolute(x_modulated)

    result_0_0_vector = pf.nodes.math.combine_xyz(
        x=result_0_0_vector_x, y=vector.y, z=vector.z
    )
    result_0_0_rotation_0 = pf.nodes.math.combine_xyz(0, 0, angle)
    result_0_0 = pf.nodes.math.vector_rotate_euler(
        vector=result_0_0_vector,
        rotation=result_0_0_rotation_0,
    )
    result_0_a_value_1 = pf.nodes.math.map_range(
        value=result_0_0_vector.x,
        from_max=0.3,
        data_type=NodeDataType.FLOAT,
    )
    result_0_a_3 = pf.nodes.math.float_curve(
        factor=1.0,
        value=result_0_a_value_1,
        curve=np.array([[0.0, 0.0], [0.5932, 0.1969], [1.0, 1.0]], dtype=np.float64),
    )
    result_0_b_a = result_0_0.x + (result_0_a_3 * 0.2)
    result_0_a_b_1 = pf.nodes.math.sign(x_modulated)
    result_0_a_2 = result_0_b_a + (result_0_a_b_1 * 0.1)
    result_0_a_value_0 = pf.nodes.texture.voronoi(
        vector=None,
        scale=8.0,
        randomness=0.7125,
        voronoi_dimensions="1D",
        w=result_0_a_2 + phase_offset,
    )
    result_0_b_1 = pf.nodes.math.vector_length(result_0_0)
    result_0_b_0 = pf.nodes.math.clamp(0.05 * result_0_b_1)
    result_0_a_from_max = pf.nodes.math.clamp(0.08 - result_0_b_0)
    result_0_a_1 = pf.nodes.math.map_range(
        value=result_0_a_value_0.distance,
        from_max=result_0_a_from_max,
        to_max=0.0,
        to_min=1.0,
        data_type=NodeDataType.FLOAT,
    )
    result_0_a_a = pf.nodes.math.absolute(x_modulated)
    result_0_a_b_0 = result_0_0_vector.y - 0.0
    result_0_a_0 = result_0_a_a < (result_0_a_b_0 * anneal)
    multiply = (result_0_a_1 * result_0_a_0) * (result_0_b_a < start)

    return multiply


class MidribResult(NamedTuple):
    result: pf.ProcNode[float]
    vector: pf.ProcNode[pf.Vector]


@pf.nodes.node_function
def midrib(
    vector: t.SocketOrVal[pf.Vector] = (0.0, 0.0, 0.0),
    angle: t.SocketOrVal[float] = 0.8238,
    vein_angle: t.SocketOrVal[float] = 0.7854,
    vein_length: t.SocketOrVal[float] = 0.2,
    vein_start: t.SocketOrVal[float] = -0.2,
    anneal: t.SocketOrVal[float] = 0.4,
    phase_offset: t.SocketOrVal[float] = 0.0,
) -> MidribResult:
    vector_rotation_0 = pf.nodes.math.combine_xyz(0, 0, angle)

    vector_1 = pf.nodes.math.vector_rotate_euler(
        vector=vector, rotation=vector_rotation_0
    )
    vein_value = pf.nodes.math.map_range(value=vector_1.y, data_type=NodeDataType.FLOAT)

    vein_x_modulated_a_a = pf.nodes.math.float_curve(
        factor=1.0,
        value=vein_value,
        curve=np.array(
            [
                [0.0, 0.5],
                [0.1432, 0.5406],
                [0.2591, 0.5062],
                [0.3705, 0.5406],
                [0.4591, 0.425],
                [0.5932, 0.4562],
                [0.7432, 0.3562],
                [0.8727, 0.5062],
                [1.0, 0.5],
            ],
            dtype=np.float64,
        ),
    )
    vein_x_modulated_b_a = pf.nodes.math.constant(0.1)
    vein_x_modulated_a = vein_x_modulated_a_a * vein_x_modulated_b_a
    vein_x_modulated = (vector_1.x + vein_x_modulated_a) - (vein_x_modulated_b_a * 0.5)
    vein_result = vein(
        vector=vector_1,
        angle=vein_angle,
        length=vein_length,
        start=vein_start,
        x_modulated=vein_x_modulated,
        anneal=anneal,
        phase_offset=phase_offset,
    )
    result_b_value_1 = pf.nodes.math.absolute(vein_x_modulated)

    b_a = pf.nodes.texture.noise(vector=vector_1, scale=10.0)

    result_b_value_0 = (b_a.fac - 0.5) * 0.01

    result_b_1 = pf.nodes.math.map_range(
        value=result_b_value_1 + result_b_value_0,
        from_max=0.01,
        to_max=0.0,
        to_min=1.0,
        data_type=NodeDataType.FLOAT,
    )
    result_b_0 = vector_1.y > 0.0
    result = pf.nodes.math.maximum(a=vein_result, b=result_b_1 * result_b_0)
    return MidribResult(result, vector_1)


class SubVeinResult(NamedTuple):
    value: pf.ProcNode[float]
    color_value: pf.ProcNode[float]


@pf.nodes.node_function
def sub_vein(
    x: t.SocketOrVal[float] = 0.5,
    y: t.SocketOrVal[float] = 0.0,
) -> SubVeinResult:
    color_value_x = pf.nodes.math.absolute(x)

    color_value_0_b = pf.nodes.math.combine_xyz(x=color_value_x, y=y)
    color_value_0_a = pf.nodes.texture.noise(color_value_0_b)
    color_value_1 = pf.nodes.color.mix_rgb(
        factor=0.9,
        a=color_value_0_a.color,
        b=color_value_0_b.astype(dtype=pf.Color),
    )
    value_a_value = pf.nodes.texture.voronoi(
        vector=color_value_1.astype(dtype=pf.Vector), scale=30.0
    )

    value_a = pf.nodes.math.map_range(
        value=value_a_value.distance,
        from_max=0.1,
        to_max=2.0,
        clamp=False,
        data_type=NodeDataType.FLOAT,
    )
    color_value_value_value = pf.nodes.texture.voronoi_distance(
        vector=color_value_1.astype(dtype=pf.Vector),
        scale=150.0,
    )

    color_value_value = pf.nodes.math.map_range(
        value=color_value_value_value,
        from_max=0.1,
        data_type=NodeDataType.FLOAT,
    )
    value = (value_a + color_value_value) * -1.0

    color_value = pf.nodes.math.map_range(
        value=color_value_value,
        to_max=-1.0,
        data_type=NodeDataType.FLOAT,
    )

    return SubVeinResult(value, color_value)


@pf.nodes.node_function
def displacement_ripple(
    shape: t.SocketOrVal[float] = 0.5,
) -> pf.ProcNode[float]:
    input_position = pf.nodes.geo.input_position()

    result_0_value_0 = pf.nodes.math.vector_length(input_position)

    map_range = pf.nodes.math.map_range(
        value=result_0_value_0 * shape,
        from_max=0.0,
        from_min=-1.0,
        to_max=0.1,
        to_min=-0.1,
        clamp=False,
        data_type=NodeDataType.FLOAT,
    )

    return map_range


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

    attribute_statistic_1 = pf.nodes.geo.attribute_statistic(
        geometry=geometry,
        attribute=input_position.z,
        data_type=NodeDataType.FLOAT,
    )
    result_0_offset = pf.nodes.math.combine_xyz(
        y=0.0 - attribute_statistic.min,
        z=0.0 - attribute_statistic_1.max,
    )

    set_position = pf.nodes.geo.set_position(geometry=geometry, offset=result_0_offset)

    return set_position


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
        curve=np.array([[0.0, 0.5], [0.5031, 0.5333], [1.0, 0.5]], dtype=np.float64),
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
            [[0.0, 0.5], [0.4, 0.5431], [0.5, 0.5], [0.6, 0.5431], [1.0, 0.5]],
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


class GeoLeafMapleResult(NamedTuple):
    geometry: pf.ProcNode
    coordinate: pf.ProcNode[pf.Vector]


@pf.nodes.node_function
def geo_leaf_maple(
    angle: t.SocketOrVal[float] = -0.09350786,
    multiplier: t.SocketOrVal[float] = 1.9239038,
) -> GeoLeafMapleResult:
    grid = pf.nodes.geo.mesh_grid(
        size_x=1.0, size_y=1.0, vertices_x=_ANGULAR_VERTS, vertices_y=_RING_VERTS
    )
    silhouette = maple_silhouette(
        geometry=pf.nodes.geo.flip_faces(grid.mesh), angle=angle, multiplier=multiplier
    )
    welded = pf.nodes.geo.merge_by_distance(geometry=silhouette, distance=1e-4)

    input_position = pf.nodes.geo.input_position()

    capture_coordinate = pf.nodes.geo.capture_attribute(
        geometry=welded, attribute=input_position
    )

    maple_stem_result = maple_stem(coordinate=input_position, length=0.32)

    geometry_x_modulated_0_rotation = pf.nodes.math.combine_xyz(0, 0, angle)

    geometry_x_modulated = pf.nodes.math.vector_rotate_euler(
        vector=input_position,
        rotation=geometry_x_modulated_0_rotation,
    )
    maple_shape_coordinate_rotation = pf.nodes.math.combine_xyz(0, 0, -1.5708)

    maple_shape_coordinate = pf.nodes.math.vector_rotate_euler(
        vector=geometry_x_modulated,
        rotation=maple_shape_coordinate_rotation,
    )
    maple_shape_result = maple_shape(
        coordinate=maple_shape_coordinate, multiplier=multiplier, noise_level=0.04
    )
    valid_area_result = valid_area(value=maple_shape_result.shape)

    midrib_result = midrib(
        vector=geometry_x_modulated,
        angle=1.693,
        vein_length=0.12,
        vein_start=-0.12,
        phase_offset=37.29053,
    )

    midrib_result_1 = midrib(
        vector=geometry_x_modulated,
        angle=-1.7279,
        vein_length=0.12,
        vein_start=-0.12,
        phase_offset=93.95482,
    )
    capture_a_a_a = pf.nodes.math.maximum(
        a=midrib_result.result, b=midrib_result_1.result
    )

    midrib_result_2 = midrib(
        vector=geometry_x_modulated,
        angle=0.8901,
        vein_start=0.0,
        phase_offset=46.145657,
    )

    midrib_result_3 = midrib(
        vector=geometry_x_modulated,
        angle=-0.9041,
        vein_start=0.0,
        phase_offset=53.275673,
    )
    capture_a_a_b = pf.nodes.math.maximum(
        a=midrib_result_2.result, b=midrib_result_3.result
    )

    capture_a_a = pf.nodes.math.maximum(a=capture_a_a_a, b=capture_a_a_b)
    midrib_result_4 = midrib(
        vector=geometry_x_modulated,
        angle=0.0,
        vein_length=1.64,
        vein_start=-0.12,
        phase_offset=49.643646,
    )

    midrib_result_5 = midrib(
        vector=geometry_x_modulated,
        angle=3.1416,
        vein_angle=0.761,
        vein_length=-10.56,
        vein_start=0.02,
        anneal=10.0,
        phase_offset=92.96866,
    )
    capture_a_b = pf.nodes.math.maximum(
        a=midrib_result_4.result, b=midrib_result_5.result
    )

    capture_a = pf.nodes.math.maximum(a=capture_a_a, b=capture_a_b)
    sub_vein_result = sub_vein(x=input_position.x, y=input_position.y)

    capture_b = pf.nodes.math.map_range(
        value=sub_vein_result.color_value,
        from_max=-0.94,
        to_max=0.0,
        to_min=1.0,
        data_type=NodeDataType.FLOAT,
    )

    capture_attribute_attribute_b = pf.nodes.math.maximum(a=capture_a, b=capture_b)
    capture_attribute_attribute = 1.0 - capture_attribute_attribute_b
    capture_attribute = pf.nodes.geo.capture_attribute(
        geometry=capture_coordinate.geometry,
        attribute=valid_area_result * capture_attribute_attribute,
    )
    set_a_a_2 = pf.nodes.math.maximum(a=capture_a, b=sub_vein_result.value * -0.03)

    set_a_1 = set_a_a_2 * 0.015
    set_a_a_1 = (set_a_1 * -1.0) * valid_area_result
    valid_area_result_1 = valid_area(value=maple_stem_result.stem)

    set_a_a_0 = valid_area_result_1 * (maple_stem_result.stem_raw - 0.01)

    set_a_0 = (set_a_a_1 + set_a_a_0) * 0.5
    displacement_ripple_result = displacement_ripple(
        shape=maple_shape_result.displacement
    )

    set_position_offset = pf.nodes.math.combine_xyz(
        z=set_a_0 + displacement_ripple_result
    )

    set_position = pf.nodes.geo.set_position(
        geometry=capture_attribute.geometry,
        offset=set_position_offset,
    )
    move_to_origin_result = move_to_origin(geometry=set_position)

    apply_wave_result = apply_wave(
        geometry=move_to_origin_result,
        wave_scale_x=0.5,
        x_modulated=geometry_x_modulated.x,
    )

    return GeoLeafMapleResult(apply_wave_result, capture_coordinate.attribute)


class LeafMapleResult(NamedTuple):
    mesh: pf.MeshObject


@pf.tracer.generator
def leaf_maple(
    size: float = 0.25,
    angle_deg: float = 0.0,
    multiplier: float = 1.96,
    material: pf.Material | None = None,
) -> LeafMapleResult:
    return leaf_maple_mesh(
        size=size,
        angle=math.radians(angle_deg),
        multiplier=multiplier,
        material=material,
    )


@pf.tracer.generator
def leaf_maple_mesh(
    size: float,
    angle: float,
    multiplier: float,
    material: pf.Material | None,
) -> LeafMapleResult:
    if material is None:
        material = pf.Material(surface=pf.nodes.shader.principled_bsdf())

    geo = geo_leaf_maple(
        angle=angle,
        multiplier=multiplier,
    )
    obj = pf.nodes.to_mesh_object_with_attributes(
        geo.geometry, attributes={"coordinate": geo.coordinate}
    )[0]
    obj.item().name = leaf_maple_mesh.__name__
    normalize_leaf_coord(obj)

    pf.ops.object.set_material(obj, material=material)
    pf.ops.object.set_transform(obj, scale=(size, size, size))
    pf.ops.mesh.transform_apply(obj)
    return LeafMapleResult(mesh=obj)


def leaf_maple_rand(
    rng: pf.RNG,
    material: pf.Material | None = None,
) -> LeafMapleResult:
    rngs = rng.spawn(2)

    if material is None:
        material = leaf_material_rand(rngs[0], vector=leaf_coord())

    rng_geo = rngs[1]
    obj = leaf_maple_mesh(
        size=pf.random.uniform(rng_geo, 0.15, 0.35),
        angle=math.radians(pf.random.uniform(rng_geo, -15.0, 15.0)),
        multiplier=pf.random.uniform(rng_geo, 1.92, 2.0),
        material=material,
    ).mesh
    pf.ops.modifier.subdivide_surface(obj, levels=2, _skip_apply=True)
    return LeafMapleResult(mesh=obj)
