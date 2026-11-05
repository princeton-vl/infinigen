# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Yiming Zuo: original Infinigen leaf broadleaf (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/objects/leaves/leaf_broadleaf.py)
# - Alexander Raistrick: refactor for Infinigen2
# Acknowledgment: This file draws inspiration from https://www.youtube.com/watch?v=pfOKB1GKJHM by Dr. Blender

from typing import NamedTuple

import numpy as np
import procfunc as pf
from procfunc.nodes import types as t
from procfunc.nodes.util.bpy_node_info import NodeDataType

from infinigen2.objects.leaf_coord import leaf_coord, normalize_leaf_coord
from infinigen2.shaders.base_materials.leaf import leaf_rand as leaf_material_rand

__all__ = [
    "LeafBroadleafResult",
    "leaf_broadleaf",
    "leaf_broadleaf_rand",
]

# Coarse boundary-solved topology. The old pipeline subdivided a dense plane and
# threshold-deleted cells outside the silhouette, welding poly count to grid resolution.
# The silhouette is bilateral: shape = |x - c(y)| - w(y), affine in x, so the boundary is
# the closed form x = c(y) +- w(y). We build a small strip parameterised by (s, y) and
# place each vertex directly at x = c(y) + s * w(y), keeping topology coarse and constant.
_WIDTH_VERTS = 6
_LENGTH_VERTS = 21
_Y_HALF = 0.6


class MidribResult(NamedTuple):
    x_modulated: pf.ProcNode[float]
    midrib_value: pf.ProcNode[float]


@pf.nodes.node_function
def midrib(
    x: t.SocketOrVal[float] = 0.5,
    y: t.SocketOrVal[float] = -0.6,
    midrib_length: t.SocketOrVal[float] = 0.4,
    midrib_width: t.SocketOrVal[float] = 1.0,
    stem_length: t.SocketOrVal[float] = 0.8,
) -> MidribResult:
    stem_shape_value = pf.nodes.math.map_range(
        value=y,
        from_max=0.6,
        from_min=-0.6,
        data_type=NodeDataType.FLOAT,
    )

    stem_shape = pf.nodes.math.float_curve(
        factor=1.0,
        value=stem_shape_value,
        curve=np.array(
            [[0.0, 0.5], [0.25, 0.4963], [0.75, 0.5037], [1.0, 0.5]], dtype=np.float64
        ),
    )
    x_modulated_a = pf.nodes.math.map_range(
        value=stem_shape, to_min=-1.0, data_type=NodeDataType.FLOAT
    )

    x_modulated = x_modulated_a - x
    midrib_value_1 = pf.nodes.texture.noise(vector=None, scale=20.0)

    midrib_value_a_a = pf.nodes.math.map_range(
        value=midrib_value_1.fac,
        to_min=-1.0,
        data_type=NodeDataType.FLOAT,
    )
    midrib_value_a_b = pf.nodes.math.map_range(
        value=y,
        from_max=midrib_length,
        from_min=-70.0,
        to_max=0.0,
        to_min=midrib_width,
        data_type=NodeDataType.FLOAT,
    )
    midrib_value_a = (midrib_value_a_a * 0.01) + midrib_value_a_b
    midrib_value_b = pf.nodes.math.absolute(x_modulated)
    midrib_value_value_2 = pf.nodes.math.absolute(y)
    midrib_value_value_0_b = pf.nodes.math.map_range(
        value=midrib_value_value_2,
        from_max=stem_length,
        to_max=0.0,
        to_min=1.0,
        data_type=NodeDataType.FLOAT,
    )
    midrib_value_value_1 = pf.nodes.math.smooth_minimum(
        a=midrib_value_a - midrib_value_b,
        b=midrib_value_value_0_b,
        distance=0.06,
    )
    midrib_value_value_0 = pf.nodes.math.clamp(1.0 / midrib_value_value_1)
    midrib_value = pf.nodes.math.map_range(
        value=midrib_value_value_0,
        from_max=0.03,
        from_min=0.001,
        to_max=0.0,
        to_min=1.0,
        data_type=NodeDataType.FLOAT,
    )
    return MidribResult(x_modulated, midrib_value)


class ShapeResult(NamedTuple):
    leaf_shape: pf.ProcNode[float]
    value: pf.ProcNode[float]


@pf.nodes.node_function
def shape(
    x_modulated: t.SocketOrVal[float] = 0.0,
    y: t.SocketOrVal[float] = 0.0,
) -> ShapeResult:
    leaf_shape_a_vector_1 = pf.nodes.math.combine_xyz(x=x_modulated, y=y)

    leaf_shape_a_vector_y = pf.nodes.math.clamp(value=y, min=-0.6, max=0.6)
    leaf_shape_a_vector_0 = pf.nodes.math.combine_xyz(y=leaf_shape_a_vector_y)
    leaf_shape_a = pf.nodes.math.vector_length(
        leaf_shape_a_vector_1 - leaf_shape_a_vector_0
    )
    value_value = pf.nodes.math.map_range(
        value=y,
        from_max=0.6,
        from_min=-0.6,
        data_type=NodeDataType.FLOAT,
    )

    leaf_shape_1 = pf.nodes.math.float_curve(
        factor=1.0,
        value=value_value,
        curve=np.array(
            [[0.0, 0.0], [0.2143, 0.2259], [0.7134, 0.2185], [1.0, 0.0]],
            dtype=np.float64,
        ),
    )

    leaf_shape = leaf_shape_a - leaf_shape_1
    return ShapeResult(leaf_shape, leaf_shape_1)


@pf.nodes.node_function
def leaf_silhouette(
    geometry: pf.ProcNode,
    midrib_length: t.SocketOrVal[float] = 0.051,
    midrib_width: t.SocketOrVal[float] = 0.051,
    stem_length: t.SocketOrVal[float] = 0.713,
) -> pf.ProcNode:
    pos = pf.nodes.geo.input_position()
    s = pos.x * 2.0
    y = pos.y * (2.0 * _Y_HALF)

    c = midrib(
        x=0.0,
        y=y,
        midrib_length=midrib_length,
        midrib_width=midrib_width,
        stem_length=stem_length,
    ).x_modulated
    w = pf.nodes.math.maximum(shape(x_modulated=0.0, y=y).leaf_shape * -1.0, 0.0)

    position = pf.nodes.math.combine_xyz(c + s * w, y, 0.0)
    return pf.nodes.geo.set_position(geometry=geometry, position=position)


@pf.nodes.node_function
def vein_coord(
    x_modulated: t.SocketOrVal[float] = 0.5,
    y: t.SocketOrVal[float] = 0.5,
    vein_asymmetry: t.SocketOrVal[float] = 0.0,
    vein_angle: t.SocketOrVal[float] = 2.0,
    leaf_shape: t.SocketOrVal[float] = 0.0,
) -> pf.ProcNode[float]:
    result_0_a_0 = pf.nodes.math.sign(x_modulated)

    result_0_b_a = pf.nodes.math.map_range(
        value=y, from_min=-1.0, data_type=NodeDataType.FLOAT
    )
    vein_numerator_1 = pf.nodes.math.absolute(x_modulated)

    vein_numerator_0 = pf.nodes.math.clamp(vein_numerator_1)
    vein_shape_value = pf.nodes.math.clamp(vein_numerator_0 / leaf_shape)
    vein_shape = pf.nodes.math.float_curve(
        factor=1.0,
        value=vein_shape_value,
        curve=np.array(
            [
                [0.0, 0.0],
                [0.0182, 0.05],
                [0.3364, 0.2386],
                [0.8091, 0.7312],
                [1.0, 0.9937],
            ],
            dtype=np.float64,
        ),
    )
    result_0_b_b_0 = pf.nodes.math.map_range(
        value=vein_shape, to_max=1.9, data_type=NodeDataType.FLOAT
    )

    result_0_b = result_0_b_a * (result_0_b_b_0 * vein_angle)
    add = (vein_asymmetry * result_0_a_0) + (result_0_b - y)

    return add


@pf.nodes.node_function
def vein_coord_1(
    x_modulated: t.SocketOrVal[float] = 0.5,
    y: t.SocketOrVal[float] = 0.5,
    vein_asymmetry: t.SocketOrVal[float] = 0.0,
    vein_angle: t.SocketOrVal[float] = 2.0,
    leaf_shape: t.SocketOrVal[float] = 0.0,
) -> pf.ProcNode[float]:
    result_0_a_0 = pf.nodes.math.sign(x_modulated)

    result_0_b_a = pf.nodes.math.map_range(
        value=y, from_min=-1.0, data_type=NodeDataType.FLOAT
    )
    vein_numerator_1 = pf.nodes.math.absolute(x_modulated)

    vein_numerator_0 = pf.nodes.math.clamp(vein_numerator_1)
    vein_shape_value = pf.nodes.math.clamp(vein_numerator_0 / leaf_shape)
    vein_shape = pf.nodes.math.float_curve(
        factor=1.0,
        value=vein_shape_value,
        curve=np.array(
            [
                [0.0, 0.0],
                [0.0182, 0.05],
                [0.3364, 0.2386],
                [0.6045, 0.4812],
                [0.7, 0.725],
                [0.8273, 0.8437],
                [1.0, 1.0],
            ],
            dtype=np.float64,
        ),
    )
    result_0_b_b_0 = pf.nodes.math.map_range(
        value=vein_shape, to_max=1.9, data_type=NodeDataType.FLOAT
    )

    result_0_b = result_0_b_a * (result_0_b_b_0 * vein_angle)
    add = (vein_asymmetry * result_0_a_0) + (result_0_b - y)

    return add


@pf.nodes.node_function
def vein_coord_2(
    x_modulated: t.SocketOrVal[float] = 0.5,
    y: t.SocketOrVal[float] = 0.5,
    vein_asymmetry: t.SocketOrVal[float] = 0.0,
    vein_angle: t.SocketOrVal[float] = 2.0,
    leaf_shape: t.SocketOrVal[float] = 0.0,
) -> pf.ProcNode[float]:
    result_0_a_0 = pf.nodes.math.sign(x_modulated)

    result_0_b_a = pf.nodes.math.map_range(
        value=y, from_min=-1.0, data_type=NodeDataType.FLOAT
    )
    vein_numerator_1 = pf.nodes.math.absolute(x_modulated)

    vein_numerator_0 = pf.nodes.math.clamp(vein_numerator_1)
    vein_shape_value = pf.nodes.math.clamp(vein_numerator_0 / leaf_shape)
    vein_shape = pf.nodes.math.float_curve(
        factor=1.0,
        value=vein_shape_value,
        curve=np.array(
            [
                [0.0, 0.0],
                [0.0182, 0.05],
                [0.2909, 0.2199],
                [0.4182, 0.3063],
                [0.7045, 0.3],
                [1.0, 0.8562],
            ],
            dtype=np.float64,
        ),
    )
    result_0_b_b_0 = pf.nodes.math.map_range(
        value=vein_shape, to_max=1.9, data_type=NodeDataType.FLOAT
    )

    result_0_b = result_0_b_a * (result_0_b_b_0 * vein_angle)
    add = (vein_asymmetry * result_0_a_0) + (result_0_b - y)

    return add


@pf.nodes.node_function
def random_mask_vein(
    coord: t.SocketOrVal[float] = 0.0,
    shape_val: t.SocketOrVal[float] = 0.5,
    density: t.SocketOrVal[float] = 0.5,
    random_scale_seed: t.SocketOrVal[float] = 0.5,
) -> pf.ProcNode[float]:
    vein = pf.nodes.texture.voronoi(
        vector=None,
        scale=density,
        randomness=0.2,
        voronoi_dimensions="1D",
        w=coord,
    )

    vein_1 = pf.nodes.texture.voronoi(
        vector=None,
        scale=density * random_scale_seed,
        voronoi_dimensions="1D",
        w=coord,
    )
    result_0_value_value = pf.nodes.math.round(vein_1.distance + 0.35)

    result_0_value_0 = pf.nodes.math.map_range(
        value=vein.distance + result_0_value_value,
        from_max=0.02,
        to_max=0.0,
        to_min=0.95,
        data_type=NodeDataType.FLOAT,
    )
    map_range = pf.nodes.math.map_range(
        value=shape_val * result_0_value_0,
        from_max=0.005,
        from_min=0.001,
        to_max=0.0,
        to_min=1.0,
        data_type=NodeDataType.FLOAT,
    )

    return map_range


@pf.nodes.node_function
def apply_vein_midrib(
    midrib_value: t.SocketOrVal[float] = 0.5,
    leaf_shape: t.SocketOrVal[float] = 1.0,
    vein_density: t.SocketOrVal[float] = 6.0,
    vein_coord_main: t.SocketOrVal[float] = 0.0,
    vein_coord_1_val: t.SocketOrVal[float] = 0.0,
    vein_coord_2_val: t.SocketOrVal[float] = 0.0,
) -> pf.ProcNode[float]:
    random_mask_vein_shape = pf.nodes.math.map_range(
        value=leaf_shape,
        from_max=0.05,
        from_min=-0.3,
        to_max=0.0,
        to_min=0.015,
        data_type=NodeDataType.FLOAT,
    )

    random_mask_vein_result = random_mask_vein(
        coord=vein_coord_2_val,
        shape_val=random_mask_vein_shape,
        density=vein_density,
        random_scale_seed=26.670334,
    )
    random_mask_vein_result_1 = random_mask_vein(
        coord=vein_coord_1_val,
        shape_val=random_mask_vein_shape,
        density=vein_density,
        random_scale_seed=9.877902,
    )
    vein = pf.nodes.texture.voronoi(
        vector=None,
        scale=vein_density,
        randomness=0.2,
        voronoi_dimensions="1D",
        w=vein_coord_main,
    )

    input_position = pf.nodes.geo.input_position()

    result_0_value = pf.nodes.texture.noise(vector=input_position, scale=20.0)

    result_0_b_value_a = pf.nodes.math.map_range(
        value=result_0_value.fac,
        to_min=-1.0,
        data_type=NodeDataType.FLOAT,
    )
    result_0_b_value = result_0_b_value_a * 0.02
    result_0_b_1 = pf.nodes.math.map_range(
        value=vein.distance + result_0_b_value,
        from_max=0.03,
        to_max=0.0,
        to_min=1.0,
        data_type=NodeDataType.FLOAT,
    )
    result_0_b_b = pf.nodes.math.map_range(
        value=random_mask_vein_shape * result_0_b_1,
        from_max=0.01,
        from_min=0.001,
        to_max=0.0,
        to_min=1.0,
        data_type=NodeDataType.FLOAT,
    )
    result_0_b_0 = random_mask_vein_result_1 * result_0_b_b
    multiply = midrib_value * (random_mask_vein_result * result_0_b_0)

    return multiply


@pf.nodes.node_function
def vein_coord_3(
    x_modulated: t.SocketOrVal[float] = 0.5,
    y: t.SocketOrVal[float] = 0.5,
    vein_asymmetry: t.SocketOrVal[float] = 0.0,
    vein_angle: t.SocketOrVal[float] = 2.0,
    leaf_shape: t.SocketOrVal[float] = 0.0,
) -> pf.ProcNode[float]:
    result_0_a_0 = pf.nodes.math.sign(x_modulated)

    result_0_b_a = pf.nodes.math.map_range(
        value=y, from_min=-1.0, data_type=NodeDataType.FLOAT
    )
    vein_shape = pf.nodes.math.float_curve(
        factor=1.0,
        value=x_modulated,
        curve=np.array(
            [[0.0, 0.0], [0.0182, 0.05], [0.3364, 0.2386], [0.7227, 0.75], [1.0, 1.0]],
            dtype=np.float64,
        ),
    )

    result_0_b_b_0 = pf.nodes.math.map_range(
        value=vein_shape, to_max=1.9, data_type=NodeDataType.FLOAT
    )

    result_0_b = result_0_b_a * (result_0_b_b_0 * vein_angle)
    add = (vein_asymmetry * result_0_a_0) + (result_0_b - y)

    return add


class LeafGenResult(NamedTuple):
    mesh: pf.ProcNode[pf.Vector]
    attribute: pf.ProcNode[float]
    x_modulated: pf.ProcNode[float]
    vein_coord: pf.ProcNode[float]
    vein_value: pf.ProcNode[float]


@pf.nodes.node_function
def leaf_gen(
    mesh: pf.ProcNode,
    displancement_scale: t.SocketOrVal[float] = 0.5,
    vein_asymmetry: t.SocketOrVal[float] = 0.0,
    vein_density: t.SocketOrVal[float] = 6.0,
    vein_angle: t.SocketOrVal[float] = 1.0,
    sub_vein_displacement: t.SocketOrVal[float] = 0.5,
    sub_vein_scale: t.SocketOrVal[float] = 50.0,
    wave_displacement: t.SocketOrVal[float] = 0.1,
    midrib_length: t.SocketOrVal[float] = 0.4,
    midrib_width: t.SocketOrVal[float] = 1.0,
    stem_length: t.SocketOrVal[float] = 0.8,
) -> LeafGenResult:
    input_position = pf.nodes.geo.input_position()

    midrib_result = midrib(
        x=input_position.x,
        y=input_position.y,
        midrib_length=midrib_length,
        midrib_width=midrib_width,
        stem_length=stem_length,
    )

    shape_result = shape(x_modulated=midrib_result.x_modulated, y=input_position.y)

    vein_coord_001_result_1 = vein_coord(
        x_modulated=midrib_result.x_modulated,
        y=input_position.y,
        vein_asymmetry=vein_asymmetry,
        vein_angle=vein_angle,
        leaf_shape=shape_result.value,
    )

    vein_coord_result = vein_coord_1(
        x_modulated=midrib_result.x_modulated,
        y=input_position.y,
        vein_asymmetry=vein_asymmetry,
        vein_angle=vein_angle,
        leaf_shape=shape_result.value,
    )
    vein_coord_002_result = vein_coord_2(
        x_modulated=midrib_result.x_modulated,
        y=input_position.y,
        vein_asymmetry=vein_asymmetry,
        vein_angle=vein_angle,
        leaf_shape=shape_result.value,
    )
    apply_vein_midrib_result = apply_vein_midrib(
        midrib_value=midrib_result.midrib_value,
        leaf_shape=shape_result.leaf_shape,
        vein_density=vein_density,
        vein_coord_main=vein_coord_001_result_1,
        vein_coord_1_val=vein_coord_result,
        vein_coord_2_val=vein_coord_002_result,
    )

    set_position_1_offset = pf.nodes.math.combine_xyz(
        z=displancement_scale * apply_vein_midrib_result
    )

    set_position_1 = pf.nodes.geo.set_position(
        geometry=mesh, offset=set_position_1_offset
    )
    capture_attribute = pf.nodes.geo.capture_attribute(
        geometry=set_position_1, attribute=apply_vein_midrib_result
    )

    input_position_1 = pf.nodes.geo.input_position()

    mesh_value_1 = pf.nodes.math.map_range(
        value=input_position_1.y,
        from_max=0.6,
        from_min=-0.6,
        data_type=NodeDataType.FLOAT,
    )

    mesh_offset_z_a = pf.nodes.math.float_curve(
        factor=1.0,
        value=mesh_value_1,
        curve=np.array([[0.0, 0.0], [0.5182, 1.0], [1.0, 1.0]], dtype=np.float64),
    )
    mesh_value_0 = pf.nodes.math.map_range(
        value=shape_result.leaf_shape,
        from_max=-1.0,
        data_type=NodeDataType.FLOAT,
    )
    mesh_offset_z_b = pf.nodes.math.float_curve(
        factor=1.0,
        value=mesh_value_0,
        curve=np.array(
            [[0.0045, 0.0063], [0.0409, 0.0375], [0.4182, 0.05], [1.0, 0.0]],
            dtype=np.float64,
        ),
    )
    mesh_offset_z = mesh_offset_z_a * mesh_offset_z_b
    mesh_offset = pf.nodes.math.combine_xyz(z=mesh_offset_z * 0.7)
    set_position = pf.nodes.geo.set_position(
        geometry=capture_attribute.geometry, offset=mesh_offset
    )

    vein_coord_001_result = vein_coord_3(
        x_modulated=midrib_result.x_modulated,
        y=input_position.y,
        vein_asymmetry=vein_asymmetry,
        vein_angle=vein_angle,
    )

    return LeafGenResult(
        set_position,
        capture_attribute.attribute,
        midrib_result.x_modulated,
        vein_coord_001_result,
        apply_vein_midrib_result,
    )


class SubVeinResult(NamedTuple):
    value: pf.ProcNode[float]
    color_value: pf.ProcNode[float]


@pf.nodes.node_function
def sub_vein(
    x: t.SocketOrVal[float] = 0.5,
    y: t.SocketOrVal[float] = 0.0,
) -> SubVeinResult:
    color_value_x = pf.nodes.math.absolute(x)

    color_value_value_value_vector = pf.nodes.math.combine_xyz(x=color_value_x, y=y)
    value_a_value = pf.nodes.texture.voronoi(
        vector=color_value_value_value_vector, scale=30.0
    )

    value_a = pf.nodes.math.map_range(
        value=value_a_value.distance,
        from_max=0.1,
        to_max=2.0,
        clamp=False,
        data_type=NodeDataType.FLOAT,
    )
    color_value_value_value = pf.nodes.texture.voronoi_distance(
        vector=color_value_value_value_vector,
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
        curve=np.array([[0.0, 0.5], [0.3781, 0.5309], [1.0, 0.5]], dtype=np.float64),
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
            [[0.0, 0.5], [0.4, 0.522], [0.5, 0.5], [0.6, 0.522], [1.0, 0.5]],
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


class GeoLeafBroadleafResult(NamedTuple):
    geometry: pf.ProcNode
    offset: pf.ProcNode[float]
    coordinate: pf.ProcNode[pf.Vector]
    subvein_offset: pf.ProcNode[float]
    vein_value: pf.ProcNode[float]


@pf.nodes.node_function
def geo_leaf_broadleaf(
    vein_asymmetry: t.SocketOrVal[float] = 0.071194224,
    vein_density: t.SocketOrVal[float] = 6.898336,
    vein_angle: t.SocketOrVal[float] = 0.91024345,
    midrib_length: t.SocketOrVal[float] = 0.05139456,
    midrib_width: t.SocketOrVal[float] = 0.05139456,
    stem_length: t.SocketOrVal[float] = 0.7134866,
) -> GeoLeafBroadleafResult:
    grid = pf.nodes.geo.mesh_grid(
        size_x=1.0, size_y=1.0, vertices_x=_WIDTH_VERTS, vertices_y=_LENGTH_VERTS
    )
    silhouette = leaf_silhouette(
        geometry=grid.mesh,
        midrib_length=midrib_length,
        midrib_width=midrib_width,
        stem_length=stem_length,
    )
    welded = pf.nodes.geo.merge_by_distance(geometry=silhouette, distance=1e-4)

    input_position = pf.nodes.geo.input_position()

    capture_attribute = pf.nodes.geo.capture_attribute(
        geometry=welded, attribute=input_position
    )

    leaf_gen_result = leaf_gen(
        mesh=capture_attribute.geometry,
        displancement_scale=0.005,
        vein_asymmetry=vein_asymmetry,
        vein_density=vein_density,
        vein_angle=vein_angle,
        midrib_length=midrib_length,
        midrib_width=midrib_width,
        stem_length=stem_length,
    )

    sub_vein_result = sub_vein(
        x=leaf_gen_result.x_modulated, y=leaf_gen_result.vein_coord
    )

    set_position_offset = pf.nodes.math.combine_xyz(z=sub_vein_result.value * 0.0002)

    set_position = pf.nodes.geo.set_position(
        geometry=leaf_gen_result.mesh, offset=set_position_offset
    )
    capture_attribute_1 = pf.nodes.geo.capture_attribute(
        geometry=set_position,
        attribute=sub_vein_result.color_value,
    )

    capture_attribute_2 = pf.nodes.geo.capture_attribute(
        geometry=capture_attribute_1.geometry,
        attribute=leaf_gen_result.vein_value,
    )
    apply_wave_result = apply_wave(
        geometry=capture_attribute_2.geometry,
        wave_scale_x=0.2,
        x_modulated=leaf_gen_result.x_modulated,
    )

    move_to_origin_result = move_to_origin(geometry=apply_wave_result)

    return GeoLeafBroadleafResult(
        move_to_origin_result,
        leaf_gen_result.attribute,
        capture_attribute.attribute,
        capture_attribute_1.attribute,
        capture_attribute_2.attribute,
    )


class LeafBroadleafResult(NamedTuple):
    mesh: pf.MeshObject


@pf.tracer.generator
def leaf_broadleaf(
    size: float = 0.15,
    vein_asymmetry: float = 0.0712,
    vein_density: float = 6.9,
    vein_angle: float = 0.91,
    midrib_length: float = 0.051,
    midrib_width: float = 0.051,
    stem_length: float = 0.713,
    material: pf.Material | None = None,
) -> LeafBroadleafResult:
    if material is None:
        material = pf.Material(surface=pf.nodes.shader.principled_bsdf())

    geo = geo_leaf_broadleaf(
        vein_asymmetry=vein_asymmetry,
        vein_density=vein_density,
        vein_angle=vein_angle,
        midrib_length=midrib_length,
        midrib_width=midrib_width,
        stem_length=stem_length,
    )
    obj = pf.nodes.to_mesh_object_with_attributes(
        geo.geometry, attributes={"coordinate": geo.coordinate}
    )[0]
    obj.item().name = leaf_broadleaf.__name__
    normalize_leaf_coord(obj)

    pf.ops.object.set_material(obj, material=material)
    pf.ops.object.set_transform(obj, scale=(size, size, size))
    pf.ops.mesh.transform_apply(obj)
    return LeafBroadleafResult(mesh=obj)


def leaf_broadleaf_rand(
    rng: pf.RNG,
    material: pf.Material | None = None,
) -> LeafBroadleafResult:
    rngs = rng.spawn(2)

    if material is None:
        material = leaf_material_rand(rngs[0], vector=leaf_coord())

    rng_geo = rngs[1]
    obj = leaf_broadleaf(
        size=pf.random.uniform(rng_geo, 0.06, 0.16),
        vein_asymmetry=pf.random.uniform(rng_geo, 0.0, 1.0),
        vein_density=pf.random.uniform(rng_geo, 3.0, 8.0),
        vein_angle=pf.random.uniform(rng_geo, 0.4, 1.0),
        midrib_length=pf.random.uniform(rng_geo, 0.0, 0.8),
        midrib_width=pf.random.uniform(rng_geo, 0.5, 1.0),
        stem_length=pf.random.uniform(rng_geo, 0.7, 0.9),
        material=material,
    ).mesh
    pf.ops.modifier.subdivide_surface(obj, levels=2, _skip_apply=True)
    return LeafBroadleafResult(mesh=obj)
