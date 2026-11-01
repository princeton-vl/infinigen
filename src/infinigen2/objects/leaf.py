# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Yiming Zuo: original Infinigen leaf (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/objects/leaves/leaf_v2.py)
# - Alexander Raistrick: refactor for Infinigen2

from typing import NamedTuple

import numpy as np
import procfunc as pf
from procfunc.nodes import types as t
from procfunc.nodes.util.bpy_node_info import NodeDataType

from infinigen2.objects.leaf_broadleaf import leaf_broadleaf_rand
from infinigen2.objects.leaf_coord import leaf_coord, normalize_leaf_coord
from infinigen2.objects.leaf_ginko import leaf_ginkgo_rand
from infinigen2.objects.leaf_maple import leaf_maple_rand
from infinigen2.shaders.base_materials.leaf import leaf_rand as leaf_material_rand

__all__ = [
    "LeafResult",
    "leaf_rand",
    "leaf_simple",
    "leaf_simple_rand",
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
            [[0.0, 0.5], [0.25, 0.4953], [0.75, 0.5117], [1.0, 0.5]], dtype=np.float64
        ),
    )
    x_modulated_a = pf.nodes.math.map_range(
        value=stem_shape, to_min=-1.0, data_type=NodeDataType.FLOAT
    )

    x_modulated = x_modulated_a - x
    midrib_value_value_3 = pf.nodes.math.map_range(
        value=y,
        from_max=midrib_length,
        from_min=-70.0,
        to_max=0.0,
        to_min=midrib_width,
        data_type=NodeDataType.FLOAT,
    )

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
        a=midrib_value_value_3 - midrib_value_b,
        b=midrib_value_value_0_b,
        distance=0.06,
    )
    midrib_value_value_0 = pf.nodes.math.clamp(
        midrib_value_value_3 / midrib_value_value_1
    )
    midrib_value = pf.nodes.math.map_range(
        value=midrib_value_value_0,
        from_max=0.03,
        from_min=0.001,
        to_max=0.0,
        to_min=1.0,
        data_type=NodeDataType.FLOAT,
    )
    return MidribResult(x_modulated, midrib_value)


@pf.nodes.node_function
def vein_coord(
    x_modulated: t.SocketOrVal[float] = 0.5,
    y: t.SocketOrVal[float] = 0.5,
    vein_asymmetry: t.SocketOrVal[float] = 0.0,
    vein_angle: t.SocketOrVal[float] = 2.0,
) -> pf.ProcNode[float]:
    result_0_a_0 = pf.nodes.math.sign(x_modulated)

    result_0_b_a = pf.nodes.math.map_range(
        value=y, from_min=-1.0, data_type=NodeDataType.FLOAT
    )
    vein_shape_value_1 = pf.nodes.math.absolute(x_modulated)

    vein_shape_value_0 = pf.nodes.math.clamp(vein_shape_value_1)
    vein_shape = pf.nodes.math.float_curve(
        factor=1.0,
        value=vein_shape_value_0,
        curve=np.array(
            [[0.0, 0.0], [0.25, 0.1261], [0.75, 0.6061], [1.0, 1.0]], dtype=np.float64
        ),
    )
    result_0_b_b_0 = pf.nodes.math.map_range(
        value=vein_shape,
        from_max=0.9,
        to_max=1.9,
        data_type=NodeDataType.FLOAT,
    )

    result_0_b = result_0_b_a * (result_0_b_b_0 * vein_angle)
    add = (result_0_a_0 * vein_asymmetry) + (result_0_b - y)

    return add


@pf.nodes.node_function
def shape(
    x_modulated: t.SocketOrVal[float] = 0.0,
    y: t.SocketOrVal[float] = 0.0,
) -> pf.ProcNode[float]:
    result_0_a_vector_1 = pf.nodes.math.combine_xyz(x=x_modulated, y=y)

    result_0_a_vector_y = pf.nodes.math.clamp(value=y, min=-0.6, max=0.6)
    result_0_a_vector_0 = pf.nodes.math.combine_xyz(y=result_0_a_vector_y)
    result_0_a = pf.nodes.math.vector_length(result_0_a_vector_1 - result_0_a_vector_0)
    leaf_shape_value = pf.nodes.math.map_range(
        value=y,
        from_max=0.6,
        from_min=-0.6,
        data_type=NodeDataType.FLOAT,
    )

    leaf_shape = pf.nodes.math.float_curve(
        factor=1.0,
        value=leaf_shape_value,
        curve=np.array(
            [[0.0, 0.0], [0.3058, 0.2704], [0.7851, 0.1213], [1.0, 0.0]],
            dtype=np.float64,
        ),
    )
    subtract = result_0_a - leaf_shape

    return subtract


@pf.nodes.node_function
def leaf_silhouette(
    geometry: pf.ProcNode,
    midrib_length: t.SocketOrVal[float] = 0.44,
    midrib_width: t.SocketOrVal[float] = 0.86,
    stem_length: t.SocketOrVal[float] = 0.82,
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
    w = pf.nodes.math.maximum(shape(x_modulated=0.0, y=y) * -1.0, 0.0)

    position = pf.nodes.math.combine_xyz(c + s * w, y, 0.0)
    return pf.nodes.geo.set_position(geometry=geometry, position=position)


@pf.nodes.node_function
def apply_vein_midrib(
    vein_coord_val: t.SocketOrVal[float] = 0.0,
    midrib_value: t.SocketOrVal[float] = 0.5,
    leaf_shape: t.SocketOrVal[float] = 1.0,
    vein_density: t.SocketOrVal[float] = 6.0,
) -> pf.ProcNode[float]:
    result_0_b_value_1 = pf.nodes.math.map_range(
        value=leaf_shape,
        from_max=0.0,
        from_min=-0.3,
        to_max=0.0,
        to_min=0.015,
        data_type=NodeDataType.FLOAT,
    )

    vein = pf.nodes.texture.voronoi(
        vector=None,
        scale=vein_density,
        randomness=0.2,
        voronoi_dimensions="1D",
        w=vein_coord_val,
    )

    result_0_b_value_0 = pf.nodes.math.map_range(
        value=vein.distance,
        from_max=0.05,
        from_min=0.001,
        to_max=0.0,
        to_min=1.0,
        data_type=NodeDataType.FLOAT,
    )

    result_0_b = pf.nodes.math.map_range(
        value=result_0_b_value_1 * result_0_b_value_0,
        from_max=0.01,
        from_min=0.001,
        to_max=0.0,
        to_min=1.0,
        data_type=NodeDataType.FLOAT,
    )
    multiply = midrib_value * result_0_b

    return multiply


class LeafGenResult(NamedTuple):
    mesh: pf.ProcNode
    attribute: pf.ProcNode[float]
    x_modulated: pf.ProcNode[float]
    vein_coord: pf.ProcNode[float]


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

    vein_coord_result = vein_coord(
        x_modulated=midrib_result.x_modulated,
        y=input_position.y,
        vein_asymmetry=vein_asymmetry,
        vein_angle=vein_angle,
    )

    shape_result = shape(x_modulated=midrib_result.x_modulated, y=input_position.y)

    apply_vein_midrib_result = apply_vein_midrib(
        vein_coord_val=vein_coord_result,
        midrib_value=midrib_result.midrib_value,
        leaf_shape=shape_result,
        vein_density=vein_density,
    )

    set_position_offset = pf.nodes.math.combine_xyz(
        z=displancement_scale * apply_vein_midrib_result
    )

    set_position = pf.nodes.geo.set_position(geometry=mesh, offset=set_position_offset)
    capture_attribute = pf.nodes.geo.capture_attribute(
        geometry=set_position, attribute=apply_vein_midrib_result
    )

    return LeafGenResult(
        capture_attribute.geometry,
        capture_attribute.attribute,
        midrib_result.x_modulated,
        vein_coord_result,
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
        curve=np.array([[0.0, 0.5], [0.6663, 0.5778], [1.0, 0.5]], dtype=np.float64),
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
            [[0.0, 0.5], [0.4, 0.5696], [0.5, 0.5], [0.6, 0.5696], [1.0, 0.5]],
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


class GeoLeafV2Result(NamedTuple):
    geometry: pf.ProcNode
    attribute: pf.ProcNode[float]
    coordinate: pf.ProcNode[pf.Vector]


@pf.nodes.node_function
def geo_leaf_v2(
    vein_asymmetry: t.SocketOrVal[float] = 0.545,
    vein_density: t.SocketOrVal[float] = 14.688,
    vein_angle: t.SocketOrVal[float] = 0.963,
    midrib_length: t.SocketOrVal[float] = 0.439,
    midrib_width: t.SocketOrVal[float] = 0.858,
    stem_length: t.SocketOrVal[float] = 0.821,
    wave_scale_x: t.SocketOrVal[float] = 0.15,
    wave_scale_y: t.SocketOrVal[float] = 1.5,
) -> GeoLeafV2Result:
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

    apply_wave_result = apply_wave(
        geometry=leaf_gen_result.mesh,
        wave_scale_y=wave_scale_y,
        wave_scale_x=wave_scale_x,
        x_modulated=leaf_gen_result.x_modulated,
    )

    move_to_origin_result = move_to_origin(geometry=apply_wave_result)

    return GeoLeafV2Result(
        move_to_origin_result, leaf_gen_result.attribute, capture_attribute.attribute
    )


class LeafResult(NamedTuple):
    mesh: pf.MeshObject


@pf.tracer.generator
def leaf_simple(
    size: float = 0.12,
    vein_asymmetry: float = 0.545,
    vein_density: float = 14.688,
    vein_angle: float = 0.963,
    midrib_length: float = 0.439,
    midrib_width: float = 0.858,
    stem_length: float = 0.821,
    wave_scale_x: float = 0.15,
    wave_scale_y: float = 1.5,
    material: pf.Material | None = None,
) -> LeafResult:
    if material is None:
        material = pf.Material(surface=pf.nodes.shader.principled_bsdf())

    geo = geo_leaf_v2(
        vein_asymmetry=vein_asymmetry,
        vein_density=vein_density,
        vein_angle=vein_angle,
        midrib_length=midrib_length,
        midrib_width=midrib_width,
        stem_length=stem_length,
        wave_scale_x=wave_scale_x,
        wave_scale_y=wave_scale_y,
    )
    obj = pf.nodes.to_mesh_object_with_attributes(
        geo.geometry, attributes={"coordinate": geo.coordinate}
    )[0]
    obj.item().name = leaf_simple.__name__
    normalize_leaf_coord(obj)

    pf.ops.object.set_material(obj, material=material)
    pf.ops.object.set_transform(obj, scale=(size, size, size))
    pf.ops.mesh.transform_apply(obj)
    return LeafResult(mesh=obj)


def leaf_simple_rand(
    rng: pf.RNG,
    material: pf.Material | None = None,
) -> LeafResult:
    rngs = rng.spawn(2)

    if material is None:
        material = leaf_material_rand(rngs[0], vector=leaf_coord())

    rng_geo = rngs[1]
    obj = leaf_simple(
        size=pf.random.uniform(rng_geo, 0.06, 0.16),
        vein_asymmetry=pf.random.uniform(rng_geo, 0.0, 1.0),
        vein_density=pf.random.uniform(rng_geo, 5.0, 20.0),
        vein_angle=pf.random.uniform(rng_geo, 0.2, 2.0),
        midrib_length=pf.random.uniform(rng_geo, 0.0, 0.8),
        midrib_width=pf.random.uniform(rng_geo, 0.5, 1.0),
        stem_length=pf.random.uniform(rng_geo, 0.7, 0.9),
        material=material,
    ).mesh
    pf.ops.modifier.subdivide_surface(obj, levels=2, _skip_apply=True)
    return LeafResult(mesh=obj)


def leaf_rand(
    rng: pf.RNG,
    material: pf.Material | None = None,
) -> LeafResult:
    rng_choice, rng_leaf = rng.spawn(2)
    leaf_fn = pf.control.choice(
        rng_choice,
        [
            (leaf_simple_rand, 1.0),
            (leaf_broadleaf_rand, 1.0),
            (leaf_ginkgo_rand, 1.0),
            (leaf_maple_rand, 1.0),
        ],
    )
    return LeafResult(mesh=leaf_fn(rng_leaf, material=material).mesh)
