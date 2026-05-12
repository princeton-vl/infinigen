# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

from typing import NamedTuple

import numpy as np
import procfunc as pf
from procfunc.nodes import types as t


class FlowerMaterials(NamedTuple):
    petal: pf.Material
    center: pf.Material


class FlowerResult(NamedTuple):
    mesh: pf.MeshObject


def flower_color_distribution(rng: pf.RNG) -> pf.Color:
    h = pf.random.normal(rng, 0.95, 1.2) % 1.0
    s = pf.random.uniform(rng, 0.2, 0.85)
    v = pf.random.uniform(rng, 0.2, 0.75)
    return pf.color.hsv_color(hue=h, saturation=s, value=v)


def petal_material(
    color: t.SocketOrVal[pf.Color],
    specular: t.SocketOrVal[float] = 0.6,
    roughness: t.SocketOrVal[float] = 0.4,
    translucent_amt: t.SocketOrVal[float] = 0.3,
) -> pf.Material:
    translucent = pf.nodes.shader.translucent_bsdf(color=color)

    principled = pf.nodes.shader.principled_bsdf(
        base_color=color,
        specular_ior_level=specular,
        roughness=roughness,
    )

    surface = pf.nodes.shader.mix_shader(
        factor=translucent_amt, a=principled, b=translucent
    )

    return pf.Material(surface=surface)


def petal_material_distribution(rng: pf.RNG) -> pf.Material:
    rngs = rng.spawn(4)

    petal_color = flower_color_distribution(rngs[0])

    specular = pf.random.clip_gaussian(rngs[1], 0.6, 0.1, 0.3, 0.9)
    roughness = pf.random.clip_gaussian(rngs[2], 0.4, 0.05, 0.2, 0.6)
    translucent_amt = pf.random.clip_gaussian(rngs[3], 0.3, 0.05, 0.1, 0.5)

    return petal_material(
        color=petal_color,
        specular=specular,
        roughness=roughness,
        translucent_amt=translucent_amt,
    )


def center_material() -> pf.Material:
    ao = pf.nodes.shader.ambient_occlusion()

    color = pf.nodes.shader.color_ramp(
        fac=ao.color,
        points=[
            (0.4841, (0.0127, 0.0075, 0.0026, 1.0)),
            (0.8591, (0.0848, 0.0066, 0.0007, 1.0)),
            (1.0, (1.0, 0.6228, 0.1069, 1.0)),
        ],
    )

    surface = pf.nodes.shader.principled_bsdf(base_color=color.color)

    return pf.Material(surface=surface)


def flower_material_distribution(rng: pf.RNG) -> FlowerMaterials:
    rngs = rng.spawn(2)

    petal_mat = petal_material_distribution(rngs[0])
    center_mat = center_material()

    return FlowerMaterials(petal=petal_mat, center=center_mat)


@pf.nodes.node_function
def _follow_curve(
    geometry: t.SocketOrVal[pf.MeshObject],
    curve: t.SocketOrVal[pf.CurveObject],
    curve_min: t.SocketOrVal[float] = 0.0,
    curve_max: t.SocketOrVal[float] = 1.0,
) -> pf.ProcNode[pf.MeshObject]:
    input_position = pf.nodes.geo.input_position()

    capture_attribute = pf.nodes.geo.capture_attribute(
        geometry=geometry, attribute=input_position
    )

    attribute_statistic = pf.nodes.geo.attribute_statistic(
        geometry=capture_attribute.geometry,
        attribute=capture_attribute.attribute.z,
    )

    result_0_position_length = pf.nodes.func.map_range(
        value=capture_attribute.attribute.z,
        from_max=attribute_statistic.max,
        from_min=attribute_statistic.min,
        to_max=curve_max,
        to_min=curve_min,
    )

    curve_length = pf.nodes.geo.curve_length(curve)

    sample_curve_length = pf.nodes.geo.sample_curve_length(
        curves=curve,
        length=result_0_position_length * curve_length,
        value=0.0,
    )

    result_0_offset_vector = pf.nodes.math.vector_cross_product(
        a=sample_curve_length.tangent,
        b=sample_curve_length.normal,
    )
    result_0_offset_1 = pf.nodes.math.vector_scale(
        vector=result_0_offset_vector,
        scale=capture_attribute.attribute.x,
    )
    result_0_offset_0 = pf.nodes.math.vector_scale(
        vector=sample_curve_length.normal,
        scale=capture_attribute.attribute.y,
    )

    set_position = pf.nodes.geo.set_position(
        geometry=capture_attribute.geometry,
        position=sample_curve_length.position,
        offset=result_0_offset_1 + result_0_offset_0,
    )
    return set_position


@pf.nodes.node_function
def _polar_to_cart(
    addend: t.SocketOrVal[pf.Vector],
    value: t.SocketOrVal[float],
    vector: t.SocketOrVal[pf.Vector],
) -> pf.ProcNode[pf.Vector]:
    result_0_b_y = pf.nodes.math.cos(value)
    result_0_b_z = pf.nodes.math.sin(value)
    result_0_b = pf.nodes.func.combine_xyz(y=result_0_b_y, z=result_0_b_z)

    vector_multiply_add = pf.nodes.math.vector_multiply_add(
        a=vector, b=result_0_b, addend=addend
    )
    return vector_multiply_add


@pf.nodes.node_function
def _flower_petal(
    length: t.SocketOrVal[float],
    point: t.SocketOrVal[float],
    point_height: t.SocketOrVal[float],
    bevel: t.SocketOrVal[float],
    base_width: t.SocketOrVal[float],
    upper_width: t.SocketOrVal[float],
    resolution_h: t.SocketOrVal[int],
    resolution_v: t.SocketOrVal[int],
    wrinkle: t.SocketOrVal[float],
    curl: t.SocketOrVal[float],
) -> pf.ProcNode:
    grid_vertices_y = pf.nodes.math.multiply_add(
        a=resolution_h.astype(dtype=float), b=2.0, addend=1.0
    )
    grid = pf.nodes.geo.mesh_grid(
        vertices_x=resolution_v,
        vertices_y=grid_vertices_y.astype(dtype=int),
    )

    input_position = pf.nodes.geo.input_position()

    capture_attribute = pf.nodes.geo.capture_attribute(
        geometry=grid.mesh, attribute=input_position
    )

    noise_vector = pf.nodes.func.combine_xyz(
        x=capture_attribute.attribute.x * 0.05,
        y=capture_attribute.attribute.y,
    )
    noise = pf.nodes.shader.noise(
        vector=noise_vector,
        scale=7.9,
        detail=0.0,
        distortion=0.2,
        noise_dimensions="2D",
    )

    set_a_1 = noise.fac + -0.5
    set_a_0 = capture_attribute.attribute.x + 0.5
    set_base_a = pf.nodes.math.absolute(capture_attribute.attribute.y)
    set_base = set_base_a * 2.0
    set_b_b_0 = pf.nodes.math.multiply_add(a=set_base**bevel, b=-1.0, addend=1.0)
    set_b_1 = pf.nodes.math.multiply_add(
        a=set_a_0 * set_b_b_0, b=upper_width, addend=base_width
    )
    set_b_a_a = pf.nodes.math.multiply_add(a=set_base_a**point, b=-1.0, addend=1.0)
    set_b_a_1 = set_b_a_a * point_height
    set_b_a_0 = pf.nodes.math.multiply_add(a=point_height, b=-1.0, addend=1.0)
    set_b_0 = (set_b_a_1 + set_b_a_0) * set_b_b_0
    set_position_position = pf.nodes.func.combine_xyz(
        x=set_a_1 * wrinkle,
        y=capture_attribute.attribute.y * set_b_1,
        z=set_a_0 * set_b_0,
    )
    set_position = pf.nodes.geo.set_position(
        geometry=capture_attribute.geometry,
        position=set_position_position,
    )

    curve_bezier_middle_y = length * 0.5
    curve_bezier_middle = pf.nodes.func.combine_xyz(y=curve_bezier_middle_y)

    polar_to_cart_result = _polar_to_cart(
        addend=curve_bezier_middle,
        value=curl,
        vector=curve_bezier_middle_y.astype(dtype=pf.Vector),
    )

    curve_bezier = pf.nodes.geo.curve_bezier(
        resolution=8,
        start=(0.0, 0.0, 0.0),
        middle=curve_bezier_middle,
        end=polar_to_cart_result,
    )

    follow_curve_result = _follow_curve(
        geometry=set_position, curve=curve_bezier, curve_min=0.0, curve_max=1.0
    )

    result_0_selection = pf.nodes.func.constant(1.0)

    store_named_attribute = pf.nodes.geo.store_named_attribute(
        geometry=follow_curve_result,
        name="TAG_petal",
        selection=result_0_selection.astype(dtype=bool),
        value=True,
        domain="FACE",
    )
    return store_named_attribute


class _PhylloPointsResult(NamedTuple):
    points: pf.ProcNode
    rotation: pf.ProcNode[pf.Vector]


@pf.nodes.node_function
def _phyllo_points(
    count: t.SocketOrVal[int],
    min_radius: t.SocketOrVal[float],
    max_radius: t.SocketOrVal[float],
    radius_exp: t.SocketOrVal[float],
    min_angle: t.SocketOrVal[float],
    max_angle: t.SocketOrVal[float],
    min_z: t.SocketOrVal[float],
    max_z: t.SocketOrVal[float],
    clamp_z: t.SocketOrVal[float],
    yaw_offset: t.SocketOrVal[float],
) -> _PhylloPointsResult:
    line = pf.nodes.geo.mesh_line(count=count)

    to_points = pf.nodes.geo.mesh_to_points(line)

    input_position = pf.nodes.geo.input_position()

    capture_attribute = pf.nodes.geo.capture_attribute(
        geometry=to_points, attribute=input_position
    )

    input_index = pf.nodes.geo.input_index()

    points_x = pf.nodes.math.cos(input_index.astype(dtype=float))
    points_y = pf.nodes.math.sin(input_index.astype(dtype=float))
    points_position_x_a = pf.nodes.func.combine_xyz(x=points_x, y=points_y)

    rotation_x_value = input_index.astype(dtype=float) / count.astype(dtype=float)

    points_0 = pf.nodes.func.map_range(
        value=rotation_x_value**radius_exp,
        to_max=max_radius,
        to_min=min_radius,
    )
    points_position_x = points_position_x_a * points_0.astype(dtype=pf.Vector)
    points_position_z = pf.nodes.func.map_range(
        value=rotation_x_value,
        from_max=clamp_z,
        to_max=max_z,
        to_min=min_z,
    )
    points_position = pf.nodes.func.combine_xyz(
        x=points_position_x.x,
        y=points_position_x.y,
        z=points_position_z,
    )

    set_position = pf.nodes.geo.set_position(
        geometry=capture_attribute.geometry,
        position=points_position,
    )

    rotation_x = pf.nodes.func.map_range(
        value=rotation_x_value, to_max=max_angle, to_min=min_angle
    )
    rotation_y = pf.nodes.func.random_value(min=-0.1, max=0.1)
    rotation = pf.nodes.func.combine_xyz(
        x=rotation_x,
        y=rotation_y,
        z=input_index.astype(dtype=float) + yaw_offset,
    )
    return _PhylloPointsResult(
        points=set_position,
        rotation=rotation,
    )


@pf.nodes.node_function
def _norm_index(
    count: t.SocketOrVal[int],
) -> pf.ProcNode[float]:
    input_index = pf.nodes.geo.input_index()

    divide = input_index.astype(dtype=float) / count.astype(dtype=float)
    return divide


@pf.nodes.node_function
def _plant_seed(
    dimensions: t.SocketOrVal[pf.Vector],
    u: t.SocketOrVal[int],
    v: t.SocketOrVal[int],
) -> pf.ProcNode:
    curve_bezier_end = pf.nodes.func.combine_xyz(dimensions.x)
    curve_bezier_middle = pf.nodes.math.vector_multiply_add(
        a=curve_bezier_end,
        b=(0.5, 0.5, 0.5),
        addend=(0.0, 0.0, 0.0),
    )
    curve_bezier = pf.nodes.geo.curve_bezier(
        resolution=u,
        start=(0.0, 0.0, 0.0),
        middle=curve_bezier_middle,
        end=curve_bezier_end,
    )

    norm_index_result = _norm_index(count=u)

    curve_value = pf.nodes.func.float_curve(
        value=norm_index_result,
        curve=np.array([[0.0, 0.0], [0.3159, 0.4469], [1.0, 0.0156]]),
    )
    curve_to_curve_radius = pf.nodes.func.map_range(value=curve_value, to_max=3.0)

    set_curve_radius = pf.nodes.geo.set_curve_radius(
        curve=curve_bezier, radius=curve_to_curve_radius
    )

    curve_circle = pf.nodes.geo.curve_circle(resolution=v, radius=dimensions.y)
    curve_to = pf.nodes.geo.curve_to_mesh(
        curve=set_curve_radius,
        profile_curve=curve_circle,
        fill_caps=True,
    )

    result_0_selection = pf.nodes.func.constant(1.0)

    store_named_attribute = pf.nodes.geo.store_named_attribute(
        geometry=curve_to,
        name="TAG_seed",
        selection=result_0_selection.astype(dtype=bool),
        value=True,
        domain="FACE",
    )
    return store_named_attribute


@pf.nodes.node_function
def flower(
    center_rad: t.SocketOrVal[float],
    petal_dims: t.SocketOrVal[pf.Vector],
    seed_size: t.SocketOrVal[float],
    min_petal_angle: t.SocketOrVal[float],
    max_petal_angle: t.SocketOrVal[float],
    wrinkle: t.SocketOrVal[float],
    curl: t.SocketOrVal[float],
    petal_material: t.SocketOrVal[pf.Material],
    center_material: t.SocketOrVal[pf.Material],
) -> pf.ProcNode[pf.MeshObject]:
    uv_sphere = pf.nodes.geo.mesh_uv_sphere(
        segments=8,
        rings=8,
        radius=center_rad,
    )

    center_transform = pf.nodes.geo.transform(
        geometry=uv_sphere.mesh, scale=(1.0, 1.0, 0.05)
    )

    distribute_points = pf.nodes.geo.distribute_points_on_faces(
        mesh=center_transform,
        density=5000.0,
    )

    seed_length = seed_size * 10.0
    seed_dimensions = pf.nodes.func.combine_xyz(x=seed_length, y=seed_size)

    plant_seed_result = _plant_seed(dimensions=seed_dimensions, u=6, v=6)

    input_position = pf.nodes.geo.input_position()
    seed_noise = pf.nodes.shader.noise(
        vector=input_position,
        w=13.8,
        scale=2.41,
        noise_dimensions="4D",
    )

    seed_scale_x = pf.nodes.func.map_range(
        value=seed_noise.fac, to_min=0.34, to_max=1.21
    )
    seed_scale = pf.nodes.func.combine_xyz(x=seed_scale_x, y=1.0, z=1.0)

    seeds_instance = pf.nodes.geo.instance_on_points(
        points=distribute_points.points,
        instance=plant_seed_result,
        rotation=(0.0, -1.5708, 0.0541),
        scale=seed_scale,
    )

    seeds_realized = pf.nodes.geo.realize_instances(seeds_instance)

    center_geometry = pf.nodes.geo.join_geometry([seeds_realized, center_transform])

    center_with_material = pf.nodes.geo.set_material(
        geometry=center_geometry, material=center_material
    )

    petal_count_base = center_rad * 6.2832
    petal_count = pf.nodes.math.ceil(petal_count_base / petal_dims.y) * 1.2

    phyllo_result = _phyllo_points(
        count=petal_count.astype(dtype=int),
        min_radius=center_rad,
        max_radius=center_rad,
        radius_exp=0.0,
        min_angle=min_petal_angle,
        max_angle=max_petal_angle,
        min_z=0.0,
        max_z=0.0,
        clamp_z=1.0,
        yaw_offset=-1.5708,
    )

    petal_upper_width = pf.nodes.math.subtract(petal_dims.z, petal_dims.y)
    petal_upper_width_clamped = pf.nodes.math.clamp(petal_upper_width, min=0.0)

    petal_result = _flower_petal(
        length=petal_dims.x,
        point=0.56,
        point_height=-0.1,
        bevel=1.83,
        base_width=petal_dims.y,
        upper_width=petal_upper_width_clamped,
        resolution_h=8,
        resolution_v=16,
        wrinkle=wrinkle,
        curl=curl,
    )

    petals_instance = pf.nodes.geo.instance_on_points(
        points=phyllo_result.points,
        instance=petal_result,
        rotation=phyllo_result.rotation,
    )

    petals_realized = pf.nodes.geo.realize_instances(petals_instance)

    petal_noise = pf.nodes.shader.noise(scale=3.73, detail=5.41, distortion=-1.0)

    petal_offset = pf.nodes.math.vector_scale(
        vector=pf.nodes.math.vector_subtract(
            petal_noise.color.astype(dtype=pf.Vector), (0.5, 0.5, 0.5)
        ),
        scale=0.025,
    )

    petals_displaced = pf.nodes.geo.set_position(
        geometry=petals_realized, offset=petal_offset
    )

    petals_with_material = pf.nodes.geo.set_material(
        geometry=petals_displaced, material=petal_material
    )

    flower_geometry = pf.nodes.geo.join_geometry(
        [center_with_material, petals_with_material]
    )

    flower_smooth = pf.nodes.geo.set_shade_smooth(
        geometry=flower_geometry, shade_smooth=False
    )

    return flower_smooth


def flower_distribution(
    rng: pf.RNG,
    materials: FlowerMaterials | None = None,
) -> FlowerResult:
    rngs = rng.spawn(2)

    if materials is None:
        materials = flower_material_distribution(rngs[0])

    petal_mat = materials.petal
    center_mat = materials.center

    rng_geo = rngs[1]

    overall_rad = pf.random.uniform(rng_geo, 0.05, 0.2)

    pct_inner = pf.random.uniform(rng_geo, 0.05, 0.2)
    center_rad = overall_rad * pct_inner

    num_petals = pf.random.clip_gaussian(rng_geo, 20, 5, 10, 40)
    base_width = 2 * np.pi * overall_rad * pct_inner / num_petals
    top_width = pf.random.clip_gaussian(
        rng_geo, overall_rad * 0.7, overall_rad * 0.3, base_width * 1.2, overall_rad * 2
    )

    petal_length = overall_rad * (1 - pct_inner)
    petal_dims = (petal_length, base_width, top_width)

    seed_size = pf.random.uniform(rng_geo, 0.0005, 0.002)

    angle1 = pf.random.uniform(rng_geo, np.deg2rad(-20), np.deg2rad(60))
    angle2 = pf.random.uniform(rng_geo, angle1, np.deg2rad(100))
    min_petal_angle = min(angle1, angle2)
    max_petal_angle = max(angle1, angle2)

    wrinkle = pf.random.uniform(rng_geo, 0.003, 0.02)
    curl = pf.random.normal(rng_geo, np.deg2rad(30), np.deg2rad(50))

    res = flower(
        center_rad=center_rad,
        petal_dims=petal_dims,
        seed_size=seed_size,
        min_petal_angle=min_petal_angle,
        max_petal_angle=max_petal_angle,
        wrinkle=wrinkle,
        curl=curl,
        petal_material=petal_mat,
        center_material=center_mat,
    )

    obj = pf.nodes.to_mesh_object(res)
    pf.ops.modifier.subdivide_surface(obj, levels=2)
    return FlowerResult(mesh=obj)
