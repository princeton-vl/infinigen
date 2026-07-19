# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Yiming Zuo: original Infinigen v1 nodegroups (office_chair, curvy_seats, round_seats, wheeled leg)
# - Alexander Raistrick: transpile to procfunc/v2, split into top/bottom part distributions

from typing import NamedTuple

import procfunc as pf
from procfunc.nodes import types as t
from procfunc.nodes.util.bpy_node_info import NodeDataType

from infinigen2.objects.table import (
    base_square_rand,
    base_straight_rand,
)
from infinigen2.shaders.functionality_lists import (
    castor_wheel_material_rand,
    furniture_fabric,
    furniture_material_rand,
)
from infinigen2.util import mesh
from infinigen2.util.curve import curve_to_mesh_with_uv

__all__ = [
    "ChairResult",
    "DiningSeatResult",
    "chair_back",
    "chair_back_rand",
    "chair_back_solid",
    "chair_rand",
    "curvy_seat",
    "curvy_seat_rand",
    "dining_chair_dimensions_rand",
    "dining_chair_rand",
    "dining_seat",
    "office_chair_dimensions_rand",
    "office_chair_rand",
    "round_seat",
    "round_seat_rand",
    "wheeled_base",
    "wheeled_base_rand",
]


class ChairResult(NamedTuple):
    mesh: pf.MeshObject


# ==== from src/infinigen2/objects/_scratch_curvy.py ====


@pf.nodes.node_function
def _bent(
    geometry: pf.ProcNode, amount: t.SocketOrVal[float] = -0.1
) -> pf.ProcNode[pf.Vector]:
    input_position = pf.nodes.geo.input_position()
    result_0_position_angle_a = pf.nodes.math.vector_length(input_position)
    result_0_position_angle_0 = result_0_position_angle_a * input_position.x
    result_0_position = pf.nodes.math.vector_rotate_axis_angle(
        vector=input_position, angle=result_0_position_angle_0 * amount
    )
    set_position = pf.nodes.geo.set_position(
        geometry=geometry, position=result_0_position, offset=(0.0, 0.0, 0.0)
    )
    return set_position


@pf.nodes.node_function
def _warp_around_curve(
    geometry: pf.ProcNode,
    curve: pf.ProcNode,
    curve_resolution: t.SocketOrVal[int] = 1024,
) -> pf.ProcNode[pf.Vector]:
    resample = curve_resolution.astype(dtype=float) + 1.0
    resample_curve_count = pf.nodes.geo.resample_curve_count(
        curve=curve, count=resample.astype(dtype=int)
    )
    input_position = pf.nodes.geo.input_position()
    input_position_1 = pf.nodes.geo.input_position()
    bound_box = pf.nodes.geo.bound_box(geometry)
    sample_b = pf.nodes.math.map_range(
        value=input_position_1.z,
        from_max=bound_box.max.z,
        from_min=bound_box.min.z,
        data_type=NodeDataType.FLOAT,
    )
    sample_index_index = pf.nodes.math.round(
        curve_resolution.astype(dtype=float) * sample_b
    )
    sample_index = pf.nodes.geo.sample_index(
        geometry=resample_curve_count,
        index=sample_index_index.astype(dtype=int),
        value=input_position,
        data_type=NodeDataType.FLOAT_VECTOR,
    )
    input_normal = pf.nodes.geo.input_normal()
    sample_index_1 = pf.nodes.geo.sample_index(
        geometry=resample_curve_count,
        index=sample_index_index.astype(dtype=int),
        value=input_normal,
        data_type=NodeDataType.FLOAT_VECTOR,
    )
    input_position_2 = pf.nodes.geo.input_position()
    input_tangent = pf.nodes.geo.input_tangent()
    sample_index_2 = pf.nodes.geo.sample_index(
        geometry=resample_curve_count,
        index=sample_index_index.astype(dtype=int),
        value=input_tangent,
        data_type=NodeDataType.FLOAT_VECTOR,
    )
    result_0_position_b = pf.nodes.math.vector_cross_product(
        a=sample_index_2, b=sample_index_1
    )
    result_0_position = (
        sample_index_1 * input_position_2.x + result_0_position_b * input_position_2.y
    )
    set_position = pf.nodes.geo.set_position(
        geometry=geometry,
        position=sample_index + result_0_position,
        offset=(0.0, 0.0, 0.0),
    )
    return set_position


@pf.nodes.node_function
def _curvy_seat_geometry(
    u_resolution: t.SocketOrVal[int] = 256,
    v_resolution: t.SocketOrVal[int] = 128,
    width: t.SocketOrVal[float] = 0.5,
    thickness: t.SocketOrVal[float] = 0.03,
    front_relative_width: t.SocketOrVal[float] = 0.5,
    front_bent: t.SocketOrVal[float] = -0.38,
    seat_bent: t.SocketOrVal[float] = -0.56,
    mid_relative_width: t.SocketOrVal[float] = 0.5,
    mid_bent: t.SocketOrVal[float] = -0.7,
    back_relative_width: t.SocketOrVal[float] = 0.5,
    back_bent: t.SocketOrVal[float] = -0.2,
    top_relative_width: t.SocketOrVal[float] = 0.5,
    top_bent: t.SocketOrVal[float] = -0.2,
    seat_height: t.SocketOrVal[float] = 0.6,
    mid_pos: t.SocketOrVal[float] = 0.5,
) -> pf.ProcNode[pf.MeshObject]:
    curve_circle = pf.nodes.geo.curve_circle(resolution=u_resolution, radius=0.5)
    transform = pf.nodes.geo.transform(geometry=curve_circle, scale=(0.0, 0.005, 1.0))
    curve_circle_1 = pf.nodes.geo.curve_circle(resolution=u_resolution, radius=0.5)
    transform_1_scale = pf.nodes.math.combine_xyz(
        x=width * front_relative_width, y=0.005, z=1.0
    )
    transform_1 = pf.nodes.geo.transform(
        geometry=curve_circle_1, translation=(0.0, 0.0, 0.06), scale=transform_1_scale
    )
    bent_result = _bent(geometry=transform_1, amount=front_bent)
    curve_circle_2 = pf.nodes.geo.curve_circle(resolution=u_resolution, radius=0.5)
    transform_2_scale = pf.nodes.math.combine_xyz(x=width, y=thickness, z=1.0)
    transform_2 = pf.nodes.geo.transform(
        geometry=curve_circle_2, translation=(0.0, 0.0, 0.5), scale=transform_2_scale
    )
    bent_result_1 = _bent(geometry=transform_2, amount=seat_bent)
    curve_circle_3 = pf.nodes.geo.curve_circle(resolution=u_resolution, radius=0.5)
    transform_3_scale = pf.nodes.math.combine_xyz(
        x=width * mid_relative_width, y=thickness, z=1.0
    )
    transform_3 = pf.nodes.geo.transform(
        geometry=curve_circle_3, translation=(0.0, 0.0, 1.0), scale=transform_3_scale
    )
    bent_result_2 = _bent(geometry=transform_3, amount=mid_bent)
    join = pf.nodes.geo.join_geometry(
        [transform, bent_result, bent_result_1, bent_result_2]
    )
    curve_circle_4 = pf.nodes.geo.curve_circle(resolution=u_resolution, radius=0.5)
    transform_4_scale = pf.nodes.math.combine_xyz(
        x=width * back_relative_width, y=thickness, z=1.0
    )
    transform_4 = pf.nodes.geo.transform(
        geometry=curve_circle_4, translation=(0.0, 0.0, 1.5), scale=transform_4_scale
    )
    bent_result_3 = _bent(geometry=transform_4, amount=back_bent)
    curve_circle_5 = pf.nodes.geo.curve_circle(resolution=u_resolution, radius=0.5)
    transform_5_scale = pf.nodes.math.combine_xyz(
        x=width * top_relative_width, y=0.005, z=1.0
    )
    transform_5 = pf.nodes.geo.transform(
        geometry=curve_circle_5, translation=(0.0, 0.0, 2.02), scale=transform_5_scale
    )
    bent_result_4 = _bent(geometry=transform_5, amount=top_bent)
    curve_circle_6 = pf.nodes.geo.curve_circle(resolution=u_resolution, radius=0.5)
    transform_6 = pf.nodes.geo.transform(
        geometry=curve_circle_6, translation=(0.0, 0.0, 2.1), scale=(0.0, 0.005, 1.0)
    )
    join_1 = pf.nodes.geo.join_geometry([bent_result_3, bent_result_4, transform_6])
    join_2 = pf.nodes.geo.join_geometry([join, join_1])
    lofting_result = mesh.lofting(
        profile_curves=join_2, u_resolution=u_resolution, v_resolution=v_resolution
    )
    curve_bezier_segment_start = pf.nodes.math.combine_xyz(y=width * -0.5, z=0.03)
    curve_bezier_segment_start_handle = pf.nodes.math.combine_xyz(y=mid_pos, z=-0.05)
    curve_bezier_segment_end = pf.nodes.math.combine_xyz(y=width * 0.5, z=seat_height)
    curve_bezier_segment = pf.nodes.geo.curve_bezier_segment(
        start=curve_bezier_segment_start,
        start_handle=curve_bezier_segment_start_handle,
        end_handle=(0.0, 0.1, 0.1),
        end=curve_bezier_segment_end,
        resolution=128,
    )
    warp_around_curve_result = _warp_around_curve(
        geometry=lofting_result.geometry, curve=curve_bezier_segment
    )
    return warp_around_curve_result


def curvy_seat(
    u_resolution: int = 256,
    v_resolution: int = 128,
    width: float = 0.5,
    thickness: float = 0.03,
    front_relative_width: float = 0.5,
    front_bent: float = -0.38,
    seat_bent: float = -0.56,
    mid_relative_width: float = 0.5,
    mid_bent: float = -0.7,
    back_relative_width: float = 0.5,
    back_bent: float = -0.2,
    top_relative_width: float = 0.5,
    top_bent: float = -0.2,
    seat_height: float = 0.6,
    mid_pos: float = 0.5,
) -> ChairResult:
    """Curved shell seat with an integrated backrest (v1 curvy_seats)."""
    geo = _curvy_seat_geometry(
        u_resolution=u_resolution,
        v_resolution=v_resolution,
        width=width,
        thickness=thickness,
        front_relative_width=front_relative_width,
        front_bent=front_bent,
        seat_bent=seat_bent,
        mid_relative_width=mid_relative_width,
        mid_bent=mid_bent,
        back_relative_width=back_relative_width,
        back_bent=back_bent,
        top_relative_width=top_relative_width,
        top_bent=top_bent,
        seat_height=seat_height,
        mid_pos=mid_pos,
    )
    return ChairResult(mesh=pf.nodes.to_mesh_object(geo))


# ==== from src/infinigen2/objects/_scratch_round.py ====


@pf.nodes.node_function
def _round_seat_n_gon_profile(
    profile_n_gon: t.SocketOrVal[int] = 4,
    profile_width: t.SocketOrVal[float] = 1.0,
    profile_aspect_ratio: t.SocketOrVal[float] = 1.0,
    profile_fillet_ratio: t.SocketOrVal[float] = 0.2,
) -> pf.ProcNode[pf.CurveObject]:
    curve_circle_radius = pf.nodes.math.constant(0.5)
    curve_circle = pf.nodes.geo.curve_circle(
        resolution=profile_n_gon, radius=curve_circle_radius
    )
    transform_rotation = pf.nodes.math.combine_xyz(
        z=3.1416 / profile_n_gon.astype(dtype=float)
    )
    transform = pf.nodes.geo.transform(
        geometry=curve_circle, rotation=transform_rotation.astype(dtype=pf.Euler)
    )
    transform_1 = pf.nodes.geo.transform(
        geometry=transform, rotation=(0.0, 0.0, -1.5708)
    )
    transform_2_scale = pf.nodes.math.combine_xyz(
        x=profile_width, y=profile_aspect_ratio * profile_width, z=1.0
    )
    transform_2 = pf.nodes.geo.transform(geometry=transform_1, scale=transform_2_scale)
    fillet_curve_poly = pf.nodes.geo.fillet_curve_poly(
        curve=transform_2,
        radius=profile_width * profile_fillet_ratio,
        count=8,
        limit_radius=True,
    )
    return fillet_curve_poly


class _RoundSeatNGonCylinderResult(NamedTuple):
    mesh: pf.ProcNode[pf.MeshObject]
    profile_curve: pf.ProcNode[pf.CurveObject]
    caps: pf.ProcNode[pf.MeshObject]


@pf.nodes.node_function
def _round_seat_n_gon_cylinder(
    radius_curve: pf.ProcNode,
    height: t.SocketOrVal[float] = 0.5,
    n_gon: t.SocketOrVal[int] = 0,
    profile_width: t.SocketOrVal[float] = 0.5,
    aspect_ratio: t.SocketOrVal[float] = 0.5,
    fillet_ratio: t.SocketOrVal[float] = 0.2,
    profile_resolution: t.SocketOrVal[int] = 64,
    resolution: t.SocketOrVal[int] = 128,
) -> _RoundSeatNGonCylinderResult:
    mesh_position_z_to_min = height * -1.0
    curve_line_end = pf.nodes.math.combine_xyz(z=mesh_position_z_to_min)
    curve_line = pf.nodes.geo.curve_line(start=(0.0, 0.0, 0.0), end=curve_line_end)
    set_curve_tilt = pf.nodes.geo.set_curve_tilt(curve=curve_line, tilt=3.1416)
    resample_curve_count_1 = pf.nodes.geo.resample_curve_count(
        curve=set_curve_tilt, count=resolution
    )
    spline_parameter = pf.nodes.geo.spline_parameter()
    capture_attribute = pf.nodes.geo.capture_attribute(
        geometry=resample_curve_count_1, attribute=spline_parameter.factor
    )
    n_gon_profile_result = _round_seat_n_gon_profile(
        profile_n_gon=n_gon,
        profile_width=profile_width,
        profile_aspect_ratio=aspect_ratio,
        profile_fillet_ratio=fillet_ratio,
    )
    resample_curve_count = pf.nodes.geo.resample_curve_count(
        curve=n_gon_profile_result, count=profile_resolution
    )
    curve_to = pf.nodes.geo.curve_to_mesh(
        curve=capture_attribute.geometry,
        profile_curve=resample_curve_count,
        fill_caps=True,
    )
    input_position = pf.nodes.geo.input_position()
    sample_curve = pf.nodes.geo.sample_curve(
        curves=radius_curve,
        factor=capture_attribute.attribute,
        use_all_curves=True,
        data_type=NodeDataType.FLOAT,
    )
    mesh_position_x_vector = pf.nodes.math.combine_xyz(
        x=sample_curve.position.x, y=sample_curve.position.y
    )
    mesh_position_x = pf.nodes.math.vector_length(mesh_position_x_vector)
    input_position_1 = pf.nodes.geo.input_position()
    attribute_statistic = pf.nodes.geo.attribute_statistic(
        geometry=radius_curve,
        attribute=input_position_1.z,
    )
    mesh_position_z = pf.nodes.math.map_range(
        value=sample_curve.position.z,
        from_max=attribute_statistic.max,
        from_min=attribute_statistic.min,
        to_max=0.0,
        to_min=mesh_position_z_to_min,
        data_type=NodeDataType.FLOAT,
    )
    mesh_position = pf.nodes.math.combine_xyz(
        x=input_position.x * mesh_position_x,
        y=input_position.y * mesh_position_x,
        z=mesh_position_z,
    )
    set_position = pf.nodes.geo.set_position(
        geometry=curve_to, position=mesh_position, offset=(0.0, 0.0, 0.0)
    )
    input_index = pf.nodes.geo.input_index()
    attribute_domain_size = pf.nodes.geo.attribute_domain_size(curve_to)
    caps_selection_b = attribute_domain_size.face_count.astype(dtype=float) - 2.0
    caps_selection = pf.nodes.func.less_than(
        a=input_index, b=caps_selection_b.astype(dtype=int)
    )
    delete = pf.nodes.geo.delete_geometry(
        geometry=curve_to, selection=caps_selection, domain="FACE"
    )
    return _RoundSeatNGonCylinderResult(set_position, resample_curve_count, delete)


class _RoundSeatTableTopResult(NamedTuple):
    geometry: pf.ProcNode[pf.MeshObject]
    curve: pf.ProcNode


@pf.nodes.node_function
def _round_seat_generate_table_top(
    thickness: t.SocketOrVal[float] = 0.5,
    n_gon: t.SocketOrVal[int] = 0,
    profile_width: t.SocketOrVal[float] = 0.5,
    aspect_ratio: t.SocketOrVal[float] = 0.5,
    fillet_ratio: t.SocketOrVal[float] = 0.2,
    fillet_radius_vertical: t.SocketOrVal[float] = 0.0,
) -> _RoundSeatTableTopResult:
    curve_line = pf.nodes.geo.curve_line(start=(1.0, 0.0, 1.0), end=(1.0, 0.0, -1.0))
    n_gon_cylinder_result = _round_seat_n_gon_cylinder(
        radius_curve=curve_line,
        height=thickness,
        n_gon=n_gon,
        profile_width=profile_width,
        aspect_ratio=aspect_ratio,
        fillet_ratio=fillet_ratio,
        profile_resolution=512,
        resolution=10,
    )
    input_index = pf.nodes.geo.input_index()
    store_named_attribute = pf.nodes.geo.store_named_attribute(
        geometry=n_gon_cylinder_result.caps,
        name="TAG_support",
        selection=input_index == 0,
        value=True,
        domain="FACE",
        data_type=NodeDataType.BOOLEAN,
    )
    curve_arc = pf.nodes.geo.curve_arc(resolution=4, radius=0.7071, sweep_angle=4.7124)
    transform_1 = pf.nodes.geo.transform(
        geometry=curve_arc, rotation=(0.0, 0.0, -0.7854)
    )
    transform_2 = pf.nodes.geo.transform(
        geometry=transform_1, rotation=(0.0, 1.5708, 0.0)
    )
    transform_3 = pf.nodes.geo.transform(
        geometry=transform_2, translation=(0.0, 0.5, 0.0)
    )
    transform_4_scale = pf.nodes.math.combine_xyz(
        x=1.0, y=fillet_radius_vertical, z=1.0
    )
    transform_4 = pf.nodes.geo.transform(geometry=transform_3, scale=transform_4_scale)
    fillet_curve_poly = pf.nodes.geo.fillet_curve_poly(
        curve=transform_4, radius=fillet_radius_vertical, count=8, limit_radius=True
    )
    transform_5 = pf.nodes.geo.transform(
        geometry=fillet_curve_poly,
        rotation=(1.5708, 1.5708, 0.0),
        scale=thickness.astype(dtype=pf.Vector),
    )
    curve_to = pf.nodes.geo.curve_to_mesh(
        curve=n_gon_cylinder_result.profile_curve, profile_curve=transform_5
    )
    transform_6_translation = pf.nodes.math.combine_xyz(z=thickness * -0.5)
    transform_6 = pf.nodes.geo.transform(
        geometry=curve_to, translation=transform_6_translation
    )
    join = pf.nodes.geo.join_geometry([store_named_attribute, transform_6])
    flip_faces = pf.nodes.geo.flip_faces(join)
    geometry_translation = pf.nodes.math.combine_xyz(z=thickness)
    transform = pf.nodes.geo.transform(
        geometry=flip_faces, translation=geometry_translation
    )
    return _RoundSeatTableTopResult(transform, n_gon_cylinder_result.profile_curve)


@pf.nodes.node_function
def _round_seat_create_cap(
    radius: t.SocketOrVal[float] = 1.0, resolution: t.SocketOrVal[int] = 64
) -> pf.ProcNode[pf.MeshObject]:
    uv_sphere_rings = radius * 257.0
    uv_sphere = pf.nodes.geo.mesh_uv_sphere(
        segments=resolution, rings=uv_sphere_rings.astype(dtype=int), radius=radius
    )
    store_named_attribute = pf.nodes.geo.store_named_attribute(
        geometry=uv_sphere.mesh,
        name="uv_map",
        value=uv_sphere.uv_map,
        domain="CORNER",
        data_type=NodeDataType.FLOAT_VECTOR,
    )
    transform_a = radius**2.0
    transform_translation_z = pf.nodes.math.sqrt(transform_a - 1.0)
    transform_translation = pf.nodes.math.combine_xyz(z=transform_translation_z * -1.0)
    transform = pf.nodes.geo.transform(
        geometry=store_named_attribute, translation=transform_translation
    )
    input_position = pf.nodes.geo.input_position()
    result_0_selection = pf.nodes.func.less_than(a=input_position.z, b=0.0)
    delete = pf.nodes.geo.delete_geometry(
        geometry=transform, selection=result_0_selection
    )
    return delete


@pf.nodes.node_function
def _round_seat_capped_cylinder(
    thickness: t.SocketOrVal[float] = 0.5,
    radius: t.SocketOrVal[float] = 0.2,
    cap_flatness: t.SocketOrVal[float] = 4.0,
    fillet_radius_vertical: t.SocketOrVal[float] = 0.4,
    cap_relative_scale: t.SocketOrVal[float] = 1.0,
    cap_relative_z_offset: t.SocketOrVal[float] = 0.0,
    resolution: t.SocketOrVal[int] = 64,
) -> pf.ProcNode[pf.Vector]:
    generate_table_top_thickness = thickness * 2.0
    generate_table_top_result = _round_seat_generate_table_top(
        thickness=generate_table_top_thickness,
        n_gon=resolution,
        profile_width=radius * 1.0,
        aspect_ratio=1.0,
        fillet_ratio=0.0,
        fillet_radius_vertical=fillet_radius_vertical,
    )
    create_cap_1 = _round_seat_create_cap(radius=cap_flatness, resolution=resolution)
    transform_translation = pf.nodes.math.combine_xyz(
        z=generate_table_top_thickness + cap_relative_z_offset
    )
    transform_scale = radius * 0.5 + cap_relative_scale
    transform = pf.nodes.geo.transform(
        geometry=create_cap_1,
        translation=transform_translation,
        scale=transform_scale.astype(dtype=pf.Vector),
    )
    join = pf.nodes.geo.join_geometry([generate_table_top_result.geometry, transform])
    return join


@pf.nodes.node_function
def _round_seat_geometry(
    thickness: t.SocketOrVal[float] = 0.1,
    radius: t.SocketOrVal[float] = 0.37,
    cap_radius: t.SocketOrVal[float] = 2.8,
    bevel_factor: t.SocketOrVal[float] = 0.028,
) -> pf.ProcNode[pf.MeshObject]:
    return _round_seat_capped_cylinder(
        thickness=thickness,
        radius=radius,
        cap_flatness=cap_radius,
        fillet_radius_vertical=bevel_factor / thickness,
        cap_relative_scale=0.014,
        cap_relative_z_offset=-0.002,
        resolution=128,
    )


def round_seat(
    thickness: float = 0.1,
    radius: float = 0.37,
    cap_radius: float = 2.8,
    bevel_factor: float = 0.028,
) -> ChairResult:
    """Backless round pad seat (v1 round_seats)."""
    geo = _round_seat_geometry(
        thickness=thickness,
        radius=radius,
        cap_radius=cap_radius,
        bevel_factor=bevel_factor,
    )
    return ChairResult(mesh=pf.nodes.to_mesh_object(geo))


# ==== from src/infinigen2/objects/_scratch_wheeled.py ====


@pf.nodes.node_function
def _wheeled_base_n_gon_profile(
    profile_n_gon: t.SocketOrVal[int] = 4,
    profile_width: t.SocketOrVal[float] = 1.0,
    profile_aspect_ratio: t.SocketOrVal[float] = 1.0,
    profile_fillet_ratio: t.SocketOrVal[float] = 0.2,
) -> pf.ProcNode[pf.CurveObject]:
    curve_circle_radius = pf.nodes.math.constant(0.5)
    curve_circle = pf.nodes.geo.curve_circle(
        resolution=profile_n_gon, radius=curve_circle_radius
    )
    transform_rotation = pf.nodes.math.combine_xyz(
        z=3.1416 / profile_n_gon.astype(dtype=float)
    )
    transform = pf.nodes.geo.transform(
        geometry=curve_circle, rotation=transform_rotation.astype(dtype=pf.Euler)
    )
    transform_1 = pf.nodes.geo.transform(
        geometry=transform, rotation=(0.0, 0.0, -1.5708)
    )
    transform_2_scale = pf.nodes.math.combine_xyz(
        x=profile_width, y=profile_aspect_ratio * profile_width, z=1.0
    )
    transform_2 = pf.nodes.geo.transform(geometry=transform_1, scale=transform_2_scale)
    fillet_curve_poly = pf.nodes.geo.fillet_curve_poly(
        curve=transform_2,
        radius=profile_width * profile_fillet_ratio,
        count=8,
        limit_radius=True,
    )
    return fillet_curve_poly


@pf.nodes.node_function
def _wheeled_base_create_anchors(
    profile_n_gon: t.SocketOrVal[int] = 0,
    profile_width: t.SocketOrVal[float] = 0.5,
    profile_aspect_ratio: t.SocketOrVal[float] = 0.5,
    profile_rotation: t.SocketOrVal[float] = 0.0,
) -> pf.ProcNode[pf.PointCloudObject]:
    n_gon_profile_result = _wheeled_base_n_gon_profile(
        profile_n_gon=profile_n_gon,
        profile_width=profile_width,
        profile_aspect_ratio=profile_aspect_ratio,
        profile_fillet_ratio=0.0,
    )
    curve_to_points_evaluated = pf.nodes.geo.curve_to_points_evaluated(
        n_gon_profile_result
    )
    curve_line_start = pf.nodes.math.combine_xyz(profile_width * 0.3535)
    curve_line_end = pf.nodes.math.combine_xyz(profile_width * -0.3535)
    curve_line = pf.nodes.geo.curve_line(start=curve_line_start, end=curve_line_end)
    curve_to_points_evaluated_1 = pf.nodes.geo.curve_to_points_evaluated(curve_line)
    set_a = pf.nodes.func.switch(
        switch=profile_n_gon == 2,
        a=curve_to_points_evaluated.points,
        b=curve_to_points_evaluated_1.points,
        data_type=NodeDataType.GEOMETRY,
    )
    points = pf.nodes.geo.points((0.0, 0.0, 0.0))
    set_point_radius_points = pf.nodes.func.switch(
        switch=profile_n_gon == 1, a=set_a, b=points, data_type=NodeDataType.GEOMETRY
    )
    set_point_radius = pf.nodes.geo.set_point_radius(set_point_radius_points)
    result_0_rotation = pf.nodes.math.combine_xyz(z=profile_rotation)
    transform = pf.nodes.geo.transform(
        geometry=set_point_radius, rotation=result_0_rotation.astype(dtype=pf.Euler)
    )
    return transform


class _WheeledBaseNGonCylinderResult(NamedTuple):
    mesh: pf.ProcNode[pf.MeshObject]
    profile_curve: pf.ProcNode[pf.CurveObject]
    caps: pf.ProcNode[pf.MeshObject]


@pf.nodes.node_function
def _wheeled_base_n_gon_cylinder(
    radius_curve: pf.ProcNode,
    height: t.SocketOrVal[float] = 0.5,
    n_gon: t.SocketOrVal[int] = 0,
    profile_width: t.SocketOrVal[float] = 0.5,
    aspect_ratio: t.SocketOrVal[float] = 0.5,
    fillet_ratio: t.SocketOrVal[float] = 0.2,
    profile_resolution: t.SocketOrVal[int] = 64,
    resolution: t.SocketOrVal[int] = 128,
) -> _WheeledBaseNGonCylinderResult:
    mesh_position_z_to_min = height * -1.0
    curve_line_end = pf.nodes.math.combine_xyz(z=mesh_position_z_to_min)
    curve_line = pf.nodes.geo.curve_line(start=(0.0, 0.0, 0.0), end=curve_line_end)
    set_curve_tilt = pf.nodes.geo.set_curve_tilt(curve=curve_line, tilt=3.1416)
    resample_curve_count_1 = pf.nodes.geo.resample_curve_count(
        curve=set_curve_tilt, count=resolution
    )
    spline_parameter = pf.nodes.geo.spline_parameter()
    capture_attribute = pf.nodes.geo.capture_attribute(
        geometry=resample_curve_count_1, attribute=spline_parameter.factor
    )
    n_gon_profile_result = _wheeled_base_n_gon_profile(
        profile_n_gon=n_gon,
        profile_width=profile_width,
        profile_aspect_ratio=aspect_ratio,
        profile_fillet_ratio=fillet_ratio,
    )
    resample_curve_count = pf.nodes.geo.resample_curve_count(
        curve=n_gon_profile_result, count=profile_resolution
    )
    curve_to = pf.nodes.geo.curve_to_mesh(
        curve=capture_attribute.geometry,
        profile_curve=resample_curve_count,
        fill_caps=True,
    )
    input_position = pf.nodes.geo.input_position()
    sample_curve = pf.nodes.geo.sample_curve(
        curves=radius_curve,
        factor=capture_attribute.attribute,
        use_all_curves=True,
        data_type=NodeDataType.FLOAT,
    )
    mesh_position_x_vector = pf.nodes.math.combine_xyz(
        x=sample_curve.position.x, y=sample_curve.position.y
    )
    mesh_position_x = pf.nodes.math.vector_length(mesh_position_x_vector)
    input_position_1 = pf.nodes.geo.input_position()
    attribute_statistic = pf.nodes.geo.attribute_statistic(
        geometry=radius_curve,
        attribute=input_position_1.z,
    )
    mesh_position_z = pf.nodes.math.map_range(
        value=sample_curve.position.z,
        from_max=attribute_statistic.max,
        from_min=attribute_statistic.min,
        to_max=0.0,
        to_min=mesh_position_z_to_min,
        data_type=NodeDataType.FLOAT,
    )
    mesh_position = pf.nodes.math.combine_xyz(
        x=input_position.x * mesh_position_x,
        y=input_position.y * mesh_position_x,
        z=mesh_position_z,
    )
    set_position = pf.nodes.geo.set_position(
        geometry=curve_to, position=mesh_position, offset=(0.0, 0.0, 0.0)
    )
    input_index = pf.nodes.geo.input_index()
    attribute_domain_size = pf.nodes.geo.attribute_domain_size(curve_to)
    caps_selection_b = attribute_domain_size.face_count.astype(dtype=float) - 2.0
    caps_selection = pf.nodes.func.less_than(
        a=input_index, b=caps_selection_b.astype(dtype=int)
    )
    delete = pf.nodes.geo.delete_geometry(
        geometry=curve_to, selection=caps_selection, domain="FACE"
    )
    return _WheeledBaseNGonCylinderResult(set_position, resample_curve_count, delete)


@pf.nodes.node_function
def _wheeled_base_arc_top(
    diameter: t.SocketOrVal[float] = 1.0, sweep_angle: t.SocketOrVal[float] = 180.0
) -> pf.ProcNode[pf.CurveObject]:
    curve_a = pf.nodes.math.multiply_add(a=sweep_angle, b=0.5, addend=-90.0)
    curve_arc_start_angle = pf.nodes.math.deg_to_rad(curve_a * -1.0)
    curve_arc_sweep_angle = pf.nodes.math.deg_to_rad(sweep_angle)
    curve_arc = pf.nodes.geo.curve_arc(
        resolution=32,
        radius=diameter / 2.0,
        start_angle=curve_arc_start_angle,
        sweep_angle=curve_arc_sweep_angle,
    )
    transform = pf.nodes.geo.transform(geometry=curve_arc, rotation=(1.5708, 0.0, 0.0))
    return transform


class _WheeledBaseTableTopResult(NamedTuple):
    geometry: pf.ProcNode[pf.MeshObject]
    curve: pf.ProcNode


@pf.nodes.node_function
def _wheeled_base_generate_table_top(
    thickness: t.SocketOrVal[float] = 0.5,
    n_gon: t.SocketOrVal[int] = 0,
    profile_width: t.SocketOrVal[float] = 0.5,
    aspect_ratio: t.SocketOrVal[float] = 0.5,
    fillet_ratio: t.SocketOrVal[float] = 0.2,
    fillet_radius_vertical: t.SocketOrVal[float] = 0.0,
) -> _WheeledBaseTableTopResult:
    curve_line = pf.nodes.geo.curve_line(start=(1.0, 0.0, 1.0), end=(1.0, 0.0, -1.0))
    n_gon_cylinder_result = _wheeled_base_n_gon_cylinder(
        radius_curve=curve_line,
        height=thickness,
        n_gon=n_gon,
        profile_width=profile_width,
        aspect_ratio=aspect_ratio,
        fillet_ratio=fillet_ratio,
        profile_resolution=512,
        resolution=10,
    )
    input_index = pf.nodes.geo.input_index()
    store_named_attribute = pf.nodes.geo.store_named_attribute(
        geometry=n_gon_cylinder_result.caps,
        name="TAG_support",
        selection=input_index == 0,
        value=True,
        domain="FACE",
        data_type=NodeDataType.BOOLEAN,
    )
    curve_arc = pf.nodes.geo.curve_arc(resolution=4, radius=0.7071, sweep_angle=4.7124)
    transform_1 = pf.nodes.geo.transform(
        geometry=curve_arc, rotation=(0.0, 0.0, -0.7854)
    )
    transform_2 = pf.nodes.geo.transform(
        geometry=transform_1, rotation=(0.0, 1.5708, 0.0)
    )
    transform_3 = pf.nodes.geo.transform(
        geometry=transform_2, translation=(0.0, 0.5, 0.0)
    )
    transform_4_scale = pf.nodes.math.combine_xyz(
        x=1.0, y=fillet_radius_vertical, z=1.0
    )
    transform_4 = pf.nodes.geo.transform(geometry=transform_3, scale=transform_4_scale)
    fillet_curve_poly = pf.nodes.geo.fillet_curve_poly(
        curve=transform_4, radius=fillet_radius_vertical, count=8, limit_radius=True
    )
    transform_5 = pf.nodes.geo.transform(
        geometry=fillet_curve_poly,
        rotation=(1.5708, 1.5708, 0.0),
        scale=thickness.astype(dtype=pf.Vector),
    )
    curve_to = pf.nodes.geo.curve_to_mesh(
        curve=n_gon_cylinder_result.profile_curve, profile_curve=transform_5
    )
    transform_6_translation = pf.nodes.math.combine_xyz(z=thickness * -0.5)
    transform_6 = pf.nodes.geo.transform(
        geometry=curve_to, translation=transform_6_translation
    )
    join = pf.nodes.geo.join_geometry([store_named_attribute, transform_6])
    flip_faces = pf.nodes.geo.flip_faces(join)
    geometry_translation = pf.nodes.math.combine_xyz(z=thickness)
    transform = pf.nodes.geo.transform(
        geometry=flip_faces, translation=geometry_translation
    )
    return _WheeledBaseTableTopResult(transform, n_gon_cylinder_result.profile_curve)


@pf.nodes.node_function
def _wheeled_base_create_cap(
    radius: t.SocketOrVal[float] = 1.0, resolution: t.SocketOrVal[int] = 64
) -> pf.ProcNode[pf.MeshObject]:
    uv_sphere_rings = radius * 257.0
    uv_sphere = pf.nodes.geo.mesh_uv_sphere(
        segments=resolution, rings=uv_sphere_rings.astype(dtype=int), radius=radius
    )
    store_named_attribute = pf.nodes.geo.store_named_attribute(
        geometry=uv_sphere.mesh,
        name="uv_map",
        value=uv_sphere.uv_map,
        domain="CORNER",
        data_type=NodeDataType.FLOAT_VECTOR,
    )
    transform_a = radius**2.0
    transform_translation_z = pf.nodes.math.sqrt(transform_a - 1.0)
    transform_translation = pf.nodes.math.combine_xyz(z=transform_translation_z * -1.0)
    transform = pf.nodes.geo.transform(
        geometry=store_named_attribute, translation=transform_translation
    )
    input_position = pf.nodes.geo.input_position()
    result_0_selection = pf.nodes.func.less_than(a=input_position.z, b=0.0)
    delete = pf.nodes.geo.delete_geometry(
        geometry=transform, selection=result_0_selection
    )
    return delete


@pf.nodes.node_function
def _wheeled_base_capped_cylinder(
    thickness: t.SocketOrVal[float] = 0.5,
    radius: t.SocketOrVal[float] = 0.2,
    cap_flatness: t.SocketOrVal[float] = 4.0,
    fillet_radius_vertical: t.SocketOrVal[float] = 0.4,
    cap_relative_scale: t.SocketOrVal[float] = 1.0,
    cap_relative_z_offset: t.SocketOrVal[float] = 0.0,
    resolution: t.SocketOrVal[int] = 64,
) -> pf.ProcNode[pf.Vector]:
    generate_table_top_thickness = thickness * 2.0
    generate_table_top_result = _wheeled_base_generate_table_top(
        thickness=generate_table_top_thickness,
        n_gon=resolution,
        profile_width=radius * 1.0,
        aspect_ratio=1.0,
        fillet_ratio=0.0,
        fillet_radius_vertical=fillet_radius_vertical,
    )
    create_cap_1 = _wheeled_base_create_cap(radius=cap_flatness, resolution=resolution)
    transform_translation = pf.nodes.math.combine_xyz(
        z=generate_table_top_thickness + cap_relative_z_offset
    )
    transform_scale = radius * 0.5 + cap_relative_scale
    transform = pf.nodes.geo.transform(
        geometry=create_cap_1,
        translation=transform_translation,
        scale=transform_scale.astype(dtype=pf.Vector),
    )
    join = pf.nodes.geo.join_geometry([generate_table_top_result.geometry, transform])
    return join


@pf.nodes.node_function
def _wheeled_base_wheel(
    arc_sweep_angle: t.SocketOrVal[float] = 240.0,
    wheel_width: t.SocketOrVal[float] = 0.0,
    wheel_rotation: t.SocketOrVal[float] = 0.5,
    pole_width: t.SocketOrVal[float] = 0.0,
    pole_aspect_ratio: t.SocketOrVal[float] = 0.6,
    pole_length: t.SocketOrVal[float] = 3.0,
    material: pf.ProcNode[pf.Material] | None = None,
    wheel_material: pf.ProcNode[pf.Material] | None = None,
) -> pf.ProcNode[pf.MeshObject]:
    curve_line = pf.nodes.geo.curve_line(start=(1.0, 0.0, -1.0), end=(1.0, 0.0, 1.0))
    n_gon_cylinder_result = _wheeled_base_n_gon_cylinder(
        radius_curve=curve_line,
        height=pole_length,
        n_gon=4,
        profile_width=pole_width,
        aspect_ratio=pole_aspect_ratio,
        fillet_ratio=0.15,
        resolution=32,
    )
    transform_1 = pf.nodes.geo.transform(
        geometry=n_gon_cylinder_result.mesh, rotation=(0.0, -1.5708, 0.0)
    )
    subdivision_surface = pf.nodes.geo.subdivision_surface(mesh=transform_1, level=0)
    if material is not None:
        subdivision_surface = pf.nodes.geo.set_material(subdivision_surface, material)
    transform_a_a = pf.nodes.math.constant(0.5)
    cylinder = pf.nodes.geo.mesh_cylinder(
        side_segments=8,
        fill_segments=4,
        radius=transform_a_a * 0.1,
        depth=transform_a_a * 0.4,
    )
    transform_a_0 = transform_a_a * 0.44
    transform_2_translation = pf.nodes.math.combine_xyz(
        x=transform_a_0, z=transform_a_a * 0.45
    )
    transform_2 = pf.nodes.geo.transform(
        geometry=cylinder.mesh, translation=transform_2_translation
    )
    arc_top_result = _wheeled_base_arc_top(
        diameter=transform_a_a + 0.08, sweep_angle=arc_sweep_angle
    )
    curve_quadrilateral = pf.nodes.geo.curve_quadrilateral(
        width=wheel_width * 2.0, height=0.02
    )
    fillet_curve_poly = pf.nodes.geo.fillet_curve_poly(
        curve=curve_quadrilateral, radius=0.03, count=4, limit_radius=True
    )
    curve_to = pf.nodes.geo.curve_to_mesh(
        curve=arc_top_result, profile_curve=fillet_curve_poly, fill_caps=True
    )
    curve_line_1_start = pf.nodes.math.combine_xyz(y=wheel_width)
    curve_line_1_end = pf.nodes.math.combine_xyz(y=wheel_width * -1.0)
    curve_line_1 = pf.nodes.geo.curve_line(
        start=curve_line_1_start, end=curve_line_1_end
    )
    capped_cylinder_thickness = pf.nodes.math.constant(0.02)
    capped_cylinder_result = _wheeled_base_capped_cylinder(
        thickness=capped_cylinder_thickness,
        radius=transform_a_a,
        cap_relative_scale=0.01,
    )
    transform_3_translation = pf.nodes.math.combine_xyz(
        y=capped_cylinder_thickness * -1.0
    )
    transform_3 = pf.nodes.geo.transform(
        geometry=capped_cylinder_result,
        translation=transform_3_translation,
        rotation=(-1.5708, 0.0, 0.0),
    )
    input_position = pf.nodes.geo.input_position()
    instance = pf.nodes.func.align_euler_to_vector(
        factor=1.0, vector=input_position, axis="Y"
    )
    instance_on_points = pf.nodes.geo.instance_on_points(
        points=curve_line_1,
        instance=transform_3,
        rotation=instance.astype(dtype=pf.Euler),
    )
    join = pf.nodes.geo.join_geometry([transform_2, curve_to, instance_on_points])
    if wheel_material is not None:
        join = pf.nodes.geo.set_material(join, wheel_material)
    transform_4_translation = pf.nodes.math.combine_xyz(transform_a_0 * -1.0)
    transform_4 = pf.nodes.geo.transform(
        geometry=join, translation=transform_4_translation
    )
    transform_5_translation_z = pf.nodes.math.multiply_add(
        a=pole_width, b=-0.3535, addend=-0.3
    )
    transform_5_translation = pf.nodes.math.combine_xyz(
        x=pole_length - 0.15, z=transform_5_translation_z
    )
    transform_5_rotation_z = pf.nodes.math.deg_to_rad(wheel_rotation)
    transform_5_rotation = pf.nodes.math.combine_xyz(z=transform_5_rotation_z)
    transform_5 = pf.nodes.geo.transform(
        geometry=transform_4,
        translation=transform_5_translation,
        rotation=transform_5_rotation.astype(dtype=pf.Euler),
    )
    join_1 = pf.nodes.geo.join_geometry([subdivision_surface, transform_5])
    result_0_scale = pf.nodes.math.constant(0.15)
    transform = pf.nodes.geo.transform(
        geometry=join_1, scale=result_0_scale.astype(dtype=pf.Vector)
    )
    return transform


@pf.nodes.node_function
def _wheeled_base_create_legs_and_strechers(
    anchors: pf.ProcNode,
    leg_instance: pf.ProcNode,
    strecher_instance: pf.ProcNode,
    keep_legs: t.SocketOrVal[bool] = False,
    table_height: t.SocketOrVal[float] = 0.0,
    leg_bottom_relative_scale: t.SocketOrVal[float] = 0.0,
    leg_bottom_relative_rotation: t.SocketOrVal[float] = 0.0,
    keep_odd_strechers: t.SocketOrVal[bool] = True,
    keep_even_strechers: t.SocketOrVal[bool] = True,
    strecher_index_increment: t.SocketOrVal[int] = 0,
    strecher_relative_position: t.SocketOrVal[float] = 0.5,
    leg_bottom_offset: t.SocketOrVal[float] = 0.0,
    align_leg_x_rot: t.SocketOrVal[bool] = False,
) -> pf.ProcNode[pf.MeshObject | pf.CurveObject | t.Instances | pf.VolumeObject]:
    transform_translation = pf.nodes.math.combine_xyz(z=table_height)
    transform = pf.nodes.geo.transform(
        geometry=anchors, translation=transform_translation
    )
    input_position = pf.nodes.geo.input_position()
    set_b_vector_b = pf.nodes.math.combine_xyz(z=leg_bottom_offset)
    set_b_vector = transform_translation - set_b_vector_b
    set_b_rotation_0 = pf.nodes.math.combine_xyz(0, 0, leg_bottom_relative_rotation)
    set_b_1 = pf.nodes.math.vector_rotate_euler(
        vector=input_position - set_b_vector, rotation=set_b_rotation_0
    )
    set_b_0 = pf.nodes.math.combine_xyz(
        x=leg_bottom_relative_scale, y=leg_bottom_relative_scale, z=1.0
    )
    set_position_position_vector = input_position - set_b_1 * set_b_0
    set_position_position = set_position_position_vector * (
        strecher_relative_position * -1.0
    )
    input_position_1 = pf.nodes.geo.input_position()
    set_position = pf.nodes.geo.set_position(
        geometry=transform,
        position=set_position_position + input_position_1,
        offset=(0.0, 0.0, 0.0),
    )
    input_index = pf.nodes.geo.input_index()
    instance_2 = input_index.astype(dtype=float) % 2.0
    instance_a_a = pf.nodes.func.boolean_and(
        a=instance_2.astype(dtype=bool), b=keep_odd_strechers
    )
    instance_a_b_b = pf.nodes.func.boolean_not(instance_2.astype(dtype=bool))
    instance_a_b = pf.nodes.func.boolean_and(a=keep_even_strechers, b=instance_a_b_b)
    instance_a = pf.nodes.func.boolean_or(a=instance_a_a, b=instance_a_b)
    attribute_domain_size = pf.nodes.geo.attribute_domain_size(
        geometry=transform, component="POINTCLOUD"
    )
    instance_b_switch = attribute_domain_size.point_count.astype(
        dtype=float
    ) / strecher_index_increment.astype(dtype=float)
    instance_b_a = pf.nodes.math.constant(True)
    input_index_1 = pf.nodes.geo.input_index()
    instance_1 = attribute_domain_size.point_count.astype(dtype=float) / 2.0
    instance_b_b = pf.nodes.func.less_than(
        a=input_index_1, b=instance_1.astype(dtype=int)
    )
    instance_b = pf.nodes.func.switch(
        switch=instance_b_switch == 2.0,
        a=instance_b_a,
        b=instance_b_b,
        data_type=NodeDataType.BOOLEAN,
    )
    instance_on_points_selection = pf.nodes.func.boolean_and(a=instance_a, b=instance_b)
    input_position_2 = pf.nodes.geo.input_position()
    field = (
        input_index.astype(dtype=float) + strecher_index_increment.astype(dtype=float)
    ) % attribute_domain_size.point_count.astype(dtype=float)
    field_at_index = pf.nodes.geo.field_at_index(
        value=input_position_2,
        index=field.astype(dtype=int),
        data_type=NodeDataType.FLOAT_VECTOR,
    )
    instance_z_vector = input_position_2 - field_at_index
    instance_0_rotation = pf.nodes.func.align_euler_to_vector(
        factor=1.0, vector=instance_z_vector, axis="Z"
    )
    instance_0 = pf.nodes.func.align_euler_to_vector(
        factor=1.0, vector=(0.0, 0.0, 1.0), rotation=instance_0_rotation, pivot_axis="Z"
    )
    instance_z = pf.nodes.math.vector_length(instance_z_vector)
    instance_on_points_scale = pf.nodes.math.combine_xyz(x=1.0, y=1.0, z=instance_z)
    instance_on_points = pf.nodes.geo.instance_on_points(
        points=set_position,
        instance=strecher_instance,
        rotation=instance_0.astype(dtype=pf.Euler),
        scale=instance_on_points_scale,
        selection=instance_on_points_selection,
    )
    realize_instances = pf.nodes.geo.realize_instances(instance_on_points)
    instance_rotation_a = pf.nodes.func.align_euler_to_vector(
        factor=1.0, vector=set_position_position_vector, axis="Z"
    )
    instance_rotation_b = pf.nodes.func.align_euler_to_vector(
        factor=1.0, vector=input_position, rotation=instance_rotation_a, pivot_axis="Z"
    )
    instance_rotation = pf.nodes.func.switch(
        switch=align_leg_x_rot,
        a=instance_rotation_a,
        b=instance_rotation_b,
        data_type=NodeDataType.FLOAT_VECTOR,
    )
    instance_scale_z = pf.nodes.math.vector_length(set_position_position_vector)
    instance_scale = pf.nodes.math.combine_xyz(x=1.0, y=1.0, z=instance_scale_z)
    instance_on_points_1 = pf.nodes.geo.instance_on_points(
        points=transform,
        instance=leg_instance,
        rotation=instance_rotation.astype(dtype=pf.Euler),
        scale=instance_scale,
    )
    realize_instances_1 = pf.nodes.geo.realize_instances(instance_on_points_1)
    result_0_geometries = pf.nodes.func.switch(
        switch=keep_legs, b=realize_instances_1, data_type=NodeDataType.GEOMETRY
    )
    join = pf.nodes.geo.join_geometry([realize_instances, result_0_geometries])
    return join


class _WheeledBaseAlignBottomResult(NamedTuple):
    geometry: pf.ProcNode[pf.Vector]
    offset: pf.ProcNode[float]


@pf.nodes.node_function
def _wheeled_base_align_bottom_to_floor(
    geometry: pf.ProcNode,
) -> _WheeledBaseAlignBottomResult:
    bound_box = pf.nodes.geo.bound_box(geometry)
    offset = bound_box.min.z * -1.0
    geometry_translation = pf.nodes.math.combine_xyz(z=offset)
    transform = pf.nodes.geo.transform(
        geometry=geometry, translation=geometry_translation
    )
    return _WheeledBaseAlignBottomResult(transform, offset)


@pf.nodes.node_function
def _wheeled_base_geometry(
    joint_height: t.SocketOrVal[float] = 0.0,
    leg_diameter: t.SocketOrVal[float] = 0.0,
    top_height: t.SocketOrVal[float] = 0.0,
    arc_sweep_angle: t.SocketOrVal[float] = 240.0,
    wheel_width: t.SocketOrVal[float] = 0.13,
    wheel_rotation: t.SocketOrVal[float] = 0.5,
    pole_length: t.SocketOrVal[float] = 1.8,
    leg_number: t.SocketOrVal[int] = 5,
    material: pf.ProcNode[pf.Material] | None = None,
    wheel_material: pf.ProcNode[pf.Material] | None = None,
) -> pf.ProcNode[pf.MeshObject]:
    cylinder_1_radius = leg_diameter * 0.5
    cylinder_depth = top_height - joint_height
    cylinder = pf.nodes.geo.mesh_cylinder(
        vertices=64, radius=cylinder_1_radius - 0.0025, depth=cylinder_depth
    )
    transform_translation_z = 0.5 * cylinder_depth
    transform_translation = pf.nodes.math.combine_xyz(
        z=top_height - transform_translation_z
    )
    transform = pf.nodes.geo.transform(
        geometry=cylinder.mesh, translation=transform_translation
    )
    create_anchors_profile_width = pf.nodes.math.constant(0.001)
    create_anchors_result = _wheeled_base_create_anchors(
        profile_n_gon=leg_number,
        profile_width=create_anchors_profile_width,
        profile_aspect_ratio=1.0,
    )
    chair_wheel_result = _wheeled_base_wheel(
        arc_sweep_angle=arc_sweep_angle,
        wheel_width=wheel_width,
        wheel_rotation=wheel_rotation,
        pole_width=0.5,
        pole_length=pole_length,
        material=material,
        wheel_material=wheel_material,
    )
    transform_1 = pf.nodes.geo.transform(
        geometry=chair_wheel_result, rotation=(0.0, 1.5708, 0.0)
    )
    create_legs_and_strechers_result = _wheeled_base_create_legs_and_strechers(
        anchors=create_anchors_result,
        keep_legs=True,
        leg_instance=transform_1,
        table_height=0.025,
        leg_bottom_relative_scale=2.0 / create_anchors_profile_width,
        strecher_instance=pf.nodes.geo.points(position=(0, 0, 0)),
        strecher_index_increment=1,
        strecher_relative_position=1.0,
        leg_bottom_offset=0.025,
        align_leg_x_rot=True,
    )
    align_bottom_to_floor_result = _wheeled_base_align_bottom_to_floor(
        geometry=create_legs_and_strechers_result
    )
    cylinder_1_depth = joint_height - align_bottom_to_floor_result.offset
    cylinder_1 = pf.nodes.geo.mesh_cylinder(
        vertices=64, radius=cylinder_1_radius, depth=cylinder_1_depth
    )
    transform_a = cylinder_1_depth * 0.5
    transform_2_translation = pf.nodes.math.combine_xyz(
        z=transform_a + align_bottom_to_floor_result.offset
    )
    transform_2 = pf.nodes.geo.transform(
        geometry=cylinder_1.mesh, translation=transform_2_translation
    )
    column = pf.nodes.geo.join_geometry([transform, transform_2])
    if material is not None:
        column = pf.nodes.geo.set_material(column, material)
    join = pf.nodes.geo.join_geometry([column, align_bottom_to_floor_result.geometry])
    return join


def wheeled_base(
    joint_height: float = 0.0,
    leg_diameter: float = 0.0,
    top_height: float = 0.0,
    arc_sweep_angle: float = 240.0,
    wheel_width: float = 0.13,
    wheel_rotation: float = 0.5,
    pole_length: float = 1.8,
    leg_number: int = 5,
    material: pf.Material | None = None,
    wheel_material: pf.Material | None = None,
) -> ChairResult:
    """Five-star caster base with a central gas-lift pole (v1 wheeled leg)."""
    geo = _wheeled_base_geometry(
        joint_height=joint_height,
        leg_diameter=leg_diameter,
        top_height=top_height,
        arc_sweep_angle=arc_sweep_angle,
        wheel_width=wheel_width,
        wheel_rotation=wheel_rotation,
        pole_length=pole_length,
        leg_number=leg_number,
        material=material,
        wheel_material=wheel_material,
    )
    return ChairResult(mesh=pf.nodes.to_mesh_object(geo))


def office_chair_dimensions_rand(
    rng: pf.RNG,
    width: float | None = None,
    seat_elevation: float | None = None,
) -> pf.Vector:
    """Footprint + seat elevation. v1 derived elevation as total(1.0-1.4) minus
    shell(0.5-0.7), whose independent tails gave bar-height 0.9m seats; sample
    the real-world seat elevation directly instead."""
    rng, rng_w = rng.spawn(2)
    if width is None:
        width = pf.random.uniform(rng_w, 0.5, 0.6)
    if seat_elevation is None:
        seat_elevation = pf.random.uniform(rng, 0.42, 0.65)
    return (width, width, seat_elevation)


def curvy_seat_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    material: pf.Material | None = None,
) -> ChairResult:
    """Curved shell seat with an integrated backrest (v1 curvy_seats)."""
    rng, rng_dims, rng_mat = rng.spawn(3)
    if dimensions is None:
        dimensions = office_chair_dimensions_rand(rng_dims)
    x, y, z = dimensions
    geo = _curvy_seat_geometry(
        width=x,
        u_resolution=96,
        v_resolution=64,
        front_relative_width=pf.random.uniform(rng, 0.5, 0.8),
        front_bent=pf.random.uniform(rng, -1.5, -0.4),
        seat_bent=pf.random.uniform(rng, -1.5, -0.4),
        mid_bent=pf.random.uniform(rng, -2.4, -0.5),
        mid_relative_width=pf.random.uniform(rng, 0.5, 0.9),
        back_bent=pf.random.uniform(rng, -1.0, -0.1),
        back_relative_width=pf.random.uniform(rng, 0.6, 0.9),
        mid_pos=pf.random.uniform(rng, 0.4, 0.6),
        seat_height=pf.random.uniform(rng, 0.5, 0.7),
    )
    geo = pf.nodes.geo.transform(
        geo, translation=(0.0, 0.0, z), rotation=(0, 0, 1.5708), scale=(1, 1, 1)
    )
    obj = pf.nodes.to_mesh_object(geo)
    if material is None:
        material = furniture_material_rand(rng_mat, pf.nodes.shader.coord().uv)
    pf.ops.object.set_material(
        obj, surface=material.surface, displacement=material.displacement
    )
    pf.ops.uv.cube_project(obj, uv_name="UVMap")
    return ChairResult(mesh=obj)


def round_seat_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    material: pf.Material | None = None,
) -> ChairResult:
    """Backless round pad seat (v1 round_seats)."""
    rng, rng_dims, rng_mat = rng.spawn(3)
    if dimensions is None:
        dimensions = office_chair_dimensions_rand(rng_dims)
    x, y, z = dimensions
    thickness = pf.random.uniform(rng, 0.05, 0.12)
    geo = _round_seat_geometry(
        thickness=thickness,
        radius=pf.random.uniform(rng, 0.35, 0.45) * (x / 0.55),
        cap_radius=pf.random.uniform(rng, 2.0, 3.2),
        bevel_factor=pf.random.uniform(rng, 0.01, 0.04),
    )
    geo = pf.nodes.geo.transform(
        geo, translation=(0.0, 0.0, z), rotation=(0, 0, 0), scale=(1, 1, 1)
    )
    obj = pf.nodes.to_mesh_object(geo)
    if material is None:
        material = furniture_material_rand(rng_mat, pf.nodes.shader.coord().uv)
    pf.ops.object.set_material(
        obj, surface=material.surface, displacement=material.displacement
    )
    pf.ops.uv.cube_project(obj, uv_name="UVMap")
    return ChairResult(mesh=obj)


def wheeled_base_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    material: pf.Material | None = None,
    wheel_material: pf.Material | None = None,
) -> ChairResult:
    """Five-star caster base with a central gas-lift pole (v1 wheeled leg)."""
    rng, rng_dims, rng_mat, rng_wheel_mat = rng.spawn(4)
    if dimensions is None:
        dimensions = office_chair_dimensions_rand(rng_dims)
    x, y, z = dimensions
    vec = pf.nodes.shader.coord().uv
    if material is None:
        material = furniture_material_rand(rng_mat, vec)
    if wheel_material is None:
        wheel_material = castor_wheel_material_rand(rng_wheel_mat, vec, material)
    geo = _wheeled_base_geometry(
        top_height=z,
        joint_height=pf.random.uniform(rng, 0.5, 0.8) * z,
        leg_diameter=pf.random.uniform(rng, 0.015, 0.065),
        arc_sweep_angle=pf.random.uniform(rng, 120.0, 240.0),
        wheel_width=pf.random.uniform(rng, 0.11, 0.15),
        wheel_rotation=pf.random.uniform(rng, 0.0, 360.0),
        pole_length=pf.random.uniform(rng, 1.6, 2.0),
        leg_number=pf.control.choice(rng, [(4, 0.5), (5, 0.5)]),
        material=material,
        wheel_material=wheel_material,
    )
    geo = pf.nodes.geo.transform(
        geo, translation=(0, 0, 0), rotation=(0, 0, 1.5708), scale=(1, 1, 1)
    )
    obj = pf.nodes.to_mesh_object(geo)
    pf.ops.uv.cube_project(obj, uv_name="UVMap")
    return ChairResult(mesh=obj)


@pf.nodes.node_function
def _chair_back_geometry(
    dimensions: t.SocketOrVal[pf.Vector],
    n_slats: t.SocketOrVal[int],
    slat_width: t.SocketOrVal[float],
    slat_depth: t.SocketOrVal[float],
    crest_height: t.SocketOrVal[float],
    crest_offset: t.SocketOrVal[pf.Vector],
    crest_margin: t.SocketOrVal[float],
    crest_loops: t.SocketOrVal[int] = 0,
    slat_span: t.SocketOrVal[float] = 1.0,
    slat_material: pf.ProcNode[pf.Material] | None = None,
    crest_material: pf.ProcNode[pf.Material] | None = None,
) -> pf.ProcNode[pf.MeshObject]:
    """Flat slatted back panel: vertical slats (swept profile curves) arrayed
    along a straight line in Y, capped by a corner-box top rail. dimensions =
    (depth, width, height); built straight, bow deferred to the seat. slat_span
    scales the slat row inward from the full width (1 = outermost slats at the
    edges). Reusable as a bed headboard."""
    slat_height = dimensions.z - crest_height
    slat_line = pf.nodes.geo.curve_line(
        start=(0, 0, 0), end=pf.nodes.math.combine_xyz(z=slat_height)
    )
    slat_line = pf.nodes.geo.resample_curve_count(curve=slat_line, count=17)
    profile = pf.nodes.geo.curve_circle(resolution=12, radius=0.5)
    profile = pf.nodes.geo.transform(
        geometry=profile,
        translation=(0, 0, 0),
        rotation=(0, 0, 0),
        scale=pf.nodes.math.combine_xyz(x=slat_depth, y=slat_width, z=1.0),
    )
    slat = curve_to_mesh_with_uv(curve=slat_line, profile=profile, fill_caps=True).mesh

    half = ((dimensions.y - slat_width) * 0.5 - crest_offset.y) * slat_span
    forward = (dimensions.x - slat_depth) * 0.5
    span = pf.nodes.geo.curve_line(
        start=pf.nodes.math.combine_xyz(x=forward, y=half * -1.0),
        end=pf.nodes.math.combine_xyz(x=forward, y=half),
    )
    points = pf.nodes.geo.resample_curve_count(curve=span, count=n_slats)
    points = pf.nodes.geo.curve_to_mesh(curve=points)
    slats = pf.nodes.geo.instance_on_points(points=points, instance=slat)
    slats = pf.nodes.geo.realize_instances(slats)

    crest_width = dimensions.y + crest_margin * 2.0
    crest = mesh.corner_box(
        size=pf.nodes.math.combine_xyz(x=dimensions.x, y=crest_width, z=crest_height),
        loops_y=crest_loops,
        support_loop_offset=crest_offset,
    ).mesh
    crest = pf.nodes.geo.transform(
        geometry=crest,
        translation=pf.nodes.math.combine_xyz(z=dimensions.z - crest_height * 0.5),
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    crest_uv = pf.nodes.geo.input_named_attribute(
        "uv_map", data_type="FLOAT_VECTOR"
    ).attribute
    crest = pf.nodes.geo.store_named_attribute(
        geometry=crest,
        name="UVMap",
        value=crest_uv,
        domain="CORNER",
        data_type="FLOAT2",
    )

    if slat_material is not None:
        slats = pf.nodes.geo.set_material(slats, slat_material)
    if crest_material is not None:
        crest = pf.nodes.geo.set_material(crest, crest_material)

    geometry = pf.nodes.geo.join_geometry([slats, crest])
    return pf.nodes.geo.set_shade_smooth(geometry=geometry, shade_smooth=True)


def chair_back(
    dimensions: pf.Vector,
    n_slats: int,
    slat_width: float,
    slat_depth: float,
    crest_height: float,
    crest_offset: pf.Vector,
    crest_margin: float,
    crest_loops: int = 0,
    slat_span: float = 1.0,
) -> ChairResult:
    """Flat slatted back panel: vertical slats arrayed along a straight line in
    Y, capped by a corner-box top rail. Reusable as a bed headboard."""
    geo = _chair_back_geometry(
        dimensions=dimensions,
        n_slats=n_slats,
        slat_width=slat_width,
        slat_depth=slat_depth,
        crest_height=crest_height,
        crest_offset=crest_offset,
        crest_margin=crest_margin,
        crest_loops=crest_loops,
        slat_span=slat_span,
    )
    obj = pf.nodes.to_mesh_object(geo)
    return ChairResult(mesh=obj)


def chair_back_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    material: pf.Material | None = None,
) -> ChairResult:
    """Slatted wooden back. `aspect` runs the slat cross-section from flat (wide
    in Y) to near-square; the support-loop offset then rounds it toward a spindle
    once the subdivision modifier is applied."""
    rng, rng_dims, rng_mat = rng.spawn(3)
    if dimensions is None:
        dimensions = (
            pf.random.uniform(rng, 0.02, 0.04),
            pf.random.uniform(rng, 0.4, 0.52),
            pf.random.uniform(rng, 0.35, 0.6),
        )
    depth = dimensions[0]

    aspect = pf.random.clip_gaussian(rng, 0.9, 0.1, 0.5, 1.0)
    slat_depth = depth * pf.random.uniform(rng, 0.5, 1.0)
    slat_width = slat_depth / aspect

    crest_height = pf.random.uniform(rng, 0.05, 0.1)
    corner_r = pf.random.uniform(rng, 0.0, 0.49) * crest_height
    depth_r = pf.random.uniform(rng, 0.2, 0.49) * depth

    two_span = pf.random.clip_gaussian(rng, 0.85, 0.2, 0.4, 1.0)
    slats_choice = pf.control.choice(
        rng,
        [
            ((pf.random.randint(rng, 4, 10), 1.0), 3.0),
            ((2, two_span), 1.0),
        ],
    )
    n_slats = slats_choice[0]
    slat_span = slats_choice[1]
    obj = chair_back(
        dimensions=dimensions,
        n_slats=n_slats,
        slat_span=slat_span,
        slat_width=slat_width,
        slat_depth=slat_depth,
        crest_height=crest_height,
        crest_offset=(depth_r, corner_r, corner_r),
        crest_margin=0.0,
    ).mesh
    if material is None:
        material = furniture_material_rand(rng_mat, pf.nodes.shader.coord().uv)
    pf.ops.object.set_material(
        obj, surface=material.surface, displacement=material.displacement
    )
    return ChairResult(mesh=obj)


@pf.nodes.node_function
def chair_back_solid(
    dimensions: t.SocketOrVal[pf.Vector],
    top_rise: t.SocketOrVal[float] = 0.0,
    n_points: t.SocketOrVal[int] = 24,
) -> pf.ProcNode[pf.MeshObject]:
    """Solid back: panel filled between a flat bottom edge and a top edge that
    bows up by top_rise (0 = square), then extruded for depth. The many fill
    points give it geometry to bend. dimensions = (depth, width, height)."""
    hw = dimensions.y * 0.5
    bottom = pf.nodes.geo.curve_line(
        start=pf.nodes.math.combine_xyz(y=hw * -1.0),
        end=pf.nodes.math.combine_xyz(y=hw),
    )
    top = pf.nodes.geo.curve_bezier_segment(
        start=pf.nodes.math.combine_xyz(y=hw * -1.0, z=dimensions.z),
        start_handle=pf.nodes.math.combine_xyz(y=hw * -0.33, z=dimensions.z + top_rise),
        end_handle=pf.nodes.math.combine_xyz(y=hw * 0.33, z=dimensions.z + top_rise),
        end=pf.nodes.math.combine_xyz(y=hw, z=dimensions.z),
        resolution=16,
    )
    panel = mesh.fill_between_curves(
        curve_left=bottom, curve_right=top, n_points=n_points, n_rows=17
    )
    solid = pf.nodes.geo.extrude_mesh(
        mesh=panel, offset=pf.nodes.math.combine_xyz(x=dimensions.x), individual=False
    )
    geo = pf.nodes.geo.join_geometry([panel, pf.nodes.geo.flip_faces(solid.mesh)])
    geo = pf.nodes.geo.merge_by_distance(geo, distance=1e-5)
    geo = pf.nodes.geo.transform(
        geometry=geo,
        translation=pf.nodes.math.combine_xyz(x=dimensions.x * -0.5),
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    return pf.nodes.geo.set_shade_smooth(geo, shade_smooth=False)


@pf.nodes.node_function
def _seat_edge(
    x: t.SocketOrVal[float],
    half_width: t.SocketOrVal[float],
    bow: t.SocketOrVal[float],
    dip: t.SocketOrVal[float] = 0.0,
) -> pf.ProcNode[pf.CurveObject]:
    return pf.nodes.geo.curve_bezier_segment(
        start=pf.nodes.math.combine_xyz(x=x, y=half_width * -1.0),
        start_handle=pf.nodes.math.combine_xyz(
            x=x + bow, y=half_width * -0.33, z=dip * -1.0
        ),
        end_handle=pf.nodes.math.combine_xyz(
            x=x + bow, y=half_width * 0.33, z=dip * -1.0
        ),
        end=pf.nodes.math.combine_xyz(x=x, y=half_width),
        resolution=16,
    )


class DiningSeatResult(NamedTuple):
    seat: pf.ProcNode[pf.MeshObject]
    back_edge: pf.ProcNode[pf.CurveObject]


@pf.nodes.node_function
def dining_seat(
    depth: t.SocketOrVal[float],
    half_width: t.SocketOrVal[float],
    thickness: t.SocketOrVal[float],
    front_bow: t.SocketOrVal[float] = 0.0,
    back_bow: t.SocketOrVal[float] = 0.0,
    front_dip: t.SocketOrVal[float] = 0.0,
    n_points: t.SocketOrVal[int] = 16,
    material: pf.ProcNode[pf.Material] | None = None,
) -> DiningSeatResult:
    """Seat pan bridged between a rear edge and a front edge and extruded down by
    thickness. Both edges are bezier curves spanning the width at x = -/+depth/2;
    front_bow/back_bow bow each edge forward in x, front_dip scoops the front edge
    center down. Returns the pan mesh plus the rear edge curve (so the back can be
    bent to match it)."""
    back_edge = _seat_edge(x=depth * -0.5, half_width=half_width, bow=back_bow * -1.0)
    front_edge = _seat_edge(
        x=depth * 0.5, half_width=half_width, bow=front_bow, dip=front_dip
    )
    top = mesh.fill_between_curves(
        curve_left=back_edge, curve_right=front_edge, n_points=n_points, n_rows=9
    )
    solid = pf.nodes.geo.extrude_mesh(
        mesh=top, offset=pf.nodes.math.combine_xyz(z=thickness * -1.0), individual=False
    )
    # downward extrude leaves bottom and walls wound inward; bevel offsets invert
    flipped = pf.nodes.geo.flip_faces(solid.mesh)
    geo = pf.nodes.geo.join_geometry([top, flipped])
    geo = pf.nodes.geo.merge_by_distance(geo, distance=1e-5)
    if material is not None:
        geo = pf.nodes.geo.set_material(geo, material)
    return DiningSeatResult(geo, back_edge)


@pf.nodes.node_function
def _bend_to_edge(
    geometry: pf.ProcNode,
    edge: t.SocketOrVal[pf.CurveObject],
    half_width: t.SocketOrVal[float],
) -> pf.ProcNode[pf.MeshObject]:
    """Bend a flat panel in x to follow the seat's rear edge: offset each vertex
    in x by the edge curve's x at the matching y."""
    pos = pf.nodes.geo.input_position()
    factor = pf.nodes.math.map_range(
        value=pos.y,
        from_min=half_width * -1.0,
        from_max=half_width,
        to_min=0.0,
        to_max=1.0,
    )
    sampled = pf.nodes.geo.sample_curve(curves=edge, factor=factor)
    return pf.nodes.geo.set_position(
        geometry=geometry, offset=pf.nodes.math.combine_xyz(x=sampled.position.x)
    )


@pf.nodes.node_function
def _shear_back(
    geometry: pf.ProcNode,
    slant: t.SocketOrVal[float],
    base_z: t.SocketOrVal[float],
    height: t.SocketOrVal[float] = 0.45,
    bend: t.SocketOrVal[float] = 0.0,
) -> pf.ProcNode[pf.MeshObject]:
    """Lean a back by warping x along a bezier from the base to the top lean
    (slant * height). bend displaces both handles in x: 0 = straight shear,
    same sign as the lean = convex (bows into the lean), opposite = concave."""
    top_x = slant * height
    spline = pf.nodes.geo.curve_bezier_segment(
        start=(0.0, 0.0, 0.0),
        start_handle=pf.nodes.math.combine_xyz(x=top_x * 0.33 + bend, z=height * 0.33),
        end_handle=pf.nodes.math.combine_xyz(x=top_x * 0.67 + bend, z=height * 0.67),
        end=pf.nodes.math.combine_xyz(x=top_x, z=height),
        resolution=32,
    )
    pos = pf.nodes.geo.input_position()
    factor = pf.nodes.math.map_range(
        value=pos.z, from_min=base_z, from_max=base_z + height, to_min=0.0, to_max=1.0
    )
    sampled = pf.nodes.geo.sample_curve(curves=spline, factor=factor)
    return pf.nodes.geo.set_position(
        geometry=geometry,
        offset=pf.nodes.math.combine_xyz(x=sampled.position.x),
    )


def dining_chair_dimensions_rand(rng: pf.RNG) -> pf.Vector:
    """Real dining-chair footprint (m): seat depth 0.41-0.46, width 0.40-0.50,
    seat height 0.45-0.50 (standard 45-50cm, tables 71-76cm)."""
    return (
        pf.random.uniform(rng, 0.41, 0.46),
        pf.random.uniform(rng, 0.40, 0.50),
        pf.random.uniform(rng, 0.45, 0.50),
    )


def _dining_slat_back(
    rng: pf.RNG,
    width: float,
    height: float,
    detail_material: pf.Material,
    slat_material: pf.Material,
) -> tuple[pf.ProcNode, float]:
    """Slatted back; slat count follows width so thin spindles and broad slats
    both space out, or occasionally just two stiles biased toward the outside.
    Slats take slat_material, the crest rail the backrest detail_material."""
    slat_width = pf.random.uniform(rng, 0.0225, 0.061875)
    aspect = pf.random.clip_gaussian(rng, 0.8, 0.15, 0.4, 1.0)
    slat_depth = slat_width * aspect
    gap = pf.random.uniform(rng, 1.4, 2.4)
    n_wide = pf.nodes.math.clamp(
        pf.nodes.math.round(width / (slat_width * gap)), 4.0, 9.0
    )
    two_span = pf.random.clip_gaussian(rng, 0.85, 0.2, 0.4, 1.0)
    slats_choice = pf.control.choice(
        rng,
        [
            ((n_wide, 1.0), 3.0),
            ((2.0, two_span), 1.0),
        ],
    )
    n_slats = slats_choice[0]
    slat_span = slats_choice[1]
    crest_height = pf.random.clip_gaussian(rng, 0.0675, 0.035, 0.0375, 0.15)
    corner_r = pf.random.uniform(rng, 0.0, 0.49) * crest_height
    return _chair_back_geometry(
        dimensions=pf.nodes.math.combine_xyz(x=slat_depth, y=width, z=height),
        n_slats=n_slats,
        slat_span=slat_span,
        slat_width=slat_width,
        slat_depth=slat_depth,
        crest_height=crest_height,
        crest_offset=(
            pf.random.uniform(rng, 0.2, 0.49) * slat_depth,
            corner_r,
            corner_r,
        ),
        crest_margin=0.0,
        crest_loops=16,
        slat_material=slat_material,
        crest_material=detail_material,
    ), 0.001


def _dining_solid_back(
    rng: pf.RNG,
    width: float,
    height: float,
    detail_material: pf.Material,
    slat_material: pf.Material,
) -> tuple[pf.ProcNode, float]:
    """Solid filled back with a flat bottom and an up-bowed (or flat) top, with
    a v1-style generous round-over (bevel up to half depth). The whole panel is
    backrest, so it takes detail_material; slat_material is unused."""
    depth = pf.random.clip_gaussian(rng, 0.05, 0.015, 0.03, 0.10)
    geo = chair_back_solid(
        dimensions=pf.nodes.math.combine_xyz(x=depth, y=width, z=height),
        top_rise=pf.random.clip_gaussian(rng, 0.03, 0.04, 0.0, 0.15),
    )
    round_width = pf.random.uniform(rng, 0.1, 0.45) * depth
    return pf.nodes.geo.set_material(geo, detail_material), round_width


def _dining_seat_with_back(
    rng: pf.RNG,
    dimensions: pf.Vector,
    thickness: float,
    front_bow: float,
    back_bow: float,
    front_dip: float,
    seat_material: pf.Material,
    back: pf.ProcNode,
    back_round: float,
    back_height: float,
    slant: float,
    back_bend: float,
) -> pf.MeshObject:
    """The seat: a pan with the given back seated on its rear edge. Builds the pan,
    sets the back's rear face on the pan's rear edge, bends the back to follow that
    edge and leans it, then bevels pan and back (separate amounts) and merges them
    into one uv-mapped object. dimensions = (depth, width, elevation)."""
    x, y, z = dimensions
    half_width = y * 0.5
    seat_res = dining_seat(
        depth=x,
        half_width=half_width,
        thickness=thickness,
        front_bow=front_bow,
        back_bow=back_bow,
        front_dip=front_dip,
        material=seat_material,
    )
    seat = pf.nodes.geo.transform(
        seat_res.seat, translation=(0, 0, z), rotation=(0, 0, 0), scale=(1, 1, 1)
    )
    bb = pf.nodes.geo.bound_box(back)
    # seat the back's rear face on the seat rear edge so slats don't overhang behind
    forward = bb.min.x * -1.0
    back = pf.nodes.geo.transform(
        back,
        translation=pf.nodes.math.combine_xyz(x=forward, z=z),
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    back = _bend_to_edge(back, edge=seat_res.back_edge, half_width=half_width)
    back = _shear_back(
        back, slant=slant * -1.0, base_z=z, height=back_height, bend=back_bend
    )

    obj = pf.nodes.to_mesh_object(seat)
    seat_round = pf.random.uniform(rng, 0.05, 0.49) * thickness
    pf.ops.modifier.bevel(obj, width=seat_round, segments=4)
    back_obj = pf.nodes.to_mesh_object(back)
    pf.ops.modifier.bevel(back_obj, width=back_round, segments=4)
    pf.ops.object.join(obj, back_obj)
    pf.ops.uv.cube_project(obj, uv_name="UVMap")
    return obj


def dining_chair_rand(rng: pf.RNG, dimensions: pf.Vector | None = None) -> ChairResult:
    """Wooden dining chair: bezier-outline seat pan + slat or solid back bent to
    follow the seat's rear edge, on straight legs. Real chair dimensions."""
    (
        rng,
        rng_dims,
        rng_back_sel,
        rng_back,
        rng_base_sel,
        rng_base,
        rng_mat1,
        rng_mat2,
        rng_fabric,
        rng_seat_sel,
        rng_backrest_sel,
        rng_leg_sel,
        rng_slat_sel,
        rng_dip_sel,
        rng_dip,
        rng_back_bend,
        rng_seat,
    ) = rng.spawn(17)
    if dimensions is None:
        dimensions = dining_chair_dimensions_rand(rng_dims)
    x, y, z = dimensions

    vec = pf.nodes.shader.coord().uv
    material1 = furniture_material_rand(rng_mat1, vec)
    material2 = furniture_material_rand(rng_mat2, vec)
    fabric = furniture_fabric(rng_fabric, vec, translucency=0.0)
    seat_material = pf.control.choice(rng_seat_sel, [(material1, 2.0), (fabric, 1.0)])
    backrest_material = pf.control.choice(
        rng_backrest_sel, [(material1, 2.0), (fabric, 1.0)]
    )
    leg_material = pf.control.choice(rng_leg_sel, [(material1, 1.0), (material2, 1.0)])
    slat_material = pf.control.choice(
        rng_slat_sel, [(material1, 1.0), (material2, 1.0)]
    )

    front_bow = pf.random.clip_gaussian(rng, 0.08, 0.1, 0.0, 0.3) * x
    back_bow = pf.random.uniform(rng, 0.0, 0.25) * x
    dip_active = pf.control.choice(rng_dip_sel, [(0.0, 0.5), (1.0, 0.5)])
    front_dip = dip_active * pf.random.uniform(rng_dip, 0.0, 0.15) * x
    thickness = pf.random.clip_gaussian(rng, 0.06, 0.04, 0.02, 0.2)

    back_height = pf.random.uniform(rng, 0.35, 0.55)
    back_fn = pf.control.choice(
        rng_back_sel, [(_dining_slat_back, 0.6), (_dining_solid_back, 0.4)]
    )
    back_res = back_fn(rng_back, y, back_height, backrest_material, slat_material)
    back = back_res[0]
    back_round = back_res[1]
    slant = pf.random.clip_gaussian(rng, 0.23, 0.067, 0.0, 0.4)
    bend_frac = pf.random.clip_gaussian(rng_back_bend, 0.15, 0.35, -0.4, 1.2)

    seat = _dining_seat_with_back(
        rng_seat,
        dimensions,
        thickness,
        front_bow,
        back_bow,
        front_dip,
        seat_material,
        back,
        back_round,
        back_height,
        slant=slant,
        back_bend=bend_frac * 0.15 * back_height * -1.0,
    )

    leg_spread = pf.random.uniform(rng, 0.8, 0.95)
    seat_bottom = z - thickness * 0.5
    # dipped pan center top sits 0.375*dip below z, center poles must stop below it
    center_bottom = seat_bottom - 0.375 * front_dip

    def wheeled_fn(
        rng: pf.RNG, dimensions: pf.Vector, material: pf.Material
    ) -> ChairResult:
        return wheeled_base_rand(
            rng, (dimensions[0], dimensions[1], center_bottom), material
        )

    base_fn = pf.control.choice(
        rng_base_sel,
        [
            (base_straight_rand, 3.0),
            (base_square_rand, 1.0),
            (wheeled_fn, 1.0),
        ],
    )
    base_dimensions = (x * leg_spread, y * leg_spread, seat_bottom)
    base = base_fn(rng_base, base_dimensions, leg_material).mesh

    pf.ops.object.join(seat, base)
    return ChairResult(mesh=seat)


def chair_rand(rng: pf.RNG, dimensions: pf.Vector | None = None) -> ChairResult:
    """Any chair: randomly an office chair (curvy/round seat on a wheeled/pedestal/
    leg base) or a wooden dining chair (bezier seat + slat back)."""
    rng, rng_sel, rng_gen = rng.spawn(3)
    chair_fn = pf.control.choice(
        rng_sel, [(office_chair_rand, 1.0), (dining_chair_rand, 1.0)]
    )
    return chair_fn(rng=rng_gen, dimensions=dimensions)


def office_chair_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    seat_material: pf.Material | None = None,
    base_material: pf.Material | None = None,
) -> ChairResult:
    (
        rng,
        rng_dims,
        rng_top_sel,
        rng_top,
        rng_base_sel,
        rng_base,
        rng_seat_mat,
        rng_base_mat,
        rng_bevel,
    ) = rng.spawn(9)
    if dimensions is None:
        dimensions = office_chair_dimensions_rand(rng_dims)

    top_fn = pf.control.choice(
        rng_top_sel,
        [
            (curvy_seat_rand, 1.5),
            (round_seat_rand, 1.0),
        ],
    )
    top = top_fn(rng=rng_top, dimensions=dimensions).mesh

    x, y, z = dimensions
    # Tuck the splayed legs under the seat and poke them 1cm up to connect.
    leg_spread = pf.random.uniform(rng, 0.5, 0.7)
    base_dimensions = (x * leg_spread, y * leg_spread, z + 0.01)

    vec = pf.nodes.shader.coord().uv
    if seat_material is None:
        seat_material = furniture_fabric(rng_seat_mat, vec, translucency=0.0)
    if base_material is None:
        base_material = furniture_material_rand(rng_base_mat, vec)

    base_fn = pf.control.choice(
        rng_base_sel,
        [
            (base_straight_rand, 3.0),
            (wheeled_base_rand, 1.0),
            (base_square_rand, 1.0),
        ],
    )
    base = base_fn(rng_base, base_dimensions, base_material).mesh

    pf.ops.object.set_material(
        top, surface=seat_material.surface, displacement=seat_material.displacement
    )

    pf.ops.object.join(top, base)
    pf.ops.modifier.bevel(
        top, width=pf.random.uniform(rng_bevel, 0.001, 0.005), segments=2
    )
    return ChairResult(mesh=top)
