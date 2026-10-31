# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Yiming Zuo: original Infinigen chair (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/objects/seating/chairs/office_chair.py; https://github.com/princeton-vl/infinigen/tree/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/objects/seating/chairs/seats; https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/objects/tables/legs/wheeled.py)
# - Lingjie Mei: original Infinigen chair (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/objects/seating/chairs/chair.py)
# - Alexander Raistrick: refactor for Infinigen2

from collections.abc import Callable
from functools import cache, partial
from typing import NamedTuple

import procfunc as pf
from procfunc.nodes import types as t
from procfunc.nodes.util.bpy_node_info import NodeDataType

from infinigen2.objects import storage
from infinigen2.objects.furniture_bases import (
    TableResult,
    base_square_rand,
    base_straight_rand,
)
from infinigen2.shaders.functionality_lists import (
    castor_wheel_material_rand,
    fabric_sturdy_rand,
    furniture_material_rand,
    furniture_surface_material_rand,
)
from infinigen2.util import mesh
from infinigen2.util.curve import curve_to_mesh_with_uv

__all__ = [
    "ChairResult",
    "DiningSeatResult",
    "chair_back",
    "chair_back_rand",
    "chair_back_solid",
    "chair_base_stable_rand",
    "chair_base_wheeled",
    "chair_base_wheeled_rand",
    "chair_bench_dimensions_rand",
    "chair_bench_rand",
    "chair_dining_dimensions_rand",
    "chair_dining_rand",
    "chair_dining_seat_slat_back",
    "chair_dining_seat_solid_back",
    "chair_office_dimensions_rand",
    "chair_office_rand",
    "chair_rand",
    "chair_seat_curvy",
    "chair_seat_curvy_rand",
    "chair_seat_dining",
    "chair_seat_round",
    "chair_seat_round_rand",
]


class ChairResult(NamedTuple):
    mesh: pf.MeshObject


# every part carries this, so whichever one a join keeps covers the whole assembly
_CHAIR_SUBDIV = 4

# a 24-vert sweep falls under this and stays smooth; a 4-vert one is 90deg and stays square
_CREASE_ANGLE = 20.0


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
    transform = pf.nodes.geo.transform(geometry=curve_circle, scale=(0.005, 0.005, 1.0))
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
    join_1 = pf.nodes.geo.join_geometry([bent_result_3, bent_result_4])
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


def _finish_part(obj: pf.MeshObject, material: pf.Material | None) -> None:
    if material is not None:
        pf.ops.object.set_material(
            obj, surface=material.surface, displacement=material.displacement
        )


@pf.tracer.generator
def chair_seat_curvy(
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
    elevation: float = 0.0,
    rotation_z: float = 0.0,
    material: pf.Material | None = None,
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
    geo = pf.nodes.geo.transform(
        geo,
        translation=(0.0, 0.0, elevation),
        rotation=(0, 0, rotation_z),
        scale=(1, 1, 1),
    )
    obj = pf.nodes.to_mesh_object(geo)
    _finish_part(obj, material)
    return ChairResult(mesh=obj)


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
        count=4,
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
    curve_to = curve_to_mesh_with_uv(
        curve=capture_attribute.geometry,
        profile=resample_curve_count,
        fill_caps=True,
    ).mesh
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
    mesh_position = pf.nodes.math.combine_xyz(
        x=input_position.x * mesh_position_x,
        y=input_position.y * mesh_position_x,
        z=input_position.z,
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
        profile_resolution=12,
        resolution=4,
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
        curve=transform_4, radius=fillet_radius_vertical, count=4, limit_radius=True
    )
    transform_5 = pf.nodes.geo.transform(
        geometry=fillet_curve_poly,
        rotation=(1.5708, 1.5708, 0.0),
        scale=thickness.astype(dtype=pf.Vector),
    )
    curve_to = curve_to_mesh_with_uv(
        curve=n_gon_cylinder_result.profile_curve, profile=transform_5
    ).mesh
    transform_6_translation = pf.nodes.math.combine_xyz(z=thickness * -0.5)
    transform_6 = pf.nodes.geo.transform(
        geometry=curve_to, translation=transform_6_translation
    )
    flip_faces = pf.nodes.geo.flip_faces(transform_6)
    bottom_fill = pf.nodes.geo.fill_curve(
        curve=n_gon_cylinder_result.profile_curve, mode="NGONS"
    )
    bottom_translation = pf.nodes.math.combine_xyz(z=thickness * -1.0)
    bottom_cap = pf.nodes.geo.transform(
        geometry=pf.nodes.geo.flip_faces(bottom_fill), translation=bottom_translation
    )
    join = pf.nodes.geo.join_geometry([flip_faces, bottom_cap])
    geometry_translation = pf.nodes.math.combine_xyz(z=thickness)
    transform = pf.nodes.geo.transform(geometry=join, translation=geometry_translation)
    return _RoundSeatTableTopResult(transform, n_gon_cylinder_result.profile_curve)


# Radii tuned so the subdivided cap lands back on the sphere it is sampled from.
_ROUND_SEAT_CAP_RINGS = (0.66, 0.24)


@pf.nodes.node_function
def _round_seat_create_cap(
    rim_curve: pf.ProcNode,
    radius: t.SocketOrVal[float] = 1.0,
    height_scale: t.SocketOrVal[float] = 1.0,
    z_offset: t.SocketOrVal[float] = 0.0,
) -> pf.ProcNode[pf.MeshObject]:
    rim_z = pf.nodes.math.sqrt(radius**2.0 - 1.0)
    mid_r, top_r = _ROUND_SEAT_CAP_RINGS
    mid_z = pf.nodes.math.sqrt(radius**2.0 - mid_r**2.0) - rim_z
    top_z = pf.nodes.math.sqrt(radius**2.0 - top_r**2.0) - rim_z
    mid_z = mid_z * height_scale + z_offset
    top_z = top_z * height_scale + z_offset
    fill_curve = pf.nodes.geo.fill_curve(curve=rim_curve, mode="NGONS")
    extrude_mid = pf.nodes.geo.extrude_mesh(
        mesh=fill_curve, offset=(0.0, 0.0, 1.0), offset_scale=mid_z, individual=False
    )
    scale_mid = pf.nodes.geo.scale_elements(
        geometry=extrude_mid.mesh, scale=mid_r, selection=extrude_mid.top
    )
    extrude_top = pf.nodes.geo.extrude_mesh(
        mesh=scale_mid,
        offset=(0.0, 0.0, 1.0),
        offset_scale=top_z - mid_z,
        individual=False,
        selection=extrude_mid.top,
    )
    scale_top = pf.nodes.geo.scale_elements(
        geometry=extrude_top.mesh, scale=top_r / mid_r, selection=extrude_top.top
    )
    cap = pf.nodes.geo.store_named_attribute(
        geometry=scale_top,
        name="TAG_support",
        value=True,
        domain="FACE",
        data_type=NodeDataType.BOOLEAN,
    )
    return mesh.metric_box_uv(cap)


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
    create_cap_1 = _round_seat_create_cap(
        rim_curve=generate_table_top_result.curve,
        radius=cap_flatness,
        height_scale=radius * 0.5 + cap_relative_scale,
        z_offset=cap_relative_z_offset,
    )
    transform_translation = pf.nodes.math.combine_xyz(z=generate_table_top_thickness)
    transform = pf.nodes.geo.transform(
        geometry=create_cap_1, translation=transform_translation
    )
    join = pf.nodes.geo.join_geometry([generate_table_top_result.geometry, transform])
    # Welds the cap rim onto the body ring it was filled from.
    return pf.nodes.geo.merge_by_distance(geometry=join, distance=0.0001)


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
        resolution=32,
    )


@pf.tracer.generator
def chair_seat_round(
    thickness: float = 0.1,
    radius: float = 0.37,
    cap_radius: float = 2.8,
    bevel_factor: float = 0.028,
    elevation: float = 0.0,
    material: pf.Material | None = None,
) -> ChairResult:
    """Backless round pad seat (v1 round_seats)."""
    geo = _round_seat_geometry(
        thickness=thickness,
        radius=radius,
        cap_radius=cap_radius,
        bevel_factor=bevel_factor,
    )
    geo = pf.nodes.geo.transform(
        geo,
        translation=(0.0, 0.0, elevation),
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    obj = pf.nodes.to_mesh_object(geo)
    _finish_part(obj, material)
    return ChairResult(mesh=obj)


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
        count=4,
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
    profile_resolution: t.SocketOrVal[int] = 12,
    resolution: t.SocketOrVal[int] = 32,
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
    curve_to = curve_to_mesh_with_uv(
        curve=capture_attribute.geometry,
        profile=resample_curve_count,
        fill_caps=True,
    ).mesh
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
    mesh_position = pf.nodes.math.combine_xyz(
        x=input_position.x * mesh_position_x,
        y=input_position.y * mesh_position_x,
        z=input_position.z,
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
        resolution=12,
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
        profile_resolution=12,
        resolution=4,
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
        curve=transform_4, radius=fillet_radius_vertical, count=4, limit_radius=True
    )
    fillet_endpoints = pf.nodes.geo.curve_endpoint_selection(start_size=1, end_size=1)
    fillet_profile = pf.nodes.geo.delete_geometry(
        geometry=fillet_curve_poly, selection=fillet_endpoints, domain="POINT"
    )
    transform_5 = pf.nodes.geo.transform(
        geometry=fillet_profile,
        rotation=(1.5708, 1.5708, 0.0),
        scale=thickness.astype(dtype=pf.Vector),
    )
    curve_to = curve_to_mesh_with_uv(
        curve=n_gon_cylinder_result.profile_curve, profile=transform_5
    ).mesh
    transform_6_translation = pf.nodes.math.combine_xyz(z=thickness * -0.5)
    transform_6 = pf.nodes.geo.transform(
        geometry=curve_to, translation=transform_6_translation
    )
    flip_faces = pf.nodes.geo.flip_faces(transform_6)
    join = pf.nodes.geo.join_geometry([store_named_attribute, flip_faces])
    geometry_translation = pf.nodes.math.combine_xyz(z=thickness)
    transform = pf.nodes.geo.transform(geometry=join, translation=geometry_translation)
    return _WheeledBaseTableTopResult(transform, n_gon_cylinder_result.profile_curve)


@pf.nodes.node_function
def _wheeled_base_create_cap(
    radius: t.SocketOrVal[float] = 1.0, resolution: t.SocketOrVal[int] = 64
) -> pf.ProcNode[pf.MeshObject]:
    uv_sphere_rings = radius * 4.0
    uv_sphere = pf.nodes.geo.mesh_uv_sphere(
        segments=resolution, rings=uv_sphere_rings.astype(dtype=int), radius=radius
    )
    transform_a = radius**2.0
    transform_translation_z = pf.nodes.math.sqrt(transform_a - 1.0)
    transform_translation = pf.nodes.math.combine_xyz(z=transform_translation_z * -1.0)
    transform = pf.nodes.geo.transform(
        geometry=uv_sphere.mesh, translation=transform_translation
    )
    input_position = pf.nodes.geo.input_position()
    result_0_selection = pf.nodes.func.less_than(a=input_position.z, b=0.0)
    delete = pf.nodes.geo.delete_geometry(
        geometry=transform, selection=result_0_selection
    )
    return mesh.metric_box_uv(delete)


@pf.nodes.node_function
def _wheeled_base_capped_cylinder(
    thickness: t.SocketOrVal[float] = 0.5,
    radius: t.SocketOrVal[float] = 0.2,
    cap_flatness: t.SocketOrVal[float] = 4.0,
    fillet_radius_vertical: t.SocketOrVal[float] = 0.4,
    cap_relative_scale: t.SocketOrVal[float] = 1.0,
    cap_relative_z_offset: t.SocketOrVal[float] = 0.0,
    resolution: t.SocketOrVal[int] = 12,
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
        resolution=8,
    )
    transform_1 = pf.nodes.geo.transform(
        geometry=n_gon_cylinder_result.mesh, rotation=(0.0, -1.5708, 0.0)
    )
    subdivision_surface = pf.nodes.geo.subdivision_surface(mesh=transform_1, level=0)
    if material is not None:
        subdivision_surface = pf.nodes.geo.set_material(subdivision_surface, material)
    transform_a_a = pf.nodes.math.constant(0.5)
    cylinder = pf.nodes.geo.mesh_cylinder(
        vertices=5,
        side_segments=1,
        fill_segments=1,
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
    transform_2 = mesh.metric_box_uv(transform_2)
    arc_top_result = _wheeled_base_arc_top(
        diameter=transform_a_a + 0.08, sweep_angle=arc_sweep_angle
    )
    curve_quadrilateral = pf.nodes.geo.curve_quadrilateral(
        width=wheel_width * 2.0, height=0.02
    )
    fillet_curve_poly = pf.nodes.geo.fillet_curve_poly(
        curve=curve_quadrilateral, radius=0.03, count=2, limit_radius=True
    )
    fork_index = pf.nodes.geo.input_index()
    fork_coincident = pf.nodes.math.modulo(fork_index.astype(dtype=float), 6.0) == 5.0
    fork_profile = pf.nodes.geo.delete_geometry(
        geometry=fillet_curve_poly, selection=fork_coincident, domain="POINT"
    )
    curve_to = curve_to_mesh_with_uv(
        curve=arc_top_result, profile=fork_profile, fill_caps=True
    ).mesh
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
def _wheeled_base_create_legs_and_stretchers(
    anchors: pf.ProcNode,
    leg_instance: pf.ProcNode,
    stretcher_instance: pf.ProcNode,
    keep_legs: t.SocketOrVal[bool] = False,
    table_height: t.SocketOrVal[float] = 0.0,
    leg_bottom_relative_scale: t.SocketOrVal[float] = 0.0,
    leg_bottom_relative_rotation: t.SocketOrVal[float] = 0.0,
    keep_odd_stretchers: t.SocketOrVal[bool] = True,
    keep_even_stretchers: t.SocketOrVal[bool] = True,
    stretcher_index_increment: t.SocketOrVal[int] = 0,
    stretcher_relative_position: t.SocketOrVal[float] = 0.5,
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
        stretcher_relative_position * -1.0
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
        a=instance_2.astype(dtype=bool), b=keep_odd_stretchers
    )
    instance_a_b_b = pf.nodes.func.boolean_not(instance_2.astype(dtype=bool))
    instance_a_b = pf.nodes.func.boolean_and(a=keep_even_stretchers, b=instance_a_b_b)
    instance_a = pf.nodes.func.boolean_or(a=instance_a_a, b=instance_a_b)
    attribute_domain_size = pf.nodes.geo.attribute_domain_size(
        geometry=transform, component="POINTCLOUD"
    )
    instance_b_switch = attribute_domain_size.point_count.astype(
        dtype=float
    ) / stretcher_index_increment.astype(dtype=float)
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
        input_index.astype(dtype=float) + stretcher_index_increment.astype(dtype=float)
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
        instance=stretcher_instance,
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
        vertices=12, radius=cylinder_1_radius - 0.0025, depth=cylinder_depth
    )
    transform_translation_z = 0.5 * cylinder_depth
    transform_translation = pf.nodes.math.combine_xyz(
        z=top_height - transform_translation_z
    )
    transform = pf.nodes.geo.transform(
        geometry=cylinder.mesh, translation=transform_translation
    )
    transform = mesh.metric_box_uv(transform)
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
    create_legs_and_stretchers_result = _wheeled_base_create_legs_and_stretchers(
        anchors=create_anchors_result,
        keep_legs=True,
        leg_instance=transform_1,
        table_height=0.025,
        leg_bottom_relative_scale=2.0 / create_anchors_profile_width,
        stretcher_instance=pf.nodes.geo.points(position=(0, 0, 0)),
        stretcher_index_increment=1,
        stretcher_relative_position=1.0,
        leg_bottom_offset=0.025,
        align_leg_x_rot=True,
    )
    align_bottom_to_floor_result = _wheeled_base_align_bottom_to_floor(
        geometry=create_legs_and_stretchers_result
    )
    cylinder_1_depth = joint_height - align_bottom_to_floor_result.offset
    cylinder_1 = pf.nodes.geo.mesh_cylinder(
        vertices=12, radius=cylinder_1_radius, depth=cylinder_1_depth
    )
    transform_a = cylinder_1_depth * 0.5
    transform_2_translation = pf.nodes.math.combine_xyz(
        z=transform_a + align_bottom_to_floor_result.offset
    )
    transform_2 = pf.nodes.geo.transform(
        geometry=cylinder_1.mesh, translation=transform_2_translation
    )
    transform_2 = mesh.metric_box_uv(transform_2)
    column = pf.nodes.geo.join_geometry([transform, transform_2])
    if material is not None:
        column = pf.nodes.geo.set_material(column, material)
    join = pf.nodes.geo.join_geometry([column, align_bottom_to_floor_result.geometry])
    return join


@pf.tracer.generator
def chair_base_wheeled(
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
    rotation_z: float = 0.0,
    crease_degrees: float | None = None,
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
    geo = pf.nodes.geo.transform(
        geo, translation=(0, 0, 0), rotation=(0, 0, rotation_z), scale=(1, 1, 1)
    )
    if crease_degrees is not None:
        geo = mesh.crease_sharp(geo, threshold_degrees=crease_degrees)
    obj = pf.nodes.to_mesh_object(geo)
    return ChairResult(mesh=obj)


def chair_office_dimensions_rand(
    rng: pf.RNG,
    width: float | None = None,
    seat_elevation: float | None = None,
) -> pf.Vector:
    """Footprint and ordinary adjustable office-chair seat elevation."""
    rng, rng_w = rng.spawn(2)
    if width is None:
        width = pf.random.uniform(rng_w, 0.5, 0.6)
    if seat_elevation is None:
        seat_elevation = pf.random.uniform(rng, 0.42, 0.55)
    return (width, width, seat_elevation)


def _curvy_bend_amount(
    wrap_angle: float, profile_z: float, width: float, relative_width: float
) -> float:
    """`_bent` rotates each profile vertex by |p| * x * amount, so a fixed amount
    bows a wide profile far harder than a narrow one and folds it back on itself
    past ~0.85 rad. Convert a target edge wrap angle instead, keeping the bow
    proportional to the profile's own half-width whatever the taper does."""
    return wrap_angle * -2.0 / (profile_z * width * relative_width)


def chair_seat_curvy_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    material: pf.Material | None = None,
) -> ChairResult:
    """Curved shell seat with an integrated backrest (v1 curvy_seats)."""
    rng, rng_dims, rng_mat = rng.spawn(3)
    if dimensions is None:
        dimensions = chair_office_dimensions_rand(rng_dims)
    back_width = pf.random.uniform(rng, 0.75, 1.05)
    top_width = back_width - pf.random.uniform(rng, 0.0, 0.5)
    mid_width = back_width * pf.random.uniform(rng, 0.7, 1.0)
    front_relative_width = pf.random.uniform(rng, 0.5, 0.8)
    front_bent = pf.random.uniform(rng, -1.5, -0.4)
    seat_bent = pf.random.uniform(rng, -1.5, -0.4)
    mid_bent = _curvy_bend_amount(
        pf.random.uniform(rng, 0.07, 0.48), 1.0, dimensions[0], mid_width
    )
    back_bent = _curvy_bend_amount(
        pf.random.uniform(rng, 0.05, 0.5), 1.5, dimensions[0], back_width
    )
    top_bent = _curvy_bend_amount(
        pf.random.uniform(rng, 0.05, 0.6), 2.02, dimensions[0], top_width
    )
    mid_pos = pf.random.uniform(rng, 0.4, 0.6)
    seat_height = pf.random.uniform(rng, 0.5, 0.7)
    if material is None:
        material = furniture_material_rand(rng_mat, pf.nodes.shader.coord().uv)
    seat = chair_seat_curvy(
        width=dimensions[0],
        u_resolution=16,
        v_resolution=11,
        front_relative_width=front_relative_width,
        front_bent=front_bent,
        seat_bent=seat_bent,
        mid_bent=mid_bent,
        mid_relative_width=mid_width,
        back_bent=back_bent,
        back_relative_width=back_width,
        top_bent=top_bent,
        top_relative_width=top_width,
        mid_pos=mid_pos,
        seat_height=seat_height,
        elevation=dimensions[2],
        rotation_z=1.5708,
        material=material,
    ).mesh
    pf.ops.modifier.subdivide_surface(seat, levels=_CHAIR_SUBDIV, _skip_apply=True)
    return ChairResult(mesh=seat)


def chair_seat_round_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    material: pf.Material | None = None,
) -> ChairResult:
    """Backless round pad seat (v1 round_seats)."""
    rng, rng_dims, rng_mat = rng.spawn(3)
    if dimensions is None:
        dimensions = chair_office_dimensions_rand(rng_dims)
    thickness = pf.random.uniform(rng, 0.05, 0.12)
    radius = pf.random.uniform(rng, 0.35, 0.45) * (dimensions[0] / 0.55)
    cap_radius = pf.random.uniform(rng, 2.0, 3.2)
    bevel_factor = pf.random.uniform(rng, 0.01, 0.04)
    if material is None:
        material = furniture_material_rand(rng_mat, pf.nodes.shader.coord().uv)
    seat = chair_seat_round(
        thickness=thickness,
        radius=radius,
        cap_radius=cap_radius,
        bevel_factor=bevel_factor,
        elevation=dimensions[2],
        material=material,
    ).mesh
    pf.ops.modifier.subdivide_surface(seat, levels=_CHAIR_SUBDIV, _skip_apply=True)
    return ChairResult(mesh=seat)


def _wheeled_base_post_diameter_rand(rng: pf.RNG) -> float:
    return pf.random.uniform(rng, 0.03, 0.065)


def chair_base_wheeled_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    material: pf.Material | None = None,
    wheel_material: pf.Material | None = None,
) -> ChairResult:
    """Five-star caster base with a central gas-lift pole (v1 wheeled leg)."""
    rng, rng_dims, rng_mat, rng_wheel_mat = rng.spawn(4)
    if dimensions is None:
        dimensions = chair_office_dimensions_rand(rng_dims)
    vec = pf.nodes.shader.coord().uv
    if material is None:
        material = furniture_material_rand(rng_mat, vec)
    if wheel_material is None:
        wheel_material = castor_wheel_material_rand(rng_wheel_mat, vec, material)
    joint_height = pf.random.uniform(rng, 0.5, 0.8) * dimensions[2]
    leg_diameter = _wheeled_base_post_diameter_rand(rng)
    arc_sweep_angle = pf.random.uniform(rng, 120.0, 240.0)
    wheel_width = pf.random.uniform(rng, 0.11, 0.15)
    wheel_rotation = pf.random.uniform(rng, 0.0, 360.0)
    pole_length = pf.random.uniform(rng, 1.6, 2.0)
    leg_number = pf.control.choice(rng, [(4, 0.5), (5, 0.5)])
    base = chair_base_wheeled(
        top_height=dimensions[2],
        joint_height=joint_height,
        leg_diameter=leg_diameter,
        arc_sweep_angle=arc_sweep_angle,
        wheel_width=wheel_width,
        wheel_rotation=wheel_rotation,
        pole_length=pole_length,
        leg_number=leg_number,
        material=material,
        wheel_material=wheel_material,
        rotation_z=1.5708,
        crease_degrees=40.0,
    ).mesh
    pf.ops.modifier.subdivide_surface(base, levels=_CHAIR_SUBDIV, _skip_apply=True)
    return ChairResult(mesh=base)


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
    slat_profile_resolution: t.SocketOrVal[int] = 24,
    crest_align: t.SocketOrVal[float] = 0.0,
    slat_material: pf.ProcNode[pf.Material] | None = None,
    crest_material: pf.ProcNode[pf.Material] | None = None,
) -> pf.ProcNode[pf.MeshObject]:
    """Flat slatted back panel: vertical slats (swept profile curves) arrayed
    along a straight line in Y, capped by a corner-box top rail. dimensions =
    (depth, width, height); built straight, bow deferred to the seat. slat_span
    scales the slat row inward from the full width (1 = outermost slats at the
    edges). Reusable as a bed headboard."""
    # run into the rail mid-plane so its rounded underside cannot expose the slat tops
    slat_height = dimensions.z - crest_height * 0.5
    slat_line = pf.nodes.geo.curve_line(
        start=(0, 0, 0), end=pf.nodes.math.combine_xyz(z=slat_height)
    )
    slat_line = pf.nodes.geo.resample_curve_count(curve=slat_line, count=5)
    profile = pf.nodes.geo.curve_circle(resolution=slat_profile_resolution, radius=0.5)
    # spin half a segment so a 4-vert section is a square rather than a diamond
    half_segment = 3.14159265 / slat_profile_resolution.astype(dtype=float)
    profile = pf.nodes.geo.transform(
        geometry=profile, rotation=pf.nodes.math.combine_xyz(z=half_segment)
    )
    flat_to_flat = pf.nodes.math.cos(half_segment)
    profile = pf.nodes.geo.transform(
        geometry=profile,
        scale=pf.nodes.math.combine_xyz(
            x=slat_depth / flat_to_flat, y=slat_width / flat_to_flat, z=1.0
        ),
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
    # proud of the slats so their buried tops never z-fight the rail, even once bent
    crest_depth = dimensions.x + 0.006
    crest = mesh.box_with_support_loops(
        size=pf.nodes.math.combine_xyz(x=crest_depth, y=crest_width, z=crest_height),
        vertices_x=4,
        vertices_y=pf.nodes.math.add(crest_loops, 4),
        vertices_z=4,
        support_loop_offset=crest_offset,
    )
    # 0 sits the rail flush with the slat fronts, 1 with their backs, 0.5 centered
    crest_shift = crest_align * (dimensions.x - slat_depth)
    crest = pf.nodes.geo.transform(
        geometry=crest,
        translation=pf.nodes.math.combine_xyz(
            x=crest_shift, z=dimensions.z - crest_height * 0.5
        ),
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    if slat_material is not None:
        slats = pf.nodes.geo.set_material(slats, slat_material)
    if crest_material is not None:
        crest = pf.nodes.geo.set_material(crest, crest_material)

    geometry = pf.nodes.geo.join_geometry([slats, crest])
    geometry = mesh.crease_sharp(geometry, threshold_degrees=_CREASE_ANGLE)
    return pf.nodes.geo.set_shade_smooth(geometry=geometry, shade_smooth=True)


@pf.tracer.generator
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
    slat_profile_resolution: int = 24,
    crest_align: float = 0.0,
    material: pf.Material | None = None,
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
        slat_profile_resolution=slat_profile_resolution,
        crest_align=crest_align,
    )
    obj = pf.nodes.to_mesh_object(geo)
    _finish_part(obj, material)
    return ChairResult(mesh=obj)


def chair_back_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    material: pf.Material | None = None,
    slat_count_rand: Callable[[pf.RNG, float, float], int] | None = None,
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

    if slat_count_rand is None:
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
    else:
        n_slats = slat_count_rand(rng, dimensions[1], slat_width)
        slat_span = 1.0
    slat_profile_resolution = pf.control.choice(rng, [(24, 2.0), (4, 1.0)])
    crest_align = pf.random.uniform(rng, 0.0, 1.0)
    if material is None:
        material = furniture_material_rand(rng_mat, pf.nodes.shader.coord().uv)
    back = chair_back(
        dimensions=dimensions,
        n_slats=n_slats,
        slat_span=slat_span,
        slat_width=slat_width,
        slat_depth=slat_depth,
        crest_height=crest_height,
        crest_offset=(depth_r, corner_r, corner_r),
        crest_margin=0.0,
        slat_profile_resolution=slat_profile_resolution,
        crest_align=crest_align,
        material=material,
    ).mesh
    pf.ops.modifier.subdivide_surface(back, levels=_CHAIR_SUBDIV, _skip_apply=True)
    return ChairResult(mesh=back)


@pf.nodes.node_function
def chair_back_solid(
    dimensions: t.SocketOrVal[pf.Vector],
    top_rise: t.SocketOrVal[float] = 0.0,
    n_points: t.SocketOrVal[int] = 7,
) -> pf.ProcNode[pf.MeshObject]:
    """Solid back: panel filled between a flat bottom edge and a top edge that
    bows up by top_rise (0 = square), then extruded for depth. The fill points
    give it enough geometry to bend. dimensions = (depth, width, height)."""
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
        curve_left=bottom, curve_right=top, n_points=n_points, n_rows=5
    )
    solid = mesh.extrude_mesh_seamless_uvs_along(
        mesh=panel,
        selection=True,
        offset_scale=1.0,
        offset=pf.nodes.math.combine_xyz(x=dimensions.x),
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
def chair_seat_dining(
    depth: t.SocketOrVal[float],
    half_width: t.SocketOrVal[float],
    thickness: t.SocketOrVal[float],
    front_bow: t.SocketOrVal[float] = 0.0,
    back_bow: t.SocketOrVal[float] = 0.0,
    front_dip: t.SocketOrVal[float] = 0.0,
    n_points: t.SocketOrVal[int] = 7,
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
        curve_left=back_edge, curve_right=front_edge, n_points=n_points, n_rows=3
    )
    solid = mesh.extrude_mesh_seamless_uvs_along(
        mesh=top,
        selection=True,
        offset_scale=1.0,
        offset=pf.nodes.math.combine_xyz(z=thickness * -1.0),
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


def chair_dining_dimensions_rand(
    rng: pf.RNG,
    depth: float | None = None,
    width: float | None = None,
    seat_elevation: float | None = None,
) -> pf.Vector:
    """Real dining-chair footprint (m): seat depth 0.41-0.46, width 0.40-0.50,
    seat height 0.45-0.50 (standard 45-50cm, tables 71-76cm)."""
    if depth is None:
        depth = pf.random.uniform(rng, 0.41, 0.46)
    if width is None:
        width = pf.random.uniform(rng, 0.40, 0.50)
    if seat_elevation is None:
        seat_elevation = pf.random.uniform(rng, 0.45, 0.50)
    return (depth, width, seat_elevation)


@pf.tracer.generator
def chair_dining_seat_slat_back(
    dimensions: pf.Vector,
    thickness: float,
    front_bow: float,
    back_bow: float,
    front_dip: float,
    seat_material: pf.Material,
    back_height: float,
    back_sink: float,
    slant: float,
    back_bend: float,
    seat_round: float,
    slat_width: float,
    slat_depth: float,
    gap: float,
    crest_height: float,
    corner_r: float,
    crest_depth_r: float,
    slat_profile_resolution: int,
    crest_align: float,
    detail_material: pf.Material,
    slat_material: pf.Material,
) -> ChairResult:
    width = dimensions[1]
    usable_span = width - slat_width - 2.0 * corner_r
    n_slats = pf.nodes.math.clamp(
        pf.nodes.math.round(1.0 + usable_span / (slat_width + gap)),
        2.0,
        64.0,
    )
    back = _chair_back_geometry(
        dimensions=pf.nodes.math.combine_xyz(x=slat_depth, y=width, z=back_height),
        n_slats=n_slats,
        slat_span=1.0,
        slat_width=slat_width,
        slat_depth=slat_depth,
        crest_height=crest_height,
        crest_offset=(crest_depth_r, corner_r, corner_r),
        crest_margin=0.0,
        crest_loops=16,
        slat_profile_resolution=slat_profile_resolution,
        crest_align=crest_align,
        slat_material=slat_material,
        crest_material=detail_material,
    )
    seat = _dining_seat_with_back(
        dimensions,
        thickness,
        front_bow,
        back_bow,
        front_dip,
        seat_material,
        back,
        0.001,
        back_height,
        back_sink=back_sink,
        slant=slant,
        back_bend=back_bend,
        seat_round=seat_round,
    )
    return ChairResult(mesh=seat)


def _dining_slat_back_rand(
    rng: pf.RNG,
    dimensions: pf.Vector,
    thickness: float,
    front_bow: float,
    back_bow: float,
    front_dip: float,
    seat_material: pf.Material,
    back_height: float,
    back_sink: float,
    slant: float,
    back_bend: float,
    seat_round: float,
    detail_material: pf.Material,
    slat_material: pf.Material,
) -> ChairResult:
    """Seat with a slatted back; slat count follows width so thin spindles and
    broad slats both space out. Slats take slat_material, the crest rail the
    backrest detail_material."""
    width = dimensions[1]
    slat_width = pf.random.uniform(rng, 0.0225, 0.061875)
    aspect = pf.random.clip_gaussian(rng, 0.8, 0.15, 0.4, 1.0)
    slat_depth = slat_width * aspect
    gap = 0.0127 + (width - 0.0127) * pf.random.uniform(rng, 0.0, 1.0) ** 3
    crest_height = pf.random.clip_gaussian(rng, 0.0675, 0.035, 0.0375, 0.15)
    # stays above the back's 1mm bevel, which otherwise collapses the support loops
    corner_r = pf.random.uniform(rng, 0.05, 0.49) * crest_height
    crest_depth_r = pf.random.uniform(rng, 0.2, 0.49) * slat_depth
    slat_profile_resolution = pf.control.choice(rng, [(24, 2.0), (4, 1.0)])
    crest_align = pf.random.uniform(rng, 0.0, 1.0)
    return chair_dining_seat_slat_back(
        dimensions=dimensions,
        thickness=thickness,
        front_bow=front_bow,
        back_bow=back_bow,
        front_dip=front_dip,
        seat_material=seat_material,
        back_height=back_height,
        back_sink=back_sink,
        slant=slant,
        back_bend=back_bend,
        seat_round=seat_round,
        slat_width=slat_width,
        slat_depth=slat_depth,
        gap=gap,
        crest_height=crest_height,
        corner_r=corner_r,
        crest_depth_r=crest_depth_r,
        slat_profile_resolution=slat_profile_resolution,
        crest_align=crest_align,
        detail_material=detail_material,
        slat_material=slat_material,
    )


@pf.tracer.generator
def chair_dining_seat_solid_back(
    dimensions: pf.Vector,
    thickness: float,
    front_bow: float,
    back_bow: float,
    front_dip: float,
    seat_material: pf.Material,
    back_height: float,
    back_sink: float,
    slant: float,
    back_bend: float,
    seat_round: float,
    back_depth: float,
    top_rise: float,
    back_round: float,
    detail_material: pf.Material,
) -> ChairResult:
    back = chair_back_solid(
        dimensions=pf.nodes.math.combine_xyz(
            x=back_depth, y=dimensions[1], z=back_height
        ),
        top_rise=top_rise,
    )
    back = pf.nodes.geo.set_material(back, detail_material)
    seat = _dining_seat_with_back(
        dimensions,
        thickness,
        front_bow,
        back_bow,
        front_dip,
        seat_material,
        back,
        back_round,
        back_height,
        back_sink=back_sink,
        slant=slant,
        back_bend=back_bend,
        seat_round=seat_round,
    )
    return ChairResult(mesh=seat)


def _dining_solid_back_rand(
    rng: pf.RNG,
    dimensions: pf.Vector,
    thickness: float,
    front_bow: float,
    back_bow: float,
    front_dip: float,
    seat_material: pf.Material,
    back_height: float,
    back_sink: float,
    slant: float,
    back_bend: float,
    seat_round: float,
    detail_material: pf.Material,
    slat_material: pf.Material,
) -> ChairResult:
    """Seat with a solid filled back: flat bottom, up-bowed (or flat) top, and a
    v1-style generous round-over (bevel up to half depth). The whole panel is
    backrest, so it takes detail_material; slat_material is unused."""
    back_depth = pf.random.clip_gaussian(rng, 0.05, 0.015, 0.03, 0.10)
    top_rise = pf.random.clip_gaussian(rng, 0.03, 0.04, 0.0, 0.15)
    back_round = pf.random.uniform(rng, 0.1, 0.45) * back_depth
    return chair_dining_seat_solid_back(
        dimensions=dimensions,
        thickness=thickness,
        front_bow=front_bow,
        back_bow=back_bow,
        front_dip=front_dip,
        seat_material=seat_material,
        back_height=back_height,
        back_sink=back_sink,
        slant=slant,
        back_bend=back_bend,
        seat_round=seat_round,
        back_depth=back_depth,
        top_rise=top_rise,
        back_round=back_round,
        detail_material=detail_material,
    )


def _dining_seat_with_back(
    dimensions: pf.Vector,
    thickness: float,
    front_bow: float,
    back_bow: float,
    front_dip: float,
    seat_material: pf.Material,
    back: pf.ProcNode,
    back_round: float,
    back_height: float,
    back_sink: float,
    slant: float,
    back_bend: float,
    seat_round: float,
) -> pf.MeshObject:
    """The seat: a pan with the given back seated on its rear edge. Builds the pan,
    sets the back's rear face on the pan's rear edge, bends the back to follow that
    edge and leans it, then bevels pan and back (separate amounts) and merges them
    into one uv-mapped object. dimensions = (depth, width, elevation). back_height
    is the full built height of the back, of which back_sink is buried in the pan."""
    half_width = dimensions[1] * 0.5
    seat_res = chair_seat_dining(
        depth=dimensions[0],
        half_width=half_width,
        thickness=thickness,
        front_bow=front_bow,
        back_bow=back_bow,
        front_dip=front_dip,
        material=seat_material,
    )
    seat = pf.nodes.geo.transform(
        seat_res.seat,
        translation=(0, 0, dimensions[2]),
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    bb = pf.nodes.geo.bound_box(back)
    # seat the back's rear face on the seat rear edge so slats don't overhang behind
    forward = bb.min.x * -1.0
    base_z = dimensions[2] - back_sink
    back = pf.nodes.geo.transform(
        back,
        translation=pf.nodes.math.combine_xyz(x=forward, z=base_z),
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    back = _bend_to_edge(back, edge=seat_res.back_edge, half_width=half_width)
    back = _shear_back(
        back, slant=slant * -1.0, base_z=base_z, height=back_height, bend=back_bend
    )

    obj = pf.nodes.to_mesh_object(seat)
    pf.ops.modifier.bevel(obj, width=seat_round, segments=2)
    back_obj = pf.nodes.to_mesh_object(back)
    pf.ops.modifier.bevel(back_obj, width=back_round, segments=2)
    pf.ops.object.join(obj, back_obj)
    return obj


def _chair_straight_base_rand(
    rng: pf.RNG,
    dimensions: pf.Vector,
    material: pf.Material,
) -> ChairResult:
    result = base_straight_rand(
        rng,
        dimensions,
        material,
        leg_placement_top_scale=1.0,
    )
    return ChairResult(mesh=result.mesh)


def _chair_square_base_rand(
    rng: pf.RNG,
    dimensions: pf.Vector,
    material: pf.Material,
) -> ChairResult:
    rng, rng_diameter = rng.spawn(2)
    diameter = pf.random.uniform(rng_diameter, 0.03, 0.14)
    result = base_square_rand(
        rng,
        (dimensions[0] - diameter, dimensions[1] - diameter, dimensions[2]),
        material,
        leg_diameter=diameter,
        leg_placement_top_scale=1.0,
    )
    return ChairResult(mesh=result.mesh)


def _grid_base_rand(
    rng: pf.RNG,
    dimensions: pf.Vector,
    material: pf.Material,
    close_edges: bool = False,
) -> ChairResult:
    footprint = pf.nodes.math.minimum(dimensions[0], dimensions[1])
    base = storage.storage_legs_grid_rand(
        rng,
        dimensions,
        dimensions[2],
        close_edges=close_edges,
        diameter=0.02 + pf.random.uniform(rng, 0.0, 0.15) * footprint,
    )
    pf.ops.object.set_transform(
        base, location=(dimensions[0] * -0.5, dimensions[1] * -0.5, 0.0)
    )
    pf.ops.object.set_material(base, material)
    return ChairResult(mesh=base)


def chair_base_stable_rand(
    rng: pf.RNG,
    dimensions: pf.Vector,
    material: pf.Material,
    close_edges: bool = False,
) -> ChairResult:
    """Chair base without oversized pedestal or wheeled options."""
    base_fn = pf.control.choice(
        rng,
        [
            (_grid_base_rand, 0.4),
            (base_straight_rand, 0.45),
            (base_square_rand, 0.15),
        ],
    )
    return base_fn(rng, dimensions, material, close_edges=close_edges)


def chair_dining_rand(rng: pf.RNG, dimensions: pf.Vector | None = None) -> ChairResult:
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
        rng_dip,
        rng_back_bend,
        rng_seat,
    ) = rng.spawn(16)
    if dimensions is None:
        dimensions = chair_dining_dimensions_rand(rng_dims)
    vec = pf.nodes.shader.coord().uv
    material1 = cache(partial(furniture_material_rand, rng_mat1, vec))
    material2 = cache(partial(furniture_material_rand, rng_mat2, vec))
    fabric = cache(partial(fabric_sturdy_rand, rng_fabric, vec))
    seat_material_fn = pf.control.choice(
        rng_seat_sel, [(material1, 2.0), (fabric, 1.0)]
    )
    seat_material = seat_material_fn()
    backrest_material_fn = pf.control.choice(
        rng_backrest_sel, [(material1, 2.0), (fabric, 1.0)]
    )
    backrest_material = backrest_material_fn()
    leg_material_fn = pf.control.choice(
        rng_leg_sel, [(material1, 1.0), (material2, 1.0)]
    )
    leg_material = leg_material_fn()
    slat_material_fn = pf.control.choice(
        rng_slat_sel, [(material1, 1.0), (material2, 1.0)]
    )
    slat_material = slat_material_fn()

    front_bow = pf.random.clip_gaussian(rng, 0.08, 0.1, 0.0, 0.3) * dimensions[0]
    back_bow = pf.random.uniform(rng, 0.0, 0.25) * dimensions[0]
    front_dip = 0.15 * pf.random.uniform(rng_dip, 0.0, 1.0) ** 2 * dimensions[0]
    thickness = pf.random.clip_gaussian(rng, 0.06, 0.04, 0.02, 0.2)

    back_height = pf.random.uniform(rng, 0.35, 0.55)
    back_fn = pf.control.choice(
        rng_back_sel, [(_dining_slat_back_rand, 0.6), (_dining_solid_back_rand, 0.4)]
    )
    # the pan mid-plane is the one depth still inside the pan for any round-over
    back_sink = thickness * 0.5
    back_built = back_height + back_sink
    slant = pf.random.clip_gaussian(rng, 0.23, 0.067, 0.0, 0.4)
    bend_frac = pf.random.clip_gaussian(rng_back_bend, 0.15, 0.35, -0.4, 1.2)
    seat_round = pf.random.uniform(rng_seat, 0.05, 0.49) * thickness
    seat = back_fn(
        rng_back,
        dimensions,
        thickness,
        front_bow,
        back_bow,
        front_dip,
        seat_material,
        back_built,
        back_sink,
        slant,
        bend_frac * 0.15 * back_height * -1.0,
        seat_round,
        backrest_material,
        slat_material,
    ).mesh

    leg_spread = pf.random.uniform(rng, 0.8, 0.95)
    leg_inset_fraction = pf.random.uniform(rng, 0.05, 0.10)
    leg_scale = 1.0 - 2.0 * leg_inset_fraction
    seat_bottom = dimensions[2] - thickness * 0.5
    # dipped pan center top sits 0.375*dip below z, center poles must stop below it
    center_bottom = seat_bottom - 0.375 * front_dip

    def wheeled_fn(
        rng: pf.RNG, base_dimensions: pf.Vector, material: pf.Material
    ) -> ChairResult:
        return chair_base_wheeled_rand(
            rng,
            (
                dimensions[0] * leg_spread,
                dimensions[1] * leg_spread,
                center_bottom,
            ),
            material,
        )

    base_fn = pf.control.choice(
        rng_base_sel,
        [
            (_chair_straight_base_rand, 3.0),
            (wheeled_fn, 1.0),
        ],
    )
    base_dimensions = (
        dimensions[0] * leg_scale,
        dimensions[1] * leg_scale,
        seat_bottom,
    )
    base = base_fn(rng_base, base_dimensions, leg_material).mesh

    pf.ops.object.join(seat, base)
    seat = pf.nodes.to_mesh_object(mesh.crease_sharp(seat, threshold_degrees=40.0))
    pf.ops.modifier.subdivide_surface(seat, levels=_CHAIR_SUBDIV, _skip_apply=True)
    return ChairResult(mesh=seat)


def chair_bench_dimensions_rand(rng: pf.RNG) -> pf.Vector:
    """Bench dimensions: seat-depth deep, several seats wide."""
    return (
        pf.random.uniform(rng, 0.4, 0.48),
        pf.random.uniform(rng, 0.9, 2.5),
        pf.random.uniform(rng, 0.42, 0.5),
    )


def chair_bench_rand(rng: pf.RNG, dimensions: pf.Vector | None = None) -> ChairResult:
    """Backed bench: a bezier-outline seat pan several seats wide, on a base
    chosen for stability under a long span rather than a chair's single seat."""
    (
        rng,
        rng_dims,
        rng_back_sel,
        rng_back,
        rng_base,
        rng_mat1,
        rng_mat2,
        rng_fabric,
        rng_seat_sel,
        rng_backrest_sel,
        rng_leg_sel,
        rng_slat_sel,
        rng_dip,
        rng_back_bend,
        rng_seat,
    ) = rng.spawn(15)
    if dimensions is None:
        dimensions = chair_bench_dimensions_rand(rng_dims)
    vec = pf.nodes.shader.coord().uv
    material1 = cache(partial(furniture_material_rand, rng_mat1, vec))
    material2 = cache(partial(furniture_material_rand, rng_mat2, vec))
    fabric = cache(partial(fabric_sturdy_rand, rng_fabric, vec))
    seat_material_fn = pf.control.choice(
        rng_seat_sel, [(material1, 2.0), (fabric, 1.0)]
    )
    seat_material = seat_material_fn()
    backrest_material_fn = pf.control.choice(
        rng_backrest_sel, [(material1, 2.0), (fabric, 1.0)]
    )
    backrest_material = backrest_material_fn()
    leg_material_fn = pf.control.choice(
        rng_leg_sel, [(material1, 1.0), (material2, 1.0)]
    )
    leg_material = leg_material_fn()
    slat_material_fn = pf.control.choice(
        rng_slat_sel, [(material1, 1.0), (material2, 1.0)]
    )
    slat_material = slat_material_fn()

    front_bow = pf.random.clip_gaussian(rng, 0.08, 0.1, 0.0, 0.3) * dimensions[0]
    back_bow = pf.random.uniform(rng, 0.0, 0.25) * dimensions[0]
    front_dip = 0.15 * pf.random.uniform(rng_dip, 0.0, 1.0) ** 2 * dimensions[0]
    thickness = pf.random.clip_gaussian(rng, 0.06, 0.04, 0.02, 0.2)

    back_height = pf.random.uniform(rng, 0.35, 0.55)
    back_fn = pf.control.choice(
        rng_back_sel, [(_dining_slat_back_rand, 0.6), (_dining_solid_back_rand, 0.4)]
    )
    back_sink = thickness * 0.5
    back_built = back_height + back_sink
    slant = pf.random.clip_gaussian(rng, 0.23, 0.067, 0.0, 0.4)
    bend_frac = pf.random.clip_gaussian(rng_back_bend, 0.15, 0.35, -0.4, 1.2)
    seat_round = pf.random.uniform(rng_seat, 0.05, 0.49) * thickness
    seat = back_fn(
        rng_back,
        dimensions,
        thickness,
        front_bow,
        back_bow,
        front_dip,
        seat_material,
        back_built,
        back_sink,
        slant,
        bend_frac * 0.15 * back_height * -1.0,
        seat_round,
        backrest_material,
        slat_material,
    ).mesh

    # a long span needs its legs near the ends, so spread further than a chair does
    leg_spread = pf.random.uniform(rng, 0.9, 0.98)
    seat_bottom = dimensions[2] - thickness * 0.5
    base = chair_base_stable_rand(
        rng_base,
        (
            dimensions[0] * leg_spread,
            dimensions[1] * leg_spread,
            seat_bottom,
        ),
        leg_material,
        close_edges=True,
    ).mesh

    pf.ops.object.join(seat, base)
    return ChairResult(mesh=seat)


def chair_rand(rng: pf.RNG, dimensions: pf.Vector | None = None) -> ChairResult:
    """Any chair: randomly an office chair (curvy/round seat on a wheeled/pedestal/
    leg base) or a wooden dining chair (bezier seat + slat back)."""
    rng, rng_sel, rng_gen = rng.spawn(3)
    chair_fn = pf.control.choice(
        rng_sel, [(chair_office_rand, 1.0), (chair_dining_rand, 1.0)]
    )
    return chair_fn(rng=rng_gen, dimensions=dimensions)


def _office_chair_base_rand(
    rng: pf.RNG,
    wheeled_fn: Callable[..., ChairResult] = chair_base_wheeled_rand,
) -> Callable[..., ChairResult | TableResult]:
    return pf.control.choice(
        rng,
        [
            (_chair_straight_base_rand, 3.0),
            (wheeled_fn, 4.0),
        ],
    )


def chair_office_rand(
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
    ) = rng.spawn(8)
    if dimensions is None:
        dimensions = chair_office_dimensions_rand(rng_dims)

    top_fn = pf.control.choice(
        rng_top_sel,
        [
            (chair_seat_curvy_rand, 1.5),
            (chair_seat_round_rand, 0.33),
        ],
    )
    leg_spread = pf.random.uniform(rng, 0.5, 0.7)
    leg_inset_fraction = pf.random.uniform(rng, 0.05, 0.10)
    leg_scale = (1.0 - 2.0 * leg_inset_fraction) / 2.0**0.5
    base_dimensions = (
        dimensions[0] * leg_scale,
        dimensions[1] * leg_scale,
        dimensions[2] + 0.01,
    )

    def wheeled_fn(
        rng: pf.RNG, base_dimensions: pf.Vector, material: pf.Material
    ) -> ChairResult:
        return chair_base_wheeled_rand(
            rng,
            (
                dimensions[0] * leg_spread,
                dimensions[1] * leg_spread,
                base_dimensions[2],
            ),
            material,
        )

    vec = pf.nodes.shader.coord().uv
    if seat_material is None:
        seat_material = furniture_surface_material_rand(rng_seat_mat, vec)
    if base_material is None:
        base_material = furniture_material_rand(rng_base_mat, vec)

    top = top_fn(rng=rng_top, dimensions=dimensions, material=seat_material).mesh

    base_fn = _office_chair_base_rand(rng_base_sel, wheeled_fn)
    base = base_fn(rng_base, base_dimensions, base_material).mesh

    # join keeps only the target's stack, so the seat's _CHAIR_SUBDIV covers the base too
    pf.ops.object.join(top, base)
    return ChairResult(mesh=top)
