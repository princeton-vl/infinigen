# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Yiming Zuo: original Infinigen v1 nodegroup (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/objects/tables/dining_table.py)
# - Alexander Raistrick: transpile to procfunc/v2

import math
from typing import NamedTuple

import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.shaders.functionality_lists import furniture_material_rand
from infinigen2.util import curve, mesh

__all__ = [
    "TableResult",
    "base_box_leg",
    "base_four_leg",
    "base_pedestal",
    "base_pedestal_column",
    "base_pedestal_column_rand",
    "base_pedestal_rand",
    "base_square",
    "base_square_rand",
    "base_straight",
    "base_straight_rand",
    "table_coffee_dimensions_rand",
    "table_dimensions_rand",
    "table_side_dimensions_rand",
]


class TableResult(NamedTuple):
    mesh: pf.MeshObject


@pf.nodes.node_function
def _n_gon_profile(
    profile_n_gon: t.SocketOrVal[int],
    profile_width: t.SocketOrVal[float],
    profile_aspect_ratio: t.SocketOrVal[float],
    profile_fillet_ratio: t.SocketOrVal[float] = 0.0,
) -> t.ProcNode[pf.CurveObject]:
    curve_circle_radius = pf.nodes.math.constant(0.5)
    curve_circle = pf.nodes.geo.curve_circle(
        resolution=profile_n_gon, radius=curve_circle_radius
    )

    transform_rotation = pf.nodes.math.combine_xyz(
        z=3.1416 / profile_n_gon.astype(dtype=float)
    )
    transform = pf.nodes.geo.transform(
        geometry=curve_circle,
        rotation=transform_rotation.astype(dtype=pf.Euler),
        translation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    transform_1 = pf.nodes.geo.transform(
        geometry=transform,
        rotation=(0.0, 0.0, -1.5708),
        translation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    transform_2_scale = pf.nodes.math.combine_xyz(
        x=profile_width,
        y=profile_aspect_ratio * profile_width,
        z=1.0,
    )
    transform_2 = pf.nodes.geo.transform(
        geometry=transform_1,
        scale=transform_2_scale,
        translation=(0, 0, 0),
        rotation=(0, 0, 0),
    )

    fillet_curve = pf.nodes.geo.fillet_curve_poly(
        curve=transform_2,
        radius=profile_width * profile_fillet_ratio,
        limit_radius=True,
        count=4,
    )
    return fillet_curve


@pf.nodes.node_function
def _merge_curve(
    curve: t.ProcNode[pf.CurveObject],
) -> t.ProcNode[pf.CurveObject]:
    curve_to = pf.nodes.geo.curve_to_mesh(curve)

    merge_by_distance = pf.nodes.geo.merge_by_distance(curve_to)

    to_curve = pf.nodes.geo.mesh_to_curve(merge_by_distance)
    return to_curve


class _NGonCylinderResult(NamedTuple):
    mesh: t.ProcNode[pf.MeshObject]
    profile_curve: t.ProcNode[pf.CurveObject]
    caps: t.ProcNode[pf.MeshObject]


@pf.nodes.node_function
def _n_gon_cylinder(
    radius_curve: t.ProcNode[pf.CurveObject],
    n_gon: t.SocketOrVal[int],
    profile_width: t.SocketOrVal[float],
    height: t.SocketOrVal[float] = 1.0,
    aspect_ratio: t.SocketOrVal[float] = 1.0,
    fillet_ratio: t.SocketOrVal[float] = 0.2,
    profile_resolution: t.SocketOrVal[int] = 16,
    resolution: t.SocketOrVal[int] = 64,
) -> _NGonCylinderResult:
    mesh_position_z_to_min = height * -1.0

    curve_line_end = pf.nodes.math.combine_xyz(z=mesh_position_z_to_min)
    curve_line = pf.nodes.geo.curve_line(end=curve_line_end, start=(0, 0, 0))

    set_curve_tilt = pf.nodes.geo.set_curve_tilt(curve=curve_line, tilt=3.1416)

    resample_curve_count_1 = pf.nodes.geo.resample_curve_count(
        curve=set_curve_tilt, count=resolution
    )

    spline_parameter = pf.nodes.geo.spline_parameter()

    capture_attribute = pf.nodes.geo.capture_attribute(
        geometry=resample_curve_count_1,
        attribute=spline_parameter.factor,
    )

    n_gon_profile_result = _n_gon_profile(
        profile_n_gon=n_gon,
        profile_width=profile_width,
        profile_aspect_ratio=aspect_ratio,
        profile_fillet_ratio=fillet_ratio,
    )

    resample_curve_count = pf.nodes.geo.resample_curve_count(
        curve=n_gon_profile_result,
        count=profile_resolution,
    )

    curve_to = curve.curve_to_mesh_with_uv(
        curve=capture_attribute.geometry,
        profile=resample_curve_count,
        fill_caps=True,
    ).mesh

    input_position = pf.nodes.geo.input_position()

    sample_curve = pf.nodes.geo.sample_curve(
        curves=radius_curve,
        factor=capture_attribute.attribute,
        value=0.0,
        use_all_curves=True,
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
    )
    mesh_position = pf.nodes.math.combine_xyz(
        x=input_position.x * mesh_position_x,
        y=input_position.y * mesh_position_x,
        z=mesh_position_z,
    )

    set_position = pf.nodes.geo.set_position(geometry=curve_to, position=mesh_position)

    input_index = pf.nodes.geo.input_index()

    attribute_domain_size = pf.nodes.geo.attribute_domain_size(curve_to)

    caps_selection_b = attribute_domain_size.face_count.astype(dtype=float) - 2.0
    caps_selection = pf.nodes.func.less_than(
        a=input_index, b=caps_selection_b.astype(dtype=int)
    )

    delete = pf.nodes.geo.delete_geometry(
        geometry=curve_to,
        selection=caps_selection,
        domain="FACE",
    )
    return _NGonCylinderResult(
        mesh=set_position,
        profile_curve=resample_curve_count,
        caps=delete,
    )


@pf.nodes.node_function
def _stretcher(
    n_gon: t.SocketOrVal[int],
    profile_width: t.SocketOrVal[float],
) -> t.ProcNode[pf.MeshObject]:
    curve_line = pf.nodes.geo.curve_line(start=(1.0, 0.0, 1.0), end=(1.0, 0.0, -1.0))

    n_gon_cylinder_result = _n_gon_cylinder(
        radius_curve=curve_line,
        height=1.0,
        n_gon=n_gon,
        profile_width=profile_width,
        aspect_ratio=1.0,
        fillet_ratio=0.2,
        profile_resolution=8,
        resolution=2,
    )
    return n_gon_cylinder_result.mesh


@pf.nodes.node_function
def _create_anchors(
    profile_n_gon: t.SocketOrVal[int],
    profile_width: t.SocketOrVal[float],
    profile_aspect_ratio: t.SocketOrVal[float],
    profile_rotation: t.SocketOrVal[float],
) -> t.ProcNode[pf.MeshObject]:
    set_switch = pf.nodes.func.equal(a=profile_n_gon, b=1)
    set_a_switch = pf.nodes.func.equal(a=profile_n_gon, b=2)

    n_gon_profile_result = _n_gon_profile(
        profile_n_gon=profile_n_gon,
        profile_width=profile_width,
        profile_aspect_ratio=profile_aspect_ratio,
        profile_fillet_ratio=0.0,
    )

    curve_to_points = pf.nodes.geo.curve_to_points_evaluated(curve=n_gon_profile_result)
    curve_line_start = pf.nodes.math.combine_xyz(profile_width * 0.3535)
    curve_line_end = pf.nodes.math.combine_xyz(profile_width * -0.3535)
    curve_line = pf.nodes.geo.curve_line(start=curve_line_start, end=curve_line_end)
    curve_to_points_1 = pf.nodes.geo.curve_to_points_evaluated(curve=curve_line)

    set_a = pf.nodes.func.switch(
        switch=set_a_switch,
        a=curve_to_points.points,
        b=curve_to_points_1.points,
    )

    points = pf.nodes.geo.points(position=(0, 0, 0))

    set_point_radius_points = pf.nodes.func.switch(switch=set_switch, a=set_a, b=points)
    set_point_radius = pf.nodes.geo.set_point_radius(set_point_radius_points)

    rotation = pf.nodes.math.combine_xyz(z=profile_rotation)

    transform = pf.nodes.geo.transform(
        geometry=set_point_radius,
        rotation=rotation.astype(dtype=pf.Euler),
        translation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    return transform


@pf.nodes.node_function
def _create_legs_and_stretchers(
    anchors: t.ProcNode[pf.MeshObject],
    keep_legs: t.SocketOrVal[bool],
    leg_instance: t.ProcNode[pf.MeshObject],
    table_height: t.SocketOrVal[float],
    leg_bottom_relative_scale: t.SocketOrVal[float],
    leg_bottom_relative_rotation: t.SocketOrVal[float],
    keep_odd_stretchers: t.SocketOrVal[bool],
    keep_even_stretchers: t.SocketOrVal[bool],
    stretcher_instance: t.ProcNode[pf.MeshObject],
    stretcher_index_increment: t.SocketOrVal[int],
    stretcher_relative_position: t.SocketOrVal[float],
    leg_bottom_offset: t.SocketOrVal[float],
) -> t.ProcNode[pf.MeshObject]:
    transform_translation = pf.nodes.math.combine_xyz(z=table_height)
    transform = pf.nodes.geo.transform(
        geometry=anchors,
        translation=transform_translation,
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )

    input_position = pf.nodes.geo.input_position()

    set_b_vector_b = pf.nodes.math.combine_xyz(z=leg_bottom_offset)
    set_b_vector = transform_translation - set_b_vector_b
    set_b_rotation = pf.nodes.math.combine_xyz(0, 0, leg_bottom_relative_rotation)
    set_b_1 = pf.nodes.math.vector_rotate_euler(
        vector=input_position - set_b_vector,
        rotation=set_b_rotation,
        center=(0, 0, 0),
    )
    splay_height_factor = pf.nodes.math.minimum(table_height / 0.4, 1.0)
    splay_scale = 1.0 + (leg_bottom_relative_scale - 1.0) * splay_height_factor
    set_b_0 = pf.nodes.math.combine_xyz(
        x=splay_scale,
        y=splay_scale,
        z=1.0,
    )
    set_position_position_vector = input_position - (set_b_1 * set_b_0)
    set_position_position = pf.nodes.math.vector_scale(
        vector=set_position_position_vector,
        scale=stretcher_relative_position * -1.0,
    )

    input_position_1 = pf.nodes.geo.input_position()

    set_position = pf.nodes.geo.set_position(
        geometry=transform,
        position=set_position_position + input_position_1,
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

    instance_b_switch = pf.nodes.func.equal(
        a=attribute_domain_size.point_count.astype(dtype=float)
        / stretcher_index_increment.astype(dtype=float),
        b=2.0,
        epsilon=0.001,
    )
    input_index_1 = pf.nodes.geo.input_index()

    instance_1 = attribute_domain_size.point_count.astype(dtype=float) / 2.0
    instance_b_b = pf.nodes.func.less_than(
        a=input_index_1, b=instance_1.astype(dtype=int)
    )
    instance_b = pf.nodes.func.switch(switch=instance_b_switch, a=True, b=instance_b_b)
    instance_on_points_selection = pf.nodes.func.boolean_and(a=instance_a, b=instance_b)

    input_position_2 = pf.nodes.geo.input_position()

    field = (
        input_index.astype(dtype=float) + stretcher_index_increment.astype(dtype=float)
    ) % attribute_domain_size.point_count.astype(dtype=float)
    field_at_index = pf.nodes.geo.field_at_index(
        value=input_position_2, index=field.astype(dtype=int)
    )

    instance_z_vector = input_position_2 - field_at_index
    instance_0_rotation = pf.nodes.func.align_euler_to_vector(
        vector=instance_z_vector,
        axis="Z",
        rotation=(0, 0, 0),
        factor=1.0,
    )
    instance = pf.nodes.func.align_euler_to_vector(
        rotation=instance_0_rotation,
        pivot_axis="Z",
        factor=1.0,
        vector=(0, 0, 1),
    )
    instance_z = pf.nodes.math.vector_length(instance_z_vector)
    instance_on_points_scale = pf.nodes.math.combine_xyz(x=1.0, y=1.0, z=instance_z)
    # increment 0 aims every anchor at itself; a zero-length stretcher is a flat shell
    spans_a_gap = pf.nodes.func.greater_than(a=instance_z, b=1e-6)
    instance_on_points = pf.nodes.geo.instance_on_points(
        points=set_position,
        instance=stretcher_instance,
        selection=pf.nodes.func.boolean_and(
            a=instance_on_points_selection, b=spans_a_gap
        ),
        rotation=instance.astype(dtype=pf.Euler),
        scale=instance_on_points_scale,
    )

    realize_instances = pf.nodes.geo.realize_instances(instance_on_points)

    leg_rotation = pf.nodes.func.align_euler_to_vector(
        vector=set_position_position_vector,
        axis="Z",
        rotation=(0, 0, 0),
        factor=1.0,
    )
    instance_scale_z = pf.nodes.math.vector_length(set_position_position_vector)
    instance_scale = pf.nodes.math.combine_xyz(x=1.0, y=1.0, z=instance_scale_z)
    instance_on_points_1 = pf.nodes.geo.instance_on_points(
        points=transform,
        instance=leg_instance,
        rotation=leg_rotation.astype(dtype=pf.Euler),
        scale=instance_scale,
    )

    realize_instances_1 = pf.nodes.geo.realize_instances(instance_on_points_1)

    geometries = pf.nodes.func.switch(switch=keep_legs, b=realize_instances_1)

    join = pf.nodes.geo.join_geometry([realize_instances, geometries])
    return join


@pf.nodes.node_function
def _pedestal_profile(
    height: t.SocketOrVal[float],
    top_radius: t.SocketOrVal[float],
    bottom_radius: t.SocketOrVal[float],
    flare: t.SocketOrVal[float],
    concavity: t.SocketOrVal[float],
    neck_scale: t.SocketOrVal[float] = 1.0,
    resolution: t.SocketOrVal[int] = 12,
) -> t.ProcNode[pf.CurveObject]:
    """Closed silhouette curve (x=radius, y=z) from the axis at the top, out to
    (top_radius, height), down to (bottom_radius, 0), and back to the axis, so
    the lathed surface is capped. flare (0-1) sets how far down the column stays
    skinny before flaring out; concavity (0-1) bends the flare from a convex
    bulge to a sharp concave sweep; neck_scale bulges (>1) or waists (<1) the
    column."""
    knee_z = height * (1.0 - flare)
    end_handle_x = bottom_radius + concavity * (top_radius - bottom_radius)
    end_handle_z = knee_z * (1.0 - concavity)
    body = pf.nodes.geo.curve_bezier_segment(
        start=pf.nodes.math.combine_xyz(x=top_radius, y=height),
        start_handle=pf.nodes.math.combine_xyz(x=top_radius * neck_scale, y=knee_z),
        end_handle=pf.nodes.math.combine_xyz(x=end_handle_x, y=end_handle_z),
        end=pf.nodes.math.combine_xyz(x=bottom_radius),
        resolution=resolution,
    )
    top_cap = pf.nodes.geo.curve_line(
        start=pf.nodes.math.combine_xyz(y=height),
        end=pf.nodes.math.combine_xyz(x=top_radius, y=height),
    )
    bottom_cap = pf.nodes.geo.curve_line(
        start=pf.nodes.math.combine_xyz(x=bottom_radius), end=(0.0, 0.0, 0.0)
    )
    return pf.nodes.geo.join_geometry([top_cap, body, bottom_cap])


@pf.nodes.node_function
def _pedestal_sweep(
    radius_curve: pf.ProcNode[pf.CurveObject],
    resolution: t.SocketOrVal[int] = 16,
) -> t.ProcNode[pf.MeshObject]:
    """Lathe a silhouette curve (x=radius, y=z) around the z axis by sweeping it
    around a circle. Profile x offsets add to the unit circle's radius, so shift
    x by -1; mirroring y maps profile y to +Z and keeps faces outward."""
    # subdivision pulls an n-gon ring in to (2 + cos(2pi/n))/3 of its radius
    ring_angle = math.tau / resolution.astype(dtype=float)
    ring_shrink = (2.0 + pf.nodes.math.cos(ring_angle)) / 3.0
    profile_scale = pf.nodes.math.combine_xyz(x=1.0 / ring_shrink, y=-1.0, z=1.0)
    profile = pf.nodes.geo.transform(
        geometry=radius_curve, translation=(-1.0, 0.0, 0.0), scale=profile_scale
    )
    circle = pf.nodes.geo.curve_circle(resolution=resolution, radius=1.0)
    swept = curve.curve_to_mesh_with_uv(curve=circle, profile=profile).mesh
    return pf.nodes.geo.merge_by_distance(swept, distance=1e-5)


@pf.nodes.node_function
def base_pedestal_column(
    height: t.SocketOrVal[float],
    top_radius: t.SocketOrVal[float],
    bottom_radius: t.SocketOrVal[float],
    flare: t.SocketOrVal[float],
    concavity: t.SocketOrVal[float],
    neck_scale: t.SocketOrVal[float] = 1.0,
    profile_resolution: t.SocketOrVal[int] = 12,
    resolution: t.SocketOrVal[int] = 16,
) -> t.ProcNode[pf.MeshObject]:
    profile = _pedestal_profile(
        height=height,
        top_radius=top_radius,
        bottom_radius=bottom_radius,
        flare=flare,
        concavity=concavity,
        neck_scale=neck_scale,
        resolution=profile_resolution,
    )
    return _pedestal_sweep(profile, resolution=resolution)


@pf.nodes.node_function
def _leg_square(
    width: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    fillet_radius: t.SocketOrVal[float],
    has_bottom_connector: t.SocketOrVal[bool],
    profile_n_gon: t.SocketOrVal[int],
    profile_width: t.SocketOrVal[float],
    profile_aspect_ratio: t.SocketOrVal[float],
    profile_fillet_ratio: t.SocketOrVal[float],
) -> t.ProcNode[pf.MeshObject]:
    curve_circle = pf.nodes.geo.curve_circle(resolution=4, radius=0.7071)
    curve_circle = pf.nodes.geo.set_spline_cyclic(
        curve_circle, cyclic=has_bottom_connector
    )

    merge_curve_result = _merge_curve(curve=curve_circle)

    set_curve_tilt_tilt = pf.nodes.math.map_range(
        value=has_bottom_connector.astype(dtype=float),
        to_max=3.1416,
        to_min=1.5708,
    )
    set_curve_tilt = pf.nodes.geo.set_curve_tilt(
        curve=merge_curve_result, tilt=set_curve_tilt_tilt
    )

    transform = pf.nodes.geo.transform(
        geometry=set_curve_tilt,
        rotation=(0.0, 0.0, 0.7854),
        translation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    transform_1 = pf.nodes.geo.transform(
        geometry=transform,
        translation=(0.0, 0.0, -0.5),
        rotation=(0.0, 1.5708, 0.0),
        scale=(1, 1, 1),
    )
    n_gon_profile_result = _n_gon_profile(
        profile_n_gon=profile_n_gon,
        profile_width=profile_width,
        profile_aspect_ratio=profile_aspect_ratio,
        profile_fillet_ratio=profile_fillet_ratio,
    )
    profile_mesh = pf.nodes.geo.curve_to_mesh(n_gon_profile_result)
    profile_z_radius = pf.nodes.geo.bound_box(profile_mesh).max.y
    bottom_profile_radius = pf.nodes.func.switch(
        switch=has_bottom_connector, a=0.0, b=profile_z_radius
    )
    centerline_height = height - profile_z_radius - bottom_profile_radius
    transform_2_scale = pf.nodes.math.combine_xyz(x=1.0, y=width, z=centerline_height)
    transform_2 = pf.nodes.geo.transform(
        geometry=transform_1,
        scale=transform_2_scale,
        translation=pf.nodes.math.combine_xyz(z=profile_z_radius * -1.0),
        rotation=(0, 0, 0),
    )

    set_curve_radius = pf.nodes.geo.set_curve_radius(curve=transform_2, radius=1.0)

    fillet_curve = pf.nodes.geo.fillet_curve_poly(
        curve=set_curve_radius,
        radius=fillet_radius,
        limit_radius=True,
        count=4,
    )

    curve_to = curve.curve_to_mesh_with_uv(
        curve=fillet_curve,
        profile=n_gon_profile_result,
        fill_caps=True,
    ).mesh

    transform_3 = pf.nodes.geo.transform(
        geometry=curve_to,
        rotation=(0.0, 0.0, 0.0),
        translation=(0, 0, 0),
        scale=(1, 1, 1),
    )

    set_shade_smooth = pf.nodes.geo.set_shade_smooth(
        geometry=transform_3, shade_smooth=False
    )
    return set_shade_smooth


@pf.nodes.node_function
def _leg_straight(
    leg_height: t.SocketOrVal[float],
    leg_diameter: t.SocketOrVal[float],
    resolution: t.SocketOrVal[int],
    n_gon: t.SocketOrVal[int],
    fillet_ratio: t.SocketOrVal[float],
) -> t.ProcNode[pf.CurveObject]:
    radius_curve_result = pf.nodes.geo.curve_bezier(
        resolution=resolution,
        start=(1.0, 0.0, 1.0),
        middle=(0.95, 0.0, 0.0),
        end=(0.5, 0.0, -1.0),
    )

    n_gon_cylinder_result = _n_gon_cylinder(
        radius_curve=radius_curve_result,
        height=leg_height,
        n_gon=n_gon,
        profile_width=leg_diameter,
        aspect_ratio=1.0,
        fillet_ratio=fillet_ratio,
        profile_resolution=8,
        resolution=resolution,
    )
    return n_gon_cylinder_result.mesh


@pf.nodes.node_function
def base_four_leg(
    dimensions: t.SocketOrVal[pf.Vector],
    leg_diameter: t.SocketOrVal[float],
    leg_inset: t.SocketOrVal[float],
    leg_placement_bottom_scale: t.SocketOrVal[float],
    stretcher_increment: t.SocketOrVal[int],
    stretcher_relative_pos: t.SocketOrVal[float],
    leg_placement_top_scale: t.SocketOrVal[float] = 1.0,
) -> t.ProcNode[pf.MeshObject]:
    """4-leg base with optional stretchers."""
    leg_span_x = (
        dimensions.x - leg_diameter - 2.0 * leg_inset
    ) * leg_placement_top_scale
    leg_span_y = (
        dimensions.y - leg_diameter - 2.0 * leg_inset
    ) * leg_placement_top_scale
    anchors = _create_anchors(
        profile_n_gon=4,
        profile_width=1.414 * leg_span_x,
        profile_aspect_ratio=leg_span_y / leg_span_x,
        profile_rotation=0.0,
    )

    leg = _leg_straight(
        leg_height=1.0,
        leg_diameter=leg_diameter,
        resolution=8,
        n_gon=4,
        fillet_ratio=0.1,
    )

    stretcher_geo = _stretcher(n_gon=4, profile_width=leg_diameter * 0.5)

    return _create_legs_and_stretchers(
        anchors=anchors,
        keep_legs=True,
        leg_instance=leg,
        table_height=dimensions.z,
        leg_bottom_relative_scale=leg_placement_bottom_scale,
        leg_bottom_relative_rotation=0.0,
        keep_odd_stretchers=True,
        keep_even_stretchers=True,
        stretcher_instance=stretcher_geo,
        stretcher_index_increment=stretcher_increment,
        stretcher_relative_position=stretcher_relative_pos,
        leg_bottom_offset=0.0,
    )


@pf.nodes.node_function
def base_box_leg(
    dimensions: t.SocketOrVal[pf.Vector],
    leg_diameter: t.SocketOrVal[float],
    leg_placement_top_scale: t.SocketOrVal[float],
    leg_placement_bottom_scale: t.SocketOrVal[float],
    has_bottom_connector: t.SocketOrVal[bool],
) -> t.ProcNode[pf.MeshObject]:
    """2 box-frame legs."""
    anchors = _create_anchors(
        profile_n_gon=2,
        profile_width=1.414 * dimensions.x * leg_placement_top_scale,
        profile_aspect_ratio=dimensions.y / dimensions.x,
        profile_rotation=0.0,
    )

    leg = _leg_square(
        height=1.0,
        width=dimensions.y * leg_placement_top_scale,
        fillet_radius=0.03,
        has_bottom_connector=has_bottom_connector,
        profile_n_gon=4,
        profile_width=leg_diameter,
        profile_aspect_ratio=1.0,
        profile_fillet_ratio=0.1,
    )

    empty_stretcher = pf.nodes.geo.points(position=(0, 0, 0))

    return _create_legs_and_stretchers(
        anchors=anchors,
        keep_legs=True,
        leg_instance=leg,
        table_height=dimensions.z,
        leg_bottom_relative_scale=leg_placement_bottom_scale,
        leg_bottom_relative_rotation=0.0,
        keep_odd_stretchers=False,
        keep_even_stretchers=False,
        stretcher_instance=empty_stretcher,
        stretcher_index_increment=1,
        stretcher_relative_position=0.0,
        leg_bottom_offset=0.0,
    )


def table_dimensions_rand(
    rng: pf.RNG,
    width: float | None = None,
    height: float | None = None,
    depth: float | None = None,
) -> pf.Vector:
    """Default dining table dimensions."""
    aspect = pf.random.clip_gaussian(rng, 0.6, 0.2, 0.4, 1)

    if width is None and depth is not None:
        width = min(max(depth * aspect, 0.8), 1.1)
    if width is None:
        width = pf.random.clip_gaussian(rng, 0.975, 0.3, 0.675, 1.5)
    if height is None:
        height = pf.random.uniform(rng, 0.72, 0.76)
    if depth is None:
        depth = width / aspect
    width = min(width, 1.875)
    return (width, depth, height)


def _finish(
    geo: pf.ProcNode[pf.MeshObject], material: pf.Material | None
) -> TableResult:
    geo = mesh.crease_sharp(geo, threshold_degrees=40.0)
    obj = pf.nodes.to_mesh_object(geo)
    if material is not None:
        pf.ops.object.set_material(
            obj, surface=material.surface, displacement=material.displacement
        )
    return TableResult(mesh=obj)


def _subdivided(obj: pf.MeshObject, levels: int) -> TableResult:
    pf.ops.modifier.subdivide_surface(obj, levels=levels, _skip_apply=True)
    return TableResult(mesh=obj)


@pf.tracer.generator
def base_straight(
    dimensions: pf.Vector | None = None,
    leg_diameter: float = 0.06,
    leg_inset: float = 0.1,
    leg_placement_bottom_scale: float = 1.0,
    stretcher_increment: int = 1,
    stretcher_relative_pos: float = 0.4,
    leg_placement_top_scale: float = 1.0,
    material: pf.Material | None = None,
) -> TableResult:
    """4-leg base with optional stretchers."""
    if dimensions is None:
        dimensions = pf.Vector((1.4, 0.8, 0.75))
    geo = base_four_leg(
        dimensions=dimensions,
        leg_diameter=leg_diameter,
        leg_inset=leg_inset,
        leg_placement_bottom_scale=leg_placement_bottom_scale,
        stretcher_increment=stretcher_increment,
        stretcher_relative_pos=stretcher_relative_pos,
        leg_placement_top_scale=leg_placement_top_scale,
    )
    return _finish(geo, material)


def base_straight_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    material: pf.Material | None = None,
    leg_diameter: float | None = None,
    leg_placement_bottom_scale: float | None = None,
    leg_diameter_range: tuple[float, float] = (0.02, 0.10),
    leg_inset_range: tuple[float, float] | None = None,
    leg_placement_top_scale: float = 0.8,
    close_edges: bool = False,
) -> TableResult:
    """4-leg base with optional stretchers. leg_placement_bottom_scale > 1 splays
    the legs outward; pass 1.0 for an upright base to sit under a carcass."""
    rng_parameters, rng_dims, rng_mat, rng_inset, rng_stretcher = rng.spawn(5)
    if dimensions is None:
        dimensions = table_dimensions_rand(rng_dims)
    # drawn unconditionally so explicit overrides do not shift the stream
    sampled_diameter = pf.random.uniform(rng_parameters, *leg_diameter_range)
    if leg_diameter is None:
        leg_diameter = sampled_diameter
    sampled_bottom_scale = pf.random.uniform(rng_parameters, 0.95, 1.25)
    if leg_placement_bottom_scale is None:
        leg_placement_bottom_scale = 1.0 if close_edges else sampled_bottom_scale
    leg_inset = 0.0
    if leg_inset_range is not None and not close_edges:
        leg_inset = pf.random.uniform(rng_inset, *leg_inset_range)
    stretcher_increment = pf.control.choice(
        rng_stretcher, [(0, 1.0), (1, 1.0), (2, 1.0)]
    )
    stretcher_relative_pos = pf.random.uniform(rng_parameters, 0.2, 0.6)
    if material is None:
        material = furniture_material_rand(rng_mat, pf.nodes.shader.coord().uv)
    obj = base_straight(
        dimensions=dimensions,
        leg_diameter=leg_diameter,
        leg_inset=leg_inset,
        leg_placement_bottom_scale=leg_placement_bottom_scale,
        stretcher_increment=stretcher_increment,
        stretcher_relative_pos=stretcher_relative_pos,
        leg_placement_top_scale=1.0 if close_edges else leg_placement_top_scale,
        material=material,
    ).mesh
    return _subdivided(obj, levels=1)


@pf.tracer.generator
def base_pedestal(
    height: float,
    top_radius: float = 0.03,
    bottom_radius: float = 0.2,
    flare: float = 0.75,
    concavity: float = 0.0,
    neck_scale: float = 1.0,
    resolution: int = 16,
    profile_resolution: int | None = None,
    material: pf.Material | None = None,
) -> TableResult:
    """Rotationally symmetric pedestal: stays skinny then flares out low
    (flare/concavity); neck_scale bulges or waists the upper column.
    `profile_resolution` defaults to `resolution`."""
    if profile_resolution is None:
        profile_resolution = resolution
    geo = base_pedestal_column(
        height=height,
        top_radius=top_radius,
        bottom_radius=bottom_radius,
        flare=flare,
        concavity=concavity,
        neck_scale=neck_scale,
        profile_resolution=profile_resolution,
        resolution=resolution,
    )
    return _finish(geo, material)


def base_pedestal_column_rand(
    rng: pf.RNG, height: float, top_radius: float, bottom_radius: float
) -> pf.ProcNode[pf.MeshObject]:
    rng_flare, rng_concavity, rng_neck = rng.spawn(3)
    return base_pedestal_column(
        height=height,
        top_radius=top_radius,
        bottom_radius=bottom_radius,
        flare=pf.random.uniform(rng_flare, 0.5, 0.85),
        concavity=pf.random.uniform(rng_concavity, 0.0, 1.0),
        neck_scale=pf.random.uniform(rng_neck, 0.5, 2.0),
    )


def base_pedestal_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    material: pf.Material | None = None,
    top_radius_range: tuple[float, float] | None = None,
    bottom_radius_range: tuple[float, float] | None = None,
) -> TableResult:
    """Rotationally symmetric pedestal spanning z in [0, dimensions.z]: a
    low-flare silhouette with random concavity and a bulged/waisted upper
    column. Radius ranges default from the footprint; callers override them."""
    rng, rng_dims, rng_mat, rng_profile = rng.spawn(4)
    if dimensions is None:
        dimensions = table_dimensions_rand(rng_dims)
    x = dimensions[0]
    y = dimensions[1]
    z = dimensions[2]
    footprint = min(x, y)
    if top_radius_range is None:
        top_radius_range = (0.015 * x, 0.06 * x)
    if bottom_radius_range is None:
        bottom_radius_range = (0.30 * footprint, 0.55 * footprint)
    top_radius = pf.random.uniform(rng, *top_radius_range)
    bottom_radius = pf.random.uniform(rng, *bottom_radius_range)
    rng_flare, rng_concavity, rng_neck = rng_profile.spawn(3)
    flare = pf.random.uniform(rng_flare, 0.5, 0.85)
    concavity = pf.random.uniform(rng_concavity, 0.0, 1.0)
    neck_scale = pf.random.uniform(rng_neck, 0.5, 2.0)
    if material is None:
        material = furniture_material_rand(rng_mat, pf.nodes.shader.coord().uv)
    obj = base_pedestal(
        z,
        top_radius=top_radius,
        bottom_radius=bottom_radius,
        flare=flare,
        concavity=concavity,
        neck_scale=neck_scale,
        resolution=16,
        profile_resolution=12,
        material=material,
    ).mesh
    return _subdivided(obj, levels=3)


@pf.tracer.generator
def base_square(
    dimensions: pf.Vector | None = None,
    leg_diameter: float = 0.085,
    leg_placement_top_scale: float = 0.8,
    leg_placement_bottom_scale: float = 1.0,
    has_bottom_connector: bool = True,
    material: pf.Material | None = None,
) -> TableResult:
    """2 box-frame legs."""
    if dimensions is None:
        dimensions = pf.Vector((1.4, 0.8, 0.75))
    geo = base_box_leg(
        dimensions=dimensions,
        leg_diameter=leg_diameter,
        leg_placement_top_scale=leg_placement_top_scale,
        leg_placement_bottom_scale=leg_placement_bottom_scale,
        has_bottom_connector=has_bottom_connector,
    )
    return _finish(geo, material)


def base_square_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    material: pf.Material | None = None,
    leg_diameter: float | None = None,
    leg_placement_bottom_scale: float | None = None,
    leg_diameter_range: tuple[float, float] = (0.03, 0.14),
    leg_placement_top_scale: float = 0.8,
    close_edges: bool = False,
) -> TableResult:
    """2 box-frame legs. leg_placement_bottom_scale > 1 splays the frames outward."""
    rng_parameters, rng_dims, rng_mat, rng_connector = rng.spawn(4)
    if dimensions is None:
        dimensions = table_dimensions_rand(rng_dims)
    # drawn unconditionally so an explicit leg_diameter does not shift the stream
    sampled_diameter = pf.random.uniform(rng_parameters, *leg_diameter_range)
    if leg_diameter is None:
        leg_diameter = sampled_diameter
    if leg_placement_bottom_scale is None:
        leg_placement_bottom_scale = 0.98 if close_edges else 1.0
    has_bottom_connector = pf.control.choice(rng_connector, [(True, 2.0), (False, 1.0)])
    if material is None:
        material = furniture_material_rand(rng_mat, pf.nodes.shader.coord().uv)
    obj = base_square(
        dimensions=dimensions,
        leg_diameter=leg_diameter,
        leg_placement_top_scale=0.94 if close_edges else leg_placement_top_scale,
        leg_placement_bottom_scale=leg_placement_bottom_scale,
        has_bottom_connector=has_bottom_connector,
        material=material,
    ).mesh
    return _subdivided(obj, levels=1)


def table_side_dimensions_rand(rng: pf.RNG) -> pf.Vector:
    """Side table dimensions."""
    return (
        pf.random.uniform(rng, 0.4, 0.6),
        pf.random.uniform(rng, 0.4, 0.6),
        pf.random.uniform(rng, 0.3, 0.5),
    )


def table_coffee_dimensions_rand(rng: pf.RNG) -> pf.Vector:
    """Coffee table dimensions."""
    return (
        pf.random.uniform(rng, 0.6, 0.9),
        pf.random.uniform(rng, 1.0, 1.5),
        pf.random.uniform(rng, 0.3, 0.5),
    )
