from typing import NamedTuple

import procfunc as pf
from procfunc.nodes import types as t

from infinigen_v2.generators.shaders.functionality_lists import (
    furniture_material_distribution,
    table_top_material_distribution,
)


class TableResult(NamedTuple):
    mesh: pf.MeshObject


@pf.nodes.node_function
def n_gon_profile(
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
        count=8,
    )
    return fillet_curve


@pf.nodes.node_function
def merge_curve(
    curve: t.ProcNode[pf.CurveObject],
) -> t.ProcNode[pf.CurveObject]:
    curve_to = pf.nodes.geo.curve_to_mesh(curve)

    merge_by_distance = pf.nodes.geo.merge_by_distance(curve_to)

    to_curve = pf.nodes.geo.mesh_to_curve(merge_by_distance)
    return to_curve


class NGonCylinderResult(NamedTuple):
    mesh: t.ProcNode[pf.MeshObject]
    profile_curve: t.ProcNode[pf.CurveObject]
    caps: t.ProcNode[pf.MeshObject]


@pf.nodes.node_function
def n_gon_cylinder(
    radius_curve: t.ProcNode[pf.CurveObject],
    n_gon: t.SocketOrVal[int],
    profile_width: t.SocketOrVal[float],
    height: t.SocketOrVal[float] = 1.0,
    aspect_ratio: t.SocketOrVal[float] = 1.0,
    fillet_ratio: t.SocketOrVal[float] = 0.2,
    profile_resolution: t.SocketOrVal[int] = 64,
    resolution: t.SocketOrVal[int] = 64,
) -> NGonCylinderResult:
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

    n_gon_profile_result = n_gon_profile(
        profile_n_gon=n_gon,
        profile_width=profile_width,
        profile_aspect_ratio=aspect_ratio,
        profile_fillet_ratio=fillet_ratio,
    )

    resample_curve_count = pf.nodes.geo.resample_curve_count(
        curve=n_gon_profile_result,
        count=profile_resolution,
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
    return NGonCylinderResult(
        mesh=set_position,
        profile_curve=resample_curve_count,
        caps=delete,
    )


@pf.nodes.node_function
def strecher(
    n_gon: t.SocketOrVal[int],
    profile_width: t.SocketOrVal[float],
) -> t.ProcNode[pf.MeshObject]:
    curve_line = pf.nodes.geo.curve_line(start=(1.0, 0.0, 1.0), end=(1.0, 0.0, -1.0))

    n_gon_cylinder_result = n_gon_cylinder(
        radius_curve=curve_line,
        height=1.0,
        n_gon=n_gon,
        profile_width=profile_width,
        aspect_ratio=1.0,
        fillet_ratio=0.2,
        profile_resolution=64,
        resolution=64,
    )
    return n_gon_cylinder_result.mesh


@pf.nodes.node_function
def create_anchors(
    profile_n_gon: t.SocketOrVal[int],
    profile_width: t.SocketOrVal[float],
    profile_aspect_ratio: t.SocketOrVal[float],
    profile_rotation: t.SocketOrVal[float],
) -> t.ProcNode[pf.MeshObject]:
    set_switch = pf.nodes.func.equal(a=profile_n_gon, b=1)
    set_a_switch = pf.nodes.func.equal(a=profile_n_gon, b=2)

    n_gon_profile_result = n_gon_profile(
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

    result_0_rotation = pf.nodes.math.combine_xyz(z=profile_rotation)

    transform = pf.nodes.geo.transform(
        geometry=set_point_radius,
        rotation=result_0_rotation.astype(dtype=pf.Euler),
        translation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    return transform


@pf.nodes.node_function
def create_legs_and_strechers(
    anchors: t.ProcNode[pf.MeshObject],
    keep_legs: t.SocketOrVal[bool],
    leg_instance: t.ProcNode[pf.MeshObject],
    table_height: t.SocketOrVal[float],
    leg_bottom_relative_scale: t.SocketOrVal[float],
    leg_bottom_relative_rotation: t.SocketOrVal[float],
    keep_odd_strechers: t.SocketOrVal[bool],
    keep_even_strechers: t.SocketOrVal[bool],
    strecher_instance: t.ProcNode[pf.MeshObject],
    strecher_index_increment: t.SocketOrVal[int],
    strecher_relative_position: t.SocketOrVal[float],
    leg_bottom_offset: t.SocketOrVal[float],
    align_leg_x_rot: t.SocketOrVal[bool],
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
    set_b_0 = pf.nodes.math.combine_xyz(
        x=leg_bottom_relative_scale,
        y=leg_bottom_relative_scale,
        z=1.0,
    )
    set_position_position_vector = input_position - (set_b_1 * set_b_0)
    set_position_position = pf.nodes.math.vector_scale(
        vector=set_position_position_vector,
        scale=strecher_relative_position * -1.0,
    )

    input_position_1 = pf.nodes.geo.input_position()

    set_position = pf.nodes.geo.set_position(
        geometry=transform,
        position=set_position_position + input_position_1,
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

    instance_b_switch = pf.nodes.func.equal(
        a=attribute_domain_size.point_count.astype(dtype=float)
        / strecher_index_increment.astype(dtype=float),
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

    field = input_index.astype(dtype=float) + strecher_index_increment.astype(
        dtype=float
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
    instance_on_points = pf.nodes.geo.instance_on_points(
        points=set_position,
        instance=strecher_instance,
        selection=instance_on_points_selection,
        rotation=instance.astype(dtype=pf.Euler),
        scale=instance_on_points_scale,
    )

    realize_instances = pf.nodes.geo.realize_instances(instance_on_points)

    instance_rotation_a = pf.nodes.func.align_euler_to_vector(
        vector=set_position_position_vector,
        axis="Z",
        rotation=(0, 0, 0),
        factor=1.0,
    )
    instance_rotation_b = pf.nodes.func.align_euler_to_vector(
        rotation=instance_rotation_a,
        vector=input_position,
        pivot_axis="Z",
        factor=1.0,
    )
    instance_rotation = pf.nodes.func.switch(
        switch=align_leg_x_rot,
        a=instance_rotation_a,
        b=instance_rotation_b,
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

    result_0_geometries = pf.nodes.func.switch(switch=keep_legs, b=realize_instances_1)

    join = pf.nodes.geo.join_geometry([realize_instances, result_0_geometries])
    return join


@pf.nodes.node_function
def single_stand(
    leg_height: t.SocketOrVal[float],
    leg_diameter: t.SocketOrVal[float],
    resolution: t.SocketOrVal[int],
    top_radius: t.SocketOrVal[float] = 0.5,
    middle_radius: t.SocketOrVal[float] = 0.7,
    bottom_radius: t.SocketOrVal[float] = 1.0,
) -> t.ProcNode[pf.MeshObject]:
    radius_curve_result = pf.nodes.geo.curve_bezier(
        resolution=resolution,
        start=pf.nodes.math.combine_xyz(x=top_radius, z=1.0),
        middle=pf.nodes.math.combine_xyz(x=middle_radius, z=0.0),
        end=pf.nodes.math.combine_xyz(x=bottom_radius, z=-1.0),
    )

    n_gon_cylinder_result = n_gon_cylinder(
        radius_curve=radius_curve_result,
        height=leg_height,
        n_gon=resolution,
        profile_width=leg_diameter,
        aspect_ratio=1.0,
        fillet_ratio=0.0,
        profile_resolution=64,
        resolution=resolution,
    )
    return n_gon_cylinder_result.mesh


@pf.nodes.node_function
def leg_square(
    width: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    fillet_radius: t.SocketOrVal[float],
    has_bottom_connector: t.SocketOrVal[bool],
    profile_n_gon: t.SocketOrVal[int],
    profile_width: t.SocketOrVal[float],
    profile_aspect_ratio: t.SocketOrVal[float],
    profile_fillet_ratio: t.SocketOrVal[float],
) -> t.ProcNode[pf.MeshObject]:
    curve_arc_resolution = has_bottom_connector.astype(dtype=float) + 4.0
    curve_arc_sweep_angle = pf.nodes.math.map_range(
        value=has_bottom_connector.astype(dtype=float),
        to_max=6.2832,
        to_min=4.7124,
    )
    curve_arc = pf.nodes.geo.curve_arc(
        resolution=curve_arc_resolution.astype(dtype=int),
        radius=0.7071,
        sweep_angle=curve_arc_sweep_angle,
    )

    merge_curve_result = merge_curve(curve=curve_arc)

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
        rotation=(0.0, 0.0, -0.7854),
        translation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    transform_1 = pf.nodes.geo.transform(
        geometry=transform,
        translation=(0.0, 0.0, -0.5),
        rotation=(1.5708, 0.0, 0.0),
        scale=(1, 1, 1),
    )
    transform_2_scale = pf.nodes.math.combine_xyz(x=width, y=1.0, z=height)
    transform_2 = pf.nodes.geo.transform(
        geometry=transform_1,
        scale=transform_2_scale,
        translation=(0, 0, 0),
        rotation=(0, 0, 0),
    )

    set_curve_radius = pf.nodes.geo.set_curve_radius(curve=transform_2, radius=1.0)

    fillet_curve = pf.nodes.geo.fillet_curve_poly(
        curve=set_curve_radius,
        radius=fillet_radius,
        limit_radius=True,
        count=8,
    )

    n_gon_profile_result = n_gon_profile(
        profile_n_gon=profile_n_gon,
        profile_width=profile_width,
        profile_aspect_ratio=profile_aspect_ratio,
        profile_fillet_ratio=profile_fillet_ratio,
    )

    curve_to = pf.nodes.geo.curve_to_mesh(
        curve=fillet_curve,
        profile_curve=n_gon_profile_result,
        fill_caps=True,
    )

    transform_3 = pf.nodes.geo.transform(
        geometry=curve_to,
        rotation=(0.0, 0.0, 1.5708),
        translation=(0, 0, 0),
        scale=(1, 1, 1),
    )

    set_shade_smooth = pf.nodes.geo.set_shade_smooth(
        geometry=transform_3, shade_smooth=False
    )
    return set_shade_smooth


@pf.nodes.node_function
def leg_straight(
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

    n_gon_cylinder_result = n_gon_cylinder(
        radius_curve=radius_curve_result,
        height=leg_height,
        n_gon=n_gon,
        profile_width=leg_diameter,
        aspect_ratio=1.0,
        fillet_ratio=fillet_ratio,
        profile_resolution=64,
        resolution=resolution,
    )
    return n_gon_cylinder_result.mesh


class TableTopResult(NamedTuple):
    geometry: t.ProcNode[pf.MeshObject]
    curve: pf.ProcNode


@pf.nodes.node_function
def table_top(
    thickness: t.SocketOrVal[float],
    n_gon: t.SocketOrVal[int],
    profile_width: t.SocketOrVal[float],
    aspect_ratio: t.SocketOrVal[float],
    fillet_ratio: t.SocketOrVal[float],
    fillet_radius_vertical: t.SocketOrVal[float],
) -> TableTopResult:
    curve_line = pf.nodes.geo.curve_line(start=(1.0, 0.0, 1.0), end=(1.0, 0.0, -1.0))

    n_gon_cylinder_result = n_gon_cylinder(
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

    join_geometries_0_selection = pf.nodes.func.equal(a=input_index, b=0)

    store_named_attribute = pf.nodes.geo.store_named_attribute(
        geometry=n_gon_cylinder_result.caps,
        name="TAG_support",
        selection=join_geometries_0_selection,
        value=True,
        domain="FACE",
    )

    curve_arc = pf.nodes.geo.curve_arc(resolution=4, radius=0.7071, sweep_angle=4.7124)

    transform_1 = pf.nodes.geo.transform(
        geometry=curve_arc,
        rotation=(0.0, 0.0, -0.7854),
        translation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    transform_2 = pf.nodes.geo.transform(
        geometry=transform_1,
        rotation=(0.0, 1.5708, 0.0),
        translation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    transform_3 = pf.nodes.geo.transform(
        geometry=transform_2,
        translation=(0.0, 0.5, 0.0),
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    transform_4_scale = pf.nodes.math.combine_xyz(
        x=1.0, y=fillet_radius_vertical, z=1.0
    )
    transform_4 = pf.nodes.geo.transform(
        geometry=transform_3,
        scale=transform_4_scale,
        translation=(0, 0, 0),
        rotation=(0, 0, 0),
    )

    fillet_curve = pf.nodes.geo.fillet_curve_poly(
        curve=transform_4,
        radius=fillet_radius_vertical,
        limit_radius=True,
        count=8,
    )

    transform_5 = pf.nodes.geo.transform(
        geometry=fillet_curve,
        rotation=(1.5708, 1.5708, 0.0),
        scale=thickness.astype(dtype=pf.Vector),
        translation=(0, 0, 0),
    )

    curve_to = pf.nodes.geo.curve_to_mesh(
        curve=n_gon_cylinder_result.profile_curve,
        profile_curve=transform_5,
    )

    transform_6_translation = pf.nodes.math.combine_xyz(z=thickness * -0.5)
    transform_6 = pf.nodes.geo.transform(
        geometry=curve_to,
        translation=transform_6_translation,
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )

    flip_edge = pf.nodes.geo.flip_faces(transform_6)

    join = pf.nodes.geo.join_geometry([store_named_attribute, flip_edge])

    geometry_translation = pf.nodes.math.combine_xyz(z=thickness)

    transform = pf.nodes.geo.transform(
        geometry=join,
        translation=geometry_translation,
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    return TableTopResult(
        geometry=transform,
        curve=n_gon_cylinder_result.profile_curve,
    )


@pf.nodes.node_function
def base_straight(
    dimensions: t.SocketOrVal[pf.Vector],
    leg_diameter: t.SocketOrVal[float],
    leg_placement_top_scale: t.SocketOrVal[float],
    leg_placement_bottom_scale: t.SocketOrVal[float],
    stretcher_increment: t.SocketOrVal[int],
    stretcher_relative_pos: t.SocketOrVal[float],
) -> t.ProcNode[pf.MeshObject]:
    """4-leg base with optional stretchers."""
    x, y, z = dimensions.x, dimensions.y, dimensions.z
    anchors = create_anchors(
        profile_n_gon=4,
        profile_width=1.414 * x * leg_placement_top_scale,
        profile_aspect_ratio=y / x,
        profile_rotation=0.0,
    )

    leg = leg_straight(
        leg_height=1.0,
        leg_diameter=leg_diameter,
        resolution=32,
        n_gon=4,
        fillet_ratio=0.1,
    )

    stretcher_geo = strecher(n_gon=4, profile_width=leg_diameter * 0.5)

    return create_legs_and_strechers(
        anchors=anchors,
        keep_legs=True,
        leg_instance=leg,
        table_height=z,
        leg_bottom_relative_scale=leg_placement_bottom_scale,
        leg_bottom_relative_rotation=0.0,
        keep_odd_strechers=True,
        keep_even_strechers=True,
        strecher_instance=stretcher_geo,
        strecher_index_increment=stretcher_increment,
        strecher_relative_position=stretcher_relative_pos,
        leg_bottom_offset=0.0,
        align_leg_x_rot=True,
    )


@pf.nodes.node_function
def base_square(
    dimensions: t.SocketOrVal[pf.Vector],
    leg_diameter: t.SocketOrVal[float],
    leg_placement_top_scale: t.SocketOrVal[float],
    leg_placement_bottom_scale: t.SocketOrVal[float],
    has_bottom_connector: t.SocketOrVal[bool],
) -> t.ProcNode[pf.MeshObject]:
    """2 box-frame legs."""
    x, y, z = dimensions.x, dimensions.y, dimensions.z
    anchors = create_anchors(
        profile_n_gon=2,
        profile_width=1.414 * x * leg_placement_top_scale,
        profile_aspect_ratio=y / x,
        profile_rotation=0.0,
    )

    leg = leg_square(
        height=1.0,
        width=y * leg_placement_top_scale,
        fillet_radius=0.03,
        has_bottom_connector=has_bottom_connector,
        profile_n_gon=4,
        profile_width=leg_diameter,
        profile_aspect_ratio=0.5,
        profile_fillet_ratio=0.1,
    )

    empty_stretcher = pf.nodes.geo.points(position=(0, 0, 0))

    return create_legs_and_strechers(
        anchors=anchors,
        keep_legs=True,
        leg_instance=leg,
        table_height=z,
        leg_bottom_relative_scale=leg_placement_bottom_scale,
        leg_bottom_relative_rotation=0.0,
        keep_odd_strechers=False,
        keep_even_strechers=False,
        strecher_instance=empty_stretcher,
        strecher_index_increment=1,
        strecher_relative_position=0.0,
        leg_bottom_offset=0.0,
        align_leg_x_rot=True,
    )


@pf.nodes.node_function
def base_single_stand(
    dimensions: t.SocketOrVal[pf.Vector],
    leg_diameter: t.SocketOrVal[float],
    leg_placement_top_scale: t.SocketOrVal[float],
    leg_placement_bottom_scale: t.SocketOrVal[float],
    top_radius: t.SocketOrVal[float] = 0.5,
    middle_radius: t.SocketOrVal[float] = 0.7,
    bottom_radius: t.SocketOrVal[float] = 1.0,
) -> t.ProcNode[pf.MeshObject]:
    """Single central pedestal."""
    x, y, z = dimensions.x, dimensions.y, dimensions.z
    anchors = create_anchors(
        profile_n_gon=1,
        profile_width=1.414 * x * leg_placement_top_scale,
        profile_aspect_ratio=y / x,
        profile_rotation=0.0,
    )

    leg = single_stand(
        leg_height=1.0,
        leg_diameter=leg_diameter,
        resolution=64,
        top_radius=top_radius,
        middle_radius=middle_radius,
        bottom_radius=bottom_radius,
    )

    empty_stretcher = pf.nodes.geo.points(position=(0, 0, 0))

    return create_legs_and_strechers(
        anchors=anchors,
        keep_legs=True,
        leg_instance=leg,
        table_height=z,
        leg_bottom_relative_scale=leg_placement_bottom_scale,
        leg_bottom_relative_rotation=0.0,
        keep_odd_strechers=False,
        keep_even_strechers=False,
        strecher_instance=empty_stretcher,
        strecher_index_increment=1,
        strecher_relative_position=0.0,
        leg_bottom_offset=0.0,
        align_leg_x_rot=True,
    )


def table_dimensions_distribution(
    rng: pf.RNG,
    width: float | None = None,
    height: float | None = None,
) -> pf.Vector:
    """Default dining table dimensions."""
    aspect = pf.random.clip_gaussian(rng, 0.6, 0.2, 0.4, 1)

    if width is None:
        width = pf.random.clip_gaussian(rng, 1.3, 0.4, 0.9, 2)
    if height is None:
        height = pf.random.uniform(rng, 0.65, 0.85)
    depth = width / aspect
    width = min(width, 2.5)
    return (width, depth, height)


def base_straight_distribution(
    rng: pf.RNG, dimensions: pf.Vector | None = None
) -> TableResult:
    """4-leg base with optional stretchers."""
    if dimensions is None:
        dimensions = table_dimensions_distribution(rng)
    geo = base_straight(
        dimensions=dimensions,
        leg_diameter=pf.random.uniform(rng, 0.05, 0.07),
        leg_placement_top_scale=0.8,
        leg_placement_bottom_scale=pf.random.uniform(rng, 1.0, 1.2),
        stretcher_increment=pf.control.choice(rng, [(0, 1.0), (1, 1.0), (2, 1.0)]),
        stretcher_relative_pos=pf.random.uniform(rng, 0.2, 0.6),
    )
    return TableResult(mesh=pf.nodes.to_mesh_object(geo))


def base_single_stand_distribution(
    rng: pf.RNG, dimensions: pf.Vector | None = None
) -> TableResult:
    """2 pedestal legs."""
    if dimensions is None:
        dimensions = table_dimensions_distribution(rng)
    geo = base_single_stand(
        dimensions=dimensions,
        leg_diameter=pf.random.uniform(rng, 0.22 * dimensions[0], 0.28 * dimensions[0]),
        leg_placement_top_scale=pf.random.uniform(rng, 0.6, 0.7),
        leg_placement_bottom_scale=1.0,
        top_radius=pf.random.uniform(rng, 0.1, 0.36),
        middle_radius=pf.random.uniform(rng, 0.1, 0.6),
        bottom_radius=pf.random.uniform(rng, 1.275, 2.1),
    )
    return TableResult(mesh=pf.nodes.to_mesh_object(geo))


def base_square_distribution(
    rng: pf.RNG, dimensions: pf.Vector | None = None
) -> TableResult:
    """2 box-frame legs."""
    if dimensions is None:
        dimensions = table_dimensions_distribution(rng)
    geo = base_square(
        dimensions=dimensions,
        leg_diameter=pf.random.uniform(rng, 0.07, 0.10),
        leg_placement_top_scale=0.8,
        leg_placement_bottom_scale=1.0,
        has_bottom_connector=pf.control.choice(rng, [(True, 2.0), (False, 1.0)]),
    )
    return TableResult(mesh=pf.nodes.to_mesh_object(geo))


def dining_table_distribution(
    rng: pf.RNG,
    dimensions: tuple[float, float, float] | None = None,
    base: pf.MeshObject | None = None,
    top_thickness: float | None = None,
    top_material: pf.Material | None = None,
    leg_material: pf.Material | None = None,
) -> TableResult:
    if dimensions is None:
        dimensions = table_dimensions_distribution(rng)

    x, y, z = dimensions

    if top_thickness is None:
        top_thickness = pf.random.uniform(rng, 0.03, 0.08)

    top_profile_width = 1.414 * x
    top_profile_aspect_ratio = y / x
    top_height = z - top_thickness

    vec = pf.nodes.shader.geometry().position
    if top_material is None:
        top_material = table_top_material_distribution(rng, vec)
    if leg_material is None:
        leg_material = furniture_material_distribution(rng, vec)

    top = table_top(
        thickness=top_thickness,
        n_gon=4,
        profile_width=top_profile_width,
        aspect_ratio=top_profile_aspect_ratio,
        fillet_ratio=pf.random.uniform(rng, 0.0, 0.02),
        fillet_radius_vertical=pf.random.uniform(rng, 0.1, 0.3),
    )
    top = pf.nodes.geo.transform(
        top.geometry,
        translation=(0, 0, top_height),
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    top = pf.nodes.to_mesh_object(top)
    pf.ops.object.set_material(
        top, surface=top_material.surface, displacement=top_material.displacement
    )

    if base is None:
        base_fn = pf.control.choice(
            rng,
            [
                (base_straight_distribution, 2.0),
                (base_single_stand_distribution, 1.0),
                (base_square_distribution, 0.6),
            ],
        )
        res = base_fn(rng=rng, dimensions=(x, y, top_height))
        base = res.mesh

    pf.ops.object.set_material(
        base, surface=leg_material.surface, displacement=leg_material.displacement
    )

    pf.ops.object.join(top, base)
    return TableResult(mesh=top)


def side_table_dimensions_distribution(rng: pf.RNG) -> pf.Vector:
    """Side table dimensions."""
    return (
        pf.random.uniform(rng, 0.4, 0.6),
        pf.random.uniform(rng, 0.4, 0.6),
        pf.random.uniform(rng, 0.3, 0.5),
    )


def side_table_distribution(rng: pf.RNG) -> TableResult:
    """Side table."""
    dimensions = side_table_dimensions_distribution(rng)
    top_thickness = pf.random.uniform(rng, 0.0, 0.04)
    return dining_table_distribution(rng, dimensions, top_thickness=top_thickness)


def coffee_table_distribution(rng: pf.RNG) -> TableResult:
    """Low rectangular coffee table."""
    dimensions = (
        pf.random.uniform(rng, 0.6, 0.9),
        pf.random.uniform(rng, 1.0, 1.5),
        pf.random.uniform(rng, 0.3, 0.5),
    )
    top_thickness = pf.random.uniform(rng, 0.02, 0.04)
    return dining_table_distribution(rng, dimensions, top_thickness=top_thickness)


def cocktail_table_distribution(rng: pf.RNG) -> TableResult:
    """Tall square cocktail/bar table with single pedestal base."""
    x = pf.random.uniform(rng, 0.5, 0.8)
    height = pf.random.uniform(rng, 1.0, 1.5)
    top_height = height - 0.04  # approximate top_thickness
    base = base_single_stand_distribution(rng, (x, x, top_height))
    return dining_table_distribution(rng, (x, x, height), base=base.mesh)


if __name__ == "__main__":
    table_top_result = table_top()
    leg_straight_result = leg_straight()
    leg_square_result = leg_square()
    single_stand_result = single_stand()

    create_legs_and_strechers_result = create_legs_and_strechers()
    create_anchors_result = create_anchors()

    strecher_result = strecher()
