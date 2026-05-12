from typing import NamedTuple

import procfunc as pf
from procfunc.nodes import types as t

from infinigen_v2.generators.shaders.functionality_lists import (
    furniture_material_distribution,
    table_top_material_distribution,
)
from infinigen_v2.generators.shaders.materials import metal_brushed


class TriangleShelfResult(NamedTuple):
    mesh: pf.MeshObject


@pf.nodes.node_function
def table_profile(
    profile_n_gon: t.SocketOrVal[int],
    profile_width: t.SocketOrVal[float],
    profile_aspect_ratio: t.SocketOrVal[float],
    profile_fillet_ratio: t.SocketOrVal[float],
) -> pf.ProcNode:
    curve_circle_radius = pf.nodes.func.constant(0.7071)
    curve_circle = pf.nodes.geo.curve_circle(
        resolution=profile_n_gon, radius=curve_circle_radius
    )

    transform_rotation = pf.nodes.func.combine_xyz(
        z=3.1416 / profile_n_gon.astype(dtype=float)
    )
    transform = pf.nodes.geo.transform(
        geometry=curve_circle,
        rotation=transform_rotation.astype(dtype=pf.Euler),
    )
    transform_1 = pf.nodes.geo.transform(
        geometry=transform, rotation=(0.0, 0.0, -1.5708)
    )
    transform_2_scale = pf.nodes.func.combine_xyz(
        x=profile_width,
        y=profile_aspect_ratio * profile_width,
        z=1.0,
    )
    transform_2 = pf.nodes.geo.transform(geometry=transform_1, scale=transform_2_scale)

    fillet_curve = pf.nodes.geo.fillet_curve(
        curve=transform_2,
        radius=profile_width * profile_fillet_ratio,
        limit_radius=True,
        count=4,
        mode="POLY",
    )
    return fillet_curve


class LegStraightResult(NamedTuple):
    mesh: pf.ProcNode
    profile_curve: pf.ProcNode


@pf.nodes.node_function
def leg_straight(
    height: t.SocketOrVal[float],
    n_gon: t.SocketOrVal[int],
    profile_width: t.SocketOrVal[float],
    aspect_ratio: t.SocketOrVal[float],
    fillet_ratio: t.SocketOrVal[float],
    resolution: t.SocketOrVal[int] = 128,
) -> LegStraightResult:
    mesh_position_z_to_min = height * -1.0

    curve_line_end = pf.nodes.func.combine_xyz(z=mesh_position_z_to_min)
    curve_line = pf.nodes.geo.curve_line(end=curve_line_end)

    set_curve_tilt = pf.nodes.geo.set_curve_tilt(curve=curve_line, tilt=3.1416)

    resample_curve_count = pf.nodes.geo.resample_curve_count(
        curve=set_curve_tilt, count=resolution
    )

    spline_parameter = pf.nodes.geo.spline_parameter()

    capture_attribute = pf.nodes.geo.capture_attribute(
        geometry=resample_curve_count,
        attribute=spline_parameter.factor,
    )

    table_profile_result = table_profile(
        profile_n_gon=n_gon,
        profile_width=profile_width,
        profile_aspect_ratio=aspect_ratio,
        profile_fillet_ratio=fillet_ratio,
    )

    curve_to = pf.nodes.geo.curve_to_mesh(
        curve=capture_attribute.geometry,
        profile_curve=table_profile_result,
        fill_caps=True,
    )

    profile_curve = pf.nodes.geo.curve_line(start=(1.0, 0.0, -1.0), end=(1.0, 0.0, 1.0))

    input_position = pf.nodes.geo.input_position()

    sample_curve = pf.nodes.geo.sample_curve(
        curves=profile_curve,
        factor=capture_attribute.attribute,
        value=0.0,
    )

    mesh_position_x_vector = pf.nodes.func.combine_xyz(
        x=sample_curve.position.x, y=sample_curve.position.y
    )
    mesh_position_x = pf.nodes.math.vector_length(mesh_position_x_vector)

    input_position_1 = pf.nodes.geo.input_position()

    attribute_statistic = pf.nodes.geo.attribute_statistic(
        geometry=profile_curve,
        attribute=input_position_1.z,
    )

    mesh_position_z = pf.nodes.func.map_range(
        value=sample_curve.position.z,
        from_max=attribute_statistic.max,
        from_min=attribute_statistic.min,
        to_max=0.0,
        to_min=mesh_position_z_to_min,
    )
    mesh_position = pf.nodes.func.combine_xyz(
        x=input_position.x * mesh_position_x,
        y=input_position.y * mesh_position_x,
        z=mesh_position_z,
    )

    set_position = pf.nodes.geo.set_position(geometry=curve_to, position=mesh_position)
    return LegStraightResult(
        mesh=set_position,
        profile_curve=table_profile_result,
    )


@pf.nodes.node_function
def curve_board(
    thickness: t.SocketOrVal[float],
    width: t.SocketOrVal[float],
    extrude_length: t.SocketOrVal[float],
    fillet_radius_vertical: t.SocketOrVal[float] = 0.01,
) -> pf.ProcNode:
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
    transform_4_scale = pf.nodes.func.combine_xyz(x=1.0, y=thickness, z=1.0)
    transform_4 = pf.nodes.geo.transform(geometry=transform_3, scale=transform_4_scale)

    fillet_curve = pf.nodes.geo.fillet_curve(
        curve=transform_4,
        radius=thickness,
        limit_radius=True,
        count=8,
        mode="POLY",
    )

    transform_5 = pf.nodes.geo.transform(
        geometry=fillet_curve,
        rotation=(1.5708, 1.5708, 0.0),
        scale=thickness.astype(dtype=pf.Vector),
    )

    curve_to = pf.nodes.geo.curve_to_mesh(curve=transform_5)

    transform_6_translation = pf.nodes.func.combine_xyz(z=thickness * -0.5)
    transform_6 = pf.nodes.geo.transform(
        geometry=curve_to, translation=transform_6_translation
    )

    curve_line = pf.nodes.geo.curve_line(start=(1.0, 0.0, -1.0), end=(1.0, 0.0, 1.0))
    curve_line_1_start = pf.nodes.func.combine_xyz(x=width, y=extrude_length)
    curve_line_1_end = pf.nodes.func.combine_xyz(x=extrude_length, y=width)
    curve_line_1 = pf.nodes.geo.curve_line(
        start=curve_line_1_start, end=curve_line_1_end
    )
    curve_line_2_start = pf.nodes.func.combine_xyz(y=width)
    curve_line_2 = pf.nodes.geo.curve_line(
        start=curve_line_2_start, end=curve_line_1_end
    )
    curve_line_3_start = pf.nodes.func.combine_xyz(width)
    curve_line_3 = pf.nodes.geo.curve_line(
        start=curve_line_3_start, end=curve_line_1_start
    )
    curve_line_4 = pf.nodes.geo.curve_line(end=curve_line_2_start)
    curve_line_5 = pf.nodes.geo.curve_line(end=curve_line_3_start)

    join = pf.nodes.geo.join_geometry(
        [curve_line_1, curve_line_2, curve_line_3, curve_line_4, curve_line_5]
    )

    curve_to_1 = pf.nodes.geo.curve_to_mesh(join)

    merge_by_distance = pf.nodes.geo.merge_by_distance(curve_to_1)

    to_curve = pf.nodes.geo.mesh_to_curve(merge_by_distance)

    # Inlined curve_to_board logic
    ctb_z_to_min = thickness * -1.0
    ctb_line_end = pf.nodes.func.combine_xyz(z=ctb_z_to_min)
    ctb_line = pf.nodes.geo.curve_line(end=ctb_line_end)
    ctb_tilt = pf.nodes.geo.set_curve_tilt(curve=ctb_line, tilt=3.1416)
    ctb_resample = pf.nodes.geo.resample_curve_count(curve=ctb_tilt, count=128)
    ctb_spline_param = pf.nodes.geo.spline_parameter()
    ctb_capture = pf.nodes.geo.capture_attribute(
        geometry=ctb_resample,
        attribute=ctb_spline_param.factor,
    )
    ctb_mesh = pf.nodes.geo.curve_to_mesh(
        curve=ctb_capture.geometry,
        profile_curve=to_curve,
        fill_caps=True,
    )
    ctb_input_pos = pf.nodes.geo.input_position()
    ctb_sample = pf.nodes.geo.sample_curve(
        curves=curve_line,
        factor=ctb_capture.attribute,
        value=0.0,
    )
    ctb_x_vec = pf.nodes.func.combine_xyz(
        x=ctb_sample.position.x, y=ctb_sample.position.y
    )
    ctb_x = pf.nodes.math.vector_length(ctb_x_vec)
    ctb_input_pos_1 = pf.nodes.geo.input_position()
    ctb_attr_stat = pf.nodes.geo.attribute_statistic(
        geometry=curve_line,
        attribute=ctb_input_pos_1.z,
    )
    ctb_z = pf.nodes.func.map_range(
        value=ctb_sample.position.z,
        from_max=ctb_attr_stat.max,
        from_min=ctb_attr_stat.min,
        to_max=0.0,
        to_min=ctb_z_to_min,
    )
    ctb_pos = pf.nodes.func.combine_xyz(
        x=ctb_input_pos.x * ctb_x,
        y=ctb_input_pos.y * ctb_x,
        z=ctb_z,
    )
    curve_to_board_result = pf.nodes.geo.set_position(
        geometry=ctb_mesh, position=ctb_pos
    )

    join_1 = pf.nodes.geo.join_geometry([transform_6, curve_to_board_result])

    merge_by_distance_1 = pf.nodes.geo.merge_by_distance(join_1)

    result_0_translation = pf.nodes.func.combine_xyz(z=thickness)

    transform = pf.nodes.geo.transform(
        geometry=merge_by_distance_1,
        translation=result_0_translation,
    )
    return transform


@pf.nodes.node_function
def side_leg(
    thickness: t.SocketOrVal[float],
    profile_width: t.SocketOrVal[float],
    aspect_ratio: t.SocketOrVal[float],
    fillet_ratio: t.SocketOrVal[float],
    fillet_radius_vertical: t.SocketOrVal[float],
    n_gon: t.SocketOrVal[int] = 4,
) -> pf.ProcNode:
    leg_straight_result = leg_straight(
        height=thickness,
        n_gon=n_gon,
        profile_width=profile_width,
        aspect_ratio=aspect_ratio,
        fillet_ratio=fillet_ratio,
        resolution=128,
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
    transform_4_scale = pf.nodes.func.combine_xyz(x=1.0, y=thickness, z=1.0)
    transform_4 = pf.nodes.geo.transform(geometry=transform_3, scale=transform_4_scale)

    fillet_curve = pf.nodes.geo.fillet_curve(
        curve=transform_4,
        radius=thickness,
        limit_radius=True,
        count=8,
        mode="POLY",
    )

    transform_5 = pf.nodes.geo.transform(
        geometry=fillet_curve,
        rotation=(1.5708, 1.5708, 0.0),
        scale=thickness.astype(dtype=pf.Vector),
    )

    curve_to = pf.nodes.geo.curve_to_mesh(
        curve=leg_straight_result.profile_curve,
        profile_curve=transform_5,
    )

    transform_6_translation = pf.nodes.func.combine_xyz(z=thickness * -0.5)
    transform_6 = pf.nodes.geo.transform(
        geometry=curve_to, translation=transform_6_translation
    )

    join = pf.nodes.geo.join_geometry([leg_straight_result.mesh, transform_6])

    merge_by_distance = pf.nodes.geo.merge_by_distance(join)

    result_0_translation = pf.nodes.func.combine_xyz(z=thickness)

    transform = pf.nodes.geo.transform(
        geometry=merge_by_distance, translation=result_0_translation
    )
    return transform


@pf.nodes.node_function
def shelf_legs(
    leg_gap: t.SocketOrVal[float],
    leg_curve_ratio: t.SocketOrVal[float],
    leg_width: t.SocketOrVal[float],
    leg_length: t.SocketOrVal[float],
    board_width: t.SocketOrVal[float],
    leg_depth: t.SocketOrVal[float],
) -> pf.ProcNode:
    side_leg_thickness = leg_width + 0.0
    side_leg_profile_width = leg_length + 0.0
    side_leg_fillet_ratio = leg_curve_ratio + 0.0
    side_leg_result = side_leg(
        thickness=side_leg_thickness,
        n_gon=4,
        profile_width=side_leg_profile_width,
        aspect_ratio=leg_depth / leg_length,
        fillet_ratio=side_leg_fillet_ratio,
        fillet_radius_vertical=side_leg_fillet_ratio,
    )

    transform_translation = pf.nodes.func.combine_xyz(z=side_leg_profile_width * 0.5)
    transform = pf.nodes.geo.transform(
        geometry=side_leg_result,
        translation=transform_translation,
        rotation=(0.0, 1.5708, 0.0),
    )
    transform_1 = pf.nodes.geo.transform(transform)
    transform_2_translation_y = board_width + 0.0
    transform_2_translation = pf.nodes.func.combine_xyz(y=transform_2_translation_y)
    transform_2 = pf.nodes.geo.transform(
        geometry=transform, translation=transform_2_translation
    )
    transform_a_0 = transform_2_translation_y - side_leg_thickness
    transform_b = leg_gap * 2.0
    transform_3_translation = pf.nodes.func.combine_xyz(transform_a_0 + transform_b)
    transform_3 = pf.nodes.geo.transform(
        geometry=transform, translation=transform_3_translation
    )

    join = pf.nodes.geo.join_geometry([transform_1, transform_2, transform_3])
    return join


@pf.nodes.node_function
def screw_head(
    leg_width: t.SocketOrVal[float],
    board_thickness: t.SocketOrVal[float],
    board_height: t.SocketOrVal[float],
    leg_gap: t.SocketOrVal[float],
    board_width: t.SocketOrVal[float],
    leg_depth: t.SocketOrVal[float],
) -> pf.ProcNode:
    cylinder = pf.nodes.geo.mesh_cylinder(radius=0.004, depth=0.003)

    transform = pf.nodes.geo.transform(
        geometry=cylinder.mesh, rotation=(1.5708, 0.0, 0.0)
    )
    transform_a_1 = board_width + 0.0
    transform_a_0 = transform_a_1 + (leg_gap * 2.0)
    transform_2_translation_x = leg_width * 0.5
    transform_1_translation_y = 0.0 - (leg_depth * 0.5)
    transform_1_translation_z = board_height + (board_thickness * 0.5)
    transform_1_translation = pf.nodes.func.combine_xyz(
        x=transform_a_0 - transform_2_translation_x,
        y=transform_1_translation_y,
        z=transform_1_translation_z,
    )
    transform_1 = pf.nodes.geo.transform(
        geometry=transform, translation=transform_1_translation
    )
    transform_b_0 = leg_depth * 0.5
    transform_2_translation = pf.nodes.func.combine_xyz(
        x=transform_2_translation_x,
        y=transform_a_1 + transform_b_0,
        z=transform_1_translation_z,
    )
    transform_2 = pf.nodes.geo.transform(
        geometry=transform, translation=transform_2_translation
    )
    transform_3_translation = pf.nodes.func.combine_xyz(
        x=transform_2_translation_x,
        y=transform_1_translation_y,
        z=transform_1_translation_z,
    )
    transform_3 = pf.nodes.geo.transform(
        geometry=transform, translation=transform_3_translation
    )

    join = pf.nodes.geo.join_geometry([transform_1, transform_2, transform_3])
    return join


@pf.nodes.node_function
def shelf_boards(
    thickness: t.SocketOrVal[float],
    bottom_z: t.SocketOrVal[float],
    mid_z: t.SocketOrVal[float],
    top_z: t.SocketOrVal[float],
    board_width: t.SocketOrVal[float],
    leg_gap: t.SocketOrVal[float],
    extrude_length: t.SocketOrVal[float],
) -> pf.ProcNode:
    curve_board_result = curve_board(
        thickness=thickness,
        fillet_radius_vertical=0.01,
        width=board_width,
        extrude_length=extrude_length,
    )

    transform_translation_x = leg_gap + 0.0
    transform_translation = pf.nodes.func.combine_xyz(
        x=transform_translation_x, z=top_z
    )
    transform = pf.nodes.geo.transform(
        geometry=curve_board_result,
        translation=transform_translation,
        rotation=(0.0, 0.0, -1.5708),
    )
    transform_1_translation = pf.nodes.func.combine_xyz(
        x=transform_translation_x, z=mid_z
    )
    transform_1 = pf.nodes.geo.transform(
        geometry=curve_board_result,
        translation=transform_1_translation,
        rotation=(0.0, 0.0, -1.5708),
    )
    transform_2_translation = pf.nodes.func.combine_xyz(
        x=transform_translation_x, z=bottom_z
    )
    transform_2 = pf.nodes.geo.transform(
        geometry=curve_board_result,
        translation=transform_2_translation,
        rotation=(0.0, 0.0, -1.5708),
    )

    join = pf.nodes.geo.join_geometry([transform, transform_1, transform_2])
    return join


@pf.nodes.node_function
def side_boards(
    y: t.SocketOrVal[float],
    z: t.SocketOrVal[float],
    x1: t.SocketOrVal[float],
    x2: t.SocketOrVal[float],
    x3: t.SocketOrVal[float],
    x4: t.SocketOrVal[float],
    x5: t.SocketOrVal[float],
) -> pf.ProcNode:
    cube_size_x = x5 + 0.0
    cube_size = pf.nodes.func.combine_xyz(x=cube_size_x, y=y, z=z)
    cube = pf.nodes.geo.mesh_cube(
        size=cube_size, vertices_x=5, vertices_y=5, vertices_z=5
    )

    transform_translation_x = (cube_size_x * 0.5) + x3
    transform_translation_z_0 = x1 * 0.5
    transform_translation = pf.nodes.func.combine_xyz(
        x=transform_translation_x,
        z=x4 - transform_translation_z_0,
    )
    transform = pf.nodes.geo.transform(
        geometry=cube.mesh, translation=transform_translation
    )
    transform_1_translation = pf.nodes.func.combine_xyz(
        x=transform_translation_x,
        z=x2 - transform_translation_z_0,
    )
    transform_1 = pf.nodes.geo.transform(
        geometry=cube.mesh, translation=transform_1_translation
    )

    join = pf.nodes.geo.join_geometry([transform, transform_1])
    return join


@pf.nodes.node_function
def triangle_shelf_geometry(
    dimensions: t.SocketOrVal[pf.Vector],
    leg_board_gap: t.SocketOrVal[float],
    leg_width: t.SocketOrVal[float],
    leg_depth: t.SocketOrVal[float],
    leg_curvature_ratio: t.SocketOrVal[float],
    board_thickness: t.SocketOrVal[float],
    board_extrude_length: t.SocketOrVal[float],
    side_board_height: t.SocketOrVal[float],
    bottom_layer_height: t.SocketOrVal[float],
    mid_layer_height: t.SocketOrVal[float],
    top_layer_height: t.SocketOrVal[float],
    leg_material: t.SocketOrVal[pf.Material],
    board_material: t.SocketOrVal[pf.Material],
    screw_material: t.SocketOrVal[pf.Material],
) -> pf.ProcNode:
    board_width_val, _, leg_length = dimensions.x, dimensions.y, dimensions.z

    legs = shelf_legs(
        leg_gap=leg_board_gap,
        leg_curve_ratio=leg_curvature_ratio,
        leg_width=leg_width,
        leg_length=leg_length,
        board_width=board_width_val,
        leg_depth=leg_depth,
    )
    legs_with_mat = pf.nodes.geo.set_material(geometry=legs, material=leg_material)

    screws1 = screw_head(
        leg_width=leg_width,
        board_thickness=board_thickness,
        board_height=bottom_layer_height,
        leg_gap=leg_board_gap,
        board_width=board_width_val,
        leg_depth=leg_depth,
    )
    screws2 = screw_head(
        leg_width=leg_width,
        board_thickness=board_thickness,
        board_height=mid_layer_height,
        leg_gap=leg_board_gap,
        board_width=board_width_val,
        leg_depth=leg_depth,
    )
    screws3 = screw_head(
        leg_width=leg_width,
        board_thickness=board_thickness,
        board_height=top_layer_height,
        leg_gap=leg_board_gap,
        board_width=board_width_val,
        leg_depth=leg_depth,
    )
    screws_joined = pf.nodes.geo.join_geometry([screws1, screws2, screws3])
    screws_with_mat = pf.nodes.geo.set_material(
        geometry=screws_joined, material=screw_material
    )

    boards = shelf_boards(
        thickness=board_thickness,
        bottom_z=bottom_layer_height,
        mid_z=mid_layer_height,
        top_z=top_layer_height,
        board_width=board_width_val,
        leg_gap=leg_board_gap,
        extrude_length=board_extrude_length,
    )
    boards_with_mat = pf.nodes.geo.set_material(
        geometry=boards, material=board_material
    )

    side_bds = side_boards(
        y=leg_depth,
        z=side_board_height,
        x1=side_board_height,
        x2=bottom_layer_height,
        x3=leg_board_gap,
        x4=top_layer_height,
        x5=board_width_val,
    )
    side_bds_with_mat = pf.nodes.geo.set_material(
        geometry=side_bds, material=leg_material
    )

    joined = pf.nodes.geo.join_geometry(
        [legs_with_mat, screws_with_mat, boards_with_mat, side_bds_with_mat]
    )
    realized = pf.nodes.geo.realize_instances(joined)
    flipped = pf.nodes.geo.transform(geometry=realized, scale=(-1.0, 1.0, 1.0))
    triangulated = pf.nodes.geo.triangulate(flipped)
    rotated = pf.nodes.geo.transform(
        geometry=triangulated, rotation=(0.0, 0.0, -1.5708)
    )
    return rotated


def triangle_shelf_distribution(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    leg_material: pf.Material | None = None,
    board_material: pf.Material | None = None,
    screw_material: pf.Material | None = None,
) -> TriangleShelfResult:
    if dimensions is None:
        board_width = pf.random.uniform(rng, 0.2, 0.4)
        depth = board_width
        leg_length = pf.random.uniform(rng, 0.45, 0.75)
        dimensions = pf.Vector((board_width, depth, leg_length))

    leg_board_gap = pf.random.uniform(rng, 0.002, 0.005)
    leg_width = pf.random.uniform(rng, 0.01, 0.03)
    leg_depth = pf.random.uniform(rng, 0.01, 0.02)
    leg_curvature_ratio = pf.random.uniform(rng, 0.0, 0.02)
    board_thickness = pf.random.uniform(rng, 0.01, 0.025)
    board_extrude_length = pf.random.uniform(rng, 0.03, 0.07)
    side_board_height = pf.random.uniform(rng, 0.02, 0.04)
    bottom_layer_height = pf.random.uniform(rng, 0.05, 0.1)
    top_layer_height = dimensions.z - pf.random.uniform(rng, 0.02, 0.07)
    mid_layer_height = (top_layer_height + bottom_layer_height) / 2.0

    vec = pf.nodes.shader.geometry().position
    if leg_material is None:
        leg_material = furniture_material_distribution(rng, vec)
    if board_material is None:
        board_material = table_top_material_distribution(rng, vec)
    if screw_material is None:
        screw_material = metal_brushed.metal_brushed_linear_distribution(rng, vec)

    geo = triangle_shelf_geometry(
        dimensions=dimensions,
        leg_board_gap=leg_board_gap,
        leg_width=leg_width,
        leg_depth=leg_depth,
        leg_curvature_ratio=leg_curvature_ratio,
        board_thickness=board_thickness,
        board_extrude_length=board_extrude_length,
        side_board_height=side_board_height,
        bottom_layer_height=bottom_layer_height,
        mid_layer_height=mid_layer_height,
        top_layer_height=top_layer_height,
        leg_material=leg_material,
        board_material=board_material,
        screw_material=screw_material,
    )
    return TriangleShelfResult(mesh=pf.nodes.to_mesh_object(geo))
