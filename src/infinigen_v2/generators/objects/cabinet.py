from typing import NamedTuple

import numpy as np
import procfunc as pf
from procfunc.nodes import types as t

from infinigen_v2.generators.shaders.functionality_lists import (
    furniture_material_distribution,
    glass_material_distribution,
)


class CabinetResult(NamedTuple):
    mesh: pf.MeshObject


@pf.nodes.node_function
def bottom_board(
    thickness: t.SocketOrVal[float],
    depth: t.SocketOrVal[float],
    y_gap: t.SocketOrVal[float],
    x_translation: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    width: t.SocketOrVal[float],
) -> pf.ProcNode:
    board_h = height + 0.0
    cube = pf.nodes.geo.mesh_cube(
        size=pf.nodes.math.combine_xyz(x=width, y=thickness, z=board_h),
    )
    tvec = pf.nodes.math.combine_xyz(
        x=x_translation,
        y=(depth * 0.5) - y_gap,
        z=board_h * 0.5,
    )
    return pf.nodes.geo.transform(
        geometry=cube.mesh, translation=tvec, rotation=(0, 0, 0), scale=(1, 1, 1)
    )


@pf.nodes.node_function
def back_board(
    width: t.SocketOrVal[float],
    thickness: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    depth: t.SocketOrVal[float],
) -> pf.ProcNode:
    h = height + 0.0
    cube = pf.nodes.geo.mesh_cube(
        size=pf.nodes.math.combine_xyz(x=width, y=thickness, z=h),
    )
    tvec = pf.nodes.math.combine_xyz(
        y=pf.nodes.math.multiply_add(a=depth + 0.0, b=-0.5, addend=thickness * -0.5),
        z=h * 0.5,
    )
    return pf.nodes.geo.transform(
        geometry=cube.mesh, translation=tvec, rotation=(0, 0, 0), scale=(1, 1, 1)
    )


@pf.nodes.node_function
def side_board(
    board_thickness: t.SocketOrVal[float],
    depth: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    x_translation: t.SocketOrVal[float],
) -> pf.ProcNode:
    h = height + 0.0
    cube = pf.nodes.geo.mesh_cube(
        size=pf.nodes.math.combine_xyz(x=board_thickness + 0.0, y=depth + 0.0, z=h),
    )
    return pf.nodes.geo.transform(
        geometry=cube.mesh,
        translation=pf.nodes.math.combine_xyz(x=x_translation, z=h * 0.5),
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )


@pf.nodes.node_function
def node_group(
    attach_height: t.SocketOrVal[float],
    door_width: t.SocketOrVal[float],
) -> pf.ProcNode:
    cube = pf.nodes.geo.mesh_cube((0.02, 0.0006, 0.012))
    transform_1 = pf.nodes.geo.transform(
        geometry=cube.mesh,
        translation=(0.008, 0.0, 0.0),
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    cylinder = pf.nodes.geo.mesh_cylinder(vertices=24, radius=0.01, depth=0.0005)
    transform_2 = pf.nodes.geo.transform(
        geometry=cylinder.mesh,
        translation=(0.005, 0.0, 0.0),
        rotation=(1.5708, 0.0, 0.0),
        scale=(1, 1, 1),
    )
    cube_1 = pf.nodes.geo.mesh_cube((0.012, 0.0006, 0.04))
    join = pf.nodes.geo.join_geometry([transform_1, transform_2, cube_1.mesh])
    tvec = pf.nodes.math.combine_xyz(x=door_width * 0.5 - 0.0181, z=attach_height)
    return pf.nodes.geo.transform(
        geometry=join, translation=tvec, rotation=(0, 0, 0), scale=(1, 1, 1)
    )


@pf.nodes.node_function
def knob_handle(
    radius: t.SocketOrVal[float],
    thickness_1: t.SocketOrVal[float],
    thickness_2: t.SocketOrVal[float],
    length: t.SocketOrVal[float],
    knob_mid_height: t.SocketOrVal[float],
    edge_width: t.SocketOrVal[float],
    door_width: t.SocketOrVal[float],
) -> pf.ProcNode:
    cyl_depth = (thickness_2 + thickness_1) + length
    cyl = pf.nodes.geo.mesh_cylinder(vertices=24, radius=radius, depth=cyl_depth)
    tvec = pf.nodes.math.combine_xyz(
        x=((door_width - edge_width) * -0.5) - 0.005,
        y=cyl_depth * 0.5,
        z=knob_mid_height,
    )
    return pf.nodes.geo.transform(
        geometry=cyl.mesh,
        translation=tvec,
        rotation=(1.5708, 0.0, 0.0),
        scale=(1, 1, 1),
    )


@pf.nodes.node_function
def double_rampled_edge(
    height: t.SocketOrVal[float],
    thickness_2: t.SocketOrVal[float],
    width: t.SocketOrVal[float],
    thickness_1: t.SocketOrVal[float],
    ramp_angle: t.SocketOrVal[float],
) -> pf.ProcNode:
    # Keep transpiled shape for parity with v1 door profile.
    curve_line_end_z = height + 0.0
    curve_line = pf.nodes.geo.curve_line(
        end=pf.nodes.math.combine_xyz(z=curve_line_end_z),
        start=(0, 0, 0),
    )
    curve_circle = pf.nodes.geo.curve_circle(resolution=3, radius=0.01)
    end_sel = pf.nodes.geo.curve_endpoint_selection(end_size=0)
    set_x_a = width + 0.0
    cube_b = pf.nodes.math.tan(ramp_angle + 0.0) * (thickness_2 + 0.0)
    cube_size_x = set_x_a - (2.0 * cube_b)
    set_y = thickness_1 + 0.0
    set_y_b = thickness_2 + 0.0

    set_pos = pf.nodes.geo.set_position(
        geometry=curve_circle,
        selection=end_sel,
        position=pf.nodes.math.combine_xyz(x=(cube_size_x * 0.5) * -1.0, y=set_y),
    )
    start_sel = pf.nodes.geo.curve_endpoint_selection(0)
    set_pos_1 = pf.nodes.geo.set_position(
        geometry=set_pos,
        selection=start_sel,
        position=pf.nodes.math.combine_xyz(
            x=(cube_size_x * 0.5) * -1.0, y=set_y + set_y_b
        ),
    )
    idx = pf.nodes.geo.input_index()
    mask = pf.nodes.func.boolean_and(
        a=(idx.astype(dtype=float) < 1.01).astype(dtype=bool),
        b=(idx.astype(dtype=float) > 0.99).astype(dtype=bool),
    )
    prof = pf.nodes.geo.set_position(
        geometry=set_pos_1,
        selection=mask,
        position=pf.nodes.math.combine_xyz(x=(set_x_a * 0.5) * -1.0, y=set_y),
    )
    prof_mirror = pf.nodes.geo.transform(
        geometry=prof, scale=(-1.0, 1.0, 1.0), translation=(0, 0, 0), rotation=(0, 0, 0)
    )
    curve_to = pf.nodes.geo.curve_to_mesh(
        curve=curve_line, profile_curve=prof_mirror, fill_caps=True
    )

    cube = pf.nodes.geo.mesh_cube(
        pf.nodes.math.combine_xyz(x=cube_size_x, y=set_y_b, z=curve_line_end_z)
    )
    cube_t = pf.nodes.geo.transform(
        geometry=cube.mesh,
        translation=pf.nodes.math.combine_xyz(y=set_y + set_y_b * 0.5),
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    cube_1 = pf.nodes.geo.mesh_cube(
        pf.nodes.math.combine_xyz(x=set_x_a, y=set_y, z=curve_line_end_z)
    )
    cube_1_t = pf.nodes.geo.transform(
        geometry=cube_1.mesh,
        translation=pf.nodes.math.combine_xyz(y=set_y * 0.5),
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    body = pf.nodes.geo.transform(
        geometry=pf.nodes.geo.join_geometry([cube_t, cube_1_t]),
        translation=pf.nodes.math.combine_xyz(z=curve_line_end_z * 0.5),
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    curve_to_1 = pf.nodes.geo.curve_to_mesh(
        curve=pf.nodes.geo.curve_line(
            end=pf.nodes.math.combine_xyz(z=curve_line_end_z),
            start=(0, 0, 0),
        ),
        profile_curve=prof,
        fill_caps=True,
    )
    join = pf.nodes.geo.join_geometry([curve_to, body, curve_to_1])
    join = pf.nodes.geo.merge_by_distance(geometry=join, distance=0.0001)
    return pf.nodes.geo.realize_instances(join)


@pf.nodes.node_function
def ramped_edge(
    height: t.SocketOrVal[float],
    thickness_2: t.SocketOrVal[float],
    width: t.SocketOrVal[float],
    thickness_1: t.SocketOrVal[float],
    ramp_angle: t.SocketOrVal[float],
) -> pf.ProcNode:
    x_full = width + 0.0
    y2 = thickness_2 + 0.0
    y1 = thickness_1 + 0.0
    z = height + 0.0
    taper = pf.nodes.math.tan(ramp_angle + 0.0) * y2
    x_inner = x_full - taper

    cube_a = pf.nodes.geo.mesh_cube(pf.nodes.math.combine_xyz(x=x_inner, y=y2, z=z))
    cube_a = pf.nodes.geo.transform(
        geometry=cube_a.mesh,
        translation=pf.nodes.math.combine_xyz(x=taper * 0.5, y=y1 + y2 * 0.5),
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    cube_b = pf.nodes.geo.mesh_cube(pf.nodes.math.combine_xyz(x=x_full, y=y1, z=z))
    cube_b = pf.nodes.geo.transform(
        geometry=cube_b.mesh,
        translation=pf.nodes.math.combine_xyz(y=y1 * 0.5),
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    body = pf.nodes.geo.transform(
        geometry=pf.nodes.geo.join_geometry([cube_a, cube_b]),
        translation=pf.nodes.math.combine_xyz(z=z * 0.5),
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )

    curve_line = pf.nodes.geo.curve_line(
        end=pf.nodes.math.combine_xyz(z=z), start=(0, 0, 0)
    )
    profile = pf.nodes.geo.curve_circle(resolution=3, radius=0.01)
    end_sel = pf.nodes.geo.curve_endpoint_selection(end_size=0)
    set_x = (x_full * 0.5) - x_inner
    p0 = pf.nodes.geo.set_position(
        geometry=profile,
        selection=end_sel,
        position=pf.nodes.math.combine_xyz(x=set_x, y=y1),
    )
    start_sel = pf.nodes.geo.curve_endpoint_selection(0)
    p1 = pf.nodes.geo.set_position(
        geometry=p0,
        selection=start_sel,
        position=pf.nodes.math.combine_xyz(x=set_x, y=y1 + y2),
    )
    idx = pf.nodes.geo.input_index()
    mid_mask = pf.nodes.func.boolean_and(
        a=(idx.astype(dtype=float) < 1.01).astype(dtype=bool),
        b=(idx.astype(dtype=float) > 0.99).astype(dtype=bool),
    )
    p2 = pf.nodes.geo.set_position(
        geometry=p1,
        selection=mid_mask,
        position=pf.nodes.math.combine_xyz(x=(x_full * 0.5) * -1.0, y=y1),
    )
    wedge = pf.nodes.geo.curve_to_mesh(
        curve=curve_line, profile_curve=p2, fill_caps=True
    )
    geo = pf.nodes.geo.join_geometry([body, wedge])
    geo = pf.nodes.geo.merge_by_distance(geometry=geo, distance=0.0001)
    geo = pf.nodes.geo.realize_instances(geo)
    return pf.nodes.geo.transform(
        geometry=geo,
        translation=pf.nodes.math.combine_xyz(x=x_full * -0.5),
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )


class PanelEdgeFrameResult(NamedTuple):
    value: pf.ProcNode[float]
    geometry: pf.ProcNode[pf.MeshObject]


@pf.nodes.node_function
def panel_edge_frame(
    vertical_edge: t.SocketOrVal[pf.MeshObject],
    door_width: t.SocketOrVal[float],
    door_height: t.SocketOrVal[float],
    horizontal_edge: t.SocketOrVal[pf.MeshObject],
) -> PanelEdgeFrameResult:
    value_a = pf.nodes.math.multiply_add(a=door_width, b=0.5, addend=0.001)
    value = value_a * -1.0
    transform = pf.nodes.geo.transform(
        geometry=vertical_edge,
        translation=pf.nodes.math.combine_xyz(value_a),
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    transform_1 = pf.nodes.geo.transform(
        geometry=transform,
        scale=(-1.0, 1.0, 1.0),
        translation=(0, 0, 0),
        rotation=(0, 0, 0),
    )
    transform_2 = pf.nodes.geo.transform(
        geometry=horizontal_edge,
        translation=(0.0, -0.0001, 0.0),
        scale=(0.9999, 1.0, 1.0),
        rotation=(0, 0, 0),
    )
    transform_3 = pf.nodes.geo.transform(
        geometry=transform_2,
        translation=pf.nodes.math.combine_xyz(value + 0.0001),
        rotation=(0.0, 1.5708, 0.0),
        scale=(1, 1, 1),
    )
    transform_4 = pf.nodes.geo.transform(
        geometry=transform_2,
        translation=pf.nodes.math.combine_xyz(
            x=value_a - 0.0001, z=door_height + 0.0001
        ),
        rotation=(0.0, -1.5708, 0.0),
        scale=(1, 1, 1),
    )
    return PanelEdgeFrameResult(
        value=value,
        geometry=pf.nodes.geo.join_geometry(
            [transform, transform_1, transform_3, transform_4]
        ),
    )


def _door_and_attach_material_branch(
    rng: pf.RNG,
    frame_material: pf.Material,
    panel_lower_material: pf.Material,
    panel_upper_material: pf.Material,
    has_mid_ramp: bool,
) -> list[pf.Material]:
    def _with_mid() -> list[pf.Material]:
        lower = pf.control.choice(
            rng,
            [
                (frame_material, 0.7),
                (panel_lower_material, 0.3),
            ],
        )
        upper = pf.control.choice(
            rng,
            [
                (lower, 0.6),
                (panel_upper_material, 0.4),
            ],
        )
        return [lower, upper]

    def _without_mid() -> list[pf.Material]:
        return [frame_material]

    return pf.control.choice(
        rng,
        [(_with_mid, 1.0), (_without_mid, 1.0)],
        chosen_idx=0 if has_mid_ramp else 1,
    )()


def _shelf_params_from_dimensions(
    dimensions: pf.Vector,
    frame_material: pf.Material,
    side_board_thickness: float,
    division_board_thickness: float,
    backboard_thickness: float,
    bottom_board_height: float,
    screw_depth_head: float,
    screw_head_radius: float,
    screw_width_gap: float,
    screw_depth_gap: float,
    attach_length: float,
    attach_width: float,
    attach_thickness: float,
    attach_gap: float,
    bottom_board_y_gap: float,
) -> dict:
    depth, width, height = float(dimensions.x), float(dimensions.y), float(dimensions.z)

    num_vertical_cells = max(1, int((height - bottom_board_height) / 0.3))
    shelf_cell_height = [
        (height - bottom_board_height) / num_vertical_cells
        for _ in range(num_vertical_cells)
    ]
    shelf_cell_width = [width]

    # Match v1 translation convention used by downstream door/cabinet placement.
    shelf_width = width
    shelf_height = (
        (len(shelf_cell_height) + 1) * division_board_thickness
        + bottom_board_height
        + sum(shelf_cell_height)
    )

    dist = -(shelf_width + side_board_thickness) / 2.0
    side_board_x_translation = [dist]
    for w in shelf_cell_width:
        dist += side_board_thickness + w
        side_board_x_translation.append(dist)
        dist += side_board_thickness + 0.001
        side_board_x_translation.append(dist)
    side_board_x_translation = side_board_x_translation[:-1]

    dist = bottom_board_height + division_board_thickness / 2.0
    division_board_z_translation = [dist]
    for h in shelf_cell_height:
        dist += h + division_board_thickness
        division_board_z_translation.append(dist)

    division_board_x_translation = [
        (side_board_x_translation[2 * i] + side_board_x_translation[2 * i + 1]) / 2.0
        for i in range(len(shelf_cell_width))
    ]

    return {
        "Dimensions": (depth, width, height),
        "shelf_depth": depth - 0.01,
        "shelf_width": shelf_width,
        "shelf_height": shelf_height,
        "shelf_cell_width": shelf_cell_width,
        "shelf_cell_height": shelf_cell_height,
        "side_board_thickness": side_board_thickness,
        "division_board_thickness": division_board_thickness,
        "bottom_board_height": bottom_board_height,
        "bottom_board_y_gap": bottom_board_y_gap,
        "backboard_thickness": backboard_thickness,
        "back_board_thickness": backboard_thickness,
        "screw_depth_head": screw_depth_head,
        "screw_head_radius": screw_head_radius,
        "screw_width_gap": screw_width_gap,
        "screw_depth_gap": screw_depth_gap,
        "attach_length": attach_length,
        "attach_width": attach_width,
        "attach_thickness": attach_thickness,
        "attach_gap": attach_gap,
        "attach_z_translation": shelf_height - division_board_thickness,
        "side_board_x_translation": side_board_x_translation,
        "division_board_x_translation": division_board_x_translation,
        "division_board_z_translation": division_board_z_translation,
        "bottom_gap_x_translation": division_board_x_translation,
        "frame_material": frame_material,
        "board_material": frame_material,
    }


def _shelf_geometry(
    shelf_depth: float,
    shelf_width: float,
    shelf_height: float,
    side_board_thickness: float,
    division_board_thickness: float,
    bottom_board_height: float,
    bottom_board_y_gap: float,
    backboard_thickness: float,
    side_board_x_translation: list[float],
    division_board_x_translation: list[float],
    division_board_z_translation: list[float],
    shelf_cell_width: list[float],
    frame_material,
    board_material,
) -> pf.ProcNode:
    side_boards = [
        side_board(
            board_thickness=side_board_thickness,
            depth=shelf_depth + 0.004,
            height=shelf_height + 0.002,
            x_translation=x,
        )
        for x in side_board_x_translation
    ]

    back_board_geo = back_board(
        width=shelf_width + side_board_thickness * 2,
        thickness=backboard_thickness,
        height=shelf_height - 0.001,
        depth=shelf_depth,
    )

    bottom_boards = [
        bottom_board(
            thickness=side_board_thickness,
            depth=shelf_depth,
            y_gap=bottom_board_y_gap,
            x_translation=division_board_x_translation[i],
            height=bottom_board_height,
            width=shelf_cell_width[i],
        )
        for i in range(len(shelf_cell_width))
    ]

    frame_geo = pf.nodes.geo.join_geometry(
        [back_board_geo] + side_boards + bottom_boards
    )
    frame_geo = pf.nodes.geo.realize_instances(frame_geo)
    frame_geo = pf.nodes.geo.set_material(
        geometry=frame_geo,
        material=frame_material,
        selection=True,
    )

    div_boards = []
    for i in range(len(shelf_cell_width)):
        for z in division_board_z_translation:
            cube = pf.nodes.geo.mesh_cube(
                size=(shelf_cell_width[i], shelf_depth, division_board_thickness),
            )
            board = pf.nodes.geo.transform(
                geometry=cube.mesh,
                translation=(division_board_x_translation[i], 0.0, z),
                rotation=(0, 0, 0),
                scale=(1, 1, 1),
            )
            div_boards.append(board)

    board_geo = pf.nodes.geo.join_geometry(div_boards)
    board_geo = pf.nodes.geo.set_material(
        geometry=board_geo,
        material=board_material,
        selection=True,
    )

    joined = pf.nodes.geo.join_geometry([frame_geo, board_geo])
    joined = pf.nodes.geo.realize_instances(joined)
    joined = pf.nodes.geo.transform(
        joined, rotation=(0.0, 0.0, -1.5708), scale=(1, 1, 1), translation=(0, 0, 0)
    )
    return joined


def _door_geometry(
    door_height: float,
    door_width: float,
    edge_thickness_1: float,
    edge_width: float,
    edge_thickness_2: float,
    edge_ramp_angle: float,
    knob_radius: float,
    knob_length: float,
    attach_height: list[float],
    has_mid_ramp: bool,
    door_left_hinge: bool,
    frame_material,
    panel_material: list,
) -> pf.ProcNode:
    vertical_edge = ramped_edge(
        height=door_height,
        thickness_2=edge_thickness_2,
        width=edge_width,
        thickness_1=edge_thickness_1,
        ramp_angle=edge_ramp_angle,
    )
    horizontal_edge = ramped_edge(
        height=door_width,
        thickness_2=edge_thickness_2,
        width=edge_width,
        thickness_1=edge_thickness_1,
        ramp_angle=edge_ramp_angle,
    )
    panel_edge = panel_edge_frame(
        vertical_edge=vertical_edge,
        door_width=door_width,
        door_height=door_height,
        horizontal_edge=horizontal_edge,
    )

    panel_thickness = edge_thickness_1 - 0.005
    panel_width = door_width - 0.0001
    panel_y = panel_thickness * 0.5 + 0.004

    if has_mid_ramp:
        mid_height = door_height * 0.5
        lower = pf.nodes.geo.mesh_cube(
            size=(panel_width, panel_thickness, mid_height - 0.0001)
        )
        lower = pf.nodes.geo.transform(
            geometry=lower.mesh,
            translation=(0.0, panel_y, mid_height * 0.5),
            rotation=(0, 0, 0),
            scale=(1, 1, 1),
        )
        lower = pf.nodes.geo.set_material(
            lower, material=panel_material[0], selection=True
        )

        upper = pf.nodes.geo.mesh_cube(
            size=(panel_width, panel_thickness, mid_height - 0.0001)
        )
        upper = pf.nodes.geo.transform(
            geometry=upper.mesh,
            translation=(0.0, panel_y, mid_height * 1.5),
            rotation=(0, 0, 0),
            scale=(1, 1, 1),
        )
        upper = pf.nodes.geo.set_material(
            upper, material=panel_material[1], selection=True
        )
        panel_geo = pf.nodes.geo.join_geometry([lower, upper])

        mid_ramp = double_rampled_edge(
            height=door_width,
            thickness_2=edge_thickness_2,
            width=edge_width,
            thickness_1=edge_thickness_1,
            ramp_angle=edge_ramp_angle,
        )
        mid_ramp_translation = pf.nodes.math.combine_xyz(
            x=panel_edge.value + 0.0001,
            y=-0.0001,
            z=mid_height,
        )
        mid_ramp = pf.nodes.geo.transform(
            geometry=mid_ramp,
            translation=mid_ramp_translation,
            rotation=(0.0, 1.5708, 0.0),
            scale=(1, 1, 1),
        )
        frame_geo = pf.nodes.geo.join_geometry([panel_edge.geometry, mid_ramp])
    else:
        mid_height = door_height
        panel = pf.nodes.geo.mesh_cube(
            size=(panel_width, panel_thickness, mid_height - 0.0001)
        )
        panel_geo = pf.nodes.geo.transform(
            geometry=panel.mesh,
            translation=(0.0, panel_y, mid_height * 0.5),
            rotation=(0, 0, 0),
            scale=(1, 1, 1),
        )
        panel_geo = pf.nodes.geo.set_material(
            panel_geo,
            material=panel_material[0],
            selection=True,
        )
        frame_geo = panel_edge.geometry

    frame_geo = pf.nodes.geo.set_material(
        frame_geo, material=frame_material, selection=True
    )

    knob = knob_handle(
        radius=knob_radius,
        thickness_1=edge_thickness_1,
        thickness_2=edge_thickness_2,
        length=knob_length,
        knob_mid_height=door_height * 0.5,
        edge_width=edge_width,
        door_width=door_width,
    )
    knob = pf.nodes.geo.set_material(knob, material=frame_material, selection=True)

    attach_geos = []
    for h in attach_height:
        g = node_group(attach_height=h, door_width=door_width)
        g = pf.nodes.geo.set_material(g, material=frame_material, selection=True)
        attach_geos.append(g)

    geo = pf.nodes.geo.join_geometry([frame_geo, knob, panel_geo] + attach_geos)
    geo = pf.nodes.geo.transform(
        geo,
        translation=(door_width * -0.5, 0.0, 0.0),
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    geo = pf.nodes.geo.realize_instances(geo)
    if door_left_hinge:
        geo = pf.nodes.geo.transform(
            geo, scale=(-1.0, 1.0, 1.0), rotation=(0, 0, 0), translation=(0, 0, 0)
        )
        geo = pf.nodes.geo.flip_faces(geo)
    geo = pf.nodes.geo.transform(
        geo, rotation=(0.0, 0.0, -1.5708), scale=(1, 1, 1), translation=(0, 0, 0)
    )
    return geo


def _door_params_from_shelf(
    rng: pf.RNG,
    shelf_params: dict,
    frame_material: pf.Material,
    panel_material: list[pf.Material],
    edge_thickness_1: float,
    edge_width: float,
    edge_thickness_2: float,
    edge_ramp_angle: float,
    knob_radius: float,
    knob_length: float,
    has_mid_ramp: bool,
    force_short_door: bool,
    door_attach_gap: float,
) -> tuple[dict, int]:
    shelf_width = shelf_params["shelf_width"] + shelf_params["side_board_thickness"] * 2

    num_doors, door_width = pf.control.choice(
        rng,
        [
            (lambda: (1, shelf_width), 1.0),
            (lambda: (2, shelf_width / 2.0 - 0.0005), 1.0),
        ],
        chosen_idx=0 if shelf_width < 0.55 else 1,
    )()

    full_door_height = (
        shelf_params["division_board_z_translation"][-1]
        - shelf_params["division_board_z_translation"][0]
        + shelf_params["division_board_thickness"]
    )

    if len(shelf_params["division_board_z_translation"]) > 3:
        short_door_height = (
            shelf_params["division_board_z_translation"][3]
            - shelf_params["division_board_z_translation"][0]
            + shelf_params["division_board_thickness"]
        )
    else:
        short_door_height = full_door_height
    short_door_eligible = len(shelf_params["division_board_z_translation"]) > 5

    door_height = pf.control.choice(
        rng,
        [
            (lambda: short_door_height, 1.0),
            (lambda: full_door_height, 1.0),
        ],
        chosen_idx=0 if (force_short_door and short_door_eligible) else 1,
    )()

    gap = door_attach_gap
    attach_height = [gap, float(door_height - gap)]

    params = {
        "door_height": float(door_height),
        "door_width": float(door_width),
        "edge_thickness_1": float(edge_thickness_1),
        "edge_width": float(edge_width),
        "edge_thickness_2": float(edge_thickness_2),
        "edge_ramp_angle": float(edge_ramp_angle),
        "knob_R": float(knob_radius),
        "knob_length": float(knob_length),
        "attach_height": attach_height,
        "has_mid_ramp": bool(has_mid_ramp),
        "frame_material": frame_material,
        "panel_material": panel_material,
    }
    return params, num_doors


def _cabinet_transform_params(
    rng: pf.RNG, shelf_params: dict, door_params: dict, num_doors: int
) -> dict:
    shelf_width = shelf_params["shelf_width"] + shelf_params["side_board_thickness"] * 2

    def _single_door():
        door_hinge_pos = [
            (
                shelf_params["shelf_depth"] / 2.0 + 0.0025,
                -shelf_width / 2.0,
                shelf_params["bottom_board_height"],
            )
        ]
        attach_pos = [
            (
                shelf_params["shelf_depth"] / 2.0,
                -shelf_params["shelf_width"] / 2.0,
                shelf_params["bottom_board_height"] + z,
            )
            for z in door_params["attach_height"]
        ]
        return door_hinge_pos, attach_pos

    def _double_door():
        door_hinge_pos = [
            (
                shelf_params["shelf_depth"] / 2.0 + 0.008,
                -shelf_width / 2.0,
                shelf_params["bottom_board_height"],
            ),
            (
                shelf_params["shelf_depth"] / 2.0 + 0.008,
                shelf_width / 2.0,
                shelf_params["bottom_board_height"],
            ),
        ]
        attach_pos = [
            (
                shelf_params["shelf_depth"] / 2.0,
                -shelf_params["shelf_width"] / 2.0,
                shelf_params["bottom_board_height"] + z,
            )
            for z in door_params["attach_height"]
        ] + [
            (
                shelf_params["shelf_depth"] / 2.0,
                shelf_params["shelf_width"] / 2.0,
                shelf_params["bottom_board_height"] + z,
            )
            for z in door_params["attach_height"]
        ]
        return door_hinge_pos, attach_pos

    door_hinge_pos, attach_pos = pf.control.choice(
        rng,
        [(_single_door, 1.0), (_double_door, 1.0)],
        chosen_idx=0 if num_doors == 1 else 1,
    )()

    return {
        "door_hinge_pos": door_hinge_pos,
        "door_open_angle": 0.0,
        "attach_pos": attach_pos,
    }


def cabinet(
    rng: pf.RNG,
    dimensions: pf.Vector,
    frame_material: pf.Material,
    panel_material: list[pf.Material],
    has_mid_ramp: bool,
    force_short_door: bool,
    side_board_thickness: float,
    division_board_thickness: float,
    backboard_thickness: float,
    bottom_board_height: float,
    screw_depth_head: float,
    screw_head_radius: float,
    screw_width_gap: float,
    screw_depth_gap: float,
    attach_length: float,
    attach_width: float,
    attach_thickness: float,
    attach_gap: float,
    bottom_board_y_gap: float,
    edge_thickness_1: float,
    edge_width: float,
    edge_thickness_2: float,
    edge_ramp_angle: float,
    knob_radius: float,
    knob_length: float,
    door_attach_gap: float,
) -> pf.MeshObject:
    shelf_params = _shelf_params_from_dimensions(
        dimensions=dimensions,
        frame_material=frame_material,
        side_board_thickness=side_board_thickness,
        division_board_thickness=division_board_thickness,
        backboard_thickness=backboard_thickness,
        bottom_board_height=bottom_board_height,
        screw_depth_head=screw_depth_head,
        screw_head_radius=screw_head_radius,
        screw_width_gap=screw_width_gap,
        screw_depth_gap=screw_depth_gap,
        attach_length=attach_length,
        attach_width=attach_width,
        attach_thickness=attach_thickness,
        attach_gap=attach_gap,
        bottom_board_y_gap=bottom_board_y_gap,
    )

    shelf_geo = _shelf_geometry(
        shelf_depth=shelf_params["shelf_depth"],
        shelf_width=shelf_params["shelf_width"],
        shelf_height=shelf_params["shelf_height"],
        side_board_thickness=shelf_params["side_board_thickness"],
        division_board_thickness=shelf_params["division_board_thickness"],
        bottom_board_height=shelf_params["bottom_board_height"],
        bottom_board_y_gap=shelf_params["bottom_board_y_gap"],
        backboard_thickness=shelf_params["backboard_thickness"],
        side_board_x_translation=shelf_params["side_board_x_translation"],
        division_board_x_translation=shelf_params["division_board_x_translation"],
        division_board_z_translation=shelf_params["division_board_z_translation"],
        shelf_cell_width=shelf_params["shelf_cell_width"],
        frame_material=shelf_params["frame_material"],
        board_material=shelf_params["board_material"],
    )
    door_params, num_doors = _door_params_from_shelf(
        rng=rng,
        shelf_params=shelf_params,
        frame_material=frame_material,
        panel_material=panel_material,
        edge_thickness_1=edge_thickness_1,
        edge_width=edge_width,
        edge_thickness_2=edge_thickness_2,
        edge_ramp_angle=edge_ramp_angle,
        knob_radius=knob_radius,
        knob_length=knob_length,
        has_mid_ramp=has_mid_ramp,
        force_short_door=force_short_door,
        door_attach_gap=door_attach_gap,
    )

    right_door_geo = _door_geometry(
        door_height=door_params["door_height"],
        door_width=door_params["door_width"],
        edge_thickness_1=door_params["edge_thickness_1"],
        edge_width=door_params["edge_width"],
        edge_thickness_2=door_params["edge_thickness_2"],
        edge_ramp_angle=door_params["edge_ramp_angle"],
        knob_radius=door_params["knob_R"],
        knob_length=door_params["knob_length"],
        attach_height=door_params["attach_height"],
        has_mid_ramp=door_params["has_mid_ramp"],
        door_left_hinge=False,
        frame_material=door_params["frame_material"],
        panel_material=door_params["panel_material"],
    )
    left_door_geo = _door_geometry(
        door_height=door_params["door_height"],
        door_width=door_params["door_width"],
        edge_thickness_1=door_params["edge_thickness_1"],
        edge_width=door_params["edge_width"],
        edge_thickness_2=door_params["edge_thickness_2"],
        edge_ramp_angle=door_params["edge_ramp_angle"],
        knob_radius=door_params["knob_R"],
        knob_length=door_params["knob_length"],
        attach_height=door_params["attach_height"],
        has_mid_ramp=door_params["has_mid_ramp"],
        door_left_hinge=True,
        frame_material=door_params["frame_material"],
        panel_material=door_params["panel_material"],
    )
    cab_params = _cabinet_transform_params(
        rng=rng,
        shelf_params=shelf_params,
        door_params=door_params,
        num_doors=num_doors,
    )
    doors = [
        pf.nodes.geo.transform(
            geometry=right_door_geo,
            translation=cab_params["door_hinge_pos"][0],
            rotation=(0.0, 0.0, cab_params["door_open_angle"]),
            scale=(1, 1, 1),
        )
    ]
    if len(cab_params["door_hinge_pos"]) > 1:
        doors.append(
            pf.nodes.geo.transform(
                geometry=left_door_geo,
                translation=cab_params["door_hinge_pos"][1],
                rotation=(0.0, 0.0, cab_params["door_open_angle"]),
                scale=(1, 1, 1),
            )
        )

    attaches = []
    for pos in cab_params["attach_pos"]:
        cube = pf.nodes.geo.mesh_cube((0.0006, 0.0200, 0.04500))
        bar = pf.nodes.geo.transform(
            geometry=cube.mesh,
            translation=(0.0, -0.0100, 0.0),
            rotation=(0, 0, 0),
            scale=(1, 1, 1),
        )
        plate = pf.nodes.geo.mesh_cube((0.0005, 0.0340, 0.0200))
        attach = pf.nodes.geo.join_geometry([bar, plate.mesh])
        attach = pf.nodes.geo.transform(
            geometry=attach,
            translation=(0.0, -0.0170, 0.0),
            rotation=(0, 0, 0),
            scale=(1, 1, 1),
        )
        attach = pf.nodes.geo.transform(
            geometry=attach,
            rotation=(0.0, 0.0, -1.5708),
            translation=(0, 0, 0),
            scale=(1, 1, 1),
        )
        attach = pf.nodes.geo.transform(
            geometry=attach, translation=pos, rotation=(0, 0, 0), scale=(1, 1, 1)
        )
        attach = pf.nodes.geo.set_material(
            geometry=attach, material=frame_material, selection=True
        )
        attaches.append(attach)

    joined = pf.nodes.geo.join_geometry([shelf_geo] + doors + attaches)
    result = pf.nodes.to_mesh_object(joined)
    pf.ops.uv.cube_project(result, uv_name="UVMap")
    return result


def cabinet_distribution(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    frame_material: pf.Material | None = None,
    panel_lower_material: pf.Material | None = None,
    panel_upper_material: pf.Material | None = None,
    has_mid_ramp: bool | None = None,
) -> CabinetResult:
    if dimensions is None:
        depth = pf.random.uniform(rng, 0.25, 0.35)
        width = pf.random.uniform(rng, 0.3, 0.7)
        height = pf.random.uniform(rng, 0.9, 1.8)
        dimensions = pf.Vector((depth, width, height))

    vec = pf.nodes.shader.coord().uv
    if frame_material is None:
        frame_material = furniture_material_distribution(rng, vec)

    if panel_lower_material is None:
        panel_lower_material = furniture_material_distribution(rng, vec)

    if panel_upper_material is None:
        panel_upper_material = glass_material_distribution(rng, vec)

    side_board_thickness = pf.random.clip_gaussian(rng, 0.02, 0.002, 0.015, 0.025)
    division_board_thickness = pf.random.clip_gaussian(rng, 0.02, 0.002, 0.015, 0.025)
    backboard_thickness = 0.01
    bottom_board_height = 0.083
    bottom_board_y_gap = pf.random.uniform(rng, 0.01, 0.05)

    screw_depth_head = pf.random.uniform(rng, 0.001, 0.004)
    screw_head_radius = pf.random.uniform(rng, 0.001, 0.004)
    screw_width_gap = pf.random.uniform(rng, 0.0, 0.02)
    screw_depth_gap = pf.random.uniform(rng, 0.025, 0.06)
    attach_length = pf.random.uniform(rng, 0.05, 0.1)
    attach_width = pf.random.uniform(rng, 0.01, 0.025)
    attach_thickness = pf.random.uniform(rng, 0.002, 0.005)
    attach_gap = pf.random.uniform(rng, 0.0, 0.05)

    edge_thickness_1 = pf.random.uniform(rng, 0.01, 0.018)
    edge_width = pf.random.uniform(rng, 0.03, 0.05)
    edge_thickness_2 = pf.random.uniform(rng, 0.005, 0.008)
    edge_ramp_angle = pf.random.uniform(rng, 0.6, 0.8)
    knob_radius = pf.random.uniform(rng, 0.003, 0.006)
    knob_length = pf.random.uniform(rng, 0.018, 0.035)
    door_attach_gap = pf.random.uniform(rng, 0.05, 0.15)

    if has_mid_ramp is None:
        has_mid_ramp = pf.control.choice(rng, [(True, 0.6), (False, 0.4)])

    num_vertical_cells = max(1, np.floor((dimensions.z - bottom_board_height) / 0.3))
    force_short_door = pf.control.choice(
        rng,
        [
            (lambda: pf.control.choice(rng, [(True, 0.5), (False, 0.5)]), 1.0),
            (lambda: False, 1.0),
        ],
        chosen_idx=0 if num_vertical_cells > 4 else 1,
    )()

    panel_material = _door_and_attach_material_branch(
        rng=rng,
        frame_material=frame_material,
        panel_lower_material=panel_lower_material,
        panel_upper_material=panel_upper_material,
        has_mid_ramp=has_mid_ramp,
    )

    return CabinetResult(
        mesh=cabinet(
            rng=rng,
            dimensions=dimensions,
            frame_material=frame_material,
            panel_material=panel_material,
            has_mid_ramp=has_mid_ramp,
            force_short_door=force_short_door,
            side_board_thickness=side_board_thickness,
            division_board_thickness=division_board_thickness,
            backboard_thickness=backboard_thickness,
            bottom_board_height=bottom_board_height,
            screw_depth_head=screw_depth_head,
            screw_head_radius=screw_head_radius,
            screw_width_gap=screw_width_gap,
            screw_depth_gap=screw_depth_gap,
            attach_length=attach_length,
            attach_width=attach_width,
            attach_thickness=attach_thickness,
            attach_gap=attach_gap,
            bottom_board_y_gap=bottom_board_y_gap,
            edge_thickness_1=edge_thickness_1,
            edge_width=edge_width,
            edge_thickness_2=edge_thickness_2,
            edge_ramp_angle=edge_ramp_angle,
            knob_radius=knob_radius,
            knob_length=knob_length,
            door_attach_gap=door_attach_gap,
        )
    )
