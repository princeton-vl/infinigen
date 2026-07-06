# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Beining Han: original Infinigen v1 nodegroup (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/objects/shelves/simple_bookcase.py)
# - Alexander Raistrick: transpile to procfunc/v2

from typing import NamedTuple

import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.shaders.base_materials import (
    metal_brushed,
)
from infinigen2.shaders.functionality_lists import (
    furniture_material_rand,
)

__all__ = [
    "BookcaseResult",
    "bookcase",
    "bookcase_rand",
]


class BookcaseResult(NamedTuple):
    mesh: pf.MeshObject


@pf.nodes.node_function
def _attach_gadget(
    division_thickness: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    attach_thickness: t.SocketOrVal[float],
    attach_width: t.SocketOrVal[float],
    attach_back_len: t.SocketOrVal[float],
    attach_top_len: t.SocketOrVal[float],
    depth: t.SocketOrVal[float],
) -> pf.ProcNode:
    width_val = attach_width + 0.0
    top_len_val = attach_top_len + 0.0
    thickness_val = attach_thickness + 0.0
    depth_val = depth + 0.0

    cube_size = pf.nodes.math.combine_xyz(x=width_val, y=top_len_val, z=thickness_val)
    cube = pf.nodes.geo.mesh_cube(size=cube_size)

    translate_y = (depth_val - top_len_val) * -0.5
    translate_z = height - division_thickness
    translate = pf.nodes.math.combine_xyz(y=translate_y, z=translate_z)
    attach1 = pf.nodes.geo.transform(
        geometry=cube.mesh, translation=translate, rotation=(0, 0, 0), scale=(1, 1, 1)
    )

    back_len_val = attach_back_len + 0.0
    cube_1_size = pf.nodes.math.combine_xyz(
        x=width_val, y=thickness_val, z=back_len_val
    )
    cube_1 = pf.nodes.geo.mesh_cube(size=cube_1_size)

    translate_1_y = depth_val * -0.5
    translate_1_z = translate_z - (back_len_val * 0.5)
    translate_1 = pf.nodes.math.combine_xyz(y=translate_1_y, z=translate_1_z)
    attach2 = pf.nodes.geo.transform(
        geometry=cube_1.mesh,
        translation=translate_1,
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )

    return pf.nodes.geo.join_geometry([attach1, attach2])


@pf.nodes.node_function
def _screw_head_bookcase(
    screw_depth: t.SocketOrVal[float],
    screw_radius: t.SocketOrVal[float],
    bottom_gap: t.SocketOrVal[float],
    division_thickness: t.SocketOrVal[float],
    width: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    shelf_depth: t.SocketOrVal[float],
    screw_gap: t.SocketOrVal[float],
) -> pf.ProcNode:
    cylinder = pf.nodes.geo.mesh_cylinder(
        radius=screw_radius, depth=screw_depth, fill_type="TRIANGLE_FAN"
    )
    rotated = pf.nodes.geo.transform(
        geometry=cylinder.mesh,
        rotation=(0.0, 1.5708, 0.0),
        translation=(0, 0, 0),
        scale=(1, 1, 1),
    )

    half_width = width * 0.5
    half_depth = shelf_depth * 0.5
    gap_val = screw_gap + 0.0
    half_div_thickness = division_thickness * 0.5
    top_z = height - half_div_thickness
    bottom_z = half_div_thickness + bottom_gap
    mid_z = (top_z + bottom_z) * 0.5

    pos_y = half_depth - gap_val
    neg_y = pos_y * -1.0

    pos1 = pf.nodes.math.combine_xyz(x=half_width, y=pos_y, z=top_z)
    t1 = pf.nodes.geo.transform(
        geometry=rotated, translation=pos1, rotation=(0, 0, 0), scale=(1, 1, 1)
    )

    pos2 = pf.nodes.math.combine_xyz(x=half_width, y=pos_y, z=bottom_z)
    t2 = pf.nodes.geo.transform(
        geometry=rotated, translation=pos2, rotation=(0, 0, 0), scale=(1, 1, 1)
    )

    pos3 = pf.nodes.math.combine_xyz(x=half_width, y=neg_y, z=top_z)
    t3 = pf.nodes.geo.transform(
        geometry=rotated, translation=pos3, rotation=(0, 0, 0), scale=(1, 1, 1)
    )

    pos4 = pf.nodes.math.combine_xyz(x=half_width, y=0.0, z=mid_z)
    t4 = pf.nodes.geo.transform(
        geometry=rotated, translation=pos4, rotation=(0, 0, 0), scale=(1, 1, 1)
    )

    pos5 = pf.nodes.math.combine_xyz(x=half_width, y=neg_y, z=bottom_z)
    t5 = pf.nodes.geo.transform(
        geometry=rotated, translation=pos5, rotation=(0, 0, 0), scale=(1, 1, 1)
    )

    one_side = pf.nodes.geo.join_geometry([t1, t2, t3, t4, t5])
    other_side = pf.nodes.geo.transform(
        geometry=one_side,
        scale=(-1.0, 1.0, 1.0),
        translation=(0, 0, 0),
        rotation=(0, 0, 0),
    )
    both_sides = pf.nodes.geo.join_geometry([one_side, other_side])

    return both_sides


@pf.nodes.node_function
def _back_board(
    width: t.SocketOrVal[float],
    thickness: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    depth: t.SocketOrVal[float],
) -> pf.ProcNode:
    thickness_val = thickness + 0.0
    height_val = height + 0.0
    depth_val = depth + 0.0

    cube_size = pf.nodes.math.combine_xyz(x=width, y=thickness_val, z=height_val)
    cube = pf.nodes.geo.mesh_cube(size=cube_size)

    translate_y = pf.nodes.math.multiply_add(
        a=depth_val, b=-0.5, addend=thickness_val * -0.5
    )
    translate_z = height_val * 0.5
    translate = pf.nodes.math.combine_xyz(y=translate_y, z=translate_z)
    result = pf.nodes.geo.transform(
        geometry=cube.mesh, translation=translate, rotation=(0, 0, 0), scale=(1, 1, 1)
    )

    return result


@pf.nodes.node_function
def _all_division_boards(
    board_thickness: t.SocketOrVal[float],
    depth: t.SocketOrVal[float],
    width: t.SocketOrVal[float],
    side_thickness: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    gap: t.SocketOrVal[float],
) -> pf.ProcNode:
    inner_width = width - (side_thickness * 2.0)
    depth_val = depth + 0.0

    cube_size = pf.nodes.math.combine_xyz(x=inner_width, y=depth_val, z=board_thickness)
    cube = pf.nodes.geo.mesh_cube(size=cube_size)

    half_thickness = board_thickness * 0.5
    bottom_z = gap + half_thickness
    translate_1 = pf.nodes.math.combine_xyz(z=bottom_z)
    board1 = pf.nodes.geo.transform(
        geometry=cube.mesh, translation=translate_1, rotation=(0, 0, 0), scale=(1, 1, 1)
    )

    top_z = height - half_thickness
    mid_z = (top_z + bottom_z) * 0.5
    translate_2 = pf.nodes.math.combine_xyz(z=mid_z)
    board2 = pf.nodes.geo.transform(
        geometry=cube.mesh, translation=translate_2, rotation=(0, 0, 0), scale=(1, 1, 1)
    )

    translate_3 = pf.nodes.math.combine_xyz(z=top_z)
    board3 = pf.nodes.geo.transform(
        geometry=cube.mesh, translation=translate_3, rotation=(0, 0, 0), scale=(1, 1, 1)
    )

    return pf.nodes.geo.join_geometry([board1, board2, board3])


@pf.nodes.node_function
def _side_board(
    board_thickness: t.SocketOrVal[float],
    depth: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    width: t.SocketOrVal[float],
) -> pf.ProcNode:
    thickness_val = board_thickness + 0.0
    depth_val = depth + 0.0
    height_val = height + 0.0
    width_val = width + 0.0

    cube_size = pf.nodes.math.combine_xyz(x=thickness_val, y=depth_val, z=height_val)
    cube = pf.nodes.geo.mesh_cube(size=cube_size)

    offset_x = (width_val - thickness_val) * -0.5
    offset_z = height_val * 0.5

    translate_1 = pf.nodes.math.combine_xyz(x=offset_x, z=offset_z)
    left = pf.nodes.geo.transform(
        geometry=cube.mesh, translation=translate_1, rotation=(0, 0, 0), scale=(1, 1, 1)
    )

    translate_2 = pf.nodes.math.combine_xyz(x=offset_x * -1.0, z=offset_z)
    right = pf.nodes.geo.transform(
        geometry=cube.mesh, translation=translate_2, rotation=(0, 0, 0), scale=(1, 1, 1)
    )

    joined = pf.nodes.geo.join_geometry([left, right])
    return joined


@pf.nodes.node_function
def _bookcase_geometry(
    dimensions: t.SocketOrVal[pf.Vector],
    side_board_thickness: t.SocketOrVal[float],
    division_board_thickness: t.SocketOrVal[float],
    bottom_gap: t.SocketOrVal[float],
    backboard_thickness: t.SocketOrVal[float],
    screw_head_depth: t.SocketOrVal[float],
    screw_head_radius: t.SocketOrVal[float],
    screw_head_dist: t.SocketOrVal[float],
    attach_thickness: t.SocketOrVal[float],
    attach_width: t.SocketOrVal[float],
    attach_top_length: t.SocketOrVal[float],
    attach_back_length: t.SocketOrVal[float],
    frame_material: t.SocketOrVal[pf.Material],
    metal_material: t.SocketOrVal[pf.Material],
) -> pf.ProcNode:
    shelf_depth, shelf_width, shelf_height = dimensions.x, dimensions.y, dimensions.z

    sides = _side_board(
        board_thickness=side_board_thickness,
        depth=shelf_depth,
        height=shelf_height,
        width=shelf_width,
    )

    div_boards = _all_division_boards(
        board_thickness=division_board_thickness,
        depth=shelf_depth,
        width=shelf_width,
        side_thickness=side_board_thickness,
        height=shelf_height,
        gap=bottom_gap,
    )

    back = _back_board(
        width=shelf_width,
        thickness=backboard_thickness,
        height=shelf_height,
        depth=shelf_depth,
    )

    frame_joined = pf.nodes.geo.join_geometry([sides, div_boards, back])
    frame_realized = pf.nodes.geo.realize_instances(frame_joined)
    frame_with_mat = pf.nodes.geo.set_material(
        geometry=frame_realized, material=frame_material
    )

    screws = _screw_head_bookcase(
        screw_depth=screw_head_depth,
        screw_radius=screw_head_radius,
        bottom_gap=bottom_gap,
        division_thickness=division_board_thickness,
        width=shelf_width,
        height=shelf_height,
        shelf_depth=shelf_depth,
        screw_gap=screw_head_dist,
    )
    screws_realized = pf.nodes.geo.realize_instances(screws)
    screws_with_mat = pf.nodes.geo.set_material(
        geometry=screws_realized, material=metal_material
    )

    attach = _attach_gadget(
        division_thickness=division_board_thickness,
        height=shelf_height,
        attach_thickness=attach_thickness,
        attach_width=attach_width,
        attach_back_len=attach_back_length,
        attach_top_len=attach_top_length,
        depth=shelf_depth,
    )
    attach_realized = pf.nodes.geo.realize_instances(attach)
    attach_with_mat = pf.nodes.geo.set_material(
        geometry=attach_realized, material=metal_material
    )

    all_joined = pf.nodes.geo.join_geometry(
        [frame_with_mat, screws_with_mat, attach_with_mat]
    )
    final_realized = pf.nodes.geo.realize_instances(all_joined)
    rotated = pf.nodes.geo.transform(
        geometry=final_realized,
        rotation=(0.0, 0.0, -1.5708),
        translation=(0, 0, 0),
        scale=(1, 1, 1),
    )

    return rotated


def bookcase(
    dimensions: pf.Vector | None = None,
    side_board_thickness: float = 0.0175,
    division_board_thickness: float = 0.015,
    bottom_gap: float = 0.1,
    backboard_thickness: float = 0.015,
    screw_head_depth: float = 0.005,
    screw_head_radius: float = 0.0055,
    screw_head_dist: float = 0.065,
    attach_thickness: float = 0.0035,
    attach_width: float = 0.025,
    attach_top_length: float = 0.065,
    attach_back_length: float = 0.035,
    frame_material: pf.Material | None = None,
    metal_material: pf.Material | None = None,
) -> BookcaseResult:
    if dimensions is None:
        dimensions = pf.Vector((0.3, 0.5, 0.75))
    if frame_material is None:
        frame_material = pf.Material(surface=pf.nodes.shader.principled_bsdf())
    if metal_material is None:
        metal_material = pf.Material(surface=pf.nodes.shader.principled_bsdf())

    geo = _bookcase_geometry(
        dimensions=dimensions,
        side_board_thickness=side_board_thickness,
        division_board_thickness=division_board_thickness,
        bottom_gap=bottom_gap,
        backboard_thickness=backboard_thickness,
        screw_head_depth=screw_head_depth,
        screw_head_radius=screw_head_radius,
        screw_head_dist=screw_head_dist,
        attach_thickness=attach_thickness,
        attach_width=attach_width,
        attach_top_length=attach_top_length,
        attach_back_length=attach_back_length,
        frame_material=frame_material,
        metal_material=metal_material,
    )
    return BookcaseResult(mesh=pf.nodes.to_mesh_object(geo))


def bookcase_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    frame_material: pf.Material | None = None,
    metal_material: pf.Material | None = None,
) -> BookcaseResult:
    if dimensions is None:
        depth = pf.random.uniform(rng, 0.15, 0.45)
        width = pf.random.uniform(rng, 0.25, 0.75)
        height = pf.random.uniform(rng, 0.5, 1.0)
        dimensions = pf.Vector((depth, width, height))

    side_board_thickness = pf.random.uniform(rng, 0.005, 0.03)
    division_board_thickness = pf.random.uniform(rng, 0.005, 0.025)
    bottom_gap = pf.random.uniform(rng, 0.0, 0.2)
    backboard_thickness = pf.random.uniform(rng, 0.01, 0.02)
    screw_head_depth = pf.random.uniform(rng, 0.002, 0.008)
    screw_head_radius = pf.random.uniform(rng, 0.003, 0.008)
    screw_head_dist = pf.random.uniform(rng, 0.03, 0.1)
    attach_thickness = pf.random.uniform(rng, 0.002, 0.005)
    attach_width = pf.random.uniform(rng, 0.01, 0.04)
    attach_top_length = pf.random.uniform(rng, 0.03, 0.1)
    attach_back_length = pf.random.uniform(rng, 0.02, 0.05)

    vec = pf.nodes.shader.geometry().position
    if frame_material is None:
        frame_material = furniture_material_rand(rng, vec)
    if metal_material is None:
        metal_material = metal_brushed.metal_brushed_linear_rand(rng, vec)

    geo = _bookcase_geometry(
        dimensions=dimensions,
        side_board_thickness=side_board_thickness,
        division_board_thickness=division_board_thickness,
        bottom_gap=bottom_gap,
        backboard_thickness=backboard_thickness,
        screw_head_depth=screw_head_depth,
        screw_head_radius=screw_head_radius,
        screw_head_dist=screw_head_dist,
        attach_thickness=attach_thickness,
        attach_width=attach_width,
        attach_top_length=attach_top_length,
        attach_back_length=attach_back_length,
        frame_material=frame_material,
        metal_material=metal_material,
    )
    return BookcaseResult(mesh=pf.nodes.to_mesh_object(geo))
