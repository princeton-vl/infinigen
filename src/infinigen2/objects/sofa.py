# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Alexander Raistrick: original Infinigen v1 sofa nodegroup (https://github.com/princeton-vl/infinigen/commit/5c016c408c76c0f1bd97b449f1a149a7a8050b3e)
# - Stamatis Alexandropoulos: added the Infinigen v1 footrest and alternate arm and leg styles (https://github.com/princeton-vl/infinigen/commit/6637750d6ff7d440384b04381bfb64f70d116cc1)
# - Yiming Zuo: added Infinigen v1 curved-arm shaping, beveled back cushions, and backrest variation (https://github.com/princeton-vl/infinigen/commit/e8f0f7bb642b9eec582aa71912b324eb2b574539)
# - Alexander Raistrick: refactor for Infinigen2

import math
from typing import NamedTuple

import numpy as np
import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.objects import cushion
from infinigen2.objects.furniture_bases import base_square_rand, base_straight_rand
from infinigen2.shaders.functionality_lists import (
    decorative_material_rand,
    fabric_sturdy_rand,
)
from infinigen2.util import mesh as mesh_util
from infinigen2.util.instance import instances_on_line

__all__ = [
    "SofaResult",
    "sofa",
    "sofa_dimensions_rand",
    "sofa_rand",
    "sofa_with_base_rand",
]


class SofaResult(NamedTuple):
    mesh: pf.MeshObject
    back_seat_line_start: pf.Vector
    back_seat_line_end: pf.Vector
    back_tilt: float


ARM_TYPE_SQUARE = 0
ARM_TYPE_ROUND = 1
ARM_TYPE_ANGULAR = 2


@pf.nodes.node_function
def _sofa_arm(
    dimensions: t.SocketOrVal[pf.Vector],
    arm_dimensions: t.SocketOrVal[pf.Vector],
    arm_type: t.SocketOrVal[int],
    arm_width: t.SocketOrVal[float],
    arm_height: t.SocketOrVal[float],
    arm_back_crease: t.SocketOrVal[float],
    body_radius: t.SocketOrVal[float],
) -> pf.ProcNode[pf.MeshObject]:
    is_arm_angular = pf.nodes.func.equal(a=arm_type, b=ARM_TYPE_ANGULAR)
    is_arm_square = pf.nodes.func.equal(a=arm_type, b=ARM_TYPE_SQUARE)
    join_b_dimensions = pf.nodes.math.combine_xyz(
        x=arm_dimensions.x,
        y=arm_dimensions.y,
        z=(  # move down by radius to prevent taller-than-average arms when adding cylinder
            arm_dimensions.z - arm_dimensions.y * 0.5
        ),
    )

    transform_numerator = join_b_dimensions.x * 1.0001

    arm_radius = join_b_dimensions.y * 0.5
    cylinder = pf.nodes.geo.mesh_cylinder(
        fill_type="TRIANGLE_FAN",
        side_segments=4,
        radius=arm_radius,
        depth=transform_numerator,
    )

    join_b_location = dimensions * (0.0, 0.5, 0.0)

    transform_12_translation = pf.nodes.math.combine_xyz(
        x=transform_numerator / 2.0,
        y=join_b_location.y - arm_radius,
        z=join_b_dimensions.z,
    )
    transform_12 = pf.nodes.geo.transform(
        geometry=cylinder.mesh,
        translation=transform_12_translation,
        rotation=(0.0, 1.5708, 0.0),
        scale=(1, 1, 1),
    )

    arm_cube = mesh_util.box_with_support_loops(
        location=join_b_location,
        anchor=(0.0, 1.0, 0.0),
        size=join_b_dimensions,
        support_loop_offset=cushion.support_loop_offset(body_radius, join_b_dimensions),
    )

    arm_round = pf.nodes.geo.join_geometry([transform_12, arm_cube])

    arm_cube_1_location = dimensions * (0.0, 0.5, 0.0)
    arm_cube_1 = mesh_util.box(
        location=arm_cube_1_location,
        anchor=(0.0, 1.0, 0.0),
        size=arm_dimensions,
        vertices_x=4,
        vertices_y=4,
        vertices_z=10,
        crease=arm_back_crease,
    )

    input_position = pf.nodes.geo.input_position()

    set_y_value = pf.nodes.math.map_range(
        value=input_position.z,
        from_min=-0.1,
        from_max=arm_dimensions.z,
        to_min=-0.1,
        to_max=0.2,
    )
    set_y_1 = pf.nodes.math.float_curve(
        factor=arm_width,
        value=set_y_value,
        curve=np.array(
            [
                [0.0092, 0.7688],
                [0.1011, 0.5937],
                [0.1494, 0.4062],
                [0.3954, 0.0781],
                [1.0, 0.2187],
            ]
        ),
    )
    set_y_0 = input_position.y - arm_cube_1_location.y

    input_position_1 = pf.nodes.geo.input_position()

    set_z_value = pf.nodes.math.map_range(
        value=input_position_1.x,
        from_min=-1.0,
        from_max=0.6,
        to_min=2.1,
        to_max=-1.1,
    )
    set_z_1 = pf.nodes.math.float_curve(
        factor=arm_height,
        value=set_z_value,
        curve=np.array([[0.1341, 0.2094], [0.7386, 1.0], [0.9682, 0.0781], [1.0, 0.0]]),
    )
    set_z_b = pf.nodes.math.constant((-2.9, 3.3, 0.0))
    set_z_0 = input_position_1.z - set_z_b.z
    set_position_offset_vector = pf.nodes.math.combine_xyz(
        y=set_y_1 * set_y_0, z=set_z_1 * set_z_0
    )
    set_position_offset = pf.nodes.math.vector_rotate_axis_angle(
        vector=set_position_offset_vector,
        axis=(1.0, 0.0, 0.0),
        center=(0, 0, 0),
        angle=0.0,
    )
    arm_angular = pf.nodes.geo.set_position(
        geometry=arm_cube_1, offset=set_position_offset
    )

    arm_switch = pf.nodes.func.switch(switch=is_arm_square, a=arm_round, b=arm_cube)
    arm_switch = pf.nodes.func.switch(
        switch=is_arm_angular,
        a=arm_switch,
        b=arm_angular,
    )

    return arm_switch


@pf.nodes.node_function
def _lean_arm(
    arm: t.SocketOrVal[pf.MeshObject],
    arm_top: t.SocketOrVal[float],
    slant_tan: t.SocketOrVal[float],
) -> pf.ProcNode[pf.MeshObject]:
    below_top = pf.nodes.math.minimum(pf.nodes.geo.input_position().z - arm_top, 0.0)
    offset = pf.nodes.math.combine_xyz(y=below_top * slant_tan)
    return pf.nodes.geo.set_position(geometry=arm, offset=offset)


@pf.nodes.node_function
def _sofa_geometry(
    dimensions: t.SocketOrVal[pf.Vector],
    arm_dimensions: t.SocketOrVal[pf.Vector],
    left_arm_thickness: t.SocketOrVal[float],
    right_arm_thickness: t.SocketOrVal[float],
    back_dimensions: t.SocketOrVal[pf.Vector],
    seat_thickness: t.SocketOrVal[float],
    seat_cushion_count: t.SocketOrVal[int],
    fabric_material: t.SocketOrVal[pf.Material],
    baseboard_height: t.SocketOrVal[float],
    backrest_width: t.SocketOrVal[float],
    seat_margin: t.SocketOrVal[float],
    backrest_angle: t.SocketOrVal[float],
    arm_width: t.SocketOrVal[float],
    arm_type: t.SocketOrVal[int],
    arm_height: t.SocketOrVal[float],
    arm_back_crease: t.SocketOrVal[float] = 0.2,
    cushion_radius: t.SocketOrVal[float] = 0.05,
    back_cushion_radius: t.SocketOrVal[float] = 0.06,
    seat_cushion_crown: t.SocketOrVal[float] = 0.0,
    back_cushion_crown: t.SocketOrVal[float] = 0.0,
    piping_radius: t.SocketOrVal[float] = 0.0,
    body_radius: t.SocketOrVal[float] = 0.02,
    arm_slant: t.SocketOrVal[float] = 0.0,
) -> pf.ProcNode[pf.MeshObject]:
    has_left_arm = left_arm_thickness > 0.0
    has_right_arm = right_arm_thickness > 0.0

    # arms lean out about their top edge, so the seat sits inside their lowered inner faces
    slant_tan = pf.nodes.math.tan(arm_slant)
    seat_top_z = baseboard_height + seat_thickness
    is_arm_square = pf.nodes.func.equal(a=arm_type, b=ARM_TYPE_SQUARE)
    square_top_drop = pf.nodes.func.switch(switch=is_arm_square, a=0.0, b=0.5)
    left_arm_top = arm_dimensions.z - left_arm_thickness * square_top_drop
    right_arm_top = arm_dimensions.z - right_arm_thickness * square_top_drop
    left_seat_shift = slant_tan * (left_arm_top - seat_top_z)
    right_seat_shift = slant_tan * (right_arm_top - seat_top_z)
    left_inset = left_arm_thickness + pf.nodes.func.switch(
        switch=has_left_arm, a=0.0, b=left_seat_shift
    )
    right_inset = right_arm_thickness + pf.nodes.func.switch(
        switch=has_right_arm, a=0.0, b=right_seat_shift
    )

    seat_depth = dimensions.x - back_dimensions.x
    inner_width = dimensions.y - left_inset - right_inset
    base_board_dimensions = pf.nodes.math.combine_xyz(
        x=seat_depth, y=inner_width, z=baseboard_height
    )
    seat_line_start = base_board_dimensions * (0.0, -0.5, 1.0)
    seat_line_end = base_board_dimensions * (0.0, 0.5, 1.0)
    back_line_offset = pf.nodes.math.combine_xyz(x=backrest_width, z=seat_thickness)
    cushion_width = inner_width / seat_cushion_count.astype(dtype=float)
    seat_cushion_dimensions = pf.nodes.math.combine_xyz(
        x=seat_depth, y=cushion_width, z=seat_thickness
    )

    back_cushion_dimensions = pf.nodes.math.combine_xyz(
        x=dimensions.z - seat_thickness - baseboard_height,
        y=cushion_width,
        z=backrest_width,
    )
    back_cushion = cushion.cushion_box_geometry(
        size=back_cushion_dimensions,
        material=fabric_material,
        anchor=(0.1, 0.5, 1.0),
        edge_radius=back_cushion_radius,
        crown=back_cushion_crown,
        piping_radius=piping_radius,
    )

    back_cushion_translation = pf.nodes.math.combine_xyz(
        back_dimensions.x + 0.1 - backrest_width
    )
    back_cushion_rotation = pf.nodes.math.combine_xyz(y=backrest_angle + -1.5708)
    cushion_scale = pf.nodes.math.combine_xyz(x=seat_margin, y=seat_margin, z=1.0)
    back_cushion = pf.nodes.geo.transform(
        geometry=back_cushion,
        translation=back_cushion_translation,
        rotation=back_cushion_rotation.astype(dtype=pf.Euler),
        scale=cushion_scale,
    )
    back_cushions = instances_on_line(
        instance=back_cushion,
        start=seat_line_start + back_line_offset,
        end=seat_line_end + back_line_offset,
        count=seat_cushion_count,
    )
    back_cushions = pf.nodes.geo.realize_instances(back_cushions)

    seat_cushion = cushion.cushion_box_geometry(
        size=seat_cushion_dimensions * (1.0, 1.03, 1.0),
        material=fabric_material,
        anchor=(0.0, 0.5, 0.0),
        edge_radius=cushion_radius,
        crown=seat_cushion_crown,
        piping_radius=piping_radius,
    )
    seat_cushion = pf.nodes.geo.store_named_attribute(
        domain="FACE",
        geometry=seat_cushion,
        name="TAG_cushion",
        value=True,
    )
    seat_cushion = pf.nodes.geo.transform(
        geometry=seat_cushion,
        scale=cushion_scale,
        translation=back_dimensions * (1.0, 0.0, 0.0),
        rotation=(0, 0, 0),
    )
    seat_cushions = instances_on_line(
        instance=seat_cushion,
        start=seat_line_start,
        end=seat_line_end,
        count=seat_cushion_count,
    )
    seat_cushions = pf.nodes.geo.realize_instances(seat_cushions)

    base_board_location = back_dimensions * (1.0, 0.0, 0.0)
    base_board = mesh_util.box_with_support_loops(
        location=base_board_location,
        anchor=(0.0, 0.5, -1.0),
        size=base_board_dimensions,
        support_loop_offset=cushion.support_loop_offset(
            body_radius, base_board_dimensions
        ),
    )

    back_board_dimensions = pf.nodes.math.combine_xyz(
        x=back_dimensions.x,
        y=dimensions.y - left_arm_thickness - right_arm_thickness,
        z=back_dimensions.z,
    )
    back_board = mesh_util.box_with_support_loops(
        location=(0.0, 0.0, 0.0),
        anchor=(0.0, 0.5, -1.0),
        size=back_board_dimensions,
        support_loop_offset=cushion.support_loop_offset(
            body_radius, back_board_dimensions
        ),
    )
    back_board = pf.nodes.geo.transform(
        geometry=back_board,
        translation=pf.nodes.math.combine_xyz(
            y=(left_arm_thickness - right_arm_thickness) * 0.5
        ),
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )

    right_arm_dimensions = pf.nodes.math.combine_xyz(
        x=arm_dimensions.x, y=right_arm_thickness, z=arm_dimensions.z
    )
    right_arm_upright = _sofa_arm(
        dimensions=dimensions,
        arm_dimensions=right_arm_dimensions,
        arm_type=arm_type,
        arm_width=arm_width,
        arm_height=arm_height,
        arm_back_crease=arm_back_crease,
        body_radius=body_radius,
    )
    right_arm = _lean_arm(right_arm_upright, right_arm_top, slant_tan)
    right_arm = pf.nodes.func.switch(switch=has_right_arm, b=right_arm)

    left_arm_dimensions = pf.nodes.math.combine_xyz(
        x=arm_dimensions.x, y=left_arm_thickness, z=arm_dimensions.z
    )
    left_arm_upright = _sofa_arm(
        dimensions=dimensions,
        arm_dimensions=left_arm_dimensions,
        arm_type=arm_type,
        arm_width=arm_width,
        arm_height=arm_height,
        arm_back_crease=arm_back_crease,
        body_radius=body_radius,
    )
    left_arm = _lean_arm(left_arm_upright, left_arm_top, slant_tan)
    left_arm = pf.nodes.geo.transform(
        geometry=left_arm,
        scale=(1.0, -1.0, 1.0),
        translation=(0, 0, 0),
        rotation=(0, 0, 0),
    )
    left_arm = pf.nodes.geo.flip_faces(left_arm)
    left_arm = pf.nodes.func.switch(switch=has_left_arm, b=left_arm)
    arms = pf.nodes.geo.join_geometry([right_arm, left_arm])

    inner = pf.nodes.geo.join_geometry([back_cushions, seat_cushions, base_board])
    inner_offset = pf.nodes.math.combine_xyz(y=(left_inset - right_inset) * 0.5)
    inner = pf.nodes.geo.transform(
        geometry=inner, translation=inner_offset, rotation=(0, 0, 0), scale=(1, 1, 1)
    )

    geometry = pf.nodes.geo.join_geometry([inner, back_board, arms])
    geometry = pf.nodes.geo.set_material(geometry, fabric_material)
    geometry = pf.nodes.geo.merge_by_distance(geometry, distance=1e-5)
    return geometry


@pf.tracer.generator
def sofa(
    dimensions: pf.Vector | None = None,
    arm_dimensions: pf.Vector = (1.0, 0.105, 0.625),
    left_arm_thickness: float | None = None,
    right_arm_thickness: float | None = None,
    back_dimensions: pf.Vector = (0.2, 0.0, 0.625),
    seat_thickness: float = 0.2,
    seat_cushion_count: int = 2,
    baseboard_height: float = 0.07,
    backrest_width: float = 0.15,
    seat_margin: float = 0.985,
    backrest_angle: float = -0.325,
    arm_width: float = 0.75,
    arm_type: int = ARM_TYPE_SQUARE,
    arm_height: float = 0.8,
    arm_back_crease: float = 0.2,
    cushion_radius: float = 0.05,
    back_cushion_radius: float = 0.06,
    seat_cushion_crown: float = 0.0,
    back_cushion_crown: float = 0.0,
    piping_radius: float = 0.0,
    body_radius: float = 0.02,
    arm_slant: float = 0.0,
    fabric_material: pf.Material | None = None,
) -> SofaResult:
    if dimensions is None:
        dimensions = pf.Vector((0.925, 1.75, 0.83))
    if fabric_material is None:
        fabric_material = pf.Material(surface=pf.nodes.shader.principled_bsdf())
    if left_arm_thickness is None:
        left_arm_thickness = arm_dimensions[1]
    if right_arm_thickness is None:
        right_arm_thickness = arm_dimensions[1]

    res = _sofa_geometry(
        dimensions=dimensions,
        arm_dimensions=arm_dimensions,
        left_arm_thickness=left_arm_thickness,
        right_arm_thickness=right_arm_thickness,
        back_dimensions=back_dimensions,
        seat_thickness=seat_thickness,
        seat_cushion_count=seat_cushion_count,
        fabric_material=fabric_material,
        baseboard_height=baseboard_height,
        backrest_width=backrest_width,
        seat_margin=seat_margin,
        backrest_angle=backrest_angle,
        arm_width=arm_width,
        arm_type=arm_type,
        arm_height=arm_height,
        arm_back_crease=arm_back_crease,
        cushion_radius=cushion_radius,
        back_cushion_radius=back_cushion_radius,
        seat_cushion_crown=seat_cushion_crown,
        back_cushion_crown=back_cushion_crown,
        piping_radius=piping_radius,
        body_radius=body_radius,
        arm_slant=arm_slant,
    )
    obj = pf.nodes.to_mesh_object(res)
    pf.ops.uv.cube_project(obj, uv_name="UVMap")

    seat_top = baseboard_height + seat_thickness
    left_inset = _seat_inset(
        left_arm_thickness, arm_dimensions[2], arm_type, arm_slant, seat_top
    )
    right_inset = _seat_inset(
        right_arm_thickness, arm_dimensions[2], arm_type, arm_slant, seat_top
    )
    back_x = back_dimensions[0] + 0.1 + backrest_width / np.cos(backrest_angle)
    return SofaResult(
        mesh=obj,
        back_seat_line_start=pf.Vector(
            (back_x, left_inset - dimensions[1] * 0.5, seat_top)
        ),
        back_seat_line_end=pf.Vector(
            (back_x, dimensions[1] * 0.5 - right_inset, seat_top)
        ),
        back_tilt=-backrest_angle,
    )


def _seat_inset(
    arm_thickness: float,
    arm_height: float,
    arm_type: int,
    arm_slant: float,
    seat_top: float,
) -> float:
    square_top = 1 - np.minimum(np.abs(arm_type - ARM_TYPE_SQUARE), 1)
    arm_top = arm_height - arm_thickness * 0.5 * square_top
    seat_shift = np.tan(arm_slant) * (arm_top - seat_top)
    return arm_thickness + seat_shift * np.sign(arm_thickness)


def sofa_dimensions_rand(rng: pf.RNG) -> pf.Vector:
    """Default sofa dimensions."""
    return (
        pf.random.uniform(rng, 0.82, 1.08),
        pf.random.clip_gaussian(rng, 1.75, 0.75, 0.9, 3),
        pf.random.uniform(rng, 0.76, 0.98),
    )


def sofa_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    material: pf.Material | None = None,
    left_arm_thickness: float | None = None,
    right_arm_thickness: float | None = None,
    seat_height: float | None = None,
) -> SofaResult:
    """Skirted sofa whose body rests on the floor, seat 16-20 in high."""
    rng, rng_fabric, rng_piping, rng_seat, rng_dims, rng_arm = rng.spawn(6)
    if dimensions is None:
        dimensions = sofa_dimensions_rand(rng_dims)
    if material is None:
        material = fabric_sturdy_rand(rng_fabric, pf.nodes.shader.coord().uv)
    depth = dimensions[0]

    arm_type = pf.control.choice(
        rng_arm,
        [(ARM_TYPE_SQUARE, 0.4), (ARM_TYPE_ROUND, 0.2), (ARM_TYPE_ANGULAR, 0.4)],
    )

    boxiness = pf.random.uniform(rng, 0.0, 1.0) ** 1.5
    plushness = (1.0 - boxiness) ** 2

    if seat_height is None:
        seat_height = pf.random.uniform(rng_seat, 0.41, 0.5)
    seat_thickness = pf.random.uniform(rng, 0.1, 0.2)
    baseboard_height = max(seat_height - seat_thickness, 0.04)
    seat_top = baseboard_height + seat_thickness
    back_cushion_height = max(dimensions[2] - seat_height, 0.3)

    arm_thickness = (0.09 + 0.17 * boxiness) * pf.random.uniform(rng, 0.85, 1.15)
    arm_thickness = min(
        arm_thickness, (dimensions[1] - 0.55) * 0.5, dimensions[1] * 0.15
    )
    backrest_width = pf.random.uniform(rng, 0.12, 0.2)
    back_thickness = (0.13 + 0.1 * boxiness) * pf.random.uniform(rng, 0.9, 1.1)
    back_thickness = min(back_thickness, depth - backrest_width - 0.5)

    low_arm_top = seat_top + pf.random.uniform(rng, 0.06, 0.28)
    back_top = seat_top + back_cushion_height
    rail_top = low_arm_top + (back_top - low_arm_top) * pf.random.uniform(rng, 0.3, 1.0)
    arm_rise = min(max(2.0 * pf.random.uniform(rng, 0.0, 1.0) - 0.5, 0.0), 1.0)
    arm_top = low_arm_top + (rail_top - low_arm_top) * arm_rise
    arm_length = depth * pf.random.uniform(rng, 0.97, 1.05)
    cushion_target_width = pf.random.uniform(rng, 0.55, 0.8)

    arm_dimensions = (arm_length, arm_thickness, arm_top)
    back_dimensions = (back_thickness, 0.0, rail_top)

    seat_margin = pf.random.uniform(rng, 0.96, 1.0)
    backrest_angle = pf.random.uniform(rng, -0.44, -0.17)
    arm_width = pf.random.uniform(rng, 0.6, 0.9)
    arm_height = pf.random.uniform(rng, 0.7, 0.9)

    arm_back_crease = pf.random.uniform(rng, 0.0, 0.4) * (1.0 - 0.5 * boxiness)
    cushion_radius = 0.03 + 0.17 * plushness * pf.random.uniform(rng, 0.7, 1.0)
    back_cushion_radius = 0.04 + 0.16 * plushness * pf.random.uniform(rng, 0.7, 1.0)
    body_radius = (0.015 + 0.05 * plushness) * pf.random.uniform(rng, 0.8, 1.2)
    seat_cushion_crown = pf.random.uniform(rng, 0.005, 0.02) + 0.025 * plushness
    back_cushion_crown = pf.random.uniform(rng, 0.008, 0.025) + 0.025 * plushness
    piping_radius = cushion.optional_piping_radius_rand(rng_piping, 0.4)
    arm_slant = 0.25 * pf.random.uniform(rng, 0.0, 1.0) ** 3

    if left_arm_thickness is None:
        left_arm_thickness = arm_thickness
    if right_arm_thickness is None:
        right_arm_thickness = arm_thickness
    left_inset = _seat_inset(left_arm_thickness, arm_top, arm_type, arm_slant, seat_top)
    right_inset = _seat_inset(
        right_arm_thickness, arm_top, arm_type, arm_slant, seat_top
    )
    inner_width = dimensions[1] - left_inset - right_inset
    seat_cushion_count = max(1, round(inner_width / cushion_target_width))

    result = sofa(
        dimensions=(dimensions[0], dimensions[1], back_top),
        arm_dimensions=arm_dimensions,
        left_arm_thickness=left_arm_thickness,
        right_arm_thickness=right_arm_thickness,
        back_dimensions=back_dimensions,
        seat_thickness=seat_thickness,
        seat_cushion_count=seat_cushion_count,
        fabric_material=material,
        baseboard_height=baseboard_height,
        backrest_width=backrest_width,
        seat_margin=seat_margin,
        backrest_angle=backrest_angle,
        arm_width=arm_width,
        arm_type=arm_type,
        arm_height=arm_height,
        arm_back_crease=arm_back_crease,
        cushion_radius=cushion_radius,
        back_cushion_radius=back_cushion_radius,
        seat_cushion_crown=seat_cushion_crown,
        back_cushion_crown=back_cushion_crown,
        piping_radius=piping_radius,
        body_radius=body_radius,
        arm_slant=arm_slant,
    )
    body = result.mesh
    pf.ops.modifier.subdivide_surface(body, levels=5, _skip_apply=True)
    return SofaResult(
        mesh=body,
        back_seat_line_start=result.back_seat_line_start,
        back_seat_line_end=result.back_seat_line_end,
        back_tilt=result.back_tilt,
    )


def sofa_with_base_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    material: pf.Material | None = None,
    base_material: pf.Material | None = None,
) -> SofaResult:
    """Sofa body raised on a table-style leg base 2-10 in high, seat 16-20 in high."""
    rng, rng_sofa, rng_dims, rng_mat, rng_base_sel, rng_base = rng.spawn(6)
    if dimensions is None:
        dimensions = sofa_dimensions_rand(rng_dims)
    base_height = pf.random.uniform(rng, 0.0508, 0.254)
    seat_height = pf.random.uniform(rng, 0.41, 0.5)
    leg_diameter = pf.random.uniform(rng, 0.1016, 0.254)

    body = sofa_rand(
        rng_sofa,
        dimensions=(dimensions[0], dimensions[1], dimensions[2] - base_height),
        material=material,
        seat_height=max(seat_height - base_height, 0.2),
    )
    pf.ops.object.set_transform(body.mesh, location=(0.0, 0.0, base_height))

    if base_material is None:
        base_material = decorative_material_rand(rng_mat, pf.nodes.shader.coord().uv)
    base_fn = pf.control.choice(
        rng_base_sel,
        [(base_straight_rand, 2.0), (base_square_rand, 1.0)],
    )
    base = base_fn(
        rng_base,
        dimensions=pf.Vector((dimensions[1], dimensions[0], base_height)),
        material=base_material,
        leg_diameter=leg_diameter,
        leg_placement_bottom_scale=1.0,
    ).mesh
    # bases separate their frames along local x, so turn that across the length
    pf.ops.object.set_transform(
        base,
        location=(dimensions[0] * 0.5, 0.0, 0.0),
        rotation_euler=(0.0, 0.0, math.pi / 2),
    )
    pf.ops.object.join(body.mesh, base)
    pf.ops.mesh.transform_apply(body.mesh)
    lift = pf.Vector((0.0, 0.0, base_height))
    return SofaResult(
        mesh=body.mesh,
        back_seat_line_start=body.back_seat_line_start + lift,
        back_seat_line_end=body.back_seat_line_end + lift,
        back_tilt=body.back_tilt,
    )
