# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Beining Han: original Infinigen v1 nodegroup (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/objects/shelves/drawers.py)
# - Alexander Raistrick: transpile to procfunc/v2

from typing import NamedTuple

import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.shaders.functionality_lists import (
    furniture_material_rand,
)

__all__ = [
    "DrawersResult",
    "drawers",
    "drawers_rand",
]


class DrawersResult(NamedTuple):
    mesh: pf.MeshObject


@pf.nodes.node_function
def _drawer_door_board(
    thickness: t.SocketOrVal[float],
    width: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
) -> pf.ProcNode:
    translation_y_a = thickness + 0.0
    translation_z_a = height + 0.0

    cube_size = pf.nodes.math.combine_xyz(
        x=width + 0.0,
        y=translation_y_a,
        z=translation_z_a,
    )
    cube = pf.nodes.geo.mesh_cube(
        size=cube_size, vertices_x=5, vertices_y=5, vertices_z=5
    )

    store_named_attribute = pf.nodes.geo.store_named_attribute(
        geometry=cube.mesh,
        name="uv_map",
        value=cube.uv_map,
        domain="CORNER",
    )

    translation = pf.nodes.math.combine_xyz(
        y=translation_y_a * -0.5,
        z=translation_z_a * 0.5,
    )

    transform = pf.nodes.geo.transform(
        geometry=store_named_attribute,
        translation=translation,
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    return transform


@pf.nodes.node_function
def _door_knob(
    radius: t.SocketOrVal[float],
    length: t.SocketOrVal[float],
    z: t.SocketOrVal[float],
) -> pf.ProcNode:
    cylinder_depth = length + 0.0
    cylinder = pf.nodes.geo.mesh_cylinder(
        vertices=24, radius=radius, depth=cylinder_depth
    )

    store_named_attribute = pf.nodes.geo.store_named_attribute(
        geometry=cylinder.mesh,
        name="uv_map",
        value=cylinder.uv_map,
        domain="CORNER",
    )

    translation_y = cylinder_depth * 0.5
    translation_z = z + 0.0
    translation = pf.nodes.math.combine_xyz(
        y=translation_y + 0.0001,
        z=translation_z * 0.5,
    )

    transform = pf.nodes.geo.transform(
        geometry=store_named_attribute,
        translation=translation,
        rotation=(1.5708, 0.0, 0.0),
        scale=(1, 1, 1),
    )
    return transform


@pf.nodes.node_function
def _kallax_drawer_frame(
    depth: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    thickness: t.SocketOrVal[float],
    width: t.SocketOrVal[float],
) -> pf.ProcNode:
    transform_a_2 = width + 0.0
    transform_a_1 = thickness + 0.0
    transform_translation_z_a = height + 0.0

    cube_size = pf.nodes.math.combine_xyz(
        x=transform_a_2,
        y=transform_a_1,
        z=transform_translation_z_a,
    )
    cube = pf.nodes.geo.mesh_cube(
        size=cube_size, vertices_x=4, vertices_y=4, vertices_z=4
    )

    store_named_attribute = pf.nodes.geo.store_named_attribute(
        geometry=cube.mesh,
        name="uv_map",
        value=cube.uv_map,
        domain="CORNER",
    )

    transform_translation_y_a = depth + 0.0
    transform_translation_y = pf.nodes.math.multiply_add(
        a=transform_translation_y_a,
        b=-1.0,
        addend=transform_a_1 * 0.5,
    )
    transform_translation_z = pf.nodes.math.multiply_add(
        a=transform_translation_z_a, b=0.5, addend=0.01
    )
    transform_translation = pf.nodes.math.combine_xyz(
        y=transform_translation_y, z=transform_translation_z
    )
    transform = pf.nodes.geo.transform(
        geometry=store_named_attribute,
        translation=transform_translation,
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )

    cube_b = transform_a_1 + -0.0001
    cube_1_size = pf.nodes.math.combine_xyz(
        x=transform_a_2 + cube_b,
        y=transform_translation_y_a,
        z=transform_a_1,
    )
    cube_1 = pf.nodes.geo.mesh_cube(
        size=cube_1_size, vertices_x=4, vertices_y=4, vertices_z=4
    )

    store_named_attribute_1 = pf.nodes.geo.store_named_attribute(
        geometry=cube_1.mesh,
        name="uv_map",
        value=cube_1.uv_map,
        domain="CORNER",
    )

    transform_1_translation_y = pf.nodes.math.multiply_add(
        a=transform_translation_y_a, b=-0.5, addend=-0.0001
    )
    transform_1_translation = pf.nodes.math.combine_xyz(
        y=transform_1_translation_y, z=0.01
    )
    transform_1 = pf.nodes.geo.transform(
        geometry=store_named_attribute_1,
        translation=transform_1_translation,
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )

    cube_2_size = pf.nodes.math.combine_xyz(
        x=transform_a_1,
        y=transform_translation_y_a,
        z=transform_translation_z_a,
    )
    cube_2 = pf.nodes.geo.mesh_cube(
        size=cube_2_size, vertices_x=4, vertices_y=4, vertices_z=4
    )

    store_named_attribute_2 = pf.nodes.geo.store_named_attribute(
        geometry=cube_2.mesh,
        name="uv_map",
        value=cube_2.uv_map,
        domain="CORNER",
    )

    transform_a_0 = transform_translation_y_a * -0.5
    transform_2_translation_z = pf.nodes.math.multiply_add(
        a=transform_translation_z_a, b=0.5, addend=0.01
    )
    transform_2_translation = pf.nodes.math.combine_xyz(
        x=transform_a_2 * 0.5,
        y=transform_a_0 + -0.0001,
        z=transform_2_translation_z,
    )
    transform_2 = pf.nodes.geo.transform(
        geometry=store_named_attribute_2,
        translation=transform_2_translation,
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    transform_3 = pf.nodes.geo.transform(
        geometry=transform_2,
        scale=(-1.0, 1.0, 1.0),
        translation=(0, 0, 0),
        rotation=(0, 0, 0),
    )

    join = pf.nodes.geo.join_geometry(
        [transform, transform_1, transform_2, transform_3]
    )
    return join


@pf.nodes.node_function
def _board_rail(
    width: t.SocketOrVal[float],
    thickness: t.SocketOrVal[float],
    depth: t.SocketOrVal[float],
) -> pf.ProcNode:
    transform_a_3 = depth + 0.0

    cylinder_depth = transform_a_3 - 0.03
    cylinder = pf.nodes.geo.mesh_cylinder(
        vertices=24, radius=0.003, depth=cylinder_depth
    )

    store_named_attribute = pf.nodes.geo.store_named_attribute(
        geometry=cylinder.mesh,
        name="uv_map",
        value=cylinder.uv_map,
        domain="CORNER",
    )

    transform_a_2 = width * 0.5
    transform_translation = pf.nodes.math.combine_xyz(z=transform_a_2)
    transform = pf.nodes.geo.transform(
        geometry=store_named_attribute,
        translation=transform_translation,
        rotation=(1.5708, 0.0, 0.0),
        scale=(1, 1, 1),
    )
    transform_1 = pf.nodes.geo.transform(
        geometry=transform,
        scale=(1.0, 1.0, -1.0),
        translation=(0, 0, 0),
        rotation=(0, 0, 0),
    )

    join_1 = pf.nodes.geo.join_geometry([transform, transform_1])

    cube_size = pf.nodes.math.combine_xyz(x=0.002, y=cylinder_depth, z=width)
    cube = pf.nodes.geo.mesh_cube(cube_size)

    store_named_attribute_1 = pf.nodes.geo.store_named_attribute(
        geometry=cube.mesh,
        name="uv_map",
        value=cube.uv_map,
        domain="CORNER",
    )

    transform_2 = pf.nodes.geo.transform(
        store_named_attribute_1,
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
        translation=(0, 0, 0),
    )

    cylinder_1 = pf.nodes.geo.mesh_cylinder(vertices=24, radius=0.004, depth=0.005)

    store_named_attribute_2 = pf.nodes.geo.store_named_attribute(
        geometry=cylinder_1.mesh,
        name="uv_map",
        value=cylinder_1.uv_map,
        domain="CORNER",
    )

    transform_a_1 = transform_a_3 * -0.5
    transform_3_translation = pf.nodes.math.combine_xyz(y=transform_a_1 + 0.02)
    transform_3 = pf.nodes.geo.transform(
        geometry=store_named_attribute_2,
        translation=transform_3_translation,
        rotation=(0.0, 1.5708, 0.0),
        scale=(1, 1, 1),
    )

    join_2 = pf.nodes.geo.join_geometry([join_1, transform_2, transform_3])

    transform_a_0 = thickness * 0.5
    transform_4_translation = pf.nodes.math.combine_xyz(
        x=transform_a_0 + 0.003,
        y=transform_a_3 * -0.5,
        z=transform_a_2 + 0.02,
    )
    transform_4 = pf.nodes.geo.transform(
        geometry=join_2,
        translation=transform_4_translation,
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    transform_5 = pf.nodes.geo.transform(
        geometry=transform_4,
        scale=(-1.0, 1.0, 1.0),
        translation=(0, 0, 0),
        rotation=(0, 0, 0),
    )

    join = pf.nodes.geo.join_geometry([transform_4, transform_5])
    return join


@pf.nodes.node_function
def _drawers_geometry(
    dimensions: t.SocketOrVal[pf.Vector],
    drawer_board_thickness: t.SocketOrVal[float],
    drawer_side_height: t.SocketOrVal[float],
    drawer_width_gap: t.SocketOrVal[float],
    knob_radius: t.SocketOrVal[float],
    knob_length: t.SocketOrVal[float],
    frame_material: t.SocketOrVal[pf.Material],
) -> pf.ProcNode:
    depth, width, height = dimensions.x, dimensions.y, dimensions.z

    drawer_board_width = width
    drawer_board_height = height
    drawer_depth = depth - drawer_board_thickness
    drawer_width = width - drawer_width_gap

    door_board = _drawer_door_board(
        thickness=drawer_board_thickness,
        width=drawer_board_width,
        height=drawer_board_height,
    )

    knob = _door_knob(
        radius=knob_radius,
        length=knob_length,
        z=drawer_board_height,
    )

    frame = _kallax_drawer_frame(
        depth=drawer_depth,
        height=drawer_side_height,
        thickness=drawer_board_thickness,
        width=drawer_width,
    )

    joined = pf.nodes.geo.join_geometry([door_board, knob, frame])
    with_mat = pf.nodes.geo.set_material(geometry=joined, material=frame_material)
    realized = pf.nodes.geo.realize_instances(with_mat)
    triangulated = pf.nodes.geo.triangulate(realized)
    rotated = pf.nodes.geo.transform(
        geometry=triangulated,
        rotation=(0.0, 0.0, -1.5708),
        translation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    return rotated


def drawers(
    dimensions: pf.Vector | None = None,
    drawer_board_thickness: float = 0.0075,
    drawer_side_height: float = 0.125,
    drawer_width_gap: float = 0.02,
    knob_radius: float = 0.0045,
    knob_length: float = 0.0265,
    frame_material: pf.Material | None = None,
) -> DrawersResult:
    if dimensions is None:
        dimensions = pf.Vector((0.35, 0.5, 0.325))
    if frame_material is None:
        frame_material = pf.Material(surface=pf.nodes.shader.principled_bsdf())

    geo = _drawers_geometry(
        dimensions=dimensions,
        drawer_board_thickness=drawer_board_thickness,
        drawer_side_height=drawer_side_height,
        drawer_width_gap=drawer_width_gap,
        knob_radius=knob_radius,
        knob_length=knob_length,
        frame_material=frame_material,
    )
    return DrawersResult(mesh=pf.nodes.to_mesh_object(geo))


def drawers_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    frame_material: pf.Material | None = None,
) -> DrawersResult:
    if dimensions is None:
        depth = pf.random.uniform(rng, 0.3, 0.4)
        width = pf.random.uniform(rng, 0.3, 0.7)
        height = pf.random.uniform(rng, 0.25, 0.4)
        dimensions = pf.Vector((depth, width, height))

    drawer_board_thickness = pf.random.uniform(rng, 0.005, 0.01)
    drawer_side_height = pf.random.uniform(rng, 0.05, 0.2)
    drawer_width_gap = pf.random.uniform(rng, 0.015, 0.025)
    knob_radius = pf.random.uniform(rng, 0.003, 0.006)
    knob_length = pf.random.uniform(rng, 0.018, 0.035)

    vec = pf.nodes.shader.geometry().position
    if frame_material is None:
        frame_material = furniture_material_rand(rng, vec)

    geo = _drawers_geometry(
        dimensions=dimensions,
        drawer_board_thickness=drawer_board_thickness,
        drawer_side_height=drawer_side_height,
        drawer_width_gap=drawer_width_gap,
        knob_radius=knob_radius,
        knob_length=knob_length,
        frame_material=frame_material,
    )
    return DrawersResult(mesh=pf.nodes.to_mesh_object(geo))
