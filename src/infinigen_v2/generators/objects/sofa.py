# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Alexander Raistrick - initial version, refactor to procfunc
# - Stamatis Alexandropolous, Yiming Zuo - add footrest and alternate arm/leg styles


from typing import NamedTuple

import numpy as np
import procfunc as pf
from procfunc.nodes import types as t

from infinigen_v2.generators.shaders.functionality_lists import (
    furniture_fabric,
    furniture_material_distribution,
)


class SofaResult(NamedTuple):
    mesh: pf.MeshObject


@pf.nodes.node_function
def array_fill_line(
    line_start: t.SocketOrVal[pf.Vector],
    line_end: t.SocketOrVal[pf.Vector],
    instance_dimensions: t.SocketOrVal[pf.Vector],
    count: t.SocketOrVal[int],
    instance: t.SocketOrVal[pf.MeshObject],
) -> pf.ProcNode[pf.MeshObject]:
    line_b = instance_dimensions * (0.0, -0.5, 0.0)
    line_from_endpoints = pf.nodes.geo.mesh_line_from_endpoints(
        count=count,
        start_location=line_end + line_b,
        end_location=line_start - line_b,
    )

    instance_on_points = pf.nodes.geo.instance_on_points(
        points=line_from_endpoints, instance=instance
    )

    realize_instances = pf.nodes.geo.realize_instances(instance_on_points)
    return realize_instances


@pf.nodes.node_function
def corner_cube(
    dimensions: t.SocketOrVal[pf.Vector],
    location: t.SocketOrVal[pf.Vector] = (0.0, 0.0, 0.0),
    centering_loc: t.SocketOrVal[pf.Vector] = (0.1, 0.5, 1.0),
    supporting_edge_fac: t.SocketOrVal[float] = 0.0,
    vertices_x: t.SocketOrVal[int] = 2,
    vertices_y: t.SocketOrVal[int] = 2,
    vertices_z: t.SocketOrVal[int] = 2,
    crease: t.SocketOrVal[float] = 0.0,
) -> pf.ProcNode[pf.MeshObject]:
    cube = pf.nodes.geo.mesh_cube(
        size=dimensions,
        vertices_x=vertices_x,
        vertices_y=vertices_y,
        vertices_z=vertices_z,
    )

    transform_translation_a = pf.nodes.math.map_range(
        value=centering_loc,
        from_min=(0.0, 0.0, 0.0),
        from_max=(1.0, 1.0, 1.0),
        to_min=(0.5, 0.5, 0.5),
        to_max=(-0.5, -0.5, -0.5),
    )
    transform_translation = pf.nodes.math.vector_multiply_add(
        a=transform_translation_a,
        b=dimensions,
        addend=location,
    )
    transform = pf.nodes.geo.transform(
        geometry=cube.mesh,
        translation=transform_translation,
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )

    return pf.nodes.geo.store_named_attribute(
        geometry=transform, domain="EDGE", name="crease_edge", value=crease
    )


ARM_TYPE_SQUARE = 0
ARM_TYPE_ROUND = 1
ARM_TYPE_ANGULAR = 2


@pf.nodes.node_function
def sofa(
    dimensions: t.SocketOrVal[pf.Vector],
    arm_dimensions: t.SocketOrVal[pf.Vector],
    back_dimensions: t.SocketOrVal[pf.Vector],
    seat_dimensions: t.SocketOrVal[pf.Vector],
    foot_dimensions: t.SocketOrVal[pf.Vector],
    fabric_material: t.SocketOrVal[pf.Material],
    foot_material: t.SocketOrVal[pf.Material],
    baseboard_height: t.SocketOrVal[float],
    backrest_width: t.SocketOrVal[float],
    seat_margin: t.SocketOrVal[float],
    backrest_angle: t.SocketOrVal[float],
    arm_width: t.SocketOrVal[float],
    arm_type: t.SocketOrVal[int],
    arm_height: t.SocketOrVal[float],
    arms_angle: t.SocketOrVal[float],
    reflection: t.SocketOrVal[int],
    leg_type: t.SocketOrVal[bool],
    leg_dimensions: t.SocketOrVal[float],
    leg_z: t.SocketOrVal[float],
    leg_faces: t.SocketOrVal[int],
    footrest: t.SocketOrVal[bool] = False,
    count: t.SocketOrVal[int] = 0,
    scaling_footrest: t.SocketOrVal[float] = 0,
    body_crease: t.SocketOrVal[float] = 0.7,
    cushion_crease: t.SocketOrVal[float] = 0.15,
    arm_back_crease: t.SocketOrVal[float] = 0.2,
) -> pf.ProcNode[pf.MeshObject]:
    join_y_numerator_addend = pf.nodes.math.vector_multiply_add(
        a=arm_dimensions,
        b=(0.0, -2.0, 0.0),
        addend=dimensions,
    )
    join_y_numerator = pf.nodes.math.vector_multiply_add(
        a=back_dimensions,
        b=(-1.0, 0.0, 0.0),
        addend=join_y_numerator_addend,
    )

    base_board_2_dimensions = pf.nodes.math.combine_xyz(
        x=join_y_numerator.x,
        y=join_y_numerator.y,
        z=baseboard_height,
    )

    join_a_2 = base_board_2_dimensions * (0.0, -0.5, 1.0)
    join_b_0 = pf.nodes.math.combine_xyz(x=backrest_width, z=seat_dimensions.z)
    join_a_1 = base_board_2_dimensions * (0.0, 0.5, 1.0)
    join_12 = pf.nodes.math.ceil(join_y_numerator.y / seat_dimensions.y)
    join_y = join_y_numerator.y / join_12
    join_1_geometries_0_instance_dimensions = pf.nodes.math.combine_xyz(
        x=seat_dimensions.x, y=join_y, z=seat_dimensions.z
    )

    seat_a = dimensions.z - seat_dimensions.z
    seat_cushion_dimensions = pf.nodes.math.combine_xyz(
        x=seat_a - baseboard_height, y=join_y, z=backrest_width
    )
    seat_cushion = corner_cube(
        location=(0.0, 0.0, 0.0),
        centering_loc=(0.1, 0.5, 1.0),
        dimensions=seat_cushion_dimensions,
        supporting_edge_fac=0.0,
        vertices_x=2,
        vertices_y=2,
        vertices_z=2,
        crease=cushion_crease,
    )

    extrude = pf.nodes.geo.extrude_mesh(mesh=seat_cushion, offset_scale=0.03)

    scale_elements = pf.nodes.geo.scale_elements(
        geometry=extrude.mesh,
        selection=extrude.top,
        scale=0.6,
        center=(0, 0, 0),
    )

    transform_translation_x_1 = backrest_width * -1.0
    transform_translation_x_0 = back_dimensions.x + 0.1
    transform_translation = pf.nodes.math.combine_xyz(
        transform_translation_x_1 + transform_translation_x_0
    )
    transform_rotation = pf.nodes.math.combine_xyz(y=backrest_angle + -1.5708)
    transform_scale = pf.nodes.math.combine_xyz(x=seat_margin, y=seat_margin, z=1.0)
    transform = pf.nodes.geo.transform(
        geometry=scale_elements,
        translation=transform_translation,
        rotation=transform_rotation.astype(dtype=pf.Euler),
        scale=transform_scale,
    )

    array_fill_line_result = array_fill_line(
        line_start=join_a_2 + join_b_0,
        line_end=join_a_1 + join_b_0,
        instance_dimensions=join_1_geometries_0_instance_dimensions,
        count=join_12.astype(dtype=int),
        instance=transform,
    )

    seat_cushion_1 = corner_cube(
        location=(0.0, 0.0, 0.0),
        centering_loc=(0.0, 0.5, 0.0),
        dimensions=join_1_geometries_0_instance_dimensions * (1.0, 1.03, 1.0),
        supporting_edge_fac=0.0,
        vertices_x=2,
        vertices_y=2,
        vertices_z=2,
        crease=cushion_crease,
    )

    transform_1_selection_0 = pf.nodes.math.constant(1.0)

    store_named_attribute_3 = pf.nodes.geo.store_named_attribute(
        domain="FACE",
        geometry=seat_cushion_1,
        selection=transform_1_selection_0.astype(dtype=bool),
        name="TAG_cushion",
        value=True,
    )

    transform_1 = pf.nodes.geo.transform(
        geometry=store_named_attribute_3,
        scale=transform_scale,
        translation=(0, 0, 0),
        rotation=(0, 0, 0),
    )

    array_fill_line_result_1 = array_fill_line(
        line_start=join_a_2,
        line_end=join_a_1,
        instance_dimensions=join_1_geometries_0_instance_dimensions,
        count=join_12.astype(dtype=int),
        instance=transform_1,
    )

    join_geometries_1_b_switch = pf.nodes.func.equal(a=count, b=4)
    join_0_switch = pf.nodes.func.equal(a=count, b=4)
    join_11 = pf.nodes.func.switch(switch=join_0_switch, a=reflection, b=1)
    join_line_end_b = pf.nodes.math.combine_xyz(
        x=1.0, y=join_11.astype(dtype=float), z=1.1
    )
    join_line_end_1 = join_a_1 * join_line_end_b

    transform_2_scale = pf.nodes.math.combine_xyz(x=scaling_footrest, y=1.0, z=1.1)
    transform_2 = pf.nodes.geo.transform(
        geometry=transform_1,
        scale=transform_2_scale,
        translation=(0, 0, 0),
        rotation=(0, 0, 0),
    )

    array_fill_line_result_2 = array_fill_line(
        line_start=join_a_2,
        line_end=join_line_end_1,
        instance_dimensions=join_1_geometries_0_instance_dimensions * join_line_end_b,
        count=count,
        instance=transform_2,
    )

    join_line_end_0 = pf.nodes.math.combine_xyz(z=join_line_end_1.z)

    transform_3_scale = pf.nodes.math.combine_xyz(x=1.0, y=join_12, z=1.0)
    transform_3 = pf.nodes.geo.transform(
        geometry=transform_2,
        scale=transform_3_scale,
        translation=(0, 0, 0),
        rotation=(0, 0, 0),
    )

    array_fill_line_result_3 = array_fill_line(
        line_start=(0.0, 0.0, 0.0),
        line_end=join_line_end_0,
        instance_dimensions=(0.0, 0.0, 0.0),
        count=1,
        instance=transform_3,
    )

    join_geometries_1_b = pf.nodes.func.switch(
        switch=join_geometries_1_b_switch,
        a=array_fill_line_result_2,
        b=array_fill_line_result_3,
    )
    join_geometries = pf.nodes.func.switch(switch=footrest, b=join_geometries_1_b)
    join = pf.nodes.geo.join_geometry([array_fill_line_result_1, join_geometries])

    subdivide_1 = pf.nodes.geo.subdivide_mesh(mesh=join, level=2)

    join_1 = pf.nodes.geo.join_geometry([array_fill_line_result, subdivide_1])

    grid = pf.nodes.geo.mesh_grid(vertices_x=2, vertices_y=2)

    transform_4_scale_1 = dimensions * (1.0, 1.0, 0.0)
    transform_4_scale_0 = foot_dimensions * (2.5, 2.5, 0.0)
    transform_4 = pf.nodes.geo.transform(
        geometry=grid.mesh,
        translation=dimensions * (0.5, 0.0, 0.0),
        scale=transform_4_scale_1 - transform_4_scale_0,
        rotation=(0, 0, 0),
    )

    cone = pf.nodes.geo.mesh_cone(
        vertices=leg_faces,
        side_segments=4,
        radius_top=0.01,
        radius_bottom=0.025,
        depth=0.07,
    )

    transform_5_scale = pf.nodes.math.combine_xyz(
        x=leg_dimensions, y=leg_dimensions, z=leg_z
    )
    transform_5 = pf.nodes.geo.transform(
        geometry=cone.mesh,
        translation=(0.0, 0.0, 0.01),
        rotation=(0.0, 3.1416, 0.0),
        scale=transform_5_scale,
    )

    foot_cube = corner_cube(
        location=(0.0, 0.0, 0.0),
        centering_loc=(0.5, 0.5, 0.9),
        dimensions=foot_dimensions,
        supporting_edge_fac=0.0,
        vertices_x=4,
        vertices_y=4,
        vertices_z=4,
    )

    transform_6 = pf.nodes.geo.transform(
        geometry=foot_cube,
        scale=(0.5, 0.8, 0.8),
        translation=(0, 0, 0),
        rotation=(0, 0, 0),
    )
    foot = pf.nodes.func.switch(switch=leg_type, a=transform_5, b=transform_6)
    foot = pf.nodes.geo.set_material(foot, foot_material)

    instance_on_points = pf.nodes.geo.instance_on_points(
        points=transform_4, instance=foot
    )
    feet = pf.nodes.geo.realize_instances(instance_on_points)

    join_7_geometries_0_switch = pf.nodes.func.equal(a=count, b=4)

    transform_switch = pf.nodes.func.equal(a=count, b=4)

    base_board_dimensions_b = pf.nodes.math.combine_xyz(x=1.0, y=join_12, z=1.0)
    base_board_dimensions = base_board_2_dimensions / base_board_dimensions_b

    transform_8_translation_a = pf.nodes.func.switch(
        switch=transform_switch,
        a=base_board_dimensions,
        b=base_board_2_dimensions,
    )

    grid_1 = pf.nodes.geo.mesh_grid(
        size_y=transform_8_translation_a.y * 0.7,
        vertices_x=1,
        vertices_y=2,
    )

    transform_8_translation_b = pf.nodes.math.combine_xyz(
        x=0.1,
        y=transform_8_translation_a.y,
        z=transform_8_translation_a.z,
    )
    transform_8_translation_1 = transform_8_translation_a - transform_8_translation_b
    transform_8_translation_0 = back_dimensions * (1.0, 0.0, 0.0)
    transform_8 = pf.nodes.geo.transform(
        geometry=grid_1.mesh,
        translation=transform_8_translation_1 + transform_8_translation_0,
        scale=(1.0, 1.0, 0.9),
        rotation=(0, 0, 0),
    )

    instance_on_points_1 = pf.nodes.geo.instance_on_points(
        points=transform_8,
        instance=foot,
        scale=(1.0, 1.0, 1.2),
    )
    footrest_feet = pf.nodes.geo.realize_instances(instance_on_points_1)

    base_a_y = pf.nodes.math.multiply_add(
        a=arm_dimensions.y, b=-2.0, addend=dimensions.y
    )
    base_a = pf.nodes.math.combine_xyz(
        x=back_dimensions.x, y=base_a_y, z=back_dimensions.z
    )
    base_board_2_location = base_a * (1.0, 0.0, 0.0)
    base_board = corner_cube(
        location=base_board_2_location,
        centering_loc=(0.0, 0.5, -1.0),
        dimensions=base_board_dimensions,
        supporting_edge_fac=0.0,
        vertices_x=2,
        vertices_y=2,
        vertices_z=2,
    )

    join_2 = pf.nodes.geo.join_geometry([footrest_feet, base_board])
    join_a_0 = base_board_dimensions_b - (1.0, 1.0, 1.0)
    join_a_translation_1 = base_board_dimensions * join_a_0 * (0.0, 0.5, 0.0)
    join_a_translation_0 = pf.nodes.math.combine_xyz(
        x=1.0, y=reflection.astype(dtype=float), z=1.0
    )
    join_a_scale = pf.nodes.math.combine_xyz(x=scaling_footrest, y=1.0, z=1.0)

    transform_9 = pf.nodes.geo.transform(
        geometry=join_2,
        translation=join_a_translation_1 * join_a_translation_0,
        scale=join_a_scale,
        rotation=(0, 0, 0),
    )

    join_7_geometries_0_a = pf.nodes.func.switch(switch=footrest, a=transform_9)

    base_board_1 = corner_cube(
        location=base_board_2_location,
        centering_loc=(0.0, 0.5, -1.0),
        dimensions=base_board_2_dimensions,
        supporting_edge_fac=0.0,
        vertices_x=3,
        vertices_y=3,
        vertices_z=3,
    )

    transform_10_scale = pf.nodes.math.combine_xyz(x=scaling_footrest, y=1.0, z=1.0)
    transform_10 = pf.nodes.geo.transform(
        geometry=base_board_1,
        scale=transform_10_scale,
        translation=(0, 0, 0),
        rotation=(0, 0, 0),
    )
    transform_11_scale = pf.nodes.math.combine_xyz(x=scaling_footrest, y=1.3, z=1.0)
    transform_11 = pf.nodes.geo.transform(
        geometry=footrest_feet,
        scale=transform_11_scale,
        translation=(0, 0, 0),
        rotation=(0, 0, 0),
    )

    join_3 = pf.nodes.geo.join_geometry([transform_10, transform_11])
    join_7_geometries_0_b = pf.nodes.func.switch(switch=footrest, b=join_3)
    join_7_geometries = pf.nodes.func.switch(
        switch=join_7_geometries_0_switch,
        a=join_7_geometries_0_a,
        b=join_7_geometries_0_b,
    )

    base_board_2 = corner_cube(
        location=base_board_2_location,
        centering_loc=(0.0, 0.5, -1.0),
        dimensions=base_board_2_dimensions,
        supporting_edge_fac=0.0,
        vertices_x=5,
        vertices_y=5,
        vertices_z=2,
        crease=body_crease,
    )

    back_board = corner_cube(
        location=(0.0, 0.0, 0.0),
        centering_loc=(0.0, 0.5, -1.0),
        dimensions=base_a,
        supporting_edge_fac=0.0,
        vertices_x=2,
        vertices_y=5,
        vertices_z=5,
        crease=arm_back_crease,
    )

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

    cylinder = pf.nodes.geo.mesh_cylinder(
        fill_type="TRIANGLE_FAN",
        side_segments=4,
        radius=join_b_dimensions.y,
        depth=transform_numerator,
    )

    # store_named_attribute_4 = pf.nodes.geo.store_named_attribute(
    #     geometry=cylinder.mesh,
    #     name="UVMap",
    #     value=cylinder.uv_map,
    # )

    join_b_location = dimensions * (0.0, 0.5, 0.0)

    transform_12_translation = pf.nodes.math.combine_xyz(
        x=transform_numerator / 2.0,
        y=join_b_location.y,
        z=join_b_dimensions.z - join_b_dimensions.y,
    )
    transform_12 = pf.nodes.geo.transform(
        geometry=cylinder.mesh,
        translation=transform_12_translation,
        rotation=(0.0, 1.5708, 0.0),
        scale=(1, 1, 1),
    )

    arm_cube = corner_cube(
        location=join_b_location,
        centering_loc=(0.0, 1.0, 0.0),
        dimensions=join_b_dimensions,
        supporting_edge_fac=0.0,
        vertices_x=4,
        vertices_y=4,
        vertices_z=4,
        crease=arm_back_crease,
    )

    arm_round = pf.nodes.geo.join_geometry([transform_12, arm_cube])

    arm_cube_1_location = dimensions * (0.0, 0.5, 0.0)
    arm_cube_1 = corner_cube(
        location=arm_cube_1_location,
        centering_loc=(0.0, 1.0, 0.0),
        dimensions=arm_dimensions,
        supporting_edge_fac=0.0,
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

    arm_sym = pf.nodes.geo.transform(
        geometry=arm_switch,
        scale=(1.0, -1.0, 1.0),
        translation=(0, 0, 0),
        rotation=(0, 0, 0),
    )
    arm_sym = pf.nodes.geo.flip_faces(arm_sym)
    arm_sym = pf.nodes.geo.join_geometry([arm_switch, arm_sym])

    join_6 = pf.nodes.geo.join_geometry([back_board, arm_sym])
    join_7 = pf.nodes.geo.join_geometry([join_7_geometries, base_board_2, join_6])

    all_fabric = pf.nodes.geo.join_geometry([join_1, join_7])
    all_fabric = pf.nodes.geo.set_material(all_fabric, fabric_material)

    geometry = pf.nodes.geo.join_geometry([all_fabric, feet])

    # TODO: this messes up the overall `dimensions`
    bbox_min_z = pf.nodes.geo.bound_box(geometry).min.z
    translation_for_legs = pf.nodes.math.combine_xyz(x=0, y=0, z=bbox_min_z * -1.0)
    geometry = pf.nodes.geo.transform(
        geometry, translation=translation_for_legs, rotation=(0, 0, 0), scale=(1, 1, 1)
    )

    return geometry


def sofa_distribution(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    material: pf.Material | None = None,
    foot_material: pf.Material | None = None,
) -> SofaResult:
    rng, rng_fabric, rng_foot = rng.spawn(3)
    if dimensions is None:
        dimensions = (
            pf.random.uniform(rng, 0.85, 1.0),
            pf.random.clip_gaussian(rng, 1.75, 0.75, 0.9, 3),
            pf.random.uniform(rng, 0.69, 0.97),
        )

    arm_type = pf.control.choice(
        rng,
        [(ARM_TYPE_SQUARE, 0.4), (ARM_TYPE_ROUND, 0.2), (ARM_TYPE_ANGULAR, 0.4)],
    )

    dim_x, dim_y, dim_z = dimensions

    reflection = pf.control.choice(rng, [(1, 0.5), (-1, 0.5)])
    leg_type = pf.control.choice(rng, [(True, 0.5), (False, 0.5)])

    arm_dimensions = pf.random.uniform(rng, (1.0, 0.06, 0.5), (1.0, 0.15, 0.75))
    back_dimensions = pf.random.uniform(rng, (0.15, 0.0, 0.5), (0.25, 0.0, 0.75))
    seat_dimensions = pf.random.uniform(rng, (dim_x, 1.2, 0.15), (dim_x, 1.5, 0.3))
    foot_dimensions = pf.random.uniform(rng, (0.07, 0.06, 0.06), (0.25, 0.06, 0.06))

    baseboard_height = pf.random.uniform(rng, 0.05, 0.09)
    backrest_width = pf.random.uniform(rng, 0.1, 0.2)
    seat_margin = pf.random.uniform(rng, 0.97, 1.0)
    backrest_angle = pf.random.uniform(rng, -0.5, -0.15)
    arm_width = pf.random.uniform(rng, 0.6, 0.9)
    arm_height = pf.random.uniform(rng, 0.7, 0.9)
    arms_angle = pf.random.uniform(rng, 0.0, 1.08)
    leg_dimensions = pf.random.uniform(rng, 0.4, 0.9)
    leg_z = pf.random.uniform(rng, 1.1, 2.5)
    leg_faces = pf.control.choice(rng, [(4, 0.5), (25, 0.5)])

    body_crease = pf.random.uniform(rng, 0.5, 0.9)
    cushion_crease = pf.random.uniform(rng, 0.0, 0.3)
    arm_back_crease = pf.random.uniform(rng, 0.0, 0.4)

    vec = pf.nodes.shader.coord().uv
    if material is None:
        material = furniture_fabric(rng_fabric, vec, translucency=0.0)
    if foot_material is None:
        foot_material = furniture_material_distribution(rng_foot, vec)

    res = sofa(
        dimensions=dimensions,
        arm_dimensions=arm_dimensions,
        back_dimensions=back_dimensions,
        seat_dimensions=seat_dimensions,
        foot_dimensions=foot_dimensions,
        fabric_material=material,
        foot_material=foot_material,
        baseboard_height=baseboard_height,
        backrest_width=backrest_width,
        seat_margin=seat_margin,
        backrest_angle=backrest_angle,
        arm_width=arm_width,
        arm_type=arm_type,
        arm_height=arm_height,
        arms_angle=arms_angle,
        reflection=reflection,
        leg_type=leg_type,
        leg_dimensions=leg_dimensions,
        leg_z=leg_z,
        leg_faces=leg_faces,
        footrest=False,  # disabled due to bugs with missing footrest seat and too tricky to assign the material
        body_crease=body_crease,
        cushion_crease=cushion_crease,
        arm_back_crease=arm_back_crease,
    )
    obj = pf.nodes.to_mesh_object(res)
    pf.ops.uv.cube_project(obj, uv_name="UVMap")
    pf.ops.modifier.subdivide_surface(obj, levels=5, _skip_apply=True)
    return SofaResult(mesh=obj)
