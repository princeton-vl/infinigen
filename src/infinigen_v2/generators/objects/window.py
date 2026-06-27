# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Hongyu Wen: primary author
# - Alexander Raistrick: refactor to procfunc/infinigen v2

import math
from typing import NamedTuple

import numpy as np
import procfunc as pf
from procfunc.nodes import types as t

from infinigen_v2.generators.shaders.functionality_lists import (
    furniture_fabric,
    furniture_material_distribution,
    glass_material_distribution,
)
from infinigen_v2.generators.util.curve import curve_to_mesh_with_uv


@pf.nodes.node_function
def line_seq(
    width: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    amount: t.SocketOrVal[float],
) -> pf.ProcNode[t.Instances]:
    curve_line_start_y = height * -0.5
    curve_line_start = pf.nodes.math.combine_xyz(x=width * 0.5, y=curve_line_start_y)
    curve_line_end = pf.nodes.math.combine_xyz(x=width * -0.5, y=curve_line_start_y)
    curve_line = pf.nodes.geo.curve_line(start=curve_line_start, end=curve_line_end)

    to_instance = pf.nodes.geo.geometry_to_instance(curve_line)

    duplicate_elements = pf.nodes.geo.duplicate_elements(
        domain="INSTANCE",
        geometry=to_instance,
        amount=amount.astype(dtype=int),
    )

    result_0_offset_y_1 = duplicate_elements.duplicate_index.astype(dtype=float) + 1.0
    result_0_offset_y_0 = height / (amount + 1.0)
    result_0_offset = pf.nodes.math.combine_xyz(
        y=result_0_offset_y_1 * result_0_offset_y_0
    )

    set_position = pf.nodes.geo.set_position(
        geometry=duplicate_elements.geometry,
        offset=result_0_offset,
    )
    return set_position


@pf.nodes.node_function
def curtain(
    width: t.SocketOrVal[float],
    depth: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    interval_number: t.SocketOrVal[float],
    radius: t.SocketOrVal[float],
    l1: t.SocketOrVal[float],
    r1: t.SocketOrVal[float],
    l2: t.SocketOrVal[float],
    r2: t.SocketOrVal[float],
    frame_depth: t.SocketOrVal[float],
    curtain_frame_material: t.SocketOrVal[pf.Material],
    curtain_material: t.SocketOrVal[pf.Material],
) -> pf.ProcNode[pf.MeshObject]:
    curve_line_start = pf.nodes.math.combine_xyz(l2)
    curve_line_end = pf.nodes.math.combine_xyz(r2)
    curve_line = pf.nodes.geo.curve_line(start=curve_line_start, end=curve_line_end)

    resample_curve_count = pf.nodes.geo.resample_curve_count(
        curve=curve_line, count=100
    )

    curve_line_1_start = pf.nodes.math.combine_xyz(l1)
    curve_line_1_end = pf.nodes.math.combine_xyz(r1)
    curve_line_1 = pf.nodes.geo.curve_line(
        start=curve_line_1_start, end=curve_line_1_end
    )

    resample_curve_count_1 = pf.nodes.geo.resample_curve_count(
        curve=curve_line_1, count=100
    )

    join: pf.ProcNode[pf.CurveObject] = pf.nodes.geo.join_geometry(
        [resample_curve_count, resample_curve_count_1]
    )

    spline_parameter = pf.nodes.geo.spline_parameter()

    set_numerator = interval_number * 6.28
    set_a_value = spline_parameter.length * (set_numerator / width)
    set_a = pf.nodes.math.sin(set_a_value + 1.68)
    set_position_offset = pf.nodes.math.combine_xyz(z=set_a * depth)
    set_position = pf.nodes.geo.set_position(
        geometry=join,
        offset=set_position_offset,
    )

    curve_quadrilateral = pf.nodes.geo.curve_quadrilateral(width=height, height=0.002)

    curve_to_result = curve_to_mesh_with_uv(
        curve=set_position,
        profile=curve_quadrilateral,
    )

    set_material = pf.nodes.geo.set_material(
        geometry=curve_to_result.mesh,
        selection=True,
        material=curtain_material,
    )

    curve_x_1 = width * 0.5
    curve_x_0 = curve_x_1 * -1.0
    curve_line_2_start = pf.nodes.math.combine_xyz(curve_x_0)
    curve_line_2_end = pf.nodes.math.combine_xyz(curve_x_1)
    curve_line_2 = pf.nodes.geo.curve_line(
        start=curve_line_2_start, end=curve_line_2_end
    )
    curve_circle = pf.nodes.geo.curve_circle(radius=radius * 1.3)
    curve_to_1 = pf.nodes.geo.curve_to_mesh(
        curve=curve_line_2, profile_curve=curve_circle
    )

    set_y = height * 0.47
    set_position_1_offset = pf.nodes.math.combine_xyz(y=set_y + radius)
    set_position_1 = pf.nodes.geo.set_position(
        geometry=curve_to_1, offset=set_position_1_offset
    )

    boolean = pf.nodes.geo.mesh_boolean(a=set_material, b=set_position_1)

    icosphere_radius = radius * 2.0
    icosphere = pf.nodes.geo.mesh_icosphere(radius=icosphere_radius, subdivisions=4)

    sample_curve = pf.nodes.geo.sample_curve(curves=curve_line_2, value=0.0, factor=0.0)

    set_position_2 = pf.nodes.geo.set_position(
        geometry=icosphere.mesh, offset=sample_curve.position
    )

    curve_line_3_end = pf.nodes.math.combine_xyz(x=curve_x_0, z=frame_depth)
    curve_line_3 = pf.nodes.geo.curve_line(
        start=curve_line_2_start, end=curve_line_3_end
    )
    curve_line_4_end = pf.nodes.math.combine_xyz(x=curve_x_1, z=frame_depth)
    curve_line_4 = pf.nodes.geo.curve_line(start=curve_line_2_end, end=curve_line_4_end)

    join_1 = pf.nodes.geo.join_geometry([curve_line_3, curve_line_4, curve_line_2])

    curve_circle_1 = pf.nodes.geo.curve_circle(radius=radius)
    curve_to_2 = pf.nodes.geo.curve_to_mesh(
        curve=join_1, profile_curve=curve_circle_1, fill_caps=True
    )

    icosphere_1 = pf.nodes.geo.mesh_icosphere(radius=icosphere_radius, subdivisions=4)

    sample_curve_1 = pf.nodes.geo.sample_curve(
        curves=curve_line_2, value=0.0, factor=1.0
    )

    set_position_3 = pf.nodes.geo.set_position(
        geometry=icosphere_1.mesh, offset=sample_curve_1.position
    )

    join_2 = pf.nodes.geo.join_geometry([set_position_2, curve_to_2, set_position_3])

    set_position_4_offset = pf.nodes.math.combine_xyz(y=set_y)
    set_position_4 = pf.nodes.geo.set_position(
        geometry=join_2, offset=set_position_4_offset
    )
    set_material_1 = pf.nodes.geo.set_material(
        geometry=set_position_4,
        selection=True,
        material=curtain_frame_material,
    )

    join_3 = pf.nodes.geo.join_geometry([boolean, set_material_1])

    set_shade_smooth = pf.nodes.geo.set_shade_smooth(
        geometry=join_3, shade_smooth=False
    )

    return set_shade_smooth


def window_dimensions_distribution(
    rng: pf.RNG,
    width: float | None = None,
    height: float | None = None,
) -> pf.Vector:
    if width is None:
        width = pf.random.uniform(rng, 1.0, 4.0)
    if height is None:
        height = pf.random.uniform(rng, 1.0, 4.0)
    return pf.Vector((pf.random.uniform(rng, 0.05, 0.12), width, height))


class CurtainResult(NamedTuple):
    mesh: pf.MeshObject


def curtain_distribution(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    material: pf.Material | None = None,
    rail_material: pf.Material | None = None,
) -> CurtainResult:
    if dimensions is None:
        dimensions = window_dimensions_distribution(rng)
    vec = pf.nodes.shader.coord().uv
    if material is None:
        material = furniture_fabric(rng, vec)

    if rail_material is None:
        rail_material = furniture_material_distribution(rng, vec)

    frame_depth = pf.random.uniform(rng, 0.05, 0.1)
    depth = frame_depth * pf.random.uniform(rng, 0.3, 1.0)
    interval_number = 1 + math.floor(dimensions.y * pf.random.uniform(rng, 5, 12))
    frame_radius = pf.random.uniform(rng, 0.01, 0.02)
    base_coverage = pf.random.uniform(rng, 0.1, 0.5)
    var_l = pf.random.uniform(rng, 0.0, 0.05)
    var_r = pf.random.uniform(rng, 0.0, 0.05)
    mid_l = -(0.5 - base_coverage - var_l) * dimensions.y
    mid_r = (0.5 - base_coverage - var_r) * dimensions.y

    curtain_r2 = dimensions.y * 0.5
    curtain_geo = curtain(
        width=dimensions.y,
        depth=depth,
        height=dimensions.z,
        interval_number=interval_number,
        radius=frame_radius,
        l1=-curtain_r2,
        r1=mid_l,
        l2=mid_r,
        r2=curtain_r2,
        frame_depth=-frame_depth,
        curtain_frame_material=rail_material,
        curtain_material=material,
    )

    return CurtainResult(mesh=pf.nodes.to_mesh_object(curtain_geo))


@pf.nodes.node_function
def window_shutter(
    width: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    frame_width: t.SocketOrVal[float],
    frame_thickness: t.SocketOrVal[float],
    panel_width: t.SocketOrVal[float],
    panel_thickness: t.SocketOrVal[float],
    shutter_width: t.SocketOrVal[float],
    shutter_thickness: t.SocketOrVal[float],
    shutter_interval: t.SocketOrVal[float],
    shutter_rotation: t.SocketOrVal[float],
    frame_material: t.SocketOrVal[pf.Material],
) -> pf.ProcNode[pf.MeshObject]:
    cube_size_y_a = height - frame_width

    set_0_a = pf.nodes.math.floor(cube_size_y_a / shutter_interval)

    shutter_true_interval = cube_size_y_a / set_0_a

    cube_size_y = shutter_true_interval * 2.0
    cube_size = pf.nodes.math.combine_xyz(
        x=panel_width,
        y=cube_size_y_a - cube_size_y,
        z=panel_thickness,
    )
    cube = pf.nodes.geo.mesh_cube(cube_size)

    curve_line_end = pf.nodes.math.combine_xyz(y=shutter_width * 0.5)
    curve_line = pf.nodes.geo.curve_line(end=curve_line_end, start=(0, 0, 0))

    to_instance = pf.nodes.geo.geometry_to_instance(curve_line)

    rotate_instances_rotation = pf.nodes.math.combine_xyz(shutter_rotation)
    rotate_instances = pf.nodes.geo.rotate_instances(
        instances=to_instance,
        rotation=rotate_instances_rotation.astype(dtype=pf.Euler),
        pivot_point=(0, 0, 0),
    )

    realize_instances_1 = pf.nodes.geo.realize_instances(rotate_instances)

    sample_curve = pf.nodes.geo.sample_curve(
        curves=realize_instances_1, value=0.0, factor=1.0
    )

    set_position = pf.nodes.geo.set_position(
        geometry=cube.mesh, offset=sample_curve.position
    )

    cube_1_size = pf.nodes.math.combine_xyz(
        x=width - frame_width, y=shutter_width, z=shutter_thickness
    )
    cube_1 = pf.nodes.geo.mesh_cube(cube_1_size)

    to_instance_1 = pf.nodes.geo.geometry_to_instance(cube_1.mesh)

    shutter_number = set_0_a - 1.0

    duplicate_elements = pf.nodes.geo.duplicate_elements(
        domain="INSTANCE",
        geometry=to_instance_1,
        amount=shutter_number.astype(dtype=int),
    )

    set_y_1 = (
        duplicate_elements.duplicate_index.astype(dtype=float) * shutter_true_interval
    )
    set_y_0 = (cube_size_y_a * -0.5) + shutter_true_interval
    set_position_1_offset = pf.nodes.math.combine_xyz(y=set_y_1 + set_y_0)
    set_position_1 = pf.nodes.geo.set_position(
        geometry=duplicate_elements.geometry,
        offset=set_position_1_offset,
    )

    rotate = pf.nodes.math.combine_xyz(shutter_rotation)
    rotate_instances_1 = pf.nodes.geo.rotate_instances(
        instances=set_position_1,
        rotation=rotate.astype(dtype=pf.Euler),
        pivot_point=(0, 0, 0),
    )

    curve_quadrilateral = pf.nodes.geo.curve_quadrilateral(width=width, height=height)
    curve_b = pf.nodes.math.sqrt(2.0)
    curve_quadrilateral_1 = pf.nodes.geo.curve_quadrilateral(
        width=frame_width * curve_b,
        height=frame_thickness,
    )
    curve_to = pf.nodes.geo.curve_to_mesh(
        curve=curve_quadrilateral,
        profile_curve=curve_quadrilateral_1,
    )

    join = pf.nodes.geo.join_geometry([set_position, rotate_instances_1, curve_to])

    set_material = pf.nodes.geo.set_material(
        geometry=join, selection=True, material=frame_material
    )
    set_shade_smooth = pf.nodes.geo.set_shade_smooth(
        geometry=set_material, shade_smooth=False
    )

    realize_instances = pf.nodes.geo.realize_instances(set_shade_smooth)
    return realize_instances


@pf.nodes.node_function
def window_panel(
    width: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    frame_width: t.SocketOrVal[float],
    frame_thickness: t.SocketOrVal[float],
    panel_width: t.SocketOrVal[float],
    panel_thickness: t.SocketOrVal[float],
    panel_h_amount: t.SocketOrVal[int],
    panel_v_amount: t.SocketOrVal[int],
    frame_material: t.SocketOrVal[pf.Material],
) -> pf.ProcNode[pf.MeshObject]:
    line_seq_result = line_seq(
        width=height, height=width, amount=panel_v_amount.astype(dtype=float) + -1.0
    )

    transform = pf.nodes.geo.transform(
        geometry=line_seq_result,
        rotation=(0.0, 0.0, 1.5708),
        translation=(0, 0, 0),
        scale=(1, 1, 1),
    )

    curve_quadrilateral_1_height = panel_thickness - 0.001
    curve_quadrilateral = pf.nodes.geo.curve_quadrilateral(
        width=panel_width,
        height=curve_quadrilateral_1_height - 0.001,
    )
    curve_to = pf.nodes.geo.curve_to_mesh(
        curve=transform, profile_curve=curve_quadrilateral
    )

    line_seq_result_1 = line_seq(
        width=width, height=height, amount=panel_h_amount.astype(dtype=float) + -1.0
    )

    curve_quadrilateral_1 = pf.nodes.geo.curve_quadrilateral(
        width=panel_width,
        height=curve_quadrilateral_1_height,
    )
    curve_to_1 = pf.nodes.geo.curve_to_mesh(
        curve=line_seq_result_1,
        profile_curve=curve_quadrilateral_1,
    )

    join = pf.nodes.geo.join_geometry([curve_to, curve_to_1])

    curve_quadrilateral_2 = pf.nodes.geo.curve_quadrilateral(width=width, height=height)
    curve_b = pf.nodes.math.sqrt(2.0)
    curve_quadrilateral_3 = pf.nodes.geo.curve_quadrilateral(
        width=frame_width * curve_b,
        height=frame_thickness,
    )
    curve_to_2 = pf.nodes.geo.curve_to_mesh(
        curve=curve_quadrilateral_2,
        profile_curve=curve_quadrilateral_3,
    )

    join_1 = pf.nodes.geo.join_geometry([join, curve_to_2])

    set_material = pf.nodes.geo.set_material(
        geometry=join_1, selection=True, material=frame_material
    )

    set_shade_smooth = pf.nodes.geo.set_shade_smooth(
        geometry=set_material, shade_smooth=False
    )
    return set_shade_smooth


class WindowGeometryResult(NamedTuple):
    geometry: pf.ProcNode[pf.MeshObject]
    bounding_box: pf.ProcNode[pf.MeshObject]


class WindowResult(NamedTuple):
    mesh: pf.MeshObject
    light: pf.LightObject


@pf.nodes.node_function
def window_geometry(
    width: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    frame_width: t.SocketOrVal[float],
    frame_thickness: t.SocketOrVal[float],
    panel_h_amount: t.SocketOrVal[int],
    panel_v_amount: t.SocketOrVal[int],
    sub_frame_width: t.SocketOrVal[float],
    sub_frame_thickness: t.SocketOrVal[float],
    sub_panel_h_amount: t.SocketOrVal[int],
    sub_panel_v_amount: t.SocketOrVal[int],
    open_h_angle: t.SocketOrVal[float],
    open_v_angle: t.SocketOrVal[float],
    open_offset: t.SocketOrVal[float],
    oe_offset: t.SocketOrVal[float],
    shutter: t.SocketOrVal[bool],
    shutter_panel_radius: t.SocketOrVal[float],
    shutter_width: t.SocketOrVal[float],
    shutter_thickness: t.SocketOrVal[float],
    shutter_rotation: t.SocketOrVal[float],
    shutter_interval: t.SocketOrVal[float],
    frame_material: t.SocketOrVal[pf.Material],
) -> WindowGeometryResult:
    rotate_b_3 = frame_width * panel_v_amount.astype(dtype=float)
    rotate_b_a = (width - rotate_b_3) / panel_v_amount.astype(dtype=float)

    transform_a_width = rotate_b_a - sub_frame_width

    rotate_numerator = frame_width * panel_h_amount.astype(dtype=float)
    rotate_y_a_a = (height - rotate_numerator) / panel_h_amount.astype(dtype=float)

    transform_a_height = rotate_y_a_a - sub_frame_width

    window_panel_result = window_panel(
        width=transform_a_width,
        height=transform_a_height,
        frame_width=sub_frame_width,
        frame_thickness=sub_frame_thickness,
        panel_width=sub_frame_width,
        panel_thickness=sub_frame_thickness,
        panel_h_amount=sub_panel_h_amount,
        panel_v_amount=sub_panel_v_amount,
        frame_material=frame_material,
    )
    window_shutter_result = window_shutter(
        width=transform_a_width,
        height=transform_a_height,
        frame_width=frame_width,
        frame_thickness=frame_thickness,
        panel_width=shutter_panel_radius,
        panel_thickness=shutter_panel_radius,
        shutter_width=shutter_width,
        shutter_thickness=shutter_thickness,
        shutter_interval=shutter_interval,
        shutter_rotation=shutter_rotation,
        frame_material=frame_material,
    )

    transform_geometry = pf.nodes.func.switch(
        switch=shutter, a=window_panel_result, b=window_shutter_result
    )

    rotate_a_1 = width * -0.5

    transform_translation_x = width / panel_v_amount.astype(dtype=float) * 0.5

    rotate_a_0 = height * -0.5

    transform_translation_y = height / panel_h_amount.astype(dtype=float) * 0.5
    transform_translation = pf.nodes.math.combine_xyz(
        x=rotate_a_1 + transform_translation_x,
        y=rotate_a_0 + transform_translation_y,
    )
    transform = pf.nodes.geo.transform(
        geometry=transform_geometry,
        translation=transform_translation,
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )

    to_instance = pf.nodes.geo.geometry_to_instance(transform)

    set_position_amount = panel_h_amount.astype(dtype=float) * panel_v_amount.astype(
        dtype=float
    )

    duplicate_elements = pf.nodes.geo.duplicate_elements(
        domain="INSTANCE",
        geometry=to_instance,
        amount=set_position_amount.astype(dtype=int),
    )
    set_x_exponent = pf.nodes.math.floor(
        duplicate_elements.duplicate_index.astype(dtype=float)
        / panel_h_amount.astype(dtype=float)
    )
    set_b_0 = rotate_b_a + frame_width

    rotate_y_a = rotate_y_a_a + frame_width

    set_position_offset_y = (
        duplicate_elements.duplicate_index.astype(dtype=float)
        % panel_h_amount.astype(dtype=float)
        * rotate_y_a
    )
    set_a = pf.nodes.math.power(base=-1.0, exponent=set_x_exponent)
    set_position_offset = pf.nodes.math.combine_xyz(
        x=set_x_exponent * set_b_0,
        y=set_position_offset_y,
        z=set_a * oe_offset,
    )
    set_position = pf.nodes.geo.set_position(
        geometry=duplicate_elements.geometry,
        offset=set_position_offset,
    )

    rotate_b_2 = pf.nodes.math.power(base=-1.0, exponent=set_x_exponent)
    rotate_instances_rotation = pf.nodes.math.combine_xyz(y=open_v_angle * rotate_b_2)
    rotate_b_1 = rotate_b_a * (set_x_exponent % 2.0)
    rotate_b_0 = rotate_y_a_a * (set_position_offset_y % 2.0)
    rotate_instances_pivot_point = pf.nodes.math.combine_xyz(
        x=rotate_a_1 + rotate_b_1, y=rotate_a_0 + rotate_b_0
    )
    rotate_instances = pf.nodes.geo.rotate_instances(
        instances=set_position,
        rotation=rotate_instances_rotation.astype(dtype=pf.Euler),
        pivot_point=rotate_instances_pivot_point,
    )
    rotate = pf.nodes.math.combine_xyz(open_h_angle * 0.5)
    rotate_instances_1_pivot_point = pf.nodes.math.combine_xyz(y=rotate_y_a * -1.0)
    rotate_instances_1 = pf.nodes.geo.rotate_instances(
        instances=rotate_instances,
        rotation=rotate.astype(dtype=pf.Euler),
        pivot_point=rotate_instances_1_pivot_point,
    )

    set_x_0 = pf.nodes.math.power(base=-1.0, exponent=set_x_exponent)
    set_position_1_offset = pf.nodes.math.combine_xyz(set_x_0 * open_offset)
    set_position_1 = pf.nodes.geo.set_position(
        geometry=rotate_instances_1, offset=set_position_1_offset
    )

    window_panel_result_1 = window_panel(
        width=width,
        height=height,
        frame_width=frame_width,
        frame_thickness=frame_thickness,
        panel_width=frame_width,
        panel_thickness=frame_thickness,
        panel_h_amount=panel_h_amount,
        panel_v_amount=panel_v_amount,
        frame_material=frame_material,
    )

    join = pf.nodes.geo.join_geometry([set_position_1, window_panel_result_1])

    realized = pf.nodes.geo.realize_instances(join)

    # Even out the long frame polys before subsurf, then crease all frame
    # edges so the later subsurf keeps the frame crisp.
    subdivided = pf.nodes.geo.subdivide_mesh(mesh=realized, level=3)
    creased = pf.nodes.geo.store_named_attribute(
        domain="EDGE",
        geometry=subdivided,
        name="crease_edge",
        value=1.0,
    )

    bound_box = pf.nodes.geo.bound_box(creased)

    return WindowGeometryResult(
        geometry=creased,
        bounding_box=bound_box.bounding_box,
    )


@pf.nodes.node_function
def glass_pane(
    width: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    material: t.SocketOrVal[pf.Material],
) -> pf.ProcNode[pf.MeshObject]:
    curve = pf.nodes.geo.curve_line(
        start=pf.nodes.math.combine_xyz(y=height * -0.5),
        end=pf.nodes.math.combine_xyz(y=height * 0.5),
    )
    profile = pf.nodes.geo.curve_line(
        start=pf.nodes.math.combine_xyz(x=width * -0.5),
        end=pf.nodes.math.combine_xyz(x=width * 0.5),
    )
    mesh = curve_to_mesh_with_uv(curve=curve, profile=profile)
    return pf.nodes.geo.set_material(
        geometry=mesh.mesh, selection=True, material=material
    )


def window_distribution(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    frame_material: pf.Material | None = None,
    glass_material: pf.Material | None = None,
    curtain: pf.MeshObject | None = None,
    include_glass_pane: bool = True,
) -> WindowResult:
    (
        rng_param,
        rng_dim,
        rng_frame,
        rng_glass,
        rng_curtain,
    ) = rng.spawn(5)

    if dimensions is None:
        dimensions = window_dimensions_distribution(rng_dim)

    # Frame dimensions - absolute
    frame_thickness = dimensions.x
    frame_width = pf.random.uniform(rng_param, 0.02, 0.05)

    # Panel grid - fraction of window, allows single panel or multiple
    target_panel_width_pct = pf.random.clip_gaussian(rng_param, 0.7, 0.5, 0.3, 1.5)
    target_panel_aspect = (dimensions.z / dimensions.y) + pf.random.clip_gaussian(
        rng_param, 0, 0.1, -0.2, 0.2
    )
    target_panel_width = dimensions.y * target_panel_width_pct
    target_panel_height = target_panel_width * target_panel_aspect
    panel_v_amount = 1 + math.floor(dimensions.y / target_panel_width)
    panel_h_amount = 1 + math.floor(dimensions.z / target_panel_height)

    # Actual panel dimensions after grid division
    actual_panel_width = (
        dimensions.y - frame_width * (panel_v_amount + 1)
    ) / panel_v_amount
    actual_panel_height = (
        dimensions.z - frame_width * (panel_h_amount + 1)
    ) / panel_h_amount

    # Glass and sub-frame - absolute
    glass_thickness = pf.random.uniform(rng_param, 0.01, 0.03)
    sub_frame_width = pf.random.uniform(rng_param, 0.015, 0.03)
    sub_frame_thickness = glass_thickness + pf.random.uniform(rng_param, 0, 1) * (
        frame_thickness - glass_thickness
    )

    # Sub-panel counts - compute target sub-panel size, can be full panel (no dividers)
    target_panel_size_pct = pf.random.clip_gaussian(rng_param, 0.7, 0.5, 0.2, 1.2)
    target_subpanel_width = actual_panel_width * target_panel_size_pct

    subpanel_aspect = pf.random.uniform(rng_param, 0.5, 2.0)
    target_subpanel_height = target_subpanel_width * subpanel_aspect
    sub_frame_v_amount = max(1, math.floor(actual_panel_width / target_subpanel_width))
    sub_frame_h_amount = max(
        1, math.floor(actual_panel_height / target_subpanel_height)
    )

    shutter = pf.control.choice(rng_param, [(True, 0.2), (False, 0.8)])

    shutter_panel_radius = pf.random.uniform(rng_param, 0.001, 0.003)
    shutter_width = pf.random.uniform(rng_param, 0.03, 0.05)
    shutter_thickness = pf.random.uniform(rng_param, 0.003, 0.007)
    shutter_rotation = pf.random.uniform(rng_param, 0.0, 1.0) ** 0.5
    shutter_interval = shutter_width * (1 + pf.random.uniform(rng_param, 0.02, 0.1))

    vec = pf.nodes.shader.coord().uv
    if frame_material is None:
        frame_material = furniture_material_distribution(rng_frame, vec)
    if glass_material is None:
        glass_material = glass_material_distribution(
            rng_glass, vec, glass_height=dimensions.z
        )

    if curtain is None:
        curtain_fn = pf.control.choice(
            rng_curtain,
            [
                (
                    lambda: (
                        curtain_distribution(rng_curtain, dimensions=dimensions).mesh
                    ),
                    1.0,
                ),
                (lambda: pf.ops.primitives.mesh_single_vertex(), 2.0),  # none
            ],
        )
        curtain = curtain_fn()

    res = window_geometry(
        width=dimensions.y,
        height=dimensions.z,
        frame_width=frame_width,
        frame_thickness=frame_thickness,
        panel_h_amount=panel_h_amount,
        panel_v_amount=panel_v_amount,
        sub_frame_width=sub_frame_width,
        sub_frame_thickness=sub_frame_thickness,
        sub_panel_h_amount=sub_frame_h_amount,
        sub_panel_v_amount=sub_frame_v_amount,
        open_h_angle=0.0,
        open_v_angle=0.0,
        open_offset=0.0,
        oe_offset=0.0,
        shutter=shutter,
        shutter_panel_radius=shutter_panel_radius,
        shutter_width=shutter_width,
        shutter_thickness=shutter_thickness,
        shutter_rotation=shutter_rotation,
        shutter_interval=shutter_interval,
        frame_material=frame_material,
    )

    frame_obj = pf.nodes.to_mesh_object(res.geometry)
    pf.ops.uv.cube_project(frame_obj, uv_name="UVMap")

    if include_glass_pane:
        pane_obj = pf.nodes.to_mesh_object(
            glass_pane(
                width=dimensions.y,
                height=dimensions.z,
                material=glass_material,
            )
        )
        pf.ops.object.join(frame_obj, pane_obj)

    curtain_offset = frame_thickness * 0.5 + 0.07
    pf.ops.object.set_transform(curtain, location=(0.0, 0.0, curtain_offset))
    pf.ops.object.join(frame_obj, curtain)

    # Smooth the curtains; crease_edge keeps the frame edges sharp.
    pf.ops.modifier.subdivide_surface(frame_obj, levels=2, _skip_apply=True)

    portal_light = pf.ops.primitives.light.area_lamp(
        shape="RECTANGLE",
        size_x=dimensions.y,
        size_y=dimensions.z,
        energy=0.0,
        portal=True,
    )
    portal_light.item().rotation_euler = (np.pi, 0, 0)

    return WindowResult(mesh=frame_obj, light=portal_light)
