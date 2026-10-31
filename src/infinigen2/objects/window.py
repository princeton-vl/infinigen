# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lingjie Mei: original window, curtain, shutter, and panel geometry and nodegroups (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/objects/windows/window.py)
# - Hongyu Wen: original window factory and parameter integration (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/objects/windows/window.py)
# - Alexander Raistrick: refactor for Infinigen2

import math
from typing import NamedTuple, cast

import numpy as np
import procfunc as pf
from mathutils import Euler
from procfunc.nodes import types as t

from infinigen2.shaders.functionality_lists import (
    fabric_general_rand,
    furniture_material_rand,
    glass_material_rand,
)
from infinigen2.util import mesh as mesh_util
from infinigen2.util.curve import curve_to_mesh_with_uv

# parts are built flat in XY; reorients them into the wall frame like door_body
_WALL_REORIENT = (math.pi / 2, 0.0, math.pi / 2)

__all__ = [
    "CurtainResult",
    "WindowProfile",
    "WindowResult",
    "curtain",
    "curtain_rand",
    "double_rounded_profile_rand",
    "rectangular_profile",
    "rounded_profile_rand",
    "top_rounded_profile_rand",
    "window",
    "window_composite_rand",
    "window_curved_rand",
    "window_dimensions_rand",
    "window_from_profile_rand",
    "window_rand",
    "window_rectangular_rand",
    "window_rectangular_composite_rand",
]


class _VerticalProfileExtents(NamedTuple):
    top: pf.ProcNode[float]
    bottom: pf.ProcNode[float]


def _profile_quarter(width, height, profile_height, sign, start_angle, resolution):
    arc = pf.nodes.geo.curve_arc(
        resolution=resolution,
        radius=1.0,
        start_angle=start_angle,
        sweep_angle=math.pi / 2,
    )
    return pf.nodes.geo.transform(
        geometry=arc,
        translation=pf.nodes.math.combine_xyz(y=(height * 0.5 - profile_height) * sign),
        scale=pf.nodes.math.combine_xyz(x=width * 0.5, y=profile_height * sign, z=1.0),
    )


@pf.nodes.node_function
def _window_profile(
    width: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    top_profile_height: t.SocketOrVal[float],
    bottom_profile_height: t.SocketOrVal[float],
    profile_resolution: t.SocketOrVal[int] = 5,
) -> pf.ProcNode[pf.CurveObject]:
    quarter = math.pi / 2
    bottom_right = _profile_quarter(
        width, height, bottom_profile_height, -1.0, 0.0, profile_resolution
    )
    top_right = _profile_quarter(
        width, height, top_profile_height, 1.0, 0.0, profile_resolution
    )
    top_left = _profile_quarter(
        width, height, top_profile_height, 1.0, quarter, profile_resolution
    )
    bottom_left = _profile_quarter(
        width, height, bottom_profile_height, -1.0, quarter, profile_resolution
    )
    parts = [
        pf.nodes.geo.reverse_curve(bottom_right),
        top_right,
        top_left,
        pf.nodes.geo.reverse_curve(bottom_left),
    ]
    points = [pf.nodes.geo.curve_to_points_evaluated(curve=c).points for c in parts]
    unique = pf.nodes.geo.merge_by_distance(
        geometry=pf.nodes.geo.join_geometry(points), distance=1e-5
    )
    outline = pf.nodes.geo.points_to_curves(points=unique)
    return pf.nodes.geo.set_spline_cyclic(curve=outline, cyclic=True)


@pf.nodes.node_function
def _profile_vertical_extents(
    width: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    top_profile_height: t.SocketOrVal[float],
    bottom_profile_height: t.SocketOrVal[float],
    x: t.SocketOrVal[float],
) -> _VerticalProfileExtents:
    across = x * 2.0 / width
    inset = 1.0 - pf.nodes.math.sqrt(
        pf.nodes.math.clamp(1.0 - across * across, 0.0, 1.0)
    )
    return _VerticalProfileExtents(
        top=height * 0.5 - top_profile_height * inset,
        bottom=height * -0.5 + bottom_profile_height * inset,
    )


@pf.nodes.node_function
def _profile_half_width(
    width: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    top_profile_height: t.SocketOrVal[float],
    bottom_profile_height: t.SocketOrVal[float],
    y: t.SocketOrVal[float],
) -> pf.ProcNode[float]:
    over_top = pf.nodes.math.maximum(
        y - (height * 0.5 - top_profile_height), 0.0
    ) / pf.nodes.math.maximum(top_profile_height, 1e-9)
    over_bottom = pf.nodes.math.maximum(
        (height * -0.5 + bottom_profile_height) - y, 0.0
    ) / pf.nodes.math.maximum(bottom_profile_height, 1e-9)
    over = pf.nodes.math.maximum(over_top, over_bottom)
    return (
        width
        * 0.5
        * pf.nodes.math.sqrt(pf.nodes.math.clamp(1.0 - over * over, 0.0, 1.0))
    )


@pf.nodes.node_function
def _profile_beams(
    instances: pf.ProcNode,
    scale: t.SocketOrVal[pf.Vector],
    translation: t.SocketOrVal[pf.Vector],
) -> pf.ProcNode:
    scaled = pf.nodes.geo.scale_instances(instances=instances, scale=scale)
    return pf.nodes.geo.translate_instances(
        instances=scaled, translation=translation, local_space=False
    )


def _unit_instance():
    return pf.nodes.geo.geometry_to_instance(mesh_util.box(size=(1.0, 1.0, 1.0)))


def _duplicate_index(duplicates):
    return duplicates.duplicate_index.astype(dtype=float) + 1.0


def _vertical_beams(
    duplicates,
    x,
    width,
    height,
    top_profile_height,
    bottom_profile_height,
    frame_width,
    frame_thickness,
):
    extents = _profile_vertical_extents(
        width=width,
        height=height,
        top_profile_height=top_profile_height,
        bottom_profile_height=bottom_profile_height,
        x=x,
    )
    return _profile_beams(
        instances=duplicates,
        scale=pf.nodes.math.combine_xyz(
            x=frame_width, y=extents.top - extents.bottom, z=frame_thickness
        ),
        translation=pf.nodes.math.combine_xyz(
            x=x, y=(extents.top + extents.bottom) * 0.5
        ),
    )


def _horizontal_beams(
    duplicates,
    y,
    width,
    height,
    top_profile_height,
    bottom_profile_height,
    frame_width,
    frame_thickness,
):
    half_width = _profile_half_width(
        width=width,
        height=height,
        top_profile_height=top_profile_height,
        bottom_profile_height=bottom_profile_height,
        y=y,
    )
    return _profile_beams(
        instances=duplicates,
        scale=pf.nodes.math.combine_xyz(
            x=half_width * 2.0, y=frame_width, z=frame_thickness
        ),
        translation=pf.nodes.math.combine_xyz(y=y),
    )


@pf.nodes.node_function
def _profile_frame(
    outline: pf.ProcNode,
    frame_width: t.SocketOrVal[float],
    frame_thickness: t.SocketOrVal[float],
) -> pf.ProcNode[pf.MeshObject]:
    position = pf.nodes.geo.input_position()
    tangent = pf.nodes.geo.input_tangent()
    previous = pf.nodes.geo.offset_point_in_curve(
        point_index=pf.nodes.geo.input_index(), offset=-1
    )
    previous_position = pf.nodes.geo.sample_index(
        geometry=outline,
        index=previous.point_index,
        value=position,
        data_type=pf.nodes.NodeDataType.FLOAT_VECTOR,
    )
    incoming = pf.nodes.math.vector_normalize(position - previous_position)
    # miter radius keeps width exact at corners; it also scales depth, so reset z
    miter = 1.0 / pf.nodes.math.vector_dot_product(incoming, tangent)
    mitered = pf.nodes.geo.set_curve_radius(curve=outline, radius=miter)
    profile = pf.nodes.geo.curve_quadrilateral(
        width=frame_width,
        height=frame_thickness,
    )
    mesh = pf.nodes.geo.curve_to_mesh(curve=mitered, profile_curve=profile)
    flat_depth = pf.nodes.math.combine_xyz(
        x=position.x,
        y=position.y,
        z=pf.nodes.math.sign(position.z) * frame_thickness * 0.5,
    )
    return pf.nodes.geo.set_position(geometry=mesh, position=flat_depth)


@pf.nodes.node_function
def _profile_grid(
    width: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    top_profile_height: t.SocketOrVal[float],
    bottom_profile_height: t.SocketOrVal[float],
    frame_width: t.SocketOrVal[float],
    frame_thickness: t.SocketOrVal[float],
    panel_h_amount: t.SocketOrVal[int],
    panel_v_amount: t.SocketOrVal[int],
) -> pf.ProcNode[pf.MeshObject]:
    unit = _unit_instance()

    vertical = pf.nodes.geo.duplicate_elements(
        geometry=unit,
        amount=(panel_v_amount.astype(dtype=float) - 1.0).astype(dtype=int),
        domain="INSTANCE",
    )
    x = width * -0.5 + _duplicate_index(vertical) * width / panel_v_amount.astype(
        dtype=float
    )
    vertical_instances = _vertical_beams(
        vertical.geometry,
        x,
        width,
        height,
        top_profile_height,
        bottom_profile_height,
        frame_width,
        frame_thickness,
    )

    horizontal = pf.nodes.geo.duplicate_elements(
        geometry=unit,
        amount=(panel_h_amount.astype(dtype=float) - 1.0).astype(dtype=int),
        domain="INSTANCE",
    )
    y = height * -0.5 + _duplicate_index(horizontal) * height / panel_h_amount.astype(
        dtype=float
    )
    horizontal_instances = _horizontal_beams(
        horizontal.geometry,
        y,
        width,
        height,
        top_profile_height,
        bottom_profile_height,
        frame_width,
        frame_thickness,
    )
    return pf.nodes.geo.realize_instances(
        pf.nodes.geo.join_geometry([vertical_instances, horizontal_instances])
    )


@pf.nodes.node_function
def _profile_sashes(
    outline: pf.ProcNode,
    outer_width: t.SocketOrVal[float],
    outer_height: t.SocketOrVal[float],
    inner_width: t.SocketOrVal[float],
    inner_height: t.SocketOrVal[float],
    top_profile_height: t.SocketOrVal[float],
    bottom_profile_height: t.SocketOrVal[float],
    inset: t.SocketOrVal[float],
    frame_width: t.SocketOrVal[float],
    frame_thickness: t.SocketOrVal[float],
    panel_h_amount: t.SocketOrVal[int],
    panel_v_amount: t.SocketOrVal[int],
    sub_panel_h_amount: t.SocketOrVal[int],
    sub_panel_v_amount: t.SocketOrVal[int],
) -> pf.ProcNode[pf.MeshObject]:
    unit = _unit_instance()

    vertical_per_cell = sub_panel_v_amount.astype(dtype=float) + 1.0
    vertical = pf.nodes.geo.duplicate_elements(
        geometry=unit,
        amount=(panel_v_amount.astype(dtype=float) * vertical_per_cell - 2.0).astype(
            dtype=int
        ),
        domain="INSTANCE",
    )
    vertical_index = _duplicate_index(vertical)
    column = pf.nodes.math.floor(vertical_index / vertical_per_cell)
    column_index = vertical_index % vertical_per_cell
    cell_width = outer_width / panel_v_amount.astype(dtype=float)
    sash_width = cell_width - inset * 2.0
    x = (
        outer_width * -0.5
        + column * cell_width
        + inset
        + column_index * sash_width / sub_panel_v_amount.astype(dtype=float)
    )
    vertical_instances = _vertical_beams(
        vertical.geometry,
        x,
        inner_width,
        inner_height,
        top_profile_height,
        bottom_profile_height,
        frame_width,
        frame_thickness,
    )

    horizontal_per_cell = sub_panel_h_amount.astype(dtype=float) + 1.0
    horizontal = pf.nodes.geo.duplicate_elements(
        geometry=unit,
        amount=(panel_h_amount.astype(dtype=float) * horizontal_per_cell - 2.0).astype(
            dtype=int
        ),
        domain="INSTANCE",
    )
    horizontal_index = _duplicate_index(horizontal)
    row = pf.nodes.math.floor(horizontal_index / horizontal_per_cell)
    row_index = horizontal_index % horizontal_per_cell
    cell_height = outer_height / panel_h_amount.astype(dtype=float)
    sash_height = cell_height - inset * 2.0
    y = (
        outer_height * -0.5
        + row * cell_height
        + inset
        + row_index * sash_height / sub_panel_h_amount.astype(dtype=float)
    )
    horizontal_instances = _horizontal_beams(
        horizontal.geometry,
        y,
        inner_width,
        inner_height,
        top_profile_height,
        bottom_profile_height,
        frame_width,
        frame_thickness,
    )

    frame = _profile_frame(
        outline=outline,
        frame_width=frame_width,
        frame_thickness=frame_thickness,
    )
    return pf.nodes.geo.realize_instances(
        pf.nodes.geo.join_geometry([frame, vertical_instances, horizontal_instances])
    )


@pf.nodes.node_function
def _pleat_curve(
    start: t.SocketOrVal[float],
    end: t.SocketOrVal[float],
    width: t.SocketOrVal[float],
    interval_number: t.SocketOrVal[float],
) -> pf.ProcNode[pf.CurveObject]:
    # segments per sine period, so pleats resolve the same at any panel width
    curve = pf.nodes.geo.curve_line(
        start=pf.nodes.math.combine_xyz(start), end=pf.nodes.math.combine_xyz(end)
    )
    span = pf.nodes.math.absolute(end - start)
    periods = interval_number * span / width
    count = pf.nodes.math.ceil(periods * 6.0)
    return pf.nodes.geo.resample_curve_count(
        curve=curve, count=pf.nodes.math.maximum(count, 8.0).astype(dtype=int)
    )


@pf.nodes.node_function
def _curtain_geometry(
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
    join: pf.ProcNode[pf.CurveObject] = pf.nodes.geo.join_geometry(
        [
            _pleat_curve(l2, r2, width, interval_number),
            _pleat_curve(l1, r1, width, interval_number),
        ]
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
    set_y = height * 0.47

    finial_radius = radius * 2.0
    finial = pf.nodes.geo.mesh_uv_sphere(segments=32, rings=16, radius=finial_radius)
    finial_mesh = pf.nodes.geo.store_named_attribute(
        geometry=finial.mesh,
        name="UVMap",
        value=finial.uv_map,
        domain="CORNER",
        data_type="FLOAT2",
    )

    sample_curve = pf.nodes.geo.sample_curve(curves=curve_line_2, value=0.0, factor=0.0)

    set_position_2 = pf.nodes.geo.set_position(
        geometry=finial_mesh, offset=sample_curve.position
    )

    curve_line_3_end = pf.nodes.math.combine_xyz(x=curve_x_0, z=frame_depth)
    curve_line_3 = pf.nodes.geo.curve_line(
        start=curve_line_2_start, end=curve_line_3_end
    )
    curve_line_4_end = pf.nodes.math.combine_xyz(x=curve_x_1, z=frame_depth)
    curve_line_4 = pf.nodes.geo.curve_line(start=curve_line_2_end, end=curve_line_4_end)

    join_1 = pf.nodes.geo.join_geometry([curve_line_3, curve_line_4, curve_line_2])

    curve_circle_1 = pf.nodes.geo.curve_circle(resolution=12, radius=radius)
    curve_to_2 = curve_to_mesh_with_uv(
        curve=join_1, profile=curve_circle_1, fill_caps=True
    ).mesh

    sample_curve_1 = pf.nodes.geo.sample_curve(
        curves=curve_line_2, value=0.0, factor=1.0
    )

    set_position_3 = pf.nodes.geo.set_position(
        geometry=finial_mesh, offset=sample_curve_1.position
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

    join_3 = pf.nodes.geo.join_geometry([set_material, set_material_1])

    # rail circles tessellate at ~11 deg, so 60 catches only the hem's 90 deg corners
    creased = mesh_util.crease_sharp(join_3, threshold_degrees=60.0)

    set_shade_smooth = pf.nodes.geo.set_shade_smooth(
        geometry=creased, shade_smooth=False
    )

    return pf.nodes.geo.transform(geometry=set_shade_smooth, rotation=_WALL_REORIENT)


def window_dimensions_rand(
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


@pf.tracer.generator
def curtain(
    width: float = 2.5,
    depth: float = 0.04875,
    height: float = 2.5,
    interval_number: float = 22,
    radius: float = 0.015,
    l1: float = -1.25,
    r1: float = -0.4375,
    l2: float = 0.4375,
    r2: float = 1.25,
    frame_depth: float = -0.075,
    curtain_frame_material: pf.Material | None = None,
    curtain_material: pf.Material | None = None,
) -> CurtainResult:
    if curtain_frame_material is None:
        curtain_frame_material = pf.Material(surface=pf.nodes.shader.principled_bsdf())
    if curtain_material is None:
        curtain_material = pf.Material(surface=pf.nodes.shader.principled_bsdf())

    curtain_geo = _curtain_geometry(
        width=width,
        depth=depth,
        height=height,
        interval_number=interval_number,
        radius=radius,
        l1=l1,
        r1=r1,
        l2=l2,
        r2=r2,
        frame_depth=frame_depth,
        curtain_frame_material=curtain_frame_material,
        curtain_material=curtain_material,
    )
    return CurtainResult(mesh=pf.nodes.to_mesh_object(curtain_geo))


def curtain_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    material: pf.Material | None = None,
    rail_material: pf.Material | None = None,
) -> CurtainResult:
    if dimensions is None:
        dimensions = window_dimensions_rand(rng)
    vec = pf.nodes.shader.coord().uv
    rng, rng_fabric = rng.spawn(2)
    if material is None:
        translucency = pf.random.clip_gaussian(rng, 0.6, 0.2, 0.05, 0.8)
        material = fabric_general_rand(rng_fabric, vec, translucency=translucency)

    if rail_material is None:
        rail_material = furniture_material_rand(rng, vec)

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
    return curtain(
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


@pf.nodes.node_function
def _window_shutter(
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
    cube = mesh_util.box(size=cube_size)

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
        geometry=cube, offset=sample_curve.position
    )

    cube_1_size = pf.nodes.math.combine_xyz(
        x=width - frame_width, y=shutter_width, z=shutter_thickness
    )
    cube_1 = mesh_util.box(size=cube_1_size)

    to_instance_1 = pf.nodes.geo.geometry_to_instance(cube_1)

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


class _WindowGeometryResult(NamedTuple):
    geometry: pf.ProcNode[pf.MeshObject]
    bounding_box: pf.ProcNode[pf.MeshObject]


class WindowProfile(NamedTuple):
    dimensions: pf.Vector
    top_profile_height: float
    bottom_profile_height: float


class WindowResult(NamedTuple):
    mesh: pf.MeshObject
    light: pf.LightObject | None
    profile: WindowProfile


def rectangular_profile(dimensions: pf.Vector) -> WindowProfile:
    return WindowProfile(dimensions.copy(), 0.0, 0.0)


def top_rounded_profile_rand(rng: pf.RNG, dimensions: pf.Vector) -> WindowProfile:
    height = dimensions.z * pf.random.uniform(rng, 0.2, 0.4)
    return WindowProfile(dimensions.copy(), height, 0.0)


def double_rounded_profile_rand(rng: pf.RNG, dimensions: pf.Vector) -> WindowProfile:
    height = dimensions.z * pf.random.uniform(rng, 0.35, 0.45)
    return WindowProfile(dimensions.copy(), height, height)


def rounded_profile_rand(rng: pf.RNG, dimensions: pf.Vector) -> WindowProfile:
    rng_choice, rng_build = rng.spawn(2)
    profile_fn = pf.control.choice(
        rng_choice,
        [
            (top_rounded_profile_rand, 0.8),
            (double_rounded_profile_rand, 0.2),
        ],
    )
    return profile_fn(rng_build, dimensions)


@pf.nodes.node_function
def _window_geometry(
    width: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    top_profile_height: t.SocketOrVal[float],
    bottom_profile_height: t.SocketOrVal[float],
    frame_width: t.SocketOrVal[float],
    frame_thickness: t.SocketOrVal[float],
    panel_h_amount: t.SocketOrVal[int],
    panel_v_amount: t.SocketOrVal[int],
    sub_frame_width: t.SocketOrVal[float],
    sub_frame_thickness: t.SocketOrVal[float],
    sub_panel_h_amount: t.SocketOrVal[int],
    sub_panel_v_amount: t.SocketOrVal[int],
    shutter: t.SocketOrVal[bool],
    shutter_panel_radius: t.SocketOrVal[float],
    shutter_width: t.SocketOrVal[float],
    shutter_thickness: t.SocketOrVal[float],
    shutter_rotation: t.SocketOrVal[float],
    shutter_interval: t.SocketOrVal[float],
    frame_material: t.SocketOrVal[pf.Material],
) -> _WindowGeometryResult:
    width = width - frame_width
    height = height - frame_width
    top_profile_height = pf.nodes.math.maximum(
        top_profile_height - frame_width * 0.5, 0.0
    )
    bottom_profile_height = pf.nodes.math.maximum(
        bottom_profile_height - frame_width * 0.5, 0.0
    )
    outline = _window_profile(
        width=width,
        height=height,
        top_profile_height=top_profile_height,
        bottom_profile_height=bottom_profile_height,
    )
    outer_frame = _profile_frame(
        outline=outline,
        frame_width=frame_width,
        frame_thickness=frame_thickness,
    )
    primary_grid = _profile_grid(
        width=width,
        height=height,
        top_profile_height=top_profile_height,
        bottom_profile_height=bottom_profile_height,
        frame_width=frame_width,
        frame_thickness=frame_thickness,
        panel_h_amount=panel_h_amount,
        panel_v_amount=panel_v_amount,
    )

    sash_inset = (frame_width + sub_frame_width) * 0.5
    inner_width = width - sash_inset * 2.0
    inner_height = height - sash_inset * 2.0
    inner_outline = _window_profile(
        width=inner_width,
        height=inner_height,
        top_profile_height=top_profile_height,
        bottom_profile_height=bottom_profile_height,
    )
    sashes = _profile_sashes(
        outline=inner_outline,
        outer_width=width,
        outer_height=height,
        inner_width=inner_width,
        inner_height=inner_height,
        top_profile_height=top_profile_height,
        bottom_profile_height=bottom_profile_height,
        inset=sash_inset,
        frame_width=sub_frame_width,
        frame_thickness=sub_frame_thickness,
        panel_h_amount=panel_h_amount,
        panel_v_amount=panel_v_amount,
        sub_panel_h_amount=sub_panel_h_amount,
        sub_panel_v_amount=sub_panel_v_amount,
    )

    panel_v_float = panel_v_amount.astype(dtype=float)
    panel_h_float = panel_h_amount.astype(dtype=float)
    panel_width = (width - frame_width * panel_v_float) / panel_v_float
    panel_height = (height - frame_width * panel_h_float) / panel_h_float
    shutter_panel = _window_shutter(
        width=panel_width - sub_frame_width,
        height=panel_height - sub_frame_width,
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
    shutter_instance = pf.nodes.geo.geometry_to_instance(shutter_panel)
    shutter_panels = pf.nodes.geo.duplicate_elements(
        domain="INSTANCE",
        geometry=shutter_instance,
        amount=(panel_h_float * panel_v_float).astype(dtype=int),
    )
    shutter_index = shutter_panels.duplicate_index.astype(dtype=float)
    column = pf.nodes.math.floor(shutter_index / panel_h_float)
    row = shutter_index % panel_h_float
    shutter_x = width * -0.5 + width / panel_v_float * (column + 0.5)
    shutter_y = height * -0.5 + height / panel_h_float * (row + 0.5)
    shutter_panels = pf.nodes.geo.translate_instances(
        instances=shutter_panels.geometry,
        translation=pf.nodes.math.combine_xyz(x=shutter_x, y=shutter_y),
        local_space=False,
    )

    details = pf.nodes.func.switch(
        switch=shutter,
        a=sashes,
        b=shutter_panels,
    )
    joined = pf.nodes.geo.join_geometry([outer_frame, primary_grid, details])
    realized = cast(
        pf.ProcNode[pf.MeshObject],
        pf.nodes.geo.realize_instances(joined),
    )
    realized = pf.nodes.geo.set_material(
        geometry=realized,
        selection=True,
        material=frame_material,
    )

    creased = pf.nodes.geo.store_named_attribute(
        domain="EDGE",
        geometry=realized,
        name="crease_edge",
        value=1.0,
    )

    # built in the XY plane (X=width, Y=height, Z=depth); reorient into the
    # shared wall frame (X=depth out of wall, Y=width, Z=height up)
    reoriented = pf.nodes.geo.transform(geometry=creased, rotation=_WALL_REORIENT)

    bound_box = pf.nodes.geo.bound_box(cast(t.ProcNode[t.Geometry], reoriented))

    return _WindowGeometryResult(
        geometry=reoriented,
        bounding_box=bound_box.bounding_box,
    )


@pf.nodes.node_function
def _glass_pane(
    width: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    top_profile_height: t.SocketOrVal[float],
    bottom_profile_height: t.SocketOrVal[float],
    material: t.SocketOrVal[pf.Material],
) -> pf.ProcNode[pf.MeshObject]:
    outline = _window_profile(
        width=width,
        height=height,
        top_profile_height=top_profile_height,
        bottom_profile_height=bottom_profile_height,
    )
    mesh = pf.nodes.geo.fill_curve(curve=outline, mode="NGONS")
    position = pf.nodes.geo.input_position()
    uv = pf.nodes.math.combine_xyz(
        x=position.x + width * 0.5,
        y=position.y + height * 0.5,
    )
    mesh = pf.nodes.geo.store_named_attribute(
        geometry=mesh,
        name="UVMap",
        value=uv,
        domain="CORNER",
        data_type=pf.nodes.NodeDataType.FLOAT_VECTOR_2D,
    )
    # uncreased, the frame's boundary_smooth=ALL subsurf rounds the pane inwards
    creased = pf.nodes.geo.store_named_attribute(
        domain="EDGE",
        geometry=mesh,
        name="crease_edge",
        value=1.0,
    )
    glass = pf.nodes.geo.set_material(
        geometry=creased, selection=True, material=material
    )
    return pf.nodes.geo.transform(geometry=glass, rotation=_WALL_REORIENT)


@pf.tracer.generator
def window(
    dimensions: pf.Vector | None = None,
    frame_width: float = 0.035,
    panel_h_amount: int = 2,
    panel_v_amount: int = 2,
    sub_frame_width: float = 0.0225,
    sub_frame_thickness: float = 0.0525,
    sub_panel_h_amount: int = 1,
    sub_panel_v_amount: int = 1,
    shutter: bool = False,
    shutter_panel_radius: float = 0.002,
    shutter_width: float = 0.04,
    shutter_thickness: float = 0.005,
    shutter_rotation: float = 0.707,
    shutter_interval: float = 0.0424,
    frame_material: pf.Material | None = None,
    glass_material: pf.Material | None = None,
    include_glass_pane: bool = True,
    include_portal: bool = True,
    top_profile_height: float = 0.0,
    bottom_profile_height: float = 0.0,
) -> WindowResult:
    if dimensions is None:
        dimensions = pf.Vector((0.085, 2.5, 2.5))
    profile = WindowProfile(dimensions, top_profile_height, bottom_profile_height)
    if frame_material is None:
        frame_material = pf.Material(surface=pf.nodes.shader.principled_bsdf())
    if glass_material is None:
        glass_material = pf.Material(surface=pf.nodes.shader.principled_bsdf())

    res = _window_geometry(
        width=dimensions.y,
        height=dimensions.z,
        top_profile_height=top_profile_height,
        bottom_profile_height=bottom_profile_height,
        frame_width=frame_width,
        frame_thickness=dimensions.x,
        panel_h_amount=panel_h_amount,
        panel_v_amount=panel_v_amount,
        sub_frame_width=sub_frame_width,
        sub_frame_thickness=sub_frame_thickness,
        sub_panel_h_amount=sub_panel_h_amount,
        sub_panel_v_amount=sub_panel_v_amount,
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
        pane = _glass_pane(
            width=dimensions.y,
            height=dimensions.z,
            top_profile_height=top_profile_height,
            bottom_profile_height=bottom_profile_height,
            material=glass_material,
        )
        pf.ops.object.join(frame_obj, pf.nodes.to_mesh_object(pane))

    origin_offset = dimensions * 0.5
    pf.ops.object.set_transform(frame_obj, location=origin_offset)
    pf.ops.mesh.transform_apply(frame_obj)

    portal_light = None
    if include_portal:
        portal_light = pf.ops.primitives.light.area_lamp(
            shape="RECTANGLE",
            size_x=dimensions.y,
            size_y=dimensions.z,
            energy=0.0,
            portal=True,
        )
        reorient = Euler(_WALL_REORIENT).to_matrix()
        flip = Euler((np.pi, 0.0, 0.0)).to_matrix()
        pf.ops.object.set_transform(
            portal_light,
            location=origin_offset,
            rotation_euler=tuple((reorient @ flip).to_euler()),
        )
    return WindowResult(mesh=frame_obj, light=portal_light, profile=profile)


def _window_build_rand(
    rng_param: pf.RNG,
    rng_frame: pf.RNG,
    rng_glass: pf.RNG,
    profile: WindowProfile,
    shutter: bool,
    frame_material: pf.Material | None,
    glass_material: pf.Material | None,
    include_glass_pane: bool,
    include_portal: bool,
) -> WindowResult:
    dimensions = profile.dimensions
    detail_scale = min(dimensions.y, dimensions.z, 1.0)

    frame_thickness = dimensions.x
    frame_width = detail_scale * pf.random.uniform(rng_param, 0.02, 0.05)

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

    glass_thickness = detail_scale * pf.random.uniform(rng_param, 0.01, 0.03)
    sub_frame_width = detail_scale * pf.random.uniform(rng_param, 0.015, 0.03)
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

    shutter_panel_radius = detail_scale * pf.random.uniform(rng_param, 0.001, 0.003)
    shutter_width = detail_scale * pf.random.uniform(rng_param, 0.03, 0.05)
    shutter_thickness = detail_scale * pf.random.uniform(rng_param, 0.003, 0.007)
    shutter_rotation = pf.random.uniform(rng_param, 0.0, 1.0) ** 0.5
    shutter_interval = shutter_width * (1 + pf.random.uniform(rng_param, 0.02, 0.1))

    vec = pf.nodes.shader.coord().uv
    if frame_material is None:
        frame_material = furniture_material_rand(rng_frame, vec)
    if glass_material is None:
        glass_material = glass_material_rand(rng_glass, vec, glass_height=dimensions.z)

    return window(
        dimensions=dimensions,
        frame_width=frame_width,
        panel_h_amount=panel_h_amount,
        panel_v_amount=panel_v_amount,
        sub_frame_width=sub_frame_width,
        sub_frame_thickness=sub_frame_thickness,
        sub_panel_h_amount=sub_frame_h_amount,
        sub_panel_v_amount=sub_frame_v_amount,
        shutter=shutter,
        shutter_panel_radius=shutter_panel_radius,
        shutter_width=shutter_width,
        shutter_thickness=shutter_thickness,
        shutter_rotation=shutter_rotation,
        shutter_interval=shutter_interval,
        frame_material=frame_material,
        glass_material=glass_material,
        include_glass_pane=include_glass_pane,
        include_portal=include_portal,
        top_profile_height=profile.top_profile_height,
        bottom_profile_height=profile.bottom_profile_height,
    )


def window_from_profile_rand(
    rng: pf.RNG,
    profile: WindowProfile,
    frame_material: pf.Material | None = None,
    glass_material: pf.Material | None = None,
    include_glass_pane: bool = True,
    include_portal: bool = True,
) -> WindowResult:
    rng_param, _rng_dim, rng_frame, rng_glass = rng.spawn(4)
    return _window_build_rand(
        rng_param,
        rng_frame,
        rng_glass,
        profile=profile,
        shutter=False,
        frame_material=frame_material,
        glass_material=glass_material,
        include_glass_pane=include_glass_pane,
        include_portal=include_portal,
    )


def window_rectangular_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    frame_material: pf.Material | None = None,
    glass_material: pf.Material | None = None,
    include_glass_pane: bool = True,
    include_portal: bool = True,
) -> WindowResult:
    rng_param, rng_dim, rng_frame, rng_glass, rng_shutter = rng.spawn(5)
    if dimensions is None:
        dimensions = window_dimensions_rand(rng_dim)
    shutter_fn = pf.control.choice(
        rng_shutter,
        [
            (lambda: True, 0.2),
            (lambda: False, 0.8),
        ],
    )
    profile = rectangular_profile(dimensions)
    shutter = shutter_fn()
    return _window_build_rand(
        rng_param,
        rng_frame,
        rng_glass,
        profile=profile,
        shutter=shutter,
        frame_material=frame_material,
        glass_material=glass_material,
        include_glass_pane=include_glass_pane,
        include_portal=include_portal,
    )


def window_curved_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    frame_material: pf.Material | None = None,
    glass_material: pf.Material | None = None,
    include_glass_pane: bool = True,
    include_portal: bool = True,
) -> WindowResult:
    rng_param, rng_dim, rng_frame, rng_glass, rng_profile = rng.spawn(5)
    if dimensions is None:
        dimensions = window_dimensions_rand(rng_dim)
    profile = rounded_profile_rand(rng_profile, dimensions)
    return _window_build_rand(
        rng_param,
        rng_frame,
        rng_glass,
        profile=profile,
        shutter=False,
        frame_material=frame_material,
        glass_material=glass_material,
        include_glass_pane=include_glass_pane,
        include_portal=include_portal,
    )


def window_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    frame_material: pf.Material | None = None,
    glass_material: pf.Material | None = None,
    include_glass_pane: bool = True,
    include_portal: bool = True,
) -> WindowResult:
    rng_choice, rng_build = rng.spawn(2)
    window_fn = pf.control.choice(
        rng_choice,
        [
            (window_rectangular_rand, 0.92),
            (window_curved_rand, 0.08),
        ],
    )
    return window_fn(
        rng_build,
        dimensions=dimensions,
        frame_material=frame_material,
        glass_material=glass_material,
        include_glass_pane=include_glass_pane,
        include_portal=include_portal,
    )


def _hang_curtain(
    window_mesh: pf.MeshObject,
    curtain: pf.MeshObject,
    dimensions: pf.Vector,
) -> None:
    location = (dimensions.x + 0.07, dimensions.y * 0.5, dimensions.z * 0.5)
    pf.ops.object.set_transform(curtain, location=location)
    pf.ops.object.join(window_mesh, curtain)


def window_rectangular_composite_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    frame_material: pf.Material | None = None,
    glass_material: pf.Material | None = None,
    curtain: pf.MeshObject | None = None,
    include_glass_pane: bool = True,
    include_portal: bool = True,
) -> WindowResult:
    rng_streams = rng.spawn(5)
    if dimensions is None:
        dimensions = window_dimensions_rand(rng_streams[1])

    result = window_rectangular_rand(
        rng,
        dimensions=dimensions,
        frame_material=frame_material,
        glass_material=glass_material,
        include_glass_pane=include_glass_pane,
        include_portal=include_portal,
    )
    if curtain is not None:
        _hang_curtain(result.mesh, curtain, dimensions)
        return result

    rng_curtain_choice, rng_curtain_build = rng_streams[4].spawn(2)

    def hang_new_curtain() -> None:
        curtain = curtain_rand(rng_curtain_build, dimensions=dimensions)
        _hang_curtain(result.mesh, curtain.mesh, dimensions)

    curtain_fn = pf.control.choice(
        rng_curtain_choice,
        [
            (hang_new_curtain, 1.0),
            (lambda: None, 2.0),
        ],
    )
    curtain_fn()
    return result


def window_composite_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    frame_material: pf.Material | None = None,
    glass_material: pf.Material | None = None,
    include_glass_pane: bool = True,
    include_portal: bool = True,
) -> WindowResult:
    rng_choice, rng_build = rng.spawn(2)
    window_fn = pf.control.choice(
        rng_choice,
        [
            (window_rectangular_composite_rand, 0.92),
            (window_curved_rand, 0.08),
        ],
    )
    return window_fn(
        rng_build,
        dimensions=dimensions,
        frame_material=frame_material,
        glass_material=glass_material,
        include_glass_pane=include_glass_pane,
        include_portal=include_portal,
    )
