# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lingjie Mei: original Infinigen v1 bathtub, bathroom-sink, and standing-sink implementations (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/objects/bathroom/bathtub.py; https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/objects/bathroom/bathroom_sink.py)
# - Alexander Raistrick: refactor for Infinigen2

import math
from typing import NamedTuple

import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.shaders.base_materials import ceramic, metal_brushed
from infinigen2.util import mesh

__all__ = [
    "BathtubResult",
    "MIN_BACK_MARGIN",
    "SinkBathroomResult",
    "SinkPedestalResult",
    "bathtub",
    "bathtub_rand",
    "bathtub_shell",
    "sink_bathroom",
    "sink_bathroom_rand",
    "sink_pedestal",
]

MIN_BACK_MARGIN = 0.1


class BathtubResult(NamedTuple):
    mesh: pf.MeshObject


class SinkBathroomResult(NamedTuple):
    mesh: pf.MeshObject


class SinkPedestalResult(NamedTuple):
    mesh: pf.MeshObject


@pf.nodes.node_function
def bathtub_shell(
    surface_material: t.SocketOrVal[pf.Material],
    metal_material: t.SocketOrVal[pf.Material],
    width: t.SocketOrVal[float] = 1.75,
    size: t.SocketOrVal[float] = 0.85,
    depth: t.SocketOrVal[float] = 0.6,
    thickness: t.SocketOrVal[float] = 0.12,
    floor_thickness: t.SocketOrVal[float] = 0.12,
    back_margin: t.SocketOrVal[float] = 0.18,
    bowl_inset: t.SocketOrVal[float] = 0.05,
    outer_bottom_inset: t.SocketOrVal[float] = 0.05,
    corner_radius: t.SocketOrVal[float] = 0.2,
    bowl_corner_radius: t.SocketOrVal[float] = 0.17,
    top_bevel_radius: t.SocketOrVal[float] = 0.035,
    bottom_bevel_radius: t.SocketOrVal[float] = 0.035,
    inner_top_bevel_radius: t.SocketOrVal[float] = 0.025,
    inner_bottom_bevel_radius: t.SocketOrVal[float] = 0.1,
    profile_roundness: t.SocketOrVal[float] = 0.5,
    hole_radius: t.SocketOrVal[float] = 0.02,
) -> t.ProcNode[pf.MeshObject]:
    min_width_size = pf.nodes.math.minimum(width, size)
    thickness_limit = pf.nodes.math.minimum(depth * 0.35, min_width_size * 0.2)
    thickness = pf.nodes.math.clamp(thickness, 0.005, thickness_limit)
    floor_thickness = pf.nodes.math.clamp(floor_thickness, thickness, depth * 0.35)
    back_margin = pf.nodes.math.clamp(back_margin, thickness, width - thickness * 2.0)
    inner_width = width - back_margin - thickness
    inner_size = size - thickness * 2.0
    min_inner = pf.nodes.math.minimum(inner_width, inner_size)
    top_bevel_limit = pf.nodes.math.minimum(thickness * 0.95, depth * 0.35)
    top_bevel_radius = pf.nodes.math.clamp(top_bevel_radius, 0.0, top_bevel_limit)
    bottom_bevel_limit = pf.nodes.math.minimum(depth * 0.6, min_width_size * 0.2)
    bottom_bevel_radius = pf.nodes.math.clamp(
        bottom_bevel_radius, 0.0, bottom_bevel_limit
    )
    inner_top_bevel_limit = pf.nodes.math.minimum(depth * 0.35, min_inner * 0.2)
    inner_top_bevel_radius = pf.nodes.math.clamp(
        inner_top_bevel_radius, 0.0, inner_top_bevel_limit
    )
    minimum_floor_width = pf.nodes.math.clamp(
        min_inner - inner_top_bevel_radius * 2.0 - 0.01, 0.05, 0.42
    )
    fit_bowl_inset = (min_inner - minimum_floor_width) * 0.5 - inner_top_bevel_radius
    max_bowl_inset = pf.nodes.math.clamp(fit_bowl_inset, 0.0, min_inner * 0.25)
    bottom_reserve = pf.nodes.math.minimum(
        inner_bottom_bevel_radius, max_bowl_inset * 0.5
    )
    max_bowl_inset -= bottom_reserve
    bowl_inset = pf.nodes.math.clamp(bowl_inset, 0.0, max_bowl_inset)
    max_outer_inset = pf.nodes.math.minimum(min_width_size * 0.25, bowl_inset)
    outer_bottom_inset = pf.nodes.math.clamp(outer_bottom_inset, 0.0, max_outer_inset)
    inner_bottom_limit = pf.nodes.math.minimum(
        depth - floor_thickness - inner_top_bevel_radius - 0.02,
        min_inner * 0.5
        - inner_top_bevel_radius
        - bowl_inset
        - minimum_floor_width * 0.5,
    )
    inner_bottom_bevel_radius = pf.nodes.math.clamp(
        inner_bottom_bevel_radius, 0.0, inner_bottom_limit
    )
    bowl_corner_limit = min_inner * 0.5
    bowl_corner_radius = pf.nodes.math.minimum(bowl_corner_radius, bowl_corner_limit)
    corner_radius = pf.nodes.math.minimum(corner_radius, min_width_size * 0.5)
    corner_radius = pf.nodes.math.clamp(
        corner_radius, bowl_corner_radius, bowl_corner_limit + thickness
    )
    bowl_corner_radius = pf.nodes.math.maximum(
        bowl_corner_radius, corner_radius - thickness
    )
    lower_outer_width = width - outer_bottom_inset * 2.0
    lower_outer_size = size - outer_bottom_inset * 2.0
    wall_top_width = inner_width - inner_top_bevel_radius * 2.0
    wall_top_size = inner_size - inner_top_bevel_radius * 2.0
    wall_bottom_width = wall_top_width - bowl_inset * 2.0
    wall_bottom_size = wall_top_size - bowl_inset * 2.0
    cylinder = pf.nodes.geo.mesh_cylinder(vertices=24, side_segments=21)
    index = pf.nodes.geo.input_index().astype(dtype=float)
    ring = pf.nodes.math.floor(index / 24.0)
    contour = pf.nodes.math.modulo(index, 24.0)
    side = pf.nodes.math.floor(contour / 6.0)
    along = pf.nodes.math.modulo(contour, 6.0) / 6.0
    square_x = pf.nodes.func.switch(side < 3.0, -1.0, 1.0 - along * 2.0)
    square_x = pf.nodes.func.switch(side < 2.0, square_x, 1.0)
    square_x = pf.nodes.func.switch(side < 1.0, square_x, -1.0 + along * 2.0)
    square_y = pf.nodes.func.switch(side < 3.0, 1.0 - along * 2.0, 1.0)
    square_y = pf.nodes.func.switch(side < 2.0, square_y, -1.0 + along * 2.0)
    square_y = pf.nodes.func.switch(side < 1.0, square_y, -1.0)

    bottom_angle = ring * 0.3926990817
    bottom_ring_inset = bottom_bevel_radius * (1.0 - pf.nodes.math.sin(bottom_angle))
    bottom_width = lower_outer_width - bottom_ring_inset * 2.0
    bottom_size = lower_outer_size - bottom_ring_inset * 2.0
    bottom_z = bottom_bevel_radius * (1.0 - pf.nodes.math.cos(bottom_angle))
    bottom_radius = corner_radius - bottom_ring_inset

    top_angle = (ring - 5.0) * 0.3926990817
    top_ring_inset = top_bevel_radius * (1.0 - pf.nodes.math.cos(top_angle))
    top_width = width - top_ring_inset * 2.0
    top_size = size - top_ring_inset * 2.0
    top_z = depth - top_bevel_radius + top_bevel_radius * pf.nodes.math.sin(top_angle)
    top_radius = corner_radius - top_ring_inset

    inner_top_angle = (ring - 10.0) * 0.3926990817
    inner_top_inset = inner_top_bevel_radius * pf.nodes.math.sin(inner_top_angle)
    inner_top_width = inner_width - inner_top_inset * 2.0
    inner_top_size = inner_size - inner_top_inset * 2.0
    inner_top_z = depth - inner_top_bevel_radius * (
        1.0 - pf.nodes.math.cos(inner_top_angle)
    )
    inner_top_radius = bowl_corner_radius - inner_top_inset

    wall_t = (ring - 14.0) / 3.0
    wall_smooth = wall_t * wall_t * (3.0 - wall_t * 2.0)
    wall_profile = wall_t + profile_roundness * (wall_smooth - wall_t)
    wall_width = wall_top_width - bowl_inset * 2.0 * wall_profile
    wall_size = wall_top_size - bowl_inset * 2.0 * wall_profile
    wall_z = depth - inner_top_bevel_radius
    wall_z -= (
        depth - floor_thickness - inner_top_bevel_radius - inner_bottom_bevel_radius
    ) * wall_t
    wall_radius = (
        bowl_corner_radius - inner_top_bevel_radius - bowl_inset * wall_profile
    )

    inner_bottom_angle = (ring - 17.0) * 0.3926990817
    inner_bottom_inset = inner_bottom_bevel_radius * (
        1.0 - pf.nodes.math.cos(inner_bottom_angle)
    )
    bevel_bottom_width = wall_bottom_width - inner_bottom_inset * 2.0
    bevel_bottom_size = wall_bottom_size - inner_bottom_inset * 2.0
    bevel_bottom_z = floor_thickness + inner_bottom_bevel_radius * (
        1.0 - pf.nodes.math.sin(inner_bottom_angle)
    )
    bevel_bottom_radius = (
        bowl_corner_radius - inner_top_bevel_radius - bowl_inset - inner_bottom_inset
    )

    profile_width = wall_bottom_width - inner_bottom_bevel_radius * 2.0
    profile_width = pf.nodes.func.switch(ring < 21.0, profile_width, bevel_bottom_width)
    profile_width = pf.nodes.func.switch(ring < 18.0, profile_width, wall_width)
    profile_width = pf.nodes.func.switch(ring < 15.0, profile_width, inner_top_width)
    profile_width = pf.nodes.func.switch(ring < 11.0, profile_width, inner_width)
    profile_width = pf.nodes.func.switch(ring < 10.0, profile_width, top_width)
    profile_width = pf.nodes.func.switch(ring < 6.0, profile_width, width)
    profile_width = pf.nodes.func.switch(ring < 5.0, profile_width, bottom_width)

    profile_size = wall_bottom_size - inner_bottom_bevel_radius * 2.0
    profile_size = pf.nodes.func.switch(ring < 21.0, profile_size, bevel_bottom_size)
    profile_size = pf.nodes.func.switch(ring < 18.0, profile_size, wall_size)
    profile_size = pf.nodes.func.switch(ring < 15.0, profile_size, inner_top_size)
    profile_size = pf.nodes.func.switch(ring < 11.0, profile_size, inner_size)
    profile_size = pf.nodes.func.switch(ring < 10.0, profile_size, top_size)
    profile_size = pf.nodes.func.switch(ring < 6.0, profile_size, size)
    profile_size = pf.nodes.func.switch(ring < 5.0, profile_size, bottom_size)

    profile_z = pf.nodes.func.switch(ring < 21.0, floor_thickness, bevel_bottom_z)
    profile_z = pf.nodes.func.switch(ring < 18.0, profile_z, wall_z)
    profile_z = pf.nodes.func.switch(ring < 15.0, profile_z, inner_top_z)
    profile_z = pf.nodes.func.switch(ring < 11.0, profile_z, depth)
    profile_z = pf.nodes.func.switch(ring < 10.0, profile_z, top_z)
    profile_z = pf.nodes.func.switch(ring < 6.0, profile_z, depth - top_bevel_radius)
    profile_z = pf.nodes.func.switch(ring < 5.0, profile_z, bottom_z)

    profile_radius = (
        bowl_corner_radius
        - inner_top_bevel_radius
        - bowl_inset
        - inner_bottom_bevel_radius
    )
    profile_radius = pf.nodes.func.switch(
        ring < 21.0, profile_radius, bevel_bottom_radius
    )
    profile_radius = pf.nodes.func.switch(ring < 18.0, profile_radius, wall_radius)
    profile_radius = pf.nodes.func.switch(ring < 15.0, profile_radius, inner_top_radius)
    profile_radius = pf.nodes.func.switch(
        ring < 11.0, profile_radius, bowl_corner_radius
    )
    profile_radius = pf.nodes.func.switch(ring < 10.0, profile_radius, top_radius)
    profile_radius = pf.nodes.func.switch(ring < 6.0, profile_radius, corner_radius)
    profile_radius = pf.nodes.func.switch(ring < 5.0, profile_radius, bottom_radius)

    half_width = profile_width * 0.5
    half_size = profile_size * 0.5
    radius = pf.nodes.math.clamp(
        profile_radius, 0.0, pf.nodes.math.minimum(half_width, half_size)
    )
    abs_x = pf.nodes.math.absolute(square_x)
    abs_y = pf.nodes.math.absolute(square_y)
    direction_x = abs_x * half_width
    direction_y = abs_y * half_size
    corner_x = half_width - radius
    corner_y = half_size - radius
    vertical_scale = half_width / pf.nodes.math.maximum(direction_x, 1e-8)
    vertical_hit_y = vertical_scale * direction_y
    horizontal_scale = half_size / pf.nodes.math.maximum(direction_y, 1e-8)
    horizontal_hit_x = horizontal_scale * direction_x
    quadratic = direction_x * direction_x + direction_y * direction_y
    linear = direction_x * corner_x + direction_y * corner_y
    constant = corner_x * corner_x + corner_y * corner_y - radius * radius
    discriminant = pf.nodes.math.maximum(0.0, linear * linear - quadratic * constant)
    corner_scale = (linear + pf.nodes.math.sqrt(discriminant)) / pf.nodes.math.maximum(
        quadratic, 1e-8
    )
    boundary_scale = pf.nodes.func.switch(
        horizontal_hit_x <= corner_x, corner_scale, horizontal_scale
    )
    boundary_scale = pf.nodes.func.switch(
        vertical_hit_y <= corner_y, boundary_scale, vertical_scale
    )
    center_x = pf.nodes.func.switch(
        ring < 10.0,
        (-width + back_margin - thickness) * 0.5,
        -width * 0.5,
    )
    position = pf.nodes.math.combine_xyz(
        x=center_x + square_x * half_width * boundary_scale,
        y=size * 0.5 + square_y * half_size * boundary_scale,
        z=profile_z,
    )
    shell = pf.nodes.geo.set_position(cylinder.mesh, position=position)
    # rings are read bottom-up from a top-down cylinder, which inverts the winding
    shell = pf.nodes.geo.flip_faces(shell)
    shell = mesh.metric_box_uv(shell)
    shell = pf.nodes.geo.set_shade_smooth(shell)
    shell = pf.nodes.geo.set_material(shell, surface_material)

    inner_center = pf.nodes.math.combine_xyz(
        x=(-width + back_margin - thickness) * 0.5,
        y=size * 0.5,
    )
    drain = mesh.quad_cylinder(radius=hole_radius, depth=0.01, resolution=16)
    drain = pf.nodes.geo.transform(
        drain,
        translation=inner_center
        + pf.nodes.math.combine_xyz(z=floor_thickness + 0.0035),
    )
    drain = pf.nodes.geo.set_material(drain, metal_material)
    return pf.nodes.geo.join_geometry([shell, drain])


@pf.tracer.generator
def bathtub(
    width: float = 1.75,
    size: float = 0.85,
    depth: float = 0.6,
    thickness: float = 0.04,
    floor_thickness: float | None = None,
    back_margin: float = 0.18,
    bowl_inset: float = 0.05,
    outer_bottom_inset: float = 0.05,
    corner_radius: float = 0.2,
    bowl_corner_radius: float = 0.17,
    top_bevel_radius: float = 0.025,
    bottom_bevel_radius: float = 0.05,
    inner_top_bevel_radius: float = 0.04,
    inner_bottom_bevel_radius: float = 0.1,
    profile_roundness: float = 0.5,
    hole_radius: float = 0.02,
    surface_material: pf.Material | None = None,
    metal_material: pf.Material | None = None,
) -> BathtubResult:
    if floor_thickness is None:
        floor_thickness = thickness
    if surface_material is None:
        surface_material = pf.Material(
            surface=pf.nodes.shader.principled_bsdf(roughness=0.25)
        )
    if metal_material is None:
        metal_material = pf.Material(
            surface=pf.nodes.shader.principled_bsdf(
                base_color=(0.45, 0.45, 0.45, 1.0),
                metallic=1.0,
                roughness=0.25,
            )
        )
    geometry = bathtub_shell(
        surface_material,
        metal_material,
        width,
        size,
        depth,
        thickness,
        floor_thickness,
        back_margin,
        bowl_inset,
        outer_bottom_inset,
        corner_radius,
        bowl_corner_radius,
        top_bevel_radius,
        bottom_bevel_radius,
        inner_top_bevel_radius,
        inner_bottom_bevel_radius,
        profile_roundness,
        hole_radius,
    )
    obj = pf.nodes.to_mesh_object(geometry)
    pf.ops.modifier.subdivide_surface(obj, levels=2, _skip_apply=True)
    return BathtubResult(mesh=obj)


def sink_bathroom(
    width: float = 0.75,
    size: float = 0.5,
    depth: float = 0.2,
    thickness: float = 0.025,
    back_margin: float = 0.1,
    bowl_inset: float = 0.025,
    corner_radius: float = 0.02,
    bowl_corner_radius: float = 0.01,
    outer_bottom_inset: float = 0.0,
    top_bevel_radius: float = 0.01,
    bottom_bevel_radius: float = 0.04,
    inner_top_bevel_radius: float = 0.012,
    profile_roundness: float = 0.6,
    hole_radius: float = 0.0175,
    surface_material: pf.Material | None = None,
    metal_material: pf.Material | None = None,
) -> SinkBathroomResult:
    floor_thickness = max(thickness, depth * 0.12, bottom_bevel_radius * 0.5)
    result = bathtub(
        width=size,
        size=width,
        depth=depth,
        thickness=thickness,
        floor_thickness=floor_thickness,
        back_margin=back_margin,
        bowl_inset=bowl_inset,
        outer_bottom_inset=outer_bottom_inset,
        corner_radius=corner_radius,
        bowl_corner_radius=bowl_corner_radius,
        top_bevel_radius=top_bevel_radius,
        bottom_bevel_radius=bottom_bevel_radius,
        inner_top_bevel_radius=inner_top_bevel_radius,
        inner_bottom_bevel_radius=bottom_bevel_radius,
        profile_roundness=profile_roundness,
        hole_radius=hole_radius,
        surface_material=surface_material,
        metal_material=metal_material,
    )
    return SinkBathroomResult(mesh=result.mesh)


def _surface_material_rand(rng: pf.RNG) -> pf.Material:
    rng_choice, rng_material = rng.spawn(2)
    material_rand = pf.control.choice(
        rng_choice,
        [
            (ceramic.ceramic_rand, 5.0),
            (metal_brushed.metal_brushed_linear_rand, 1.0),
        ],
    )
    return material_rand(rng_material, pf.nodes.shader.coord().generated)


def _metal_material_rand(rng: pf.RNG) -> pf.Material:
    return metal_brushed.metal_brushed_linear_rand(
        rng, pf.nodes.shader.coord().generated
    )


def _bathtub_dimensions_rand(
    rng: pf.RNG,
    width: float | None,
    size: float | None,
    depth: float | None,
    thickness: float | None,
    back_margin: float | None,
) -> tuple[float, float, float, float, float]:
    if thickness is None:
        thickness = pf.random.clip_gaussian(rng, 0.025, 0.03, 0.01, 0.1)
    if depth is None:
        depth = pf.random.uniform(rng, 0.48, 0.72)
    if back_margin is None:
        back_margin = pf.random.uniform(rng, MIN_BACK_MARGIN, 0.3)
    if width is None:
        width = pf.random.uniform(rng, 1.3, 1.7) + back_margin + thickness
    if size is None:
        size = pf.random.uniform(rng, 0.58, 0.72) + thickness * 2.0
    return width, size, depth, thickness, back_margin


def _bathtub_shape_rand(
    rng: pf.RNG,
    size: float,
    thickness: float,
    bowl_inset: float | None,
    outer_bottom_inset: float | None,
    corner_radius: float | None,
    bowl_corner_radius: float | None,
) -> tuple[float, float, float, float]:
    if bowl_inset is None:
        bowl_inset = pf.random.uniform(rng, 0.0, 0.15)
    if outer_bottom_inset is None:
        outer_bottom_inset = bowl_inset * pf.random.uniform(rng, 0.0, 1.0)
    if corner_radius is None:
        corner_unit = pf.random.uniform(rng, 0.0, 1.0)
        corner_radius = size * 0.5 * corner_unit**2
    if bowl_corner_radius is None:
        corner_delta = thickness * pf.random.uniform(rng, 0.0, 1.0)
        bowl_corner_radius = max(0.001, corner_radius - corner_delta)
    return bowl_inset, outer_bottom_inset, corner_radius, bowl_corner_radius


def _bathtub_edges_rand(
    rng: pf.RNG,
    thickness: float,
    top: float | None,
    bottom: float | None,
    inner_top: float | None,
    inner_bottom: float | None,
) -> tuple[float, float, float, float]:
    if top is None:
        top = thickness * pf.random.uniform(rng, 0.0, 0.5)
    if bottom is None:
        bottom = pf.random.uniform(rng, 0.03, 0.1)
    if inner_top is None:
        inner_top = thickness * pf.random.uniform(rng, 0.6, 2.0)
    if inner_bottom is None:
        inner_bottom = pf.random.uniform(rng, 0.05, 0.2)
    return top, bottom, inner_top, inner_bottom


def bathtub_rand(
    rng: pf.RNG,
    width: float | None = None,
    size: float | None = None,
    depth: float | None = None,
    thickness: float | None = None,
    back_margin: float | None = None,
    bowl_inset: float | None = None,
    outer_bottom_inset: float | None = None,
    corner_radius: float | None = None,
    bowl_corner_radius: float | None = None,
    top_bevel_radius: float | None = None,
    bottom_bevel_radius: float | None = None,
    inner_top_bevel_radius: float | None = None,
    inner_bottom_bevel_radius: float | None = None,
    profile_roundness: float | None = None,
    hole_radius: float | None = None,
    surface_material: pf.Material | None = None,
    metal_material: pf.Material | None = None,
) -> BathtubResult:
    (
        rng_dimensions,
        rng_shape,
        rng_edges,
        rng_profile,
        rng_drain,
        rng_surface,
        rng_metal,
    ) = rng.spawn(7)
    width, size, depth, thickness, back_margin = _bathtub_dimensions_rand(
        rng_dimensions, width, size, depth, thickness, back_margin
    )
    bowl_inset, outer_bottom_inset, corner_radius, bowl_corner_radius = (
        _bathtub_shape_rand(
            rng_shape,
            size,
            thickness,
            bowl_inset,
            outer_bottom_inset,
            corner_radius,
            bowl_corner_radius,
        )
    )
    (
        top_bevel_radius,
        bottom_bevel_radius,
        inner_top_bevel_radius,
        inner_bottom_bevel_radius,
    ) = _bathtub_edges_rand(
        rng_edges,
        thickness,
        top_bevel_radius,
        bottom_bevel_radius,
        inner_top_bevel_radius,
        inner_bottom_bevel_radius,
    )
    if profile_roundness is None:
        profile_roundness = pf.random.uniform(rng_profile, 0.0, 1.0)
    if hole_radius is None:
        hole_radius = pf.random.uniform(rng_drain, 0.0175, 0.025)
    if surface_material is None:
        surface_material = ceramic.ceramic_rand(
            rng_surface, pf.nodes.shader.coord().generated
        )
    if metal_material is None:
        metal_material = _metal_material_rand(rng_metal)
    return bathtub(
        width=width,
        size=size,
        depth=depth,
        thickness=thickness,
        back_margin=back_margin,
        bowl_inset=bowl_inset,
        outer_bottom_inset=outer_bottom_inset,
        corner_radius=corner_radius,
        bowl_corner_radius=bowl_corner_radius,
        top_bevel_radius=top_bevel_radius,
        bottom_bevel_radius=bottom_bevel_radius,
        inner_top_bevel_radius=inner_top_bevel_radius,
        inner_bottom_bevel_radius=inner_bottom_bevel_radius,
        profile_roundness=profile_roundness,
        hole_radius=hole_radius,
        surface_material=surface_material,
        metal_material=metal_material,
    )


def _sink_dimensions_rand(
    rng: pf.RNG,
    width: float | None,
    size: float | None,
    depth: float | None,
    thickness: float | None,
    back_margin: float | None,
) -> tuple[float, float, float, float, float]:
    if width is None:
        width = pf.random.uniform(rng, 0.49, 0.75)
    if size is None:
        size = width * pf.random.log_uniform(rng, 0.55, 1.2)
    if depth is None:
        depth = width * pf.random.log_uniform(rng, 0.2, 0.4)
    if thickness is None:
        thickness = pf.random.uniform(rng, 0.012, 0.03)
    if back_margin is None:
        back_margin = size * pf.random.uniform(rng, 0.15, 0.28)
    return width, size, depth, thickness, back_margin


def _sink_shape_rand(
    rng: pf.RNG,
    size: float,
    thickness: float,
    bowl_inset: float | None,
    corner_radius: float | None,
    bowl_corner_radius: float | None,
) -> tuple[float, float, float, float]:
    if bowl_inset is None:
        bowl_inset = size * pf.random.uniform(rng, 0.03, 0.24)
    if bowl_corner_radius is None:
        corner_unit = pf.random.uniform(rng, 0.0, 1.0)
        bowl_corner_radius = size * 0.25 * corner_unit**2
    if corner_radius is None:
        corner_delta = thickness * pf.random.uniform(rng, 0.0, 1.0)
        corner_radius = bowl_corner_radius + corner_delta
    outer_bottom_inset = bowl_inset * pf.random.uniform(rng, 0.0, 1.0)
    return bowl_inset, corner_radius, bowl_corner_radius, outer_bottom_inset


def _sink_edges_rand(
    rng: pf.RNG,
    thickness: float,
    depth: float,
    top: float | None,
    bottom: float | None,
    inner_top: float | None,
) -> tuple[float, float, float]:
    if top is None:
        top = thickness * pf.random.uniform(rng, 0.01, 0.9)
    if bottom is None:
        bottom = depth * pf.random.uniform(rng, 0.005, 0.6)
    if inner_top is None:
        inner_top = thickness * pf.random.uniform(rng, 0.05, 1.0)
    return top, bottom, inner_top


def sink_bathroom_rand(
    rng: pf.RNG,
    width: float | None = None,
    size: float | None = None,
    depth: float | None = None,
    thickness: float | None = None,
    back_margin: float | None = None,
    bowl_inset: float | None = None,
    corner_radius: float | None = None,
    bowl_corner_radius: float | None = None,
    top_bevel_radius: float | None = None,
    bottom_bevel_radius: float | None = None,
    inner_top_bevel_radius: float | None = None,
    profile_roundness: float | None = None,
    hole_radius: float | None = None,
    surface_material: pf.Material | None = None,
    metal_material: pf.Material | None = None,
) -> SinkBathroomResult:
    (
        rng_dimensions,
        rng_shape,
        rng_edges,
        rng_profile,
        rng_drain,
        rng_surface,
        rng_metal,
    ) = rng.spawn(7)
    width, size, depth, thickness, back_margin = _sink_dimensions_rand(
        rng_dimensions, width, size, depth, thickness, back_margin
    )
    bowl_inset, corner_radius, bowl_corner_radius, outer_bottom_inset = (
        _sink_shape_rand(
            rng_shape,
            size,
            thickness,
            bowl_inset,
            corner_radius,
            bowl_corner_radius,
        )
    )
    top_bevel_radius, bottom_bevel_radius, inner_top_bevel_radius = _sink_edges_rand(
        rng_edges,
        thickness,
        depth,
        top_bevel_radius,
        bottom_bevel_radius,
        inner_top_bevel_radius,
    )
    if profile_roundness is None:
        profile_roundness = pf.random.uniform(rng_profile, 0.0, 1.0)
    if hole_radius is None:
        hole_radius = pf.random.uniform(rng_drain, 0.015, 0.02)
    if surface_material is None:
        surface_material = _surface_material_rand(rng_surface)
    if metal_material is None:
        metal_material = _metal_material_rand(rng_metal)
    return sink_bathroom(
        width=width,
        size=size,
        depth=depth,
        thickness=thickness,
        back_margin=back_margin,
        bowl_inset=bowl_inset,
        corner_radius=corner_radius,
        bowl_corner_radius=bowl_corner_radius,
        outer_bottom_inset=outer_bottom_inset,
        top_bevel_radius=top_bevel_radius,
        bottom_bevel_radius=bottom_bevel_radius,
        inner_top_bevel_radius=inner_top_bevel_radius,
        profile_roundness=profile_roundness,
        hole_radius=hole_radius,
        surface_material=surface_material,
        metal_material=metal_material,
    )


@pf.nodes.node_function
def _pedestal_geometry(
    material: t.SocketOrVal[pf.Material],
    height: t.SocketOrVal[float],
    top_radius: t.SocketOrVal[float],
    bottom_radius: t.SocketOrVal[float],
    is_circular: t.SocketOrVal[bool],
) -> t.ProcNode[pf.MeshObject]:
    circular = mesh.quad_cylinder(
        radius=top_radius, depth=height, resolution=16, insets=2
    )
    square = mesh.quad_cylinder(
        radius=top_radius * math.sqrt(2.0), depth=height, resolution=4, insets=2
    )
    square = pf.nodes.geo.transform(square, rotation=(0.0, 0.0, math.pi / 4.0))
    geometry = pf.nodes.func.switch(is_circular, square, circular)
    position = pf.nodes.geo.input_position()
    height_fraction = position.z / height + 0.5
    radius = bottom_radius + (top_radius - bottom_radius) * height_fraction
    scale = radius / top_radius
    position = pf.nodes.math.combine_xyz(
        x=position.x * scale,
        y=position.y * scale,
        z=position.z + height * 0.5,
    )
    geometry = pf.nodes.geo.set_position(geometry, position=position)
    geometry = mesh.metric_box_uv(geometry)
    geometry = pf.nodes.geo.set_shade_smooth(geometry, selection=is_circular)
    return pf.nodes.geo.set_material(geometry, material)


@pf.tracer.generator
def sink_pedestal(
    height: float = 0.65,
    top_radius: float = 0.065,
    bottom_radius: float = 0.18,
    is_circular: bool = True,
    material: pf.Material | None = None,
) -> SinkPedestalResult:
    if material is None:
        material = pf.Material(surface=pf.nodes.shader.principled_bsdf(roughness=0.25))
    obj = pf.nodes.to_mesh_object(
        _pedestal_geometry(material, height, top_radius, bottom_radius, is_circular)
    )
    pf.ops.modifier.subdivide_surface(obj, levels=2, _skip_apply=True)
    return SinkPedestalResult(mesh=obj)
