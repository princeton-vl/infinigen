# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Hongyu Wen: original Infinigen lamp (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/objects/lamp/lamp.py)
# - Alexander Raistrick: refactor for Infinigen2

from functools import partial
from math import pi, sqrt
from typing import NamedTuple

import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.objects import furniture_bases, vase
from infinigen2.shaders.functionality_lists import (
    fabric_general_rand,
    furniture_material_rand,
)
from infinigen2.util import mesh as mesh_util
from infinigen2.util.curve import curve_to_mesh_with_uv

__all__ = [
    "LampResult",
    "LampshadeShape",
    "lamp",
    "lamp_base_classic",
    "lamp_ceiling_shade_rand",
    "lamp_desk_rand",
    "lamp_floor_rand",
    "lamp_rand",
    "lampshade_shape_hanging_rand",
    "lampshade_shape_rand",
    "light_point_indoor",
    "light_point_indoor_rand",
]

# head-local z of the bulb-rack inner ring (where the bulb socket sits)
BULB_HUB_HEIGHT = 0.015
_DEFAULT_SUPPORT_SHADE_OVERLAP = 0.03
_SHADE_CORNER_ATTRIBUTE = "lamp_shade_corner"
_RACK_CORNER_ATTRIBUTE = "lamp_rack_corner"


@pf.tracer.generator
def light_point_indoor(
    energy: float = 10.0,
    temperature: float = 4500.0,
    shadow_soft_size: float = 0.02,
) -> pf.LightObject:
    """Create a point light with blackbody temperature roughly based on indoor lighting. range 3500-6500K. was expanded to 2000-8000"""
    light = pf.ops.primitives.light.point_lamp(
        energy=energy,
        shadow_soft_size=shadow_soft_size,
    )
    blackbody = pf.nodes.color.blackbody(temperature=temperature)
    emission = pf.nodes.shader.emission(color=blackbody, strength=1.0)
    pf.nodes.to_light(light, surface=emission)

    return light


def light_point_indoor_rand(
    rng: pf.RNG,
    energy: float | None = None,
    temperature: float | None = None,
    shadow_soft_size: float = 0.02,
) -> pf.LightObject:
    rng_temperature, rng_energy = rng.spawn(2)
    if temperature is None:
        temperature = pf.random.clip_gaussian(rng_temperature, 4500, 1000, 2000, 8000)
    if energy is None:
        energy = pf.random.uniform(rng_energy, 450, 1600) / 177

    return light_point_indoor(
        energy=energy, temperature=temperature, shadow_soft_size=shadow_soft_size
    )


@pf.nodes.node_function
def _bulb_rack(
    thickness: t.SocketOrVal[float],
    inner_radius: t.SocketOrVal[float],
    outer_radius: t.SocketOrVal[float],
    inner_height: t.SocketOrVal[float],
    outer_height: t.SocketOrVal[float],
    outer_resolution: t.SocketOrVal[int] = 24,
    amount: t.SocketOrVal[int] = 3,
) -> pf.ProcNode:
    curve_circle_radius = pf.nodes.math.multiply_add(
        a=thickness, b=0.5, addend=inner_radius
    )
    curve_circle = pf.nodes.geo.curve_circle(resolution=24, radius=curve_circle_radius)

    transform_translation = pf.nodes.math.combine_xyz(z=inner_height)
    transform = pf.nodes.geo.transform(
        geometry=curve_circle,
        translation=transform_translation,
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )

    curve_line = pf.nodes.geo.curve_line(start=(-1.0, 0.0, 0.0), end=(1.0, 0.0, 0.0))

    to_instance = pf.nodes.geo.geometry_to_instance(curve_line)

    duplicate_elements = pf.nodes.geo.duplicate_elements(
        geometry=to_instance,
        amount=amount,
        domain="INSTANCE",
    )

    realize_instances = pf.nodes.geo.realize_instances(duplicate_elements.geometry)

    curve_endpoint_selection = pf.nodes.geo.curve_endpoint_selection(0)
    curve_circle_1 = pf.nodes.geo.curve_circle(
        resolution=outer_resolution, radius=outer_radius
    )

    transform_1_translation = pf.nodes.math.combine_xyz(z=outer_height)
    transform_1 = pf.nodes.geo.transform(
        geometry=curve_circle_1,
        translation=transform_1_translation,
        rotation=(0, 0, pi / 4.0),
        scale=(1, 1, 1),
    )

    set_0_factor = (
        duplicate_elements.duplicate_index.astype(dtype=float)
        * 1.0
        / amount.astype(dtype=float)
    )

    sample_curve = pf.nodes.geo.sample_curve(
        curves=transform_1,
        factor=set_0_factor,
        value=0.0,
    )

    set_position = pf.nodes.geo.set_position(
        geometry=realize_instances,
        selection=curve_endpoint_selection,
        position=sample_curve.position,
    )

    curve_endpoint_selection_1 = pf.nodes.geo.curve_endpoint_selection(end_size=0)

    sample_curve_1 = pf.nodes.geo.sample_curve(
        curves=transform,
        factor=set_0_factor,
        value=0.0,
    )

    set_position_1 = pf.nodes.geo.set_position(
        geometry=set_position,
        selection=curve_endpoint_selection_1,
        position=sample_curve_1.position,
    )

    join = pf.nodes.geo.join_geometry([transform, set_position_1])

    curve_circle_2 = pf.nodes.geo.curve_circle(resolution=6, radius=thickness)
    curve_to = curve_to_mesh_with_uv(curve=join, profile=curve_circle_2, fill_caps=True)
    ring = pf.nodes.geo.capture_attribute(
        geometry=transform_1, index=pf.nodes.geo.input_index()
    )
    ring_mesh = curve_to_mesh_with_uv(curve=ring.geometry, profile=curve_circle_2).mesh
    edge_vertices = pf.nodes.geo.input_mesh_edge_vertices()
    start = pf.nodes.geo.field_at_index(
        value=ring.index, index=edge_vertices.vertex_index_1, domain="POINT"
    )
    end = pf.nodes.geo.field_at_index(
        value=ring.index, index=edge_vertices.vertex_index_2, domain="POINT"
    )
    corner = pf.nodes.func.boolean_and(
        a=pf.nodes.func.equal(a=start, b=end),
        b=pf.nodes.func.equal(a=outer_resolution, b=4),
    )
    ring_mesh = pf.nodes.geo.store_named_attribute(
        geometry=ring_mesh,
        name=_RACK_CORNER_ATTRIBUTE,
        value=corner,
        domain="EDGE",
        data_type="BOOLEAN",
    )
    return pf.nodes.geo.join_geometry([curve_to.mesh, ring_mesh])


@pf.nodes.node_function
def _lamp_head(
    shade_height: t.SocketOrVal[float],
    top_radius: t.SocketOrVal[float],
    bot_radius: t.SocketOrVal[float],
    mid_radius: t.SocketOrVal[float],
    profile_resolution: t.SocketOrVal[int],
    reverse_bulb: t.SocketOrVal[bool],
    rack_thickness: t.SocketOrVal[float],
    rack_height: t.SocketOrVal[float],
    black_material: t.SocketOrVal[pf.Material],
    lampshade_material: t.SocketOrVal[pf.Material],
    metal_material: t.SocketOrVal[pf.Material],
) -> pf.ProcNode:
    bulb_rack_outer_height_b = pf.nodes.math.multiply_add(
        a=reverse_bulb.astype(dtype=float), b=2.0, addend=-1.0
    )
    bulb_rack_outer_height = rack_height * bulb_rack_outer_height_b

    curve_line_start = pf.nodes.math.combine_xyz(z=bulb_rack_outer_height)
    curve_a = shade_height - rack_height
    curve_b = bulb_rack_outer_height_b * -1.0
    curve_line_end = pf.nodes.math.combine_xyz(z=curve_a * curve_b)
    curve_line = pf.nodes.geo.curve_line(start=curve_line_start, end=curve_line_end)
    curve_line = pf.nodes.geo.resample_curve_count(curve=curve_line, count=16)

    spline_parameter = pf.nodes.geo.spline_parameter()

    radius_profile = pf.nodes.geo.curve_bezier(
        start=pf.nodes.math.combine_xyz(x=top_radius),
        middle=pf.nodes.math.combine_xyz(x=mid_radius, y=0.5),
        end=pf.nodes.math.combine_xyz(x=bot_radius, y=1.0),
        resolution=32,
    )
    sampled_radius = pf.nodes.geo.sample_curve(
        curves=radius_profile,
        factor=spline_parameter.factor,
        value=0.0,
    )

    # meter-scale UVs: real radius lives on the profile, rail radius is a ~1 multiplier
    profile_radius = (top_radius + bot_radius) * 0.5
    set_curve_radius = pf.nodes.geo.set_curve_radius(
        curve=curve_line, radius=sampled_radius.position.x / profile_radius
    )

    curve_circle = pf.nodes.geo.curve_circle(
        resolution=profile_resolution, radius=profile_radius
    )
    curve_circle = pf.nodes.geo.transform(
        geometry=curve_circle,
        rotation=(0.0, 0.0, pi / 4.0),
    )
    shade_with_uv = curve_to_mesh_with_uv(curve=set_curve_radius, profile=curve_circle)
    curve_to = shade_with_uv.mesh

    extrude = pf.nodes.geo.extrude_mesh(
        mesh=curve_to, offset_scale=0.005, individual=False
    )

    flip_faces = pf.nodes.geo.flip_faces(curve_to)

    join_1 = pf.nodes.geo.join_geometry([extrude.mesh, flip_faces])

    merge_by_distance = pf.nodes.geo.merge_by_distance(geometry=join_1)
    edge_vertices = pf.nodes.geo.input_mesh_edge_vertices()
    changes_height = (
        pf.nodes.math.absolute(edge_vertices.position_1.z - edge_vertices.position_2.z)
        > 1e-6
    )
    square_profile = pf.nodes.func.equal(a=profile_resolution, b=4)
    shade_corner = pf.nodes.func.boolean_and(a=changes_height, b=square_profile)
    merge_by_distance = pf.nodes.geo.store_named_attribute(
        geometry=merge_by_distance,
        name=_SHADE_CORNER_ATTRIBUTE,
        value=True,
        domain="EDGE",
        data_type="BOOLEAN",
        selection=shade_corner,
    )

    set_material = pf.nodes.geo.set_material(
        geometry=merge_by_distance,
        material=lampshade_material,
        selection=True,
    )

    geometries_2_scale = top_radius * 0.8

    bulb_rack_result = _bulb_rack(
        thickness=rack_thickness,
        amount=3,
        inner_radius=geometries_2_scale * 0.15,
        outer_radius=top_radius,
        inner_height=BULB_HUB_HEIGHT,
        outer_height=bulb_rack_outer_height,
        outer_resolution=profile_resolution,
    )

    set_material_1 = pf.nodes.geo.set_material(
        geometry=bulb_rack_result,
        material=black_material,
        selection=True,
    )

    join = pf.nodes.geo.join_geometry([set_material, set_material_1])
    return join


class LampResult(NamedTuple):
    mesh: pf.MeshObject
    light: pf.LightObject


class _LampGeometryResult(NamedTuple):
    geometry: pf.ProcNode
    bounding_box: pf.ProcNode
    light_position: pf.ProcNode[pf.Vector]


def _lamp_support_height(
    height: float,
    shade_height: float,
    reverse_lamp: bool,
    support_shade_overlap: float,
) -> float:
    shade_bottom_fraction = 0.4 if reverse_lamp else 0.2
    return height - shade_height * shade_bottom_fraction + support_shade_overlap


def _lamp_table_base_dimensions_rand(
    rng: pf.RNG,
    base_radius: float,
    shade_radius: float,
    support_height: float,
) -> pf.Vector:
    min_footprint = 2.0 * base_radius
    max_footprint = max(min_footprint, 2.0 * shade_radius / sqrt(2.0))
    footprint = pf.random.uniform(rng, min_footprint, max_footprint)
    return (footprint, footprint, support_height)


def _lamp_table_member_width_rand(rng: pf.RNG) -> float:
    return pf.random.uniform(rng, 0.01, 0.06)


@pf.nodes.node_function
def _cylinder_lamp_base(
    radius: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
) -> pf.ProcNode[pf.MeshObject]:
    line_end = pf.nodes.math.combine_xyz(z=height)
    line = pf.nodes.geo.curve_line(end=line_end, start=(0, 0, 0))
    profile = pf.nodes.geo.curve_circle(resolution=16, radius=radius)
    return curve_to_mesh_with_uv(curve=line, profile=profile, fill_caps=True).mesh


@pf.nodes.node_function
def _classic_lamp_base(
    stand_radius: t.SocketOrVal[float],
    base_radius: t.SocketOrVal[float],
    base_height: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
) -> pf.ProcNode[pf.MeshObject]:
    line_end = pf.nodes.math.combine_xyz(z=height)
    line = pf.nodes.geo.curve_line(end=line_end, start=(0, 0, 0))
    profile = pf.nodes.geo.curve_circle(resolution=8, radius=stand_radius)
    stand = curve_to_mesh_with_uv(curve=line, profile=profile, fill_caps=True).mesh
    disc = _cylinder_lamp_base(radius=base_radius, height=base_height)
    return pf.nodes.geo.join_geometry([stand, disc])


@pf.tracer.generator
def lamp_base_classic(
    stand_radius: float,
    base_radius: float,
    base_height: float,
    height: float,
) -> pf.ProcNode[pf.MeshObject]:
    return _classic_lamp_base(
        stand_radius=stand_radius,
        base_radius=base_radius,
        base_height=base_height,
        height=height,
    )


def _lamp_vase_base_geometry_rand(
    rng: pf.RNG,
    base_radius: float,
    base_height: float,
) -> pf.ProcNode[pf.MeshObject]:
    rngs = rng.spawn(10)
    resolution = 32
    neck_scale = pf.random.uniform(rngs[0], 0.2, 0.8)
    inner_radius = 1.0 - 0.2 * pf.random.uniform(rngs[1], 0.0, 1.0) ** 2
    geometry = vase.vase_body(
        u_resolution=resolution,
        v_resolution=resolution,
        height=base_height,
        diameter=base_radius,
        profile_inner_radius=inner_radius,
        profile_star_points=pf.random.randint(rngs[2], 8, resolution // 2 + 1),
        top_scale=neck_scale * pf.random.uniform(rngs[3], 0.8, 1.2),
        neck_mid_position=pf.random.uniform(rngs[4], 0.7, 0.95),
        neck_position=0.5 * neck_scale + 0.5 + pf.random.uniform(rngs[5], -0.05, 0.05),
        neck_scale=neck_scale,
        shoulder_position=pf.random.uniform(rngs[6], 0.3, 0.7),
        shoulder_thickness=pf.random.uniform(rngs[7], 0.1, 0.25),
        foot_scale=pf.random.uniform(rngs[8], 0.4, 0.6),
        foot_height=pf.random.uniform(rngs[9], 0.01, 0.1),
    )
    extruded = pf.nodes.geo.extrude_mesh(
        mesh=geometry,
        offset_scale=0.03 * base_radius,
        individual=False,
    )
    inner = pf.nodes.geo.flip_faces(geometry)
    shell = pf.nodes.geo.join_geometry([extruded.mesh, inner])
    return pf.nodes.geo.merge_by_distance(geometry=shell)


def _lamp_table_straight_base_rand(
    rng: pf.RNG,
    _stand_radius: float,
    dimensions: pf.Vector,
) -> pf.ProcNode[pf.MeshObject]:
    rng_width, rng_bottom_scale, rng_stretcher, rng_stretcher_position = rng.spawn(4)
    leg_diameter = _lamp_table_member_width_rand(rng_width)
    return furniture_bases.base_four_leg(
        dimensions=dimensions,
        leg_diameter=leg_diameter,
        leg_inset=0.1 * (dimensions[0] - leg_diameter),
        leg_placement_bottom_scale=pf.random.uniform(rng_bottom_scale, 0.95, 1.25),
        stretcher_increment=pf.control.choice(
            rng_stretcher, [(0, 1.0), (1, 1.0), (2, 1.0)]
        ),
        stretcher_relative_pos=pf.random.uniform(rng_stretcher_position, 0.2, 0.6),
    )


def _lamp_table_pedestal_base_rand(
    rng: pf.RNG,
    stand_radius: float,
    dimensions: pf.Vector,
) -> pf.ProcNode[pf.MeshObject]:
    rng_top_radius, rng_bottom_radius, rng_profile = rng.spawn(3)
    width = min(dimensions[0], dimensions[1])
    top_radius = pf.random.uniform(
        rng_top_radius,
        max(1.1 * stand_radius, 0.06 * width),
        max(1.5 * stand_radius, 0.16 * width),
    )
    bottom_radius = pf.random.uniform(
        rng_bottom_radius,
        max(1.75 * stand_radius, 0.32 * width),
        max(2.25 * stand_radius, 0.48 * width),
    )
    return furniture_bases.base_pedestal_column_rand(
        rng_profile, dimensions[2], top_radius, bottom_radius
    )


def _lamp_table_square_base_rand(
    rng: pf.RNG,
    _stand_radius: float,
    dimensions: pf.Vector,
) -> pf.ProcNode[pf.MeshObject]:
    rng_width, rng_connector = rng.spawn(2)
    return furniture_bases.base_box_leg(
        dimensions=dimensions,
        leg_diameter=_lamp_table_member_width_rand(rng_width),
        leg_placement_top_scale=0.8,
        leg_placement_bottom_scale=1.0,
        has_bottom_connector=pf.control.choice(
            rng_connector, [(True, 2.0), (False, 1.0)]
        ),
    )


def _lamp_table_base_geometry_rand(
    rng: pf.RNG,
    stand_radius: float,
    dimensions: pf.Vector,
) -> pf.ProcNode[pf.MeshObject]:
    return _lamp_table_pedestal_base_rand(rng, stand_radius, dimensions)


def _lamp_cylinder_base_rand(
    _rng: pf.RNG,
    stand_radius: float,
    base_radius: float,
    _shade_radius: float,
    flat_base_height: float,
    support_height: float,
    stem_material: pf.Material,
    _vector: pf.ProcNode[pf.Vector],
) -> tuple[pf.ProcNode[pf.MeshObject], pf.Material]:
    geometry = lamp_base_classic(
        stand_radius=stand_radius,
        base_radius=base_radius,
        base_height=flat_base_height,
        height=support_height,
    )
    return geometry, stem_material


def _lamp_vase_base_rand(
    rng: pf.RNG,
    _stand_radius: float,
    base_radius: float,
    _shade_radius: float,
    _flat_base_height: float,
    support_height: float,
    _stem_material: pf.Material,
    vector: pf.ProcNode[pf.Vector],
) -> tuple[pf.ProcNode[pf.MeshObject], pf.Material]:
    rng_geometry, rng_material = rng.spawn(2)
    geometry = _lamp_vase_base_geometry_rand(rng_geometry, base_radius, support_height)
    material = vase.vase_material_rand(rng_material, vector)
    return geometry, material


def _lamp_table_base_rand(
    rng: pf.RNG,
    stand_radius: float,
    base_radius: float,
    shade_radius: float,
    _flat_base_height: float,
    support_height: float,
    stem_material: pf.Material,
    _vector: pf.ProcNode[pf.Vector],
) -> tuple[pf.ProcNode[pf.MeshObject], pf.Material]:
    rng_dimensions, rng_geometry = rng.spawn(2)
    dimensions = _lamp_table_base_dimensions_rand(
        rng_dimensions,
        base_radius,
        shade_radius,
        support_height,
    )
    geometry = _lamp_table_base_geometry_rand(rng_geometry, stand_radius, dimensions)
    return geometry, stem_material


def _lamp_base_rand(
    rng: pf.RNG,
    stand_radius: float,
    base_radius: float,
    shade_radius: float,
    flat_base_height: float,
    support_height: float,
    stem_material: pf.Material,
    vector: pf.ProcNode[pf.Vector],
) -> tuple[pf.ProcNode[pf.MeshObject], pf.Material]:
    rng_choice, rng_classic, rng_vase, rng_table = rng.spawn(4)
    base_fn = pf.control.choice(
        rng_choice,
        [
            (partial(_lamp_cylinder_base_rand, rng_classic), 0.5),
            (partial(_lamp_vase_base_rand, rng_vase), 0.25),
            (partial(_lamp_table_base_rand, rng_table), 0.25),
        ],
    )
    return base_fn(
        stand_radius,
        base_radius,
        shade_radius,
        flat_base_height,
        support_height,
        stem_material,
        vector,
    )


def _floor_lamp_base_rand(
    rng: pf.RNG,
    stand_radius: float,
    base_radius: float,
    shade_radius: float,
    flat_base_height: float,
    support_height: float,
    stem_material: pf.Material,
    vector: pf.ProcNode[pf.Vector],
) -> tuple[pf.ProcNode[pf.MeshObject], pf.Material]:
    rng_choice, rng_classic, rng_table = rng.spawn(3)
    base_fn = pf.control.choice(
        rng_choice,
        [
            (partial(_lamp_cylinder_base_rand, rng_classic), 0.5),
            (partial(_lamp_table_base_rand, rng_table), 0.25),
        ],
    )
    return base_fn(
        stand_radius,
        base_radius,
        shade_radius,
        flat_base_height,
        support_height,
        stem_material,
        vector,
    )


@pf.nodes.node_function
def _lamp_geometry(
    base_geometry: pf.ProcNode[pf.MeshObject],
    shade_height: t.SocketOrVal[float],
    head_top_radius: t.SocketOrVal[float],
    head_bot_radius: t.SocketOrVal[float],
    head_mid_radius: t.SocketOrVal[float],
    profile_resolution: t.SocketOrVal[int],
    reverse_lamp: t.SocketOrVal[bool],
    rack_thickness: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    black_material: t.SocketOrVal[pf.Material],
    base_material: t.SocketOrVal[pf.Material],
    lampshade_material: t.SocketOrVal[pf.Material],
    metal_material: t.SocketOrVal[pf.Material],
) -> _LampGeometryResult:
    lamp_head_rack_height = pf.nodes.math.multiply_add(
        a=shade_height * 0.4,
        b=reverse_lamp.astype(dtype=float),
        addend=shade_height * 0.2,
    )
    lamp_head_result = _lamp_head(
        shade_height=shade_height,
        top_radius=head_top_radius,
        bot_radius=head_bot_radius,
        mid_radius=head_mid_radius,
        profile_resolution=profile_resolution,
        reverse_bulb=reverse_lamp,
        rack_thickness=rack_thickness,
        rack_height=lamp_head_rack_height,
        black_material=black_material,
        lampshade_material=lampshade_material,
        metal_material=metal_material,
    )

    head_translation = pf.nodes.math.combine_xyz(z=height)
    transform = pf.nodes.geo.transform(
        geometry=lamp_head_result,
        translation=head_translation,
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )

    base = pf.nodes.geo.set_material(
        geometry=base_geometry, material=base_material, selection=True
    )

    join = pf.nodes.geo.join_geometry([transform, base])

    bound_box = pf.nodes.geo.bound_box(join)

    light_position = pf.nodes.math.combine_xyz(z=height + 0.1)
    return _LampGeometryResult(
        geometry=join,
        bounding_box=bound_box.bounding_box,
        light_position=light_position,
    )


class LampshadeShape(NamedTuple):
    top_radius: float
    bot_radius: float
    mid_radius: float
    profile_resolution: int


def lampshade_shape_rand(
    rng: pf.RNG, base_radius: float | None = None
) -> LampshadeShape:
    rng_top, rng_bottom, rng_mid, rng_resolution = rng.spawn(4)
    if base_radius is None:
        top_radius = pf.random.clip_gaussian(rng_top, 0.1, 0.06, 0.0, 0.3)
        slant_fac = pf.random.clip_gaussian(rng_bottom, 0.15, 0.35, 0.0, 1.0)
        bot_radius = top_radius + slant_fac * (0.4 - top_radius)
    else:
        bot_radius = pf.random.uniform(
            rng_bottom, 1.25 * base_radius, 2.5 * base_radius
        )
        top_scale = pf.random.clip_gaussian(rng_top, 0.6, 0.25, 0.25, 1.0)
        top_radius = bot_radius * top_scale
    concavity_fac = pf.random.clip_gaussian(rng_mid, 0.5, 0.45, 0.0, 1.45)
    mid_radius = top_radius + (bot_radius - top_radius) * concavity_fac
    profile_resolution = pf.control.choice(rng_resolution, [(16, 4.0), (4, 1.0)])
    return LampshadeShape(
        top_radius=top_radius,
        bot_radius=bot_radius,
        mid_radius=mid_radius,
        profile_resolution=profile_resolution,
    )


def lampshade_shape_hanging_rand(rng: pf.RNG) -> LampshadeShape:
    shape = lampshade_shape_rand(rng)
    return LampshadeShape(
        top_radius=shape.bot_radius,
        bot_radius=shape.top_radius,
        mid_radius=shape.mid_radius,
        profile_resolution=shape.profile_resolution,
    )


def _lampshade_fabric_rand(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
) -> pf.Material:
    rng, rng_fabric = rng.spawn(2)
    translucency = pf.random.clip_gaussian(rng, 0.15, 0.15, 0.0, 0.9)
    return fabric_general_rand(rng_fabric, vector, translucency=translucency)


def _lampshade_material_rand(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
) -> pf.Material:
    rng_choice, rng_material = rng.spawn(2)
    material_rand = pf.control.choice(
        rng_choice,
        [
            (_lampshade_fabric_rand, 1.0),
            (furniture_material_rand, 1.0),
        ],
    )
    return material_rand(rng_material, vector)


class _LampParameters(NamedTuple):
    stand_radius: float
    base_radius: float
    flat_base_height: float
    shade_height: float
    head_top_radius: float
    head_bot_radius: float
    head_mid_radius: float
    profile_resolution: int
    height: float
    rack_thickness: float
    reverse_lamp: bool
    support_height: float
    energy: float
    temperature: float


def _lamp_parameters_rand(
    rng: pf.RNG,
    height: float | None = None,
    temperature: float | None = None,
    energy: float | None = None,
    head_top_radius: float | None = None,
    head_bot_radius: float | None = None,
    head_mid_radius: float | None = None,
    profile_resolution: int | None = None,
    base_radius: float | None = None,
    support_shade_overlap: float = _DEFAULT_SUPPORT_SHADE_OVERLAP,
) -> _LampParameters:
    (
        rng_stand_radius,
        rng_base_radius,
        rng_flat_base_height,
        rng_shade_height,
        rng_rack_thickness,
        rng_shape,
        rng_height,
        rng_energy,
        rng_temperature,
    ) = rng.spawn(9)

    stand_radius = pf.random.uniform(rng_stand_radius, 0.005, 0.015)
    if base_radius is None:
        base_radius = pf.random.uniform(rng_base_radius, 0.05, 0.15)
    flat_base_height = pf.random.uniform(rng_flat_base_height, 0.01, 0.03)
    shade_height = pf.random.uniform(rng_shade_height, 0.18, 0.3)
    rack_thickness = pf.random.uniform(rng_rack_thickness, 0.001, 0.003)
    reverse_lamp = True

    shape = lampshade_shape_rand(rng_shape, base_radius=base_radius)
    mid_radius_offset = shape.mid_radius - (shape.top_radius + shape.bot_radius) * 0.5
    if head_top_radius is None:
        head_top_radius = shape.top_radius
    if head_bot_radius is None:
        head_bot_radius = shape.bot_radius
    if head_mid_radius is None:
        head_mid_radius = (head_top_radius + head_bot_radius) * 0.5 + mid_radius_offset
    if profile_resolution is None:
        profile_resolution = shape.profile_resolution

    if height is None:
        height = pf.random.uniform(rng_height, 0.3, 0.6)
    if energy is None:
        energy = pf.random.uniform(rng_energy, 450, 1600) / 177
    if temperature is None:
        temperature = pf.random.clip_gaussian(rng_temperature, 4500, 1000, 2000, 8000)

    support_height = _lamp_support_height(
        height,
        shade_height,
        reverse_lamp,
        support_shade_overlap,
    )
    return _LampParameters(
        stand_radius=stand_radius,
        base_radius=base_radius,
        flat_base_height=flat_base_height,
        shade_height=shade_height,
        head_top_radius=head_top_radius,
        head_bot_radius=head_bot_radius,
        head_mid_radius=head_mid_radius,
        profile_resolution=profile_resolution,
        height=height,
        rack_thickness=rack_thickness,
        reverse_lamp=reverse_lamp,
        support_height=support_height,
        energy=energy,
        temperature=temperature,
    )


def _assemble_lamp(
    parameters: _LampParameters,
    base_geometry: pf.ProcNode[pf.MeshObject],
    base_material: pf.Material,
    black_material: pf.Material,
    lampshade_material: pf.Material,
    metal_material: pf.Material,
) -> LampResult:
    result = _lamp_geometry(
        base_geometry=base_geometry,
        shade_height=parameters.shade_height,
        head_top_radius=parameters.head_top_radius,
        head_bot_radius=parameters.head_bot_radius,
        head_mid_radius=parameters.head_mid_radius,
        profile_resolution=parameters.profile_resolution,
        reverse_lamp=parameters.reverse_lamp,
        rack_thickness=parameters.rack_thickness,
        height=parameters.height,
        black_material=black_material,
        base_material=base_material,
        lampshade_material=lampshade_material,
        metal_material=metal_material,
    )

    geo = mesh_util.crease_sharp(result.geometry, threshold_degrees=70.0)
    shade_corner = pf.nodes.geo.input_named_attribute(
        name=_SHADE_CORNER_ATTRIBUTE,
        data_type="BOOLEAN",
    ).attribute
    rack_corner = pf.nodes.geo.input_named_attribute(
        name=_RACK_CORNER_ATTRIBUTE, data_type="BOOLEAN"
    ).attribute
    corner = pf.nodes.func.boolean_or(a=shade_corner, b=rack_corner)
    geo = pf.nodes.geo.store_named_attribute(
        geometry=geo,
        name="crease_edge",
        value=1.0,
        domain="EDGE",
        selection=corner,
    )
    mesh = pf.nodes.to_mesh_object(geo)

    bulb_radius = 0.02
    point_light = light_point_indoor(
        energy=parameters.energy,
        temperature=parameters.temperature,
        shadow_soft_size=bulb_radius,
    )
    pf.ops.object.set_transform(
        point_light,
        location=(0.0, 0.0, parameters.height + 1.05 * bulb_radius),
    )
    point_light.item().parent = mesh.item()
    return LampResult(mesh=mesh, light=point_light)


@pf.tracer.generator
def lamp(
    stand_radius: float = 0.01,
    base_radius: float = 0.1,
    base_height: float = 0.02,
    shade_height: float = 0.24,
    head_top_radius: float = 0.16,
    head_bot_radius: float = 0.185,
    head_mid_radius: float | None = None,
    profile_resolution: int = 16,
    height: float = 0.45,
    rack_thickness: float = 0.002,
    reverse_lamp: bool = True,
    support_shade_overlap: float = _DEFAULT_SUPPORT_SHADE_OVERLAP,
    energy: float = 7.0,
    temperature: float = 4500.0,
    black_material: pf.Material | None = None,
    lampshade_material: pf.Material | None = None,
    metal_material: pf.Material | None = None,
    base_geometry: pf.ProcNode[pf.MeshObject] | None = None,
    base_material: pf.Material | None = None,
) -> LampResult:
    if black_material is None:
        black_material = pf.Material(surface=pf.nodes.shader.principled_bsdf())
    if lampshade_material is None:
        lampshade_material = pf.Material(surface=pf.nodes.shader.principled_bsdf())
    if metal_material is None:
        metal_material = pf.Material(surface=pf.nodes.shader.principled_bsdf())
    if head_mid_radius is None:
        head_mid_radius = head_top_radius

    support_height = _lamp_support_height(
        height,
        shade_height,
        reverse_lamp,
        support_shade_overlap,
    )
    parameters = _LampParameters(
        stand_radius=stand_radius,
        base_radius=base_radius,
        flat_base_height=base_height,
        shade_height=shade_height,
        head_top_radius=head_top_radius,
        head_bot_radius=head_bot_radius,
        head_mid_radius=head_mid_radius,
        profile_resolution=profile_resolution,
        height=height,
        rack_thickness=rack_thickness,
        reverse_lamp=reverse_lamp,
        support_height=support_height,
        energy=energy,
        temperature=temperature,
    )
    if base_geometry is None:
        base_geometry = _classic_lamp_base(
            stand_radius=stand_radius,
            base_radius=base_radius,
            base_height=base_height,
            height=support_height,
        )
    if base_material is None:
        base_material = black_material
    return _assemble_lamp(
        parameters,
        base_geometry,
        base_material,
        black_material,
        lampshade_material,
        metal_material,
    )


def _lamp_from_parameters(
    parameters: _LampParameters,
    support_shade_overlap: float,
    base: tuple[pf.ProcNode[pf.MeshObject], pf.Material],
    stem_material: pf.Material,
    lampshade_material: pf.Material,
) -> LampResult:
    result = lamp(
        stand_radius=parameters.stand_radius,
        base_radius=parameters.base_radius,
        base_height=parameters.flat_base_height,
        shade_height=parameters.shade_height,
        head_top_radius=parameters.head_top_radius,
        head_bot_radius=parameters.head_bot_radius,
        head_mid_radius=parameters.head_mid_radius,
        profile_resolution=parameters.profile_resolution,
        height=parameters.height,
        rack_thickness=parameters.rack_thickness,
        reverse_lamp=parameters.reverse_lamp,
        support_shade_overlap=support_shade_overlap,
        energy=parameters.energy,
        temperature=parameters.temperature,
        black_material=stem_material,
        lampshade_material=lampshade_material,
        metal_material=stem_material,
        base_geometry=base[0],
        base_material=base[1],
    )
    mesh = result.mesh
    pf.ops.modifier.subdivide_surface(mesh, levels=2, _skip_apply=True)
    return LampResult(mesh=mesh, light=result.light)


def lamp_rand(
    rng: pf.RNG,
    height: float | None = None,
    temperature: float | None = None,
    energy: float | None = None,
    head_top_radius: float | None = None,
    head_bot_radius: float | None = None,
    head_mid_radius: float | None = None,
    profile_resolution: int | None = None,
    base_radius: float | None = None,
    support_shade_overlap: float = _DEFAULT_SUPPORT_SHADE_OVERLAP,
) -> LampResult:
    rng_parameters, rng_base, rng_stem_mat, rng_shade_mat = rng.spawn(4)
    parameters = _lamp_parameters_rand(
        rng_parameters,
        height=height,
        temperature=temperature,
        energy=energy,
        head_top_radius=head_top_radius,
        head_bot_radius=head_bot_radius,
        head_mid_radius=head_mid_radius,
        profile_resolution=profile_resolution,
        base_radius=base_radius,
        support_shade_overlap=support_shade_overlap,
    )
    vec = pf.nodes.shader.coord().uv
    stem_mat = furniture_material_rand(rng_stem_mat, vec)
    lampshade_mat = _lampshade_material_rand(rng_shade_mat, vec)
    base = _lamp_base_rand(
        rng_base,
        stand_radius=parameters.stand_radius,
        base_radius=parameters.base_radius,
        shade_radius=parameters.head_bot_radius,
        flat_base_height=parameters.flat_base_height,
        support_height=parameters.support_height,
        stem_material=stem_mat,
        vector=vec,
    )
    return _lamp_from_parameters(
        parameters, support_shade_overlap, base, stem_mat, lampshade_mat
    )


def lamp_desk_rand(
    rng: pf.RNG,
    base_radius: float | None = None,
    support_shade_overlap: float = _DEFAULT_SUPPORT_SHADE_OVERLAP,
) -> LampResult:
    rng_height, rng_lamp = rng.spawn(2)
    height = pf.random.uniform(rng_height, 0.25, 0.4)
    return lamp_rand(
        rng_lamp,
        height=height,
        base_radius=base_radius,
        support_shade_overlap=support_shade_overlap,
    )


def lamp_floor_rand(
    rng: pf.RNG,
    support_shade_overlap: float = _DEFAULT_SUPPORT_SHADE_OVERLAP,
) -> LampResult:
    rng_height, rng_parameters, rng_base, rng_stem_mat, rng_shade_mat = rng.spawn(5)
    height = pf.random.uniform(rng_height, 1.0, 2.0)
    parameters = _lamp_parameters_rand(
        rng_parameters,
        height=height,
        support_shade_overlap=support_shade_overlap,
    )
    vec = pf.nodes.shader.coord().uv
    stem_mat = furniture_material_rand(rng_stem_mat, vec)
    lampshade_mat = _lampshade_material_rand(rng_shade_mat, vec)
    base = _floor_lamp_base_rand(
        rng_base,
        stand_radius=parameters.stand_radius,
        base_radius=parameters.base_radius,
        shade_radius=parameters.head_bot_radius,
        flat_base_height=parameters.flat_base_height,
        support_height=parameters.support_height,
        stem_material=stem_mat,
        vector=vec,
    )
    return _lamp_from_parameters(
        parameters, support_shade_overlap, base, stem_mat, lampshade_mat
    )


def lamp_ceiling_shade_rand(
    rng: pf.RNG,
    energy: float | None = None,
    support_shade_overlap: float = _DEFAULT_SUPPORT_SHADE_OVERLAP,
) -> LampResult:
    (
        rng_bot_radius,
        rng_top_scale,
        rng_height,
        rng_parameters,
        rng_base,
        rng_stem_mat,
        rng_shade_mat,
    ) = rng.spawn(7)
    bot_radius = pf.random.clip_gaussian(rng_bot_radius, 0.35, 0.1, 0.1, 0.6)
    top_scale = pf.random.clip_gaussian(rng_top_scale, 1.1, 0.1, 0.9, 1.5)
    top_radius = bot_radius * top_scale
    height = pf.random.clip_gaussian(rng_height, 0.3, 0.15, 0.2, 0.6)
    parameters = _lamp_parameters_rand(
        rng_parameters,
        energy=energy,
        head_top_radius=top_radius,
        head_bot_radius=bot_radius,
        height=height,
        support_shade_overlap=support_shade_overlap,
    )
    vec = pf.nodes.shader.coord().uv
    stem_mat = furniture_material_rand(rng_stem_mat, vec)
    lampshade_mat = _lampshade_material_rand(rng_shade_mat, vec)
    base = _lamp_cylinder_base_rand(
        rng_base,
        stand_radius=parameters.stand_radius,
        base_radius=parameters.base_radius,
        _shade_radius=parameters.head_bot_radius,
        flat_base_height=parameters.flat_base_height,
        support_height=parameters.support_height,
        stem_material=stem_mat,
        _vector=vec,
    )
    result = _lamp_from_parameters(
        parameters, support_shade_overlap, base, stem_mat, lampshade_mat
    )
    pf.ops.object.set_transform(result.mesh, rotation_euler=(pi, 0, 0))
    light_location = result.light.item().location
    result.light.item().parent = None
    result.light.item().location = (
        light_location[0],
        -light_location[1],
        -light_location[2],
    )
    return result


if __name__ == "__main__":
    bulb_rack_result = _bulb_rack()

    lamp_head_result = _lamp_head()
    lamp_geometry_result = _lamp_geometry()
