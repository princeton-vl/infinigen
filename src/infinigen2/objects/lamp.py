# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Hongyu Wen: original Infinigen v1 nodegroup (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/objects/lamp/lamp.py)
# - Alexander Raistrick: point light, transpile to procfunc/v2

from functools import partial
from typing import NamedTuple

import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.shaders.base_materials import fabric
from infinigen2.shaders.composites import fabric_patterned
from infinigen2.shaders.functionality_lists import (
    furniture_material_rand,
)
from infinigen2.util import mesh as mesh_util
from infinigen2.util.curve import curve_to_mesh_with_uv

__all__ = [
    "LampResult",
    "LampshadeShape",
    "desk_lamp_rand",
    "hanging_lampshade_shape_rand",
    "point_light_indoor",
    "point_light_indoor_rand",
    "lamp",
    "lamp_rand",
    "lampshade_color_rand",
    "lampshade_fabric_rand",
    "lampshade_shape_rand",
]

# head-local z of the bulb-rack inner ring (where the bulb socket sits)
BULB_HUB_HEIGHT = 0.015


def point_light_indoor(
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
    emission = pf.nodes.shader.emission(color=blackbody, strength=energy)
    pf.nodes.to_light(light, surface=emission)

    return light


def point_light_indoor_rand(
    rng: pf.RNG,
    energy: float | None = None,
    temperature: float | None = None,
    shadow_soft_size: float = 0.02,
) -> pf.LightObject:
    if temperature is None:
        temperature = pf.random.clip_gaussian(rng, 4500, 1000, 2000, 8000)
    if energy is None:
        energy = pf.random.uniform(rng, 5, 15)

    return point_light_indoor(
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
        rotation=(0, 0, 0),
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

    join = pf.nodes.geo.join_geometry([transform, set_position_1, transform_1])

    curve_circle_2 = pf.nodes.geo.curve_circle(resolution=6, radius=thickness)
    curve_to = curve_to_mesh_with_uv(curve=join, profile=curve_circle_2, fill_caps=True)
    return curve_to.mesh


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
    shade_with_uv = curve_to_mesh_with_uv(curve=set_curve_radius, profile=curve_circle)
    curve_to = shade_with_uv.mesh

    extrude = pf.nodes.geo.extrude_mesh(
        mesh=curve_to, offset_scale=0.005, individual=False
    )

    flip_faces = pf.nodes.geo.flip_faces(curve_to)

    join_1 = pf.nodes.geo.join_geometry([extrude.mesh, flip_faces])

    merge_by_distance = pf.nodes.geo.merge_by_distance(geometry=join_1)

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


@pf.nodes.node_function
def _lamp_geometry(
    stand_radius: t.SocketOrVal[float],
    base_radius: t.SocketOrVal[float],
    base_height: t.SocketOrVal[float],
    shade_height: t.SocketOrVal[float],
    head_top_radius: t.SocketOrVal[float],
    head_bot_radius: t.SocketOrVal[float],
    head_mid_radius: t.SocketOrVal[float],
    profile_resolution: t.SocketOrVal[int],
    reverse_lamp: t.SocketOrVal[bool],
    rack_thickness: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    black_material: t.SocketOrVal[pf.Material],
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

    curve_line = pf.nodes.geo.curve_line(end=head_translation, start=(0, 0, 0))

    curve_circle = pf.nodes.geo.curve_circle(resolution=8, radius=stand_radius)
    curve_to = curve_to_mesh_with_uv(
        curve=curve_line, profile=curve_circle, fill_caps=True
    ).mesh
    curve_line_1_end = pf.nodes.math.combine_xyz(z=base_height)
    curve_line_1 = pf.nodes.geo.curve_line(end=curve_line_1_end, start=(0, 0, 0))
    curve_circle_1 = pf.nodes.geo.curve_circle(resolution=16, radius=base_radius)
    curve_to_1 = curve_to_mesh_with_uv(
        curve=curve_line_1,
        profile=curve_circle_1,
        fill_caps=True,
    ).mesh

    join_2 = pf.nodes.geo.join_geometry([curve_to, curve_to_1])

    set_material = pf.nodes.geo.set_material(
        geometry=join_2, material=black_material, selection=True
    )

    join = pf.nodes.geo.join_geometry([transform, set_material])

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


def lampshade_shape_rand(rng: pf.RNG) -> LampshadeShape:
    top_radius = pf.random.clip_gaussian(rng, 0.1, 0.06, 0.0, 0.3)
    slant_fac = pf.random.clip_gaussian(rng, 0.15, 0.35, 0.0, 1.0)
    bot_radius = top_radius + slant_fac * (0.4 - top_radius)
    concavity_fac = pf.random.clip_gaussian(rng, 0.5, 0.45, 0.0, 1.45)
    mid_radius = top_radius + (bot_radius - top_radius) * concavity_fac
    profile_resolution = pf.control.choice(rng, [(16, 4.0), (4, 1.0)])
    return LampshadeShape(
        top_radius=top_radius,
        bot_radius=bot_radius,
        mid_radius=mid_radius,
        profile_resolution=profile_resolution,
    )


def hanging_lampshade_shape_rand(rng: pf.RNG) -> LampshadeShape:
    shape = lampshade_shape_rand(rng)
    return LampshadeShape(
        top_radius=shape.bot_radius,
        bot_radius=shape.top_radius,
        mid_radius=shape.mid_radius,
        profile_resolution=shape.profile_resolution,
    )


def lampshade_color_rand(rng: pf.RNG) -> pf.Color:
    hue = pf.random.uniform(rng, 0.0, 0.2)
    saturation = pf.random.uniform(rng, 0.0, 0.25)
    value = pf.random.uniform(rng, 0.1, 0.9)
    return pf.color.hsv_color(hue=hue, saturation=saturation, value=value)


def lampshade_fabric_rand(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
) -> pf.Material:
    rngs = rng.spawn(3)
    translucency = pf.random.clip_gaussian(rngs[0], 0.15, 0.15, 0.0, 0.9)
    pattern_scale = pf.random.uniform(rngs[2], 20.0, 250.0)

    lamp_fabric_fullcolor = partial(
        fabric.fabric_translucent_rand,
        translucency=translucency,
        base_color=lampshade_color_rand(rng),
    )
    lamp_fabric_patterned = partial(
        fabric_patterned.fabric_patterned_translucent_rand,
        translucency=translucency,
        scale=pattern_scale,
    )
    lamp_fabric_lampcolor = partial(
        fabric.fabric_translucent_rand,
        base_color=lampshade_color_rand(rng),
        translucency=translucency,
    )
    lamp_fabric_patterned_lampcolor = partial(
        fabric_patterned.fabric_patterned_translucent_rand,
        color1=lampshade_color_rand(rng),
        color2=lampshade_color_rand(rng),
        color3=lampshade_color_rand(rng),
        translucency=translucency,
        scale=pattern_scale,
    )

    option = pf.control.choice(
        rngs[1],
        [
            (lamp_fabric_lampcolor, 1.0),
            (lamp_fabric_patterned_lampcolor, 1.0),
            (lamp_fabric_fullcolor, 1.0),
            (lamp_fabric_patterned, 1.0),
        ],
    )

    return option(rngs[1], vector)


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
    energy: float = 7.0,
    temperature: float = 4500.0,
    black_material: pf.Material | None = None,
    lampshade_material: pf.Material | None = None,
    metal_material: pf.Material | None = None,
) -> LampResult:
    if black_material is None:
        black_material = pf.Material(surface=pf.nodes.shader.principled_bsdf())
    if lampshade_material is None:
        lampshade_material = pf.Material(surface=pf.nodes.shader.principled_bsdf())
    if metal_material is None:
        metal_material = pf.Material(surface=pf.nodes.shader.principled_bsdf())
    if head_mid_radius is None:
        head_mid_radius = head_top_radius

    result = _lamp_geometry(
        stand_radius=stand_radius,
        base_radius=base_radius,
        base_height=base_height,
        shade_height=shade_height,
        head_top_radius=head_top_radius,
        head_bot_radius=head_bot_radius,
        head_mid_radius=head_mid_radius,
        profile_resolution=profile_resolution,
        reverse_lamp=reverse_lamp,
        rack_thickness=rack_thickness,
        height=height,
        black_material=black_material,
        lampshade_material=lampshade_material,
        metal_material=metal_material,
    )

    geo = mesh_util.crease_sharp(result.geometry, threshold_degrees=70.0)
    mesh = pf.nodes.to_mesh_object(geo)
    pf.ops.modifier.subdivide_surface(mesh, levels=2, _skip_apply=True)

    bulb_radius = 0.02
    point_light = point_light_indoor(
        energy=energy, temperature=temperature, shadow_soft_size=bulb_radius
    )
    point_light.item().location.z = height + 1.05 * bulb_radius
    point_light.item().parent = mesh.item()

    return LampResult(mesh=mesh, light=point_light)


def lamp_rand(
    rng: pf.RNG,
    height: float | None = None,
    temperature: float | None = None,
    energy: float | None = None,
    head_top_radius: float | None = None,
    head_bot_radius: float | None = None,
    head_mid_radius: float | None = None,
    profile_resolution: int | None = None,
) -> LampResult:
    stand_radius = pf.random.uniform(rng, 0.005, 0.015)
    base_radius = pf.random.uniform(rng, 0.05, 0.15)
    base_height = pf.random.uniform(rng, 0.01, 0.03)
    shade_height = pf.random.uniform(rng, 0.18, 0.3)
    rack_thickness = pf.random.uniform(rng, 0.001, 0.003)
    reverse_lamp = True

    shape = lampshade_shape_rand(rng)
    if head_top_radius is None:
        head_top_radius = shape.top_radius
    if head_bot_radius is None:
        head_bot_radius = shape.bot_radius
    if head_mid_radius is None:
        head_mid_radius = shape.mid_radius
    if profile_resolution is None:
        profile_resolution = shape.profile_resolution

    if height is None:
        height = pf.random.uniform(rng, 0.3, 0.6)

    vec = pf.nodes.shader.coord().uv

    stem_mat = furniture_material_rand(rng, vec)
    lampshade_mat = lampshade_fabric_rand(rng, vec)

    result = _lamp_geometry(
        stand_radius=stand_radius,
        base_radius=base_radius,
        base_height=base_height,
        shade_height=shade_height,
        head_top_radius=head_top_radius,
        head_bot_radius=head_bot_radius,
        head_mid_radius=head_mid_radius,
        profile_resolution=profile_resolution,
        reverse_lamp=reverse_lamp,
        rack_thickness=rack_thickness,
        height=height,
        black_material=stem_mat,
        lampshade_material=lampshade_mat,
        metal_material=stem_mat,
    )

    geo = mesh_util.crease_sharp(result.geometry, threshold_degrees=70.0)
    lamp = pf.nodes.to_mesh_object(geo)
    pf.ops.modifier.subdivide_surface(lamp, levels=2, _skip_apply=True)

    if energy is None:
        energy = pf.random.clip_gaussian(rng, 7, 4, 5, 18)
    bulb_radius = 0.02
    point_light = point_light_indoor_rand(
        rng, temperature=temperature, energy=energy, shadow_soft_size=bulb_radius
    )
    # Sit the bulb just above the stem top (the head mounts at z=height) so its
    # radius clears the post mesh by 5% instead of intersecting it.
    pf.ops.object.set_transform(
        point_light, location=(0.0, 0.0, height + 1.05 * bulb_radius)
    )
    point_light.item().parent = lamp.item()

    return LampResult(mesh=lamp, light=point_light)


def desk_lamp_rand(rng: pf.RNG) -> LampResult:
    height = pf.random.uniform(rng, 0.25, 0.4)
    return lamp_rand(rng, height=height)


if __name__ == "__main__":
    bulb_rack_result = _bulb_rack()

    lamp_head_result = _lamp_head()
    lamp_geometry_result = _lamp_geometry()
