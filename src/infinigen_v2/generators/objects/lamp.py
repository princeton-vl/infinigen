from functools import partial
from typing import NamedTuple

import numpy as np
import procfunc as pf
from procfunc.nodes import types as t

from infinigen_v2.generators.shaders.composites import fabric_patterned
from infinigen_v2.generators.shaders.functionality_lists import (
    furniture_material_distribution,
)
from infinigen_v2.generators.shaders.materials import fabric
from infinigen_v2.generators.shaders.materials.emissive_nonblocking import (
    lamp_bulb_nonemissive,
)
from infinigen_v2.generators.util.curve import curve_to_mesh_with_uv


def point_light_indoor_distribution(
    rng: pf.RNG,
    energy: float | None = None,
    temperature: float | None = None,
    shadow_soft_size: float = 0.02,
) -> pf.LightObject:
    """Create a point light with blackbody temperature roughly based on indoor lighting. range 3500-6500K. was expanded to 2000-8000"""
    if temperature is None:
        temperature = pf.random.clip_gaussian(rng, 4500, 1000, 2000, 8000)
    if energy is None:
        energy = pf.random.uniform(rng, 5, 15)

    light = pf.ops.primitives.light.point_lamp(
        energy=energy,
        shadow_soft_size=shadow_soft_size,
    )
    blackbody = pf.nodes.color.blackbody(temperature=temperature)
    emission = pf.nodes.shader.emission(color=blackbody, strength=energy)
    pf.nodes.to_light(light, surface=emission)

    return light


@pf.nodes.node_function
def bulb(
    lampshade_material: t.SocketOrVal[pf.Material],
    metal_material: t.SocketOrVal[pf.Material],
) -> pf.ProcNode:
    curve_line = pf.nodes.geo.curve_line(start=(0, 0, 0), end=(0, 0, 1))

    resample_curve_count = pf.nodes.geo.resample_curve_count(
        curve=curve_line, count=100
    )

    spline_parameter = pf.nodes.geo.spline_parameter()

    curve_to_curve_radius = pf.nodes.math.float_curve(
        value=spline_parameter.factor,
        curve=np.array(
            [
                [0.0, 0.15],
                [0.05, 0.17],
                [0.15, 0.2],
                [0.55, 0.38],
                [0.8, 0.35],
                [0.9568, 0.22],
                [1.0, 0.0],
            ]
        ),
        factor=1.0,
    )

    set_curve_radius = pf.nodes.geo.set_curve_radius(
        curve=resample_curve_count,
        radius=curve_to_curve_radius,
    )

    curve_circle = pf.nodes.geo.curve_circle(100)
    curve_to = pf.nodes.geo.curve_to_mesh(
        curve=set_curve_radius, profile_curve=curve_circle
    )

    set_material = pf.nodes.geo.set_material(
        geometry=curve_to,
        material=lampshade_material,
        selection=True,
    )

    curve_line_1 = pf.nodes.geo.curve_line(start=(0.0, 0.0, -0.2), end=(0.0, 0.0, -0.3))

    resample_curve_count_1 = pf.nodes.geo.resample_curve_count(
        curve=curve_line_1, count=100
    )

    spline_parameter_1 = pf.nodes.geo.spline_parameter()

    curve_radius = pf.nodes.math.float_curve(
        value=spline_parameter_1.factor,
        curve=np.array([[0.0, 1.0], [0.4432, 0.55], [1.0, 0.275]]),
        factor=1.0,
    )

    set_curve_radius_1 = pf.nodes.geo.set_curve_radius(
        curve=resample_curve_count_1, radius=curve_radius
    )

    curve_circle_1 = pf.nodes.geo.curve_circle(resolution=100, radius=0.15)
    curve_to_1 = pf.nodes.geo.curve_to_mesh(
        curve=set_curve_radius_1,
        profile_curve=curve_circle_1,
        fill_caps=True,
    )
    curve_spiral = pf.nodes.geo.curve_spiral(
        rotations=5.0, start_radius=0.15, end_radius=0.15, height=0.2
    )

    transform_1 = pf.nodes.geo.transform(
        geometry=curve_spiral,
        translation=(0.0, 0.0, -0.2),
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )

    curve_circle_2 = pf.nodes.geo.curve_circle(resolution=100, radius=0.015)
    curve_to_2 = pf.nodes.geo.curve_to_mesh(
        curve=transform_1,
        profile_curve=curve_circle_2,
        fill_caps=True,
    )
    curve_line_2 = pf.nodes.geo.curve_line(start=(0.0, 0.0, -0.2), end=(0.0, 0.0, 0.0))
    curve_circle_3 = pf.nodes.geo.curve_circle(resolution=100, radius=0.15)
    curve_to_3 = pf.nodes.geo.curve_to_mesh(
        curve=curve_line_2,
        profile_curve=curve_circle_3,
        fill_caps=True,
    )

    join = pf.nodes.geo.join_geometry([curve_to_1, curve_to_2, curve_to_3])

    set_material_1 = pf.nodes.geo.set_material(
        geometry=join, material=metal_material, selection=True
    )

    join_1 = pf.nodes.geo.join_geometry([set_material, set_material_1])

    transform = pf.nodes.geo.transform(
        geometry=join_1,
        translation=(0.0, 0.0, 0.3),
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    return transform


@pf.nodes.node_function
def bulb_rack(
    thickness: t.SocketOrVal[float],
    inner_radius: t.SocketOrVal[float],
    outer_radius: t.SocketOrVal[float],
    inner_height: t.SocketOrVal[float],
    outer_height: t.SocketOrVal[float],
    amount: t.SocketOrVal[int] = 3,
) -> pf.ProcNode:
    curve_circle_radius = pf.nodes.math.multiply_add(
        a=thickness, b=0.5, addend=inner_radius
    )
    curve_circle = pf.nodes.geo.curve_circle(resolution=100, radius=curve_circle_radius)

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
    curve_circle_1 = pf.nodes.geo.curve_circle(resolution=100, radius=outer_radius)

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

    curve_circle_2 = pf.nodes.geo.curve_circle(resolution=100, radius=thickness)
    curve_to = pf.nodes.geo.curve_to_mesh(
        curve=join, profile_curve=curve_circle_2, fill_caps=True
    )
    return curve_to


class ReversiableBulbResult(NamedTuple):
    geometry: pf.ProcNode
    rack_support: pf.ProcNode[float]


@pf.nodes.node_function
def reversiable_bulb(
    scale: t.SocketOrVal[float],
    black_material: t.SocketOrVal[pf.Material],
    lampshade_material: t.SocketOrVal[pf.Material],
    metal_material: t.SocketOrVal[pf.Material],
    reverse: t.SocketOrVal[bool] = False,
) -> ReversiableBulbResult:
    bulb_result = bulb(
        lampshade_material=lampshade_material, metal_material=metal_material
    )

    transform_scale = pf.nodes.math.combine_xyz(x=scale, y=scale, z=scale)
    transform = pf.nodes.geo.transform(
        geometry=bulb_result,
        scale=transform_scale,
        translation=(0, 0, 0),
        rotation=(0, 0, 0),
    )

    to_instance = pf.nodes.geo.geometry_to_instance(transform)

    geometry_rotation = pf.nodes.math.combine_xyz(
        y=reverse.astype(dtype=float) * 3.1415
    )

    rotate_instances = pf.nodes.geo.rotate_instances(
        instances=to_instance,
        rotation=geometry_rotation.astype(dtype=pf.Euler),
        pivot_point=(0, 0, 0),
    )

    rack_support_b = pf.nodes.math.multiply_add(
        a=reverse.astype(dtype=float), b=2.0, addend=-1.0
    )
    rack_support = -0.015 * rack_support_b
    return ReversiableBulbResult(
        geometry=rotate_instances,
        rack_support=rack_support,
    )


@pf.nodes.node_function
def lamp_head(
    shade_height: t.SocketOrVal[float],
    top_radius: t.SocketOrVal[float],
    bot_radius: t.SocketOrVal[float],
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

    spline_parameter = pf.nodes.geo.spline_parameter()

    curve_to_curve_radius = pf.nodes.math.map_range(
        value=spline_parameter.factor,
        to_max=bot_radius,
        to_min=top_radius,
    )

    set_curve_radius = pf.nodes.geo.set_curve_radius(
        curve=curve_line, radius=curve_to_curve_radius
    )

    curve_circle = pf.nodes.geo.curve_circle(100)
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

    reversiable_bulb_result = reversiable_bulb(
        scale=geometries_2_scale,
        reverse=False,
        black_material=black_material,
        lampshade_material=lampshade_material,
        metal_material=metal_material,
    )

    bulb_rack_result = bulb_rack(
        thickness=rack_thickness,
        amount=3,
        inner_radius=geometries_2_scale * 0.15,
        outer_radius=top_radius,
        inner_height=reversiable_bulb_result.rack_support,
        outer_height=bulb_rack_outer_height,
    )

    set_material_1 = pf.nodes.geo.set_material(
        geometry=bulb_rack_result,
        material=black_material,
        selection=True,
    )

    join = pf.nodes.geo.join_geometry(
        [set_material, set_material_1, reversiable_bulb_result.geometry]
    )
    return join


class LampResult(NamedTuple):
    mesh: pf.MeshObject
    light: pf.LightObject


class LampGeometryResult(NamedTuple):
    geometry: pf.ProcNode
    bounding_box: pf.ProcNode
    light_position: pf.ProcNode[pf.Vector]


@pf.nodes.node_function
def lamp_geometry(
    stand_radius: t.SocketOrVal[float],
    base_radius: t.SocketOrVal[float],
    base_height: t.SocketOrVal[float],
    shade_height: t.SocketOrVal[float],
    head_top_radius: t.SocketOrVal[float],
    head_bot_radius: t.SocketOrVal[float],
    reverse_lamp: t.SocketOrVal[bool],
    rack_thickness: t.SocketOrVal[float],
    curve_point1: t.SocketOrVal[pf.Vector],
    curve_point2: t.SocketOrVal[pf.Vector],
    curve_point3: t.SocketOrVal[pf.Vector],
    black_material: t.SocketOrVal[pf.Material],
    lampshade_material: t.SocketOrVal[pf.Material],
    metal_material: t.SocketOrVal[pf.Material],
) -> LampGeometryResult:
    lamp_head_rack_height = pf.nodes.math.multiply_add(
        a=shade_height * 0.4,
        b=reverse_lamp.astype(dtype=float),
        addend=shade_height * 0.2,
    )
    lamp_head_result = lamp_head(
        shade_height=shade_height,
        top_radius=head_top_radius,
        bot_radius=head_bot_radius,
        reverse_bulb=reverse_lamp,
        rack_thickness=rack_thickness,
        rack_height=lamp_head_rack_height,
        black_material=black_material,
        lampshade_material=lampshade_material,
        metal_material=metal_material,
    )

    sample_curve_curves_start = pf.nodes.math.combine_xyz(z=base_height)

    curve_bezier_segment = pf.nodes.geo.curve_bezier_segment(
        resolution=100,
        start=sample_curve_curves_start,
        start_handle=curve_point1,
        end_handle=curve_point2,
        end=curve_point3,
    )

    sample_curve = pf.nodes.geo.sample_curve(
        curves=curve_bezier_segment,
        factor=1.0,
        value=0.0,
    )

    transform_rotation = pf.nodes.func.align_euler_to_vector(
        vector=sample_curve.tangent,
        axis="Z",
        rotation=(0, 0, 0),
        factor=1.0,
    )
    transform = pf.nodes.geo.transform(
        geometry=lamp_head_result,
        translation=sample_curve.position,
        rotation=transform_rotation.astype(dtype=pf.Euler),
        scale=(1, 1, 1),
    )

    curve_line = pf.nodes.geo.curve_line(end=sample_curve_curves_start, start=(0, 0, 0))

    join_1 = pf.nodes.geo.join_geometry([curve_line, curve_bezier_segment])

    curve_circle = pf.nodes.geo.curve_circle(resolution=100, radius=stand_radius)
    curve_to = pf.nodes.geo.curve_to_mesh(
        curve=join_1, profile_curve=curve_circle, fill_caps=True
    )
    curve_line_1_end = pf.nodes.math.combine_xyz(z=base_height)
    curve_line_1 = pf.nodes.geo.curve_line(end=curve_line_1_end, start=(0, 0, 0))
    curve_circle_1 = pf.nodes.geo.curve_circle(resolution=100, radius=base_radius)
    curve_to_1 = pf.nodes.geo.curve_to_mesh(
        curve=curve_line_1,
        profile_curve=curve_circle_1,
        fill_caps=True,
    )

    join_2 = pf.nodes.geo.join_geometry([curve_to, curve_to_1])

    set_material = pf.nodes.geo.set_material(
        geometry=join_2, material=black_material, selection=True
    )

    join = pf.nodes.geo.join_geometry([transform, set_material])

    bound_box = pf.nodes.geo.bound_box(join)

    curve_line_2 = pf.nodes.geo.curve_line(end=(0.0, 0.0, 0.1), start=(0, 0, 0))

    transform_1 = pf.nodes.geo.transform(
        geometry=curve_line_2,
        translation=sample_curve.position,
        rotation=transform_rotation.astype(dtype=pf.Euler),
        scale=(1, 1, 1),
    )

    sample_curve_1 = pf.nodes.geo.sample_curve(
        curves=transform_1, factor=1.0, value=0.0
    )
    return LampGeometryResult(
        geometry=join,
        bounding_box=bound_box.bounding_box,
        light_position=sample_curve_1.position,
    )


def lampshade_color_distribution(rng: pf.RNG) -> pf.Color:
    hue = pf.random.uniform(rng, 0.0, 0.2)
    saturation = pf.random.uniform(rng, 0.0, 0.25)
    value = pf.random.uniform(rng, 0.1, 0.9)
    return pf.color.hsv_color(hue=hue, saturation=saturation, value=value)


def lampshade_fabric_distribution(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
) -> pf.Material:
    rngs = rng.spawn(2)
    translucency = pf.random.uniform(rngs[0], 0.0, 0.9)

    lamp_fabric_fullcolor = partial(
        fabric.fabric_distribution,
        translucency=translucency,
        base_color=lampshade_color_distribution(rng),
    )
    lamp_fabric_patterned = partial(
        fabric_patterned.fabric_patterned_distribution,
        translucency=translucency,
    )
    lamp_fabric_lampcolor = partial(
        fabric.fabric_distribution,
        base_color=lampshade_color_distribution(rng),
        translucency=translucency,
    )
    lamp_fabric_patterned_lampcolor = partial(
        fabric_patterned.fabric_patterned_distribution,
        color1=lampshade_color_distribution(rng),
        color2=lampshade_color_distribution(rng),
        color3=lampshade_color_distribution(rng),
        translucency=translucency,
    )

    option = pf.control.choice(
        rngs[1],
        [
            (lamp_fabric_lampcolor, 2.0),
            (lamp_fabric_patterned_lampcolor, 2.0),
            (lamp_fabric_fullcolor, 1.0),
            (lamp_fabric_patterned, 0.5),
        ],
    )

    return option(rngs[1], vector)


def lamp_distribution(
    rng: pf.RNG,
    height: float | None = None,
    temperature: float | None = None,
    energy: float | None = None,
    head_top_radius: float | None = None,
    head_bot_radius: float | None = None,
) -> LampResult:
    stand_radius = pf.random.uniform(rng, 0.005, 0.015)
    base_radius = pf.random.uniform(rng, 0.05, 0.15)
    base_height = pf.random.uniform(rng, 0.01, 0.03)
    shade_height = pf.random.uniform(rng, 0.18, 0.3)
    rack_thickness = pf.random.uniform(rng, 0.001, 0.003)
    reverse_lamp = True

    if head_top_radius is None:
        head_top_radius = pf.random.uniform(rng, 0.07, 0.25)
    if head_bot_radius is None:
        head_bot_radius = head_top_radius + pf.random.uniform(rng, 0.0, 0.05)

    if height is None:
        height = pf.random.uniform(rng, 0.3, 0.6)

    z1 = pf.random.uniform(rng, base_height, height)
    z2 = pf.random.uniform(rng, z1, height)
    z3 = height
    curve_point1 = pf.Vector((0, 0, z1))
    curve_point2 = pf.Vector((0, 0, z2))
    curve_point3 = pf.Vector((0, 0, z3))

    vec = pf.nodes.shader.uv_map(uv_map="UVMap")

    stem_mat = furniture_material_distribution(rng, vec)
    lampshade_mat = lampshade_fabric_distribution(rng, vec)

    result = lamp_geometry(
        stand_radius=stand_radius,
        base_radius=base_radius,
        base_height=base_height,
        shade_height=shade_height,
        head_top_radius=head_top_radius,
        head_bot_radius=head_bot_radius,
        reverse_lamp=reverse_lamp,
        rack_thickness=rack_thickness,
        curve_point1=curve_point1,
        curve_point2=curve_point2,
        curve_point3=curve_point3,
        black_material=stem_mat,
        lampshade_material=lampshade_mat,
        metal_material=stem_mat,
    )

    lamp = pf.nodes.to_mesh_object(result.geometry)

    bulb_radius = head_top_radius * 0.3
    bulb_mat = lamp_bulb_nonemissive()
    bulb_sphere = pf.nodes.geo.mesh_uv_sphere(radius=bulb_radius, segments=32, rings=16)
    bulb_sphere = pf.nodes.geo.set_material(bulb_sphere.mesh, bulb_mat)
    bulb_obj = pf.nodes.to_mesh_object(bulb_sphere)
    bulb_obj.item().location.z = height
    bulb_obj.item().parent = lamp.item()

    if energy is None:
        energy = pf.random.clip_gaussian(rng, 7, 4, 5, 18)
    point_light = point_light_indoor_distribution(
        rng, temperature=temperature, energy=energy
    )
    point_light.item().location.z = height
    point_light.item().parent = lamp.item()

    return LampResult(mesh=lamp, light=point_light)


def desk_lamp_distribution(rng: pf.RNG) -> LampResult:
    height = pf.random.uniform(rng, 0.25, 0.4)
    return lamp_distribution(rng, height=height)


if __name__ == "__main__":
    bulb_result = bulb()
    bulb_rack_result = bulb_rack()

    reversiable_bulb_result = reversiable_bulb()

    lamp_head_result = lamp_head()
    lamp_geometry_result = lamp_geometry()
