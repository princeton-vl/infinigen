# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Hongyu Wen - original version
# - Alexander Raistrick - add point light, refactor to procfunc

from typing import NamedTuple

import procfunc as pf
from procfunc.nodes import types as t

from infinigen_v2.generators.objects.lamp import point_light_indoor_distribution
from infinigen_v2.generators.shaders.materials.emissive_nonblocking import (
    lamp_bulb_nonemissive,
)
from infinigen_v2.generators.shaders.materials.metal_brushed import (
    metal_brushed_radial_distribution,
)
from infinigen_v2.generators.shaders.materials.plastic import (
    plastic_grayscale_distribution,
)
from infinigen_v2.generators.shaders.materials.wood_grain import wood_grain_distribution


class CeilingLightGeometryResult(NamedTuple):
    geometry: pf.ProcNode
    bounding_box: pf.ProcNode


class CeilingLightResult(NamedTuple):
    mesh: pf.MeshObject
    light: pf.LightObject | None


@pf.nodes.node_function
def ceiling_light_geometry(
    radius: t.SocketOrVal[float],
    thickness: t.SocketOrVal[float],
    inner_radius: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    inner_height: t.SocketOrVal[float],
    curvature: t.SocketOrVal[float],
    frame_material: t.SocketOrVal[pf.Material],
    bulb_material: t.SocketOrVal[pf.Material],
) -> CeilingLightGeometryResult:
    transform_translation_z = inner_height * -1.0

    curve_line_end = pf.nodes.math.combine_xyz(z=transform_translation_z)
    curve_line = pf.nodes.geo.curve_line(start=(0.0, 0.0, -0.001), end=curve_line_end)
    curve_circle = pf.nodes.geo.curve_circle(radius=inner_radius)
    curve_to = pf.nodes.geo.curve_to_mesh(
        curve=curve_line,
        profile_curve=curve_circle,
        fill_caps=True,
    )

    icosphere = pf.nodes.geo.mesh_icosphere(radius=inner_radius, subdivisions=5)

    store_named_attribute = pf.nodes.geo.store_named_attribute(
        geometry=icosphere.mesh,
        name="UVMap",
        value=icosphere.uv_map,
        domain="CORNER",
        data_type="FLOAT2",
    )

    input_position = pf.nodes.geo.input_position()

    separate_selection = input_position.z < 0.001
    separate = pf.nodes.geo.separate_geometry(
        geometry=store_named_attribute,
        selection=separate_selection.astype(dtype=bool),
    )

    transform_translation = pf.nodes.math.combine_xyz(z=transform_translation_z)
    transform_scale = pf.nodes.math.combine_xyz(x=1.0, y=1.0, z=curvature)
    transform = pf.nodes.geo.transform(
        geometry=separate.selection,
        translation=transform_translation,
        scale=transform_scale,
        rotation=(0, 0, 0),
    )

    join_1 = pf.nodes.geo.join_geometry([curve_to, transform])

    set_material = pf.nodes.geo.set_material(
        geometry=join_1, material=bulb_material, selection=True
    )

    circle = pf.nodes.geo.mesh_circle(radius=radius, fill_type="NGON")

    curve_line_1_end = pf.nodes.math.combine_xyz(z=height * -1.0)
    curve_line_1 = pf.nodes.geo.curve_line(end=curve_line_1_end, start=(0, 0, 0))
    curve_circle_1 = pf.nodes.geo.curve_circle(resolution=512, radius=radius)
    curve_to_1 = pf.nodes.geo.curve_to_mesh(
        curve=curve_line_1, profile_curve=curve_circle_1
    )

    flip_faces = pf.nodes.geo.flip_faces(curve_to_1)

    extrude = pf.nodes.geo.extrude_mesh(
        mesh=curve_to_1, offset_scale=thickness, individual=False
    )

    join_2 = pf.nodes.geo.join_geometry([flip_faces, extrude.mesh])

    set_shade_smooth = pf.nodes.geo.set_shade_smooth(
        geometry=join_2, shade_smooth=False
    )

    join_3 = pf.nodes.geo.join_geometry([circle, set_shade_smooth])

    set_material_1 = pf.nodes.geo.set_material(
        geometry=join_3, material=frame_material, selection=True
    )

    join = pf.nodes.geo.join_geometry([set_material, set_material_1])

    bound_box = pf.nodes.geo.bound_box(join)
    return CeilingLightGeometryResult(
        geometry=join,
        bounding_box=bound_box.bounding_box,
    )


@pf.nodes.node_function
def black_for_reflections(
    shader: pf.ProcNode[pf.Shader],
) -> pf.Material:
    """Wrap a shader to appear black for reflected/glossy rays."""
    light_path = pf.nodes.shader.light_path()
    surface = pf.nodes.shader.mix_shader(
        factor=light_path.is_camera_ray,
        a=None,
        b=shader,
    )
    return pf.Material(surface=surface)


def lamp_material_distribution(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
) -> pf.Material:
    material_func = pf.control.choice(
        rng,
        [
            (plastic_grayscale_distribution, 0.2),
            (wood_grain_distribution, 0.2),
            (metal_brushed_radial_distribution, 0.2),
        ],
    )
    return material_func(rng, vector)


def ceiling_light_distribution(
    rng: pf.RNG,
    energy: float | None = None,
    shadow_soft_size: float | None = None,
    temperature: float | None = None,
    turned_on: bool = True,
) -> CeilingLightResult:
    radius = pf.random.clip_gaussian(rng, 0.12, 0.04, 0.1, 0.25)
    thickness = pf.random.uniform(rng, 0.005, 0.05)
    inner_radius = radius * pf.random.uniform(rng, 0.4, 0.9)
    height = 0.7 * pf.random.clip_gaussian(rng, 0.09, 0.03, 0.07, 0.15)
    inner_height = height * pf.random.uniform(rng, 0.5, 1.1)
    curvature = pf.random.uniform(rng, 0.1, 0.5)

    vec = pf.nodes.shader.coord().uv
    frame_shader = lamp_material_distribution(rng, vec)
    frame_shader = black_for_reflections(frame_shader.surface)
    frame_material = frame_shader

    bulb_material = lamp_bulb_nonemissive()

    geo = ceiling_light_geometry(
        radius=radius,
        thickness=thickness,
        inner_radius=inner_radius,
        height=height,
        inner_height=inner_height,
        curvature=curvature,
        frame_material=frame_material,
        bulb_material=bulb_material,
    )

    obj = pf.nodes.to_mesh_object(geo.geometry)
    pf.ops.uv.cylinder_project(obj)

    light = None
    if turned_on:
        if energy is None:
            energy = pf.random.uniform(rng, 50, 100)
        if shadow_soft_size is None:
            shadow_soft_size = pf.random.uniform(rng, 0.02, 0.03)

        light = point_light_indoor_distribution(
            rng,
            energy=energy,
            temperature=temperature,
            shadow_soft_size=shadow_soft_size if shadow_soft_size is not None else 0.0,
        )
        light.item().location.z = -0.03

    return CeilingLightResult(mesh=obj, light=light)
