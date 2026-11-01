# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Hongyu Wen: original Infinigen ceiling light (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/objects/lamp/ceiling_lights.py)
# - Alexander Raistrick: refactor for Infinigen2

from typing import NamedTuple

import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.objects.lamp import light_point_indoor
from infinigen2.shaders.base_materials.emissive_nonblocking import (
    lamp_bulb_nonemissive,
)
from infinigen2.shaders.functionality_lists import (
    furniture_material_rand,
)
from infinigen2.util.curve import curve_to_mesh_with_uv
from infinigen2.util.mesh import extrude_mesh_seamless_uvs

__all__ = [
    "CeilingLightResult",
    "black_for_reflections",
    "light_ceiling",
    "light_ceiling_rand",
]


class _CeilingLightGeometryResult(NamedTuple):
    geometry: pf.ProcNode
    bounding_box: pf.ProcNode


class CeilingLightResult(NamedTuple):
    mesh: pf.MeshObject
    light: pf.LightObject | None


@pf.nodes.node_function
def _ceiling_light_geometry(
    radius: t.SocketOrVal[float],
    thickness: t.SocketOrVal[float],
    inner_radius: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    inner_height: t.SocketOrVal[float],
    curvature: t.SocketOrVal[float],
    frame_material: t.SocketOrVal[pf.Material],
    bulb_material: t.SocketOrVal[pf.Material],
) -> _CeilingLightGeometryResult:
    transform_translation_z = inner_height * -1.0

    curve_line_end = pf.nodes.math.combine_xyz(z=transform_translation_z)
    curve_line = pf.nodes.geo.curve_line(start=(0.0, 0.0, -0.001), end=curve_line_end)
    curve_circle = pf.nodes.geo.curve_circle(resolution=16, radius=inner_radius)
    curve_to = curve_to_mesh_with_uv(
        curve=curve_line,
        profile=curve_circle,
        fill_caps=True,
    ).mesh

    icosphere = pf.nodes.geo.mesh_icosphere(radius=inner_radius, subdivisions=3)

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

    circle = pf.nodes.geo.mesh_circle(vertices=16, radius=radius, fill_type="NGON")
    circle_position = pf.nodes.geo.input_position()
    circle = pf.nodes.geo.store_named_attribute(
        geometry=circle,
        name="UVMap",
        value=pf.nodes.math.combine_xyz(
            x=circle_position.x,
            y=circle_position.y,
        ),
        domain="CORNER",
        data_type="FLOAT2",
    )

    curve_line_1_end = pf.nodes.math.combine_xyz(z=height * -1.0)
    curve_line_1 = pf.nodes.geo.curve_line(end=curve_line_1_end, start=(0, 0, 0))
    curve_line_1 = pf.nodes.geo.resample_curve_count(curve=curve_line_1, count=4)
    curve_circle_1 = pf.nodes.geo.curve_circle(resolution=16, radius=radius)
    curve_to_1 = curve_to_mesh_with_uv(curve=curve_line_1, profile=curve_circle_1).mesh

    flip_faces = pf.nodes.geo.flip_faces(curve_to_1)

    extrude = extrude_mesh_seamless_uvs(
        mesh=curve_to_1,
        selection=True,
        offset_scale=thickness,
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
    return _CeilingLightGeometryResult(
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


@pf.tracer.generator
def light_ceiling(
    radius: float = 0.12,
    thickness: float = 0.0275,
    inner_radius: float = 0.078,
    height: float = 0.063,
    inner_height: float = 0.0504,
    curvature: float = 0.3,
    energy: float = 9.0,
    temperature: float = 4500.0,
    shadow_soft_size: float = 0.025,
    turned_on: bool = True,
    frame_material: pf.Material | None = None,
    bulb_material: pf.Material | None = None,
) -> CeilingLightResult:
    if frame_material is None:
        frame_material = black_for_reflections(pf.nodes.shader.principled_bsdf())
    if bulb_material is None:
        bulb_material = lamp_bulb_nonemissive()

    geo = _ceiling_light_geometry(
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

    light = None
    if turned_on:
        light = light_point_indoor(
            energy=energy,
            temperature=temperature,
            shadow_soft_size=shadow_soft_size,
        )
        pf.ops.object.set_transform(light, location=(0.0, 0.0, -0.03))

    return CeilingLightResult(mesh=obj, light=light)


def light_ceiling_rand(
    rng: pf.RNG,
    energy: float | None = None,
    shadow_soft_size: float | None = None,
    temperature: float | None = None,
    turned_on: bool = True,
) -> CeilingLightResult:
    rng, rng_frame_shader = rng.spawn(2)
    radius = pf.random.clip_gaussian(rng, 0.12, 0.04, 0.1, 0.25)
    thickness = pf.random.uniform(rng, 0.005, 0.05)
    inner_radius = radius * pf.random.uniform(rng, 0.4, 0.9)
    height = 0.7 * pf.random.clip_gaussian(rng, 0.09, 0.03, 0.07, 0.15)
    inner_height = height * pf.random.uniform(rng, 0.5, 1.1)
    curvature = pf.random.uniform(rng, 0.1, 0.5)

    vec = pf.nodes.shader.coord().uv
    frame_shader = furniture_material_rand(rng_frame_shader, vec)
    frame_shader = black_for_reflections(frame_shader.surface)
    frame_material = frame_shader

    bulb_material = lamp_bulb_nonemissive()

    if not turned_on:
        result = light_ceiling(
            radius=radius,
            thickness=thickness,
            inner_radius=inner_radius,
            height=height,
            inner_height=inner_height,
            curvature=curvature,
            turned_on=False,
            frame_material=frame_material,
            bulb_material=bulb_material,
        )
        mesh = result.mesh
        pf.ops.modifier.subdivide_surface(mesh, levels=2, _skip_apply=True)
        return CeilingLightResult(mesh=mesh, light=result.light)

    if energy is None:
        energy = pf.random.uniform(rng, 800, 3000) / 177
    if shadow_soft_size is None:
        shadow_soft_size = pf.random.uniform(rng, 0.02, 0.03)
    rng_temperature, _rng_energy = rng.spawn(2)
    if temperature is None:
        temperature = pf.random.clip_gaussian(rng_temperature, 4500, 1000, 2000, 8000)

    result = light_ceiling(
        radius=radius,
        thickness=thickness,
        inner_radius=inner_radius,
        height=height,
        inner_height=inner_height,
        curvature=curvature,
        energy=energy,
        temperature=temperature,
        shadow_soft_size=shadow_soft_size,
        turned_on=True,
        frame_material=frame_material,
        bulb_material=bulb_material,
    )
    mesh = result.mesh
    pf.ops.modifier.subdivide_surface(mesh, levels=2, _skip_apply=True)
    return CeilingLightResult(mesh=mesh, light=result.light)
